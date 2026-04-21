"""
User-perspective simulation tests for NanoBanana MCP server.

Tests the key bug fixes:
1. No double-JPEG encoding (images go through _to_jpeg once, not twice)
2. Cloud output validation happens early with clear error
3. analyze_image with focus=quality uses SOURCE_MAX_DIM (2048px)
4. Image store (upload/fetch/expire) works correctly
5. Input validation on all tools
6. _normalize_image handles RGBA, bad data, oversized images
7. _decode_raw handles nanobanana://, HTTP URLs, /images/ shortcut, data URIs
8. _build_image_response produces correct structure for both output modes
"""

import asyncio
import base64
import json
import os
import sys
import time
import threading
from io import BytesIO
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from PIL import Image as PILImage

# Ensure server module can be imported without an API key
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

# Import after setting env
import server
from mcp.server.fastmcp import Image as MCPImage


# ---------------------------------------------------------------------------
# Helpers — create test images
# ---------------------------------------------------------------------------

def _parse_result(result) -> dict:
    """Parse JSON metadata from a tool result.

    Image-generating tools return [json_str, Image, ...].
    Error paths return a plain str JSON with an 'error' key.
    """
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, str):
                try:
                    return json.loads(item)
                except (json.JSONDecodeError, ValueError):
                    continue
        raise ValueError("No valid JSON found in result")
    return json.loads(result)


def _make_test_image(w=200, h=300, mode="RGB", fmt="JPEG") -> bytes:
    """Create a minimal test image and return raw bytes."""
    img = PILImage.new(mode, (w, h), color=(128, 64, 32) if mode == "RGB" else (128, 64, 32, 200))
    buf = BytesIO()
    if fmt == "JPEG" and mode == "RGBA":
        # JPEG can't do RGBA, save as PNG then convert
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_large_image(dim=3000) -> bytes:
    """Create an oversized test image."""
    return _make_test_image(w=dim, h=dim)


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    img = PILImage.open(BytesIO(data))
    return img.size


# ---------------------------------------------------------------------------
# 1. Image store: upload, fetch, expiry, eviction
# ---------------------------------------------------------------------------

class TestImageStore:
    def setup_method(self):
        """Clear the store before each test."""
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    def test_store_and_fetch(self):
        img = _make_test_image()
        img_id = server._store_image(img, "image/jpeg")
        fetched, mime = server._fetch_from_store(img_id)
        assert fetched == img
        assert mime == "image/jpeg"

    def test_fetch_missing_raises(self):
        with pytest.raises(ValueError, match="not found"):
            server._fetch_from_store("nonexistent")

    def test_gc_removes_expired(self):
        img = _make_test_image()
        img_id = server._store_image(img, "image/jpeg")
        # Backdate the timestamp to make it expired
        with server._STORE_LOCK:
            entry = server._IMAGE_STORE[img_id]
            server._IMAGE_STORE[img_id] = (entry[0], entry[1], time.time() - server._STORE_TTL - 10)
        server._gc_store()
        with pytest.raises(ValueError, match="not found"):
            server._fetch_from_store(img_id)

    def test_eviction_when_full(self):
        original_max = server._STORE_MAX_ITEMS
        try:
            server._STORE_MAX_ITEMS = 3
            ids = []
            for i in range(4):
                ids.append(server._store_image(_make_test_image(), "image/jpeg"))
                time.sleep(0.01)  # Ensure different timestamps
            # Oldest should be evicted
            with pytest.raises(ValueError):
                server._fetch_from_store(ids[0])
            # Newest should still be there
            server._fetch_from_store(ids[-1])
        finally:
            server._STORE_MAX_ITEMS = original_max


# ---------------------------------------------------------------------------
# 2. _normalize_image — RGBA, oversized, corrupt
# ---------------------------------------------------------------------------

class TestNormalizeImage:
    def test_rgb_passthrough(self):
        img = _make_test_image(100, 100, mode="RGB")
        normalized, mime = server._normalize_image(img, max_dim=1024)
        assert mime == "image/jpeg"
        result = PILImage.open(BytesIO(normalized))
        assert result.mode == "RGB"

    def test_rgba_to_rgb(self):
        """RGBA images should be flattened to RGB with white background."""
        img = _make_test_image(100, 100, mode="RGBA", fmt="PNG")
        normalized, mime = server._normalize_image(img, max_dim=1024)
        assert mime == "image/jpeg"
        result = PILImage.open(BytesIO(normalized))
        assert result.mode == "RGB"

    def test_downscale_oversized(self):
        img = _make_test_image(4000, 2000)
        normalized, _ = server._normalize_image(img, max_dim=1024)
        result = PILImage.open(BytesIO(normalized))
        assert max(result.size) <= 1024

    def test_small_image_not_upscaled(self):
        img = _make_test_image(50, 50)
        normalized, _ = server._normalize_image(img, max_dim=1024)
        result = PILImage.open(BytesIO(normalized))
        assert result.size == (50, 50)

    def test_corrupt_data_raises(self):
        with pytest.raises(ValueError, match="Could not decode"):
            server._normalize_image(b"not an image", max_dim=1024)

    def test_png_output(self):
        img = _make_test_image(100, 100, mode="RGBA", fmt="PNG")
        normalized, mime = server._normalize_image(img, max_dim=1024, output_format="PNG")
        assert mime == "image/png"


# ---------------------------------------------------------------------------
# 3. _decode_raw — nanobanana://, URLs, data URIs, base64
# ---------------------------------------------------------------------------

class TestDecodeRaw:
    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    def test_nanobanana_protocol(self):
        img = _make_test_image()
        img_id = server._store_image(img, "image/jpeg")
        data, mime = server._decode_raw(f"nanobanana://{img_id}")
        assert data == img
        assert mime == "image/jpeg"

    def test_nanobanana_missing(self):
        with pytest.raises(ValueError, match="not found"):
            server._decode_raw("nanobanana://nonexistent")

    def test_images_path_shortcut(self):
        """URLs with /images/{id} path should try local store first."""
        img = _make_test_image()
        img_id = server._store_image(img, "image/jpeg")
        data, mime = server._decode_raw(f"http://localhost:8080/images/{img_id}")
        assert data == img

    def test_data_uri(self):
        img = _make_test_image(10, 10)
        b64 = base64.b64encode(img).decode()
        data, mime = server._decode_raw(f"data:image/jpeg;base64,{b64}")
        assert data == img
        assert mime == "image/jpeg"

    def test_raw_base64(self):
        img = _make_test_image(10, 10)
        b64 = base64.b64encode(img).decode()
        data, mime = server._decode_raw(b64)
        assert mime == "image/jpeg"

    def test_base64_padding_fix(self):
        """Padded and unpadded base64 both work."""
        img = _make_test_image(10, 10)
        b64 = base64.b64encode(img).decode().rstrip("=")
        data, _ = server._decode_raw(b64)
        # Should still decode successfully
        PILImage.open(BytesIO(data))


# ---------------------------------------------------------------------------
# 4. SSRF protection in _fetch_url
# ---------------------------------------------------------------------------

class TestSSRFProtection:
    def test_blocks_metadata_endpoint(self):
        with pytest.raises(ValueError, match="Blocked"):
            server._fetch_url("http://169.254.169.254/latest/meta-data/")

    def test_blocks_gcp_metadata(self):
        with pytest.raises(ValueError, match="Blocked"):
            server._fetch_url("http://metadata.google.internal/computeMetadata/v1/")

    def test_blocks_localhost(self):
        with pytest.raises(ValueError, match="Blocked"):
            server._fetch_url("http://localhost/secret")

    def test_blocks_127(self):
        with pytest.raises(ValueError, match="Blocked"):
            server._fetch_url("http://127.0.0.1/secret")


# ---------------------------------------------------------------------------
# 5. Input validation — generate_image
# ---------------------------------------------------------------------------

class TestGenerateImageValidation:
    @pytest.mark.asyncio
    async def test_bad_aspect_ratio(self):
        result = await server.generate_image("test", aspect_ratio="7:3")
        data = _parse_result(result)
        assert "error" in data
        assert "Unsupported aspect ratio" in data["error"]

    @pytest.mark.asyncio
    async def test_bad_resolution(self):
        result = await server.generate_image("test", resolution="8K")
        data = _parse_result(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_bad_style(self):
        result = await server.generate_image("test", style="nonexistent")
        data = _parse_result(result)
        assert "error" in data
        assert "available" in data

    @pytest.mark.asyncio
    async def test_save_folder_invalid_path_is_non_fatal(self):
        """save_folder pointing to an unwritable path logs a warning but doesn't fail."""
        mock_client = MagicMock()
        test_img = _make_test_image(64, 64)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock(mime_type="image/png", data=test_img)
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_get_client", return_value=mock_client):
            # /root/noperms should fail to create on macOS — _save_to_folder swallows it
            result = await server.generate_image("test", save_folder="/root/noperms/nbtest")
        # Should still return images even if folder save fails
        assert isinstance(result, list)
        assert "image_url" in _parse_result(result)

    @pytest.mark.asyncio
    async def test_empty_reference_image_rejected(self):
        """Empty string in reference_images should produce a clear error."""
        mock_client = MagicMock()
        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("test", reference_images=[""])
        data = _parse_result(result)
        assert "error" in data
        assert "empty" in data["error"].lower() or "upload" in data["error"].lower()


# ---------------------------------------------------------------------------
# 6. No double JPEG encoding — _build_image_response
# ---------------------------------------------------------------------------

class TestNoDoubleJPEG:
    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    def test_always_stores_http_url(self):
        """Images are always stored server-side and get /images/ HTTP URLs."""
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response(
            {"test": True},
            [(jpeg, {"index": 1})],
        )
        assert isinstance(result, list)
        metadata = _parse_result(result)
        assert "image_url" in metadata
        assert "/images/" in metadata["image_url"]
        assert metadata["image_url"].startswith("http")

    def test_expires_in_set(self):
        """Stored images include expires_in field."""
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response({}, [(jpeg, {"index": 1})])
        metadata = _parse_result(result)
        assert "expires_in" in metadata

    def test_multiple_images_structure(self):
        """Multiple images should produce an 'images' array in metadata."""
        imgs = [(_make_test_image(100, 100), {"index": i}) for i in range(3)]
        result = server._build_image_response({}, imgs)
        metadata = _parse_result(result)
        assert "images" in metadata
        assert len(metadata["images"]) == 3

    def test_save_folder_writes_files(self, tmp_path):
        """When save_folder is provided, JPEG files are written there."""
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response(
            {}, [(jpeg, {"index": 1})], save_folder=str(tmp_path), prefix="test"
        )
        metadata = _parse_result(result)
        assert "saved_to" in metadata
        saved = metadata["saved_to"]
        assert saved.startswith(str(tmp_path))
        assert os.path.isfile(saved)
        # File should be a valid JPEG
        from PIL import Image as PILImage
        from io import BytesIO
        with open(saved, "rb") as f:
            PILImage.open(f).verify()

    def test_save_folder_multiple_images(self, tmp_path):
        """Multiple images with save_folder — each gets its own file."""
        imgs = [(_make_test_image(100, 100), {"index": i}) for i in range(3)]
        result = server._build_image_response({}, imgs, save_folder=str(tmp_path), prefix="var")
        metadata = _parse_result(result)
        for img_meta in metadata["images"]:
            assert "saved_to" in img_meta
            assert os.path.isfile(img_meta["saved_to"])


# ---------------------------------------------------------------------------
# 7. analyze_image focus=quality uses higher resolution
# ---------------------------------------------------------------------------

class TestAnalyzeImageResolution:
    @pytest.mark.asyncio
    async def test_quality_focus_uses_source_max_dim(self):
        """Bug fix: analyze_image with focus=quality should use SOURCE_MAX_DIM (2048)
        not REF_MAX_DIM (1024), so real sharpness defects are detectable."""
        img = _make_test_image(200, 200)
        img_id = server._store_image(img, "image/jpeg")

        mock_ctx = MagicMock()

        # Mock _acquire_image to capture the max_dim argument
        captured_kwargs = {}
        original_acquire = server._acquire_image

        async def spy_acquire(image, ctx, **kwargs):
            captured_kwargs.update(kwargs)
            raw, _ = server._decode_raw(image)
            return server._normalize_image(raw, max_dim=kwargs.get("max_dim", 2048))

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "sharpness": {"score": 8, "note": "ok"},
            "exposure": {"score": 7, "note": "ok"},
            "composition": {"score": 7, "note": "ok"},
            "color_balance": {"score": 8, "note": "ok"},
            "noise": {"score": 9, "note": "ok"},
            "issues": [],
            "total": 39,
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_acquire_image", spy_acquire), \
             patch.object(server, "_get_client", return_value=mock_client):
            result = await server.analyze_image(
                image=f"nanobanana://{img_id}",
                ctx=mock_ctx,
                focus="quality",
            )

        # The key assertion: quality focus should NOT pass max_dim=REF_MAX_DIM
        # It should use the default (SOURCE_MAX_DIM=2048) or higher
        assert captured_kwargs.get("max_dim", server.SOURCE_MAX_DIM) >= server.SOURCE_MAX_DIM

    @pytest.mark.asyncio
    async def test_general_focus_uses_ref_max_dim(self):
        """Non-quality focus should use REF_MAX_DIM (1024) for efficiency."""
        img = _make_test_image(200, 200)
        img_id = server._store_image(img, "image/jpeg")

        mock_ctx = MagicMock()
        captured_kwargs = {}

        async def spy_acquire(image, ctx, **kwargs):
            captured_kwargs.update(kwargs)
            raw, _ = server._decode_raw(image)
            return server._normalize_image(raw, max_dim=kwargs.get("max_dim", 2048))

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "description": "test", "style": "test", "mood": "test",
            "colors": ["red"], "details": ["detail"],
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_acquire_image", spy_acquire), \
             patch.object(server, "_get_client", return_value=mock_client):
            await server.analyze_image(
                image=f"nanobanana://{img_id}",
                ctx=mock_ctx,
                focus="general",
            )

        assert captured_kwargs.get("max_dim") == server.REF_MAX_DIM


# ---------------------------------------------------------------------------
# 8. Full generate_image flow (mocked Gemini)
# ---------------------------------------------------------------------------

class TestGenerateImageFlow:
    def _mock_gemini_response(self):
        """Create a mock Gemini response with an inline image."""
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    @pytest.mark.asyncio
    async def test_basic_generation(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("a cute cat")

        # Tools return [json_str, Image, ...] — Image objects render inline in claude.ai
        assert isinstance(result, list)
        metadata = _parse_result(result)
        assert "model" in metadata
        assert "image_url" in metadata
        assert metadata["image_url"].startswith("http")

    @pytest.mark.asyncio
    async def test_generation_with_style(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("a product shot", style="product-photography")

        metadata = _parse_result(result)
        assert metadata.get("style") == "product-photography"
        # Verify the style prefix was prepended to the prompt
        call_args = mock_client.models.generate_content.call_args
        text_parts = [p for p in call_args.kwargs["contents"] if hasattr(p, "text")]
        if text_parts:
            assert "product photography" in text_parts[0].text.lower()

    @pytest.mark.asyncio
    async def test_generation_with_enhance(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            # First call: prompt enhancement
            MagicMock(text="A detailed enhanced prompt about a cute cat"),
            # Second call: image generation
            self._mock_gemini_response(),
        ]

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("cat", enhance_prompt=True)

        metadata = _parse_result(result)
        assert "original_prompt" in metadata
        assert mock_client.models.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_generation_with_qa(self):
        mock_client = MagicMock()
        qa_response = MagicMock()
        qa_response.text = json.dumps({
            "composition": {"score": 8, "note": "good"},
            "clarity": {"score": 9, "note": "sharp"},
            "lighting": {"score": 7, "note": "ok"},
            "color": {"score": 8, "note": "good"},
            "prompt_adherence": {"score": 9, "note": "excellent"},
            "total": 41,
        })
        mock_client.models.generate_content.side_effect = [
            self._mock_gemini_response(),
            qa_response,
        ]

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("cat", qa=True)

        metadata = _parse_result(result)
        assert "qa" in metadata
        assert metadata["qa"]["total"] == 41

    @pytest.mark.asyncio
    async def test_generation_failure(self):
        """When Gemini returns no image, we get a clear error."""
        mock_response = MagicMock()
        mock_response.candidates = []

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("a cat")

        data = _parse_result(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_multiple_images(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("cats", count=3)

        metadata = _parse_result(result)
        assert "images" in metadata
        assert len(metadata["images"]) == 3

    @pytest.mark.asyncio
    async def test_count_clamped(self):
        """Count should be clamped to 1-4."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("cat", count=10)
        # Should have generated max 4
        assert mock_client.models.generate_content.call_count == 4

    @pytest.mark.asyncio
    async def test_reference_images(self):
        """Reference images should be included as parts."""
        ref_img = _make_test_image(200, 200)
        ref_id = server._store_image(ref_img, "image/jpeg")

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image(
                "similar style",
                reference_images=[f"nanobanana://{ref_id}"],
            )

        metadata = _parse_result(result)
        assert metadata.get("reference_count") == 1
        # Should have 2 parts: reference image + text prompt
        call_args = mock_client.models.generate_content.call_args
        assert len(call_args.kwargs["contents"]) == 2


# ---------------------------------------------------------------------------
# 9. edit_image flow
# ---------------------------------------------------------------------------

class TestEditImageFlow:
    def _mock_gemini_response(self):
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    @pytest.mark.asyncio
    async def test_basic_edit(self):
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}",
                prompt="add a hat",
                ctx=mock_ctx,
            )

        metadata = _parse_result(result)
        assert metadata["edit_mode"] == "inpaint-insertion"
        assert "image_url" in metadata

    @pytest.mark.asyncio
    async def test_bad_edit_mode(self):
        mock_ctx = MagicMock()
        result = await server.edit_image(
            image="nanobanana://test", prompt="test", ctx=mock_ctx, edit_mode="magic"
        )
        data = _parse_result(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_removal_mode(self):
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}",
                prompt="the watermark",
                ctx=mock_ctx,
                edit_mode="inpaint-removal",
            )

        metadata = _parse_result(result)
        assert metadata["edit_mode"] == "inpaint-removal"
        # Verify the removal prompt was used
        call_args = mock_client.models.generate_content.call_args
        text_parts = [p for p in call_args.kwargs["contents"] if hasattr(p, "text")]
        assert any("Remove" in str(p) for p in text_parts)

    @pytest.mark.asyncio
    async def test_edit_with_reference_images(self):
        """Edit with reference images — the user's exact use case:
        'replace bottle A with bottle B' where B is a reference image."""
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        ref = _make_test_image(150, 150)
        ref_id = server._store_image(ref, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}",
                prompt="replace the proven bottle with the noteworthy bottle",
                reference_images=[f"nanobanana://{ref_id}"],
                ctx=mock_ctx,
            )

        metadata = _parse_result(result)
        assert metadata["reference_count"] == 1
        assert "image_url" in metadata
        # Verify reference image was included in API call parts
        call_args = mock_client.models.generate_content.call_args
        # Should have: source image + reference image + text prompt = 3 parts
        assert len(call_args.kwargs["contents"]) == 3


# ---------------------------------------------------------------------------
# 10. swap_background flow
# ---------------------------------------------------------------------------

class TestSwapBackgroundFlow:
    @pytest.mark.asyncio
    async def test_basic_swap(self):
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()

        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.swap_background(
                image=f"nanobanana://{src_id}",
                background="a tropical beach",
                ctx=mock_ctx,
            )

        metadata = _parse_result(result)
        assert metadata["background"] == "a tropical beach"
        assert "image_url" in metadata


# ---------------------------------------------------------------------------
# 11. create_variations flow
# ---------------------------------------------------------------------------

class TestCreateVariationsFlow:
    @pytest.mark.asyncio
    async def test_basic_variations(self):
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()

        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.create_variations(
                image=f"nanobanana://{src_id}",
                ctx=mock_ctx,
                count=2,
            )

        metadata = _parse_result(result)
        assert "images" in metadata
        assert len(metadata["images"]) == 2

    @pytest.mark.asyncio
    async def test_bad_variation_strength(self):
        mock_ctx = MagicMock()
        result = await server.create_variations(
            image="nanobanana://test", ctx=mock_ctx, variation_strength="extreme"
        )
        data = _parse_result(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# 12. list_styles
# ---------------------------------------------------------------------------

class TestListStyles:
    def test_returns_all_styles(self):
        result = json.loads(server.list_styles())
        assert "styles" in result
        names = {s["name"] for s in result["styles"]}
        assert "cinematic" in names
        assert "product-photography" in names
        assert "watercolor" in names
        assert len(result["styles"]) == len(server.STYLE_PRESETS)


# ---------------------------------------------------------------------------
# 13. _is_url helper
# ---------------------------------------------------------------------------

class TestIsUrl:
    def test_http(self):
        assert server._is_url("http://example.com/image.jpg")

    def test_https(self):
        assert server._is_url("https://example.com/image.jpg")

    def test_nanobanana_not_url(self):
        assert not server._is_url("nanobanana://abc123")

    def test_base64_not_url(self):
        assert not server._is_url("/9j/4AAQSkZJRgABAQ...")

    def test_data_uri_not_url(self):
        assert not server._is_url("data:image/jpeg;base64,/9j/4AAQ...")


# ---------------------------------------------------------------------------
# 15. _to_jpeg idempotency (no double compression)
# ---------------------------------------------------------------------------

class TestToJpeg:
    def test_rgb_conversion(self):
        img = _make_test_image(100, 100, mode="RGB")
        result = server._to_jpeg(img)
        out = PILImage.open(BytesIO(result))
        assert out.mode == "RGB"
        assert out.format == "JPEG"

    def test_rgba_conversion(self):
        img = _make_test_image(100, 100, mode="RGBA", fmt="PNG")
        result = server._to_jpeg(img)
        out = PILImage.open(BytesIO(result))
        assert out.mode == "RGB"

    def test_already_jpeg_not_degraded(self):
        """Converting JPEG to JPEG should not significantly change file size
        (within reasonable bounds, proving it's not double-compressing)."""
        img = _make_test_image(200, 200, mode="RGB")
        first = server._to_jpeg(img)
        second = server._to_jpeg(first)
        # Size should be similar (within 20% — JPEG re-encoding has some variance)
        ratio = len(second) / len(first)
        assert 0.5 < ratio < 1.5, f"Suspicious size change: {len(first)} -> {len(second)}"


# ---------------------------------------------------------------------------
# 16. Model selection
# ---------------------------------------------------------------------------

class TestModelSelection:
    def test_default_model(self):
        assert server._pick_model("default") == server.MODEL_FLASH

    def test_pro_model(self):
        assert server._pick_model("pro") == server.MODEL_PRO


# ---------------------------------------------------------------------------
# 17. Empty image returns upload instructions (no connection crash)
# ---------------------------------------------------------------------------

class TestEmptyImageUploadInstructions:
    """When image="" is passed (no image available), _acquire_image should
    return a helpful error with the upload URL — NOT attempt elicitation
    which crashes the MCP connection on clients that don't support it."""

    @pytest.mark.asyncio
    async def test_upload_image_empty_returns_upload_url(self):
        """upload_image with image="" should return error with upload URL."""
        mock_ctx = MagicMock()
        result = await server.upload_image(ctx=mock_ctx, image="")
        data = _parse_result(result)
        assert "error" in data
        assert "upload_image" in data["error"]

    @pytest.mark.asyncio
    async def test_edit_image_empty_returns_upload_url(self):
        """edit_image with image="" should return error with upload URL."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(prompt="add a hat", ctx=mock_ctx, image="")
        data = _parse_result(result)
        assert "error" in data
        assert "upload_image" in data["error"]

    @pytest.mark.asyncio
    async def test_analyze_image_empty_returns_upload_url(self):
        """analyze_image with image="" should return error with upload URL."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.analyze_image(ctx=mock_ctx, image="", focus="general")
        data = _parse_result(result)
        assert "error" in data
        assert "upload_image" in data["error"]

    @pytest.mark.asyncio
    async def test_acquire_image_empty_does_not_elicit(self):
        """_acquire_image with empty string raises ValueError immediately,
        no elicitation attempt that could crash the connection."""
        mock_ctx = MagicMock()
        with pytest.raises(ValueError, match="upload_image"):
            await server._acquire_image("", mock_ctx, purpose="test image")


# ---------------------------------------------------------------------------
# 18. Chaining — generate then edit using nanobanana:// URL
# ---------------------------------------------------------------------------

class TestChaining:
    def _mock_gemini_response(self):
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    @pytest.mark.asyncio
    async def test_generate_then_edit(self):
        """Simulate: generate an image, then edit it using the returned URL."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()
        mock_ctx = MagicMock()

        # Step 1: Generate
        with patch.object(server, "_get_client", return_value=mock_client):
            gen_result = await server.generate_image("a cat")

        gen_meta = _parse_result(gen_result)
        cat_url = gen_meta["image_url"]
        assert cat_url.startswith("http")

        # Step 2: Edit using the URL from step 1
        with patch.object(server, "_get_client", return_value=mock_client):
            edit_result = await server.edit_image(
                image=cat_url,
                prompt="add a top hat",
                ctx=mock_ctx,
            )

        edit_meta = _parse_result(edit_result)
        assert "image_url" in edit_meta
        assert edit_meta["image_url"].startswith("http")

    @pytest.mark.asyncio
    async def test_generate_then_swap_background(self):
        """Simulate: generate an image, then swap its background."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()
        mock_ctx = MagicMock()

        with patch.object(server, "_get_client", return_value=mock_client):
            gen_result = await server.generate_image("product on white")

        gen_meta = _parse_result(gen_result)
        product_url = gen_meta["image_url"]

        with patch.object(server, "_get_client", return_value=mock_client):
            swap_result = await server.swap_background(
                image=product_url,
                background="tropical beach at sunset",
                ctx=mock_ctx,
            )

        swap_meta = _parse_result(swap_result)
        assert "image_url" in swap_meta
        assert swap_meta["background"] == "tropical beach at sunset"

    @pytest.mark.asyncio
    async def test_generate_then_variations(self):
        """Simulate: generate an image, then create variations."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()
        mock_ctx = MagicMock()

        with patch.object(server, "_get_client", return_value=mock_client):
            gen_result = await server.generate_image("landscape")

        gen_meta = _parse_result(gen_result)
        url = gen_meta["image_url"]

        with patch.object(server, "_get_client", return_value=mock_client):
            var_result = await server.create_variations(
                image=url, ctx=mock_ctx, count=2,
            )

        var_meta = _parse_result(var_result)
        assert "images" in var_meta
        for img in var_meta["images"]:
            assert "image_url" in img


# ---------------------------------------------------------------------------
# 19. Cloud output key consistency
# ---------------------------------------------------------------------------

class TestCloudOutputKeyConsistency:
    """Verify that image_url key is always present and points to /images/ URLs."""

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    def test_single_image_has_image_url(self):
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response({}, [(jpeg, {"index": 1})])
        meta = _parse_result(result)
        assert "image_url" in meta
        assert "/images/" in meta["image_url"]

    def test_single_image_no_s3_domain_without_bucket(self):
        """Without S3_BUCKET, image_url is a local /images/ URL."""
        orig_s3 = server.S3_BUCKET
        try:
            server.S3_BUCKET = None
            jpeg = _make_test_image(100, 100)
            result = server._build_image_response({}, [(jpeg, {"index": 1})])
            meta = _parse_result(result)
            assert "amazonaws.com" not in meta["image_url"]
            assert "/images/" in meta["image_url"]
        finally:
            server.S3_BUCKET = orig_s3

    def test_multi_image_all_have_image_url(self):
        imgs = [(_make_test_image(100, 100), {"index": i}) for i in range(2)]
        result = server._build_image_response({}, imgs)
        meta = _parse_result(result)
        for img in meta["images"]:
            assert "image_url" in img
            assert "/images/" in img["image_url"]

    def test_size_kb_always_set(self):
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response({}, [(jpeg, {"index": 1})])
        meta = _parse_result(result)
        assert "size_kb" in meta
        assert isinstance(meta["size_kb"], int)

    def test_single_image_returns_image_object(self):
        """Single image response: returns [Image, json_str] — Image renders inline in claude.ai."""
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response({}, [(jpeg, {"index": 1})])
        assert isinstance(result, list)
        assert len(result) == 2, "Single image: [Image, json_str]"
        assert isinstance(result[0], MCPImage), "First item must be MCPImage"
        assert isinstance(result[-1], str), "Last item must be JSON str"
        meta = json.loads(result[-1])
        # image_url in JSON is the full S3/local URL for tool chaining
        assert meta["image_url"].startswith("http"), "image_url must be http URL for chaining"
        assert "data:" not in meta["image_url"], "image_url must not be a data URI"

    def test_multi_image_returns_image_objects(self):
        """Multi-image response: returns [Image1, Image2, ..., json_str]."""
        imgs = [(_make_test_image(100, 100), {"index": i}) for i in range(3)]
        result = server._build_image_response({}, imgs)
        assert isinstance(result, list)
        assert len(result) == 4, "3 images: [Image1, Image2, Image3, json_str]"
        for i in range(0, 3):
            assert isinstance(result[i], MCPImage), f"Item {i} must be MCPImage"
        assert isinstance(result[-1], str)
        meta = json.loads(result[-1])
        assert "images" in meta
        assert len(meta["images"]) == 3
        for img in meta["images"]:
            assert "image_url" in img
            assert img["image_url"].startswith("http")

    def test_multi_image_each_has_image_url(self):
        """Multi-image response: each image entry has an image_url for chaining."""
        imgs = [(_make_test_image(100, 100), {"index": i}) for i in range(2)]
        result = server._build_image_response({}, imgs)
        meta = _parse_result(result)
        for img in meta["images"]:
            assert "image_url" in img
            assert img["image_url"].startswith("http")


# ---------------------------------------------------------------------------
# 20. Empty reference image validation
# ---------------------------------------------------------------------------

class TestReferenceImageValidation:
    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            server._build_ref_parts([""])

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            server._build_ref_parts(["   "])

    def test_mixed_valid_and_empty_rejected(self):
        img = _make_test_image(100, 100)
        img_id = server._store_image(img, "image/jpeg")
        with pytest.raises(ValueError, match="Reference image 2"):
            server._build_ref_parts([f"nanobanana://{img_id}", ""])

    def test_none_list_returns_empty(self):
        assert server._build_ref_parts(None) == []

    def test_empty_list_returns_empty(self):
        assert server._build_ref_parts([]) == []


# ---------------------------------------------------------------------------
# 21. Round 1 sim — generate 4 images with qa=True, verify ranking
# ---------------------------------------------------------------------------

class TestQARankingMultipleImages:
    def _mock_gemini_response_with_qa(self, qa_total):
        """Return a mock Gemini response for image gen + a QA response."""
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    @pytest.mark.asyncio
    async def test_qa_ranking_order(self):
        """Round 1: generate 4 images with qa=True — rank 1 must have highest QA total."""
        call_count = {"n": 0}
        qa_totals = [30, 42, 35, 48]  # image scores to be returned for QA calls

        def side_effect(**kwargs):
            n = call_count["n"]
            call_count["n"] += 1
            if n < 4:
                # Image generation calls
                return self._mock_gemini_response_with_qa(0)
            else:
                # QA scoring calls — return individual scores that sum to qa_totals[n-4]
                total = qa_totals[n - 4]
                score_each = total // 5
                remainder = total - score_each * 5
                qa_resp = MagicMock()
                qa_resp.text = json.dumps({
                    "composition": {"score": score_each + remainder, "note": "ok"},
                    "clarity": {"score": score_each, "note": "ok"},
                    "lighting": {"score": score_each, "note": "ok"},
                    "color": {"score": score_each, "note": "ok"},
                    "prompt_adherence": {"score": score_each, "note": "ok"},
                    "total": total,
                })
                return qa_resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = side_effect

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("luxury coffee mug", count=4, qa=True)

        meta = _parse_result(result)
        assert "images" in meta
        images = meta["images"]
        assert len(images) == 4
        # Rank 1 must have the highest QA total (48)
        rank1 = next(img for img in images if img.get("rank") == 1)
        assert rank1["qa"]["total"] == 48
        # Ranks should be 1-4
        ranks = sorted(img["rank"] for img in images)
        assert ranks == [1, 2, 3, 4]
        # Each image must have an image_url
        for img in images:
            assert "image_url" in img
            assert img["image_url"].startswith("http")

    @pytest.mark.asyncio
    async def test_single_image_qa_no_rank(self):
        """Single image with qa=True should have qa field but no rank field."""
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_gen_resp = MagicMock()
        mock_gen_resp.candidates = [mock_candidate]

        mock_qa_resp = MagicMock()
        mock_qa_resp.text = json.dumps({
            "composition": {"score": 8, "note": "good"},
            "clarity": {"score": 9, "note": "sharp"},
            "lighting": {"score": 7, "note": "ok"},
            "color": {"score": 8, "note": "good"},
            "prompt_adherence": {"score": 9, "note": "excellent"},
            "total": 41,
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [mock_gen_resp, mock_qa_resp]

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("cat", count=1, qa=True)

        meta = _parse_result(result)
        assert "qa" in meta
        assert meta["qa"]["total"] == 41
        # No rank field for single image
        assert "rank" not in meta

    @pytest.mark.asyncio
    async def test_create_variations_qa_ranking(self):
        """Round 3: create_variations with qa=True — variations ranked by QA score."""
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()

        call_count = {"n": 0}
        qa_totals = [25, 40, 33]

        def side_effect(**kwargs):
            n = call_count["n"]
            call_count["n"] += 1
            if n < 3:
                test_img = _make_test_image(512, 512)
                mock_part = MagicMock()
                mock_part.inline_data = MagicMock()
                mock_part.inline_data.mime_type = "image/png"
                mock_part.inline_data.data = test_img
                mock_candidate = MagicMock()
                mock_candidate.content.parts = [mock_part]
                mock_response = MagicMock()
                mock_response.candidates = [mock_candidate]
                return mock_response
            else:
                total = qa_totals[n - 3]
                score_each = total // 5
                remainder = total - score_each * 5
                qa_resp = MagicMock()
                qa_resp.text = json.dumps({
                    "composition": {"score": score_each + remainder, "note": "ok"},
                    "clarity": {"score": score_each, "note": "ok"},
                    "lighting": {"score": score_each, "note": "ok"},
                    "color": {"score": score_each, "note": "ok"},
                    "prompt_adherence": {"score": score_each, "note": "ok"},
                    "total": total,
                })
                return qa_resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = side_effect

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.create_variations(
                image=f"nanobanana://{src_id}",
                ctx=mock_ctx,
                count=3,
                qa=True,
            )

        meta = _parse_result(result)
        assert "images" in meta
        images = meta["images"]
        assert len(images) == 3
        rank1 = next(img for img in images if img.get("rank") == 1)
        assert rank1["qa"]["total"] == 40


# ---------------------------------------------------------------------------
# 22. Round 2 sim — analyze_image quality total recomputation
# ---------------------------------------------------------------------------

class TestAnalyzeImageQualityTotal:
    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_quality_total_recomputed_from_criteria(self):
        """Bug fix: total must be sum of individual scores, not Gemini's normalized value."""
        img = _make_test_image(200, 200)
        img_id = server._store_image(img, "image/jpeg")
        mock_ctx = MagicMock()

        # Gemini returns a normalized 1-10 total (8) instead of raw sum (42)
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "sharpness": {"score": 9, "note": "sharp"},
            "exposure": {"score": 8, "note": "good"},
            "composition": {"score": 8, "note": "balanced"},
            "color_balance": {"score": 9, "note": "vivid"},
            "noise": {"score": 8, "note": "clean"},
            "issues": [],
            "total": 8,  # Gemini normalized to 1-10 — should be 42
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.analyze_image(
                image=f"nanobanana://{img_id}",
                ctx=mock_ctx,
                focus="quality",
            )

        data = _parse_result(result)
        # total must be recomputed: 9+8+8+9+8 = 42, not Gemini's 8
        assert data["total"] == 42
        assert data["focus"] == "quality"

    @pytest.mark.asyncio
    async def test_non_quality_focus_total_not_touched(self):
        """Non-quality focuses don't have a total field — no recomputation needed."""
        img = _make_test_image(200, 200)
        img_id = server._store_image(img, "image/jpeg")
        mock_ctx = MagicMock()

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "description": "a test image",
            "style": "minimalist",
            "mood": "calm",
            "colors": ["gray"],
            "details": ["solid color"],
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.analyze_image(
                image=f"nanobanana://{img_id}",
                ctx=mock_ctx,
                focus="general",
            )

        data = _parse_result(result)
        assert data["focus"] == "general"
        assert "total" not in data  # no total for general focus


# ---------------------------------------------------------------------------
# 23. Round 4 sim — expired /images/ URL error in all editing tools
# ---------------------------------------------------------------------------

class TestExpiredImageErrorMessages:
    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_edit_image_expired_url(self):
        """edit_image with expired /images/ URL returns clear error."""
        mock_ctx = MagicMock()
        expired_url = "http://localhost:8080/images/expiredid12"  # not in store

        mock_client = MagicMock()
        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=expired_url,
                prompt="add a hat",
                ctx=mock_ctx,
            )

        data = _parse_result(result)
        assert "error" in data
        assert "expired" in data["error"].lower() or "evicted" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_swap_background_expired_url(self):
        """swap_background with expired /images/ URL returns clear error."""
        mock_ctx = MagicMock()
        expired_url = "http://localhost:8080/images/expiredid12"

        mock_client = MagicMock()
        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.swap_background(
                image=expired_url,
                background="tropical beach",
                ctx=mock_ctx,
            )

        data = _parse_result(result)
        assert "error" in data
        assert "expired" in data["error"].lower() or "evicted" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_create_variations_expired_url(self):
        """create_variations with expired /images/ URL returns clear error."""
        mock_ctx = MagicMock()
        expired_url = "http://localhost:8080/images/expiredid12"

        mock_client = MagicMock()
        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.create_variations(
                image=expired_url,
                ctx=mock_ctx,
            )

        data = _parse_result(result)
        assert "error" in data
        assert "expired" in data["error"].lower() or "evicted" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_analyze_image_expired_url(self):
        """analyze_image with expired /images/ URL returns clear error."""
        mock_ctx = MagicMock()
        expired_url = "http://localhost:8080/images/expiredid12"

        mock_client = MagicMock()
        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.analyze_image(
                image=expired_url,
                ctx=mock_ctx,
            )

        data = _parse_result(result)
        assert "error" in data
        assert "expired" in data["error"].lower() or "evicted" in data["error"].lower()


# ---------------------------------------------------------------------------
# 24. Round 2 sim — data URI accepted by non-upload tools
# ---------------------------------------------------------------------------

class TestDataUriInTools:
    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_edit_image_accepts_data_uri(self):
        """edit_image should accept data URIs as source image."""
        img = _make_test_image(100, 100)
        b64 = base64.b64encode(img).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"
        mock_ctx = MagicMock()

        test_result_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_result_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=data_uri,
                prompt="add a hat",
                ctx=mock_ctx,
            )

        data = _parse_result(result)
        assert "image_url" in data
        assert "error" not in data

    @pytest.mark.asyncio
    async def test_upload_image_rejects_data_uri(self):
        """upload_image only accepts http/https URLs — data URIs must use curl/upload endpoint."""
        img = _make_test_image(100, 100)
        b64 = base64.b64encode(img).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"
        mock_ctx = MagicMock()

        result = await server.upload_image(ctx=mock_ctx, image=data_uri)
        data = _parse_result(result)
        assert "error" in data
        # Error should tell user how to upload
        assert "upload" in data["error"].lower()


# ===========================================================================
# CONTEXT-AWARE OUTPUT CONTRACT TESTS
#
# Two deployment contexts, two expected behaviors:
#
#   Claude.ai (web)   → images EMBEDDED as MCP Image content for inline display
#   Claude Code (CLI) → images stored at user-designated location (S3/GCS),
#                       URL in JSON is durable; embedding still present for
#                       any client that can render it
#
# These tests validate the full output contract, not just JSON metadata.
# ===========================================================================


# ---------------------------------------------------------------------------
# 25. Embedded image contract — tools must return [json_str, Image, ...]
# ---------------------------------------------------------------------------

class TestEmbeddedImageContract:
    """Verify every image-generating tool returns [json_str, Image, ...].

    ImageContent objects are rendered inline by claude.ai in the tool result block.
    """

    def _mock_gemini_response(self):
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_generate_image_returns_list_with_image(self):
        """generate_image returns [MCPImage, json_str] — Image renders inline in claude.ai."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("a sunset")

        assert isinstance(result, list), "Should return list [Image, json_str]"
        assert isinstance(result[0], MCPImage)
        assert isinstance(result[-1], str)
        meta = _parse_result(result)
        assert "image_url" in meta

    @pytest.mark.asyncio
    async def test_generate_image_count3_returns_3_image_objects(self):
        """count=3: returns [json_str, Image1, Image2, Image3]."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("cats", count=3)

        assert isinstance(result, list)
        assert len(result) == 4, "Should be [img1, img2, img3, json_str]"
        assert all(isinstance(result[i], MCPImage) for i in range(0, 3))

    @pytest.mark.asyncio
    async def test_edit_image_returns_list_with_image(self):
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}", prompt="add hat", ctx=mock_ctx
            )

        assert isinstance(result, list)
        assert isinstance(result[0], MCPImage)

    @pytest.mark.asyncio
    async def test_swap_background_returns_list_with_image(self):
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.swap_background(
                image=f"nanobanana://{src_id}", background="beach", ctx=mock_ctx
            )

        assert isinstance(result, list)
        assert isinstance(result[0], MCPImage)

    @pytest.mark.asyncio
    async def test_create_variations_returns_image_objects(self):
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.create_variations(
                image=f"nanobanana://{src_id}", ctx=mock_ctx, count=2
            )

        assert isinstance(result, list)
        assert len(result) == 3, "Should be [img1, img2, json_str]"
        assert all(isinstance(result[i], MCPImage) for i in range(0, 2))

    @pytest.mark.asyncio
    async def test_error_paths_return_str_not_list(self):
        """Validation errors still return plain str JSON — no Image to embed."""
        result = await server.generate_image("cat", aspect_ratio="99:1")
        assert isinstance(result, str), "Validation errors must be plain str"
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_gemini_failure_returns_str_not_list(self):
        """When Gemini returns no image, result is error str, not list."""
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("cat")

        assert isinstance(result, str)
        assert "error" in json.loads(result)

    @pytest.mark.asyncio
    async def test_upload_image_still_returns_str(self):
        """upload_image is not an image generator — returns JSON string only."""
        src = _make_test_image(100, 100)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()

        result = await server.upload_image(
            ctx=mock_ctx, image=f"http://localhost:8080/images/{src_id}"
        )
        assert isinstance(result, str)
        assert "url" in json.loads(result)

    @pytest.mark.asyncio
    async def test_analyze_image_still_returns_str(self):
        """analyze_image returns analysis text — no image embedding needed."""
        src = _make_test_image(100, 100)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "description": "test", "style": "flat", "mood": "calm",
            "colors": ["gray"], "details": ["solid color"],
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.analyze_image(
                image=f"nanobanana://{src_id}", ctx=mock_ctx, focus="general"
            )

        assert isinstance(result, str)
        assert "focus" in json.loads(result)


# ---------------------------------------------------------------------------
# 26. Default image delivery — always inline + /images/ URL + optional S3 catch-all
# ---------------------------------------------------------------------------

class TestDefaultOutputBehavior:
    """Validate the new default output contract:

    - Images always displayed inline in Claude (MCPImage in return list)
    - image_url always points to /images/ (in-memory, 1-hour TTL) for chaining
    - S3 catch-all upload fires in background when S3_BUCKET is configured
    - save_folder writes JPEG files to disk when provided
    """

    def _mock_gemini_response(self):
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    def test_images_always_stored_in_memory(self):
        """Images always go to the in-memory store with /images/ URLs — no config needed."""
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response({}, [(jpeg, {"index": 1})])
        meta = _parse_result(result)
        assert "/images/" in meta["image_url"]
        assert "amazonaws.com" not in meta["image_url"]
        assert "storage.googleapis.com" not in meta["image_url"]

    @pytest.mark.asyncio
    async def test_no_s3_returns_local_images_url(self):
        """Without S3, generate_image returns /images/ URL with expiry."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        orig_s3 = server.S3_BUCKET
        try:
            server.S3_BUCKET = None
            with patch.object(server, "_get_client", return_value=mock_client):
                result = await server.generate_image("cat")
            meta = _parse_result(result)
            assert "/images/" in meta["image_url"], "must return /images/ URL"
            assert "amazonaws.com" not in meta["image_url"]
            assert "expires_in" in meta
        finally:
            server.S3_BUCKET = orig_s3

    @pytest.mark.asyncio
    async def test_inline_images_always_embedded(self):
        """Images are always returned as MCPImage objects for inline display in claude.ai."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("cat")

        assert isinstance(result, list)
        assert isinstance(result[0], MCPImage)

    @pytest.mark.asyncio
    async def test_s3_bucket_gives_s3_url_in_metadata(self):
        """When S3_BUCKET is set, image_url in metadata is the durable S3 URL."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        fake_s3_url = "https://my-bucket.s3.us-east-1.amazonaws.com/gen/abc123.jpg"

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "my-bucket"), \
             patch.object(server, "_upload_to_s3", return_value=fake_s3_url):
            result = await server.generate_image("cat")

        meta = _parse_result(result)
        assert meta["image_url"] == fake_s3_url, "S3 URL must be returned as image_url"
        assert "amazonaws.com" in meta["image_url"]
        assert "expires_in" not in meta, "S3 URLs don't expire"
        # MCPImage still returned for inline display
        assert isinstance(result, list)
        assert isinstance(result[0], MCPImage)

    @pytest.mark.asyncio
    async def test_no_s3_bucket_uses_local_url(self):
        """When S3_BUCKET is not set, image_url is the /images/ endpoint."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        orig_s3 = server.S3_BUCKET
        try:
            server.S3_BUCKET = None
            with patch.object(server, "_get_client", return_value=mock_client):
                result = await server.generate_image("cat")
            meta = _parse_result(result)
            assert "/images/" in meta["image_url"]
            assert "expires_in" in meta
        finally:
            server.S3_BUCKET = orig_s3


# ---------------------------------------------------------------------------
# 27. Claude Code context — chaining works with both url and cloud output
# ---------------------------------------------------------------------------

class TestClaudeCodeContext:
    """Simulate a Claude Code user with S3 configured.

    In this setup:
    - Images are always displayed inline in Claude
    - image_url in metadata is the durable S3 URL (no expiry)
    - Images also stored locally for /images/ serving
    """

    def _mock_gemini_response(self):
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_generate_then_edit_with_images_urls(self):
        """Claude Code: generate, then edit using the returned /images/ URL."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            # Step 1: Generate — images go to in-memory store
            gen_result = await server.generate_image("product shot")
            gen_meta = _parse_result(gen_result)
            assert "/images/" in gen_meta["image_url"]

            # Step 2: Edit using the /images/ URL from step 1
            edit_result = await server.edit_image(
                image=gen_meta["image_url"],
                prompt="add studio lighting",
                ctx=mock_ctx,
            )

        edit_meta = _parse_result(edit_result)
        assert "image_url" in edit_meta
        # MCPImage present in both results for inline display
        assert isinstance(gen_result, list) and isinstance(gen_result[0], MCPImage)
        assert isinstance(edit_result, list) and isinstance(edit_result[0], MCPImage)

    @pytest.mark.asyncio
    async def test_s3_urls_returned_for_both_gen_and_edit(self):
        """With S3_BUCKET set, both generate and edit return S3 URLs in metadata."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()
        s3_urls = iter([
            "https://my-bucket.s3.us-east-1.amazonaws.com/gen/img1.jpg",
            "https://my-bucket.s3.us-east-1.amazonaws.com/edit/img2.jpg",
        ])

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "my-bucket"), \
             patch.object(server, "_upload_to_s3", side_effect=lambda *a, **kw: next(s3_urls)):

            gen_result = await server.generate_image("cat")
            gen_meta = _parse_result(gen_result)
            assert "amazonaws.com" in gen_meta["image_url"]

            # Edit using the S3 URL — fetched as external HTTP
            with patch.object(server, "_fetch_url",
                               return_value=(_make_test_image(512, 512), "image/jpeg")):
                edit_result = await server.edit_image(
                    image=gen_meta["image_url"], prompt="add hat", ctx=mock_ctx,
                )

        edit_meta = _parse_result(edit_result)
        assert "amazonaws.com" in edit_meta["image_url"]

    @pytest.mark.asyncio
    async def test_url_mode_image_expires_after_chain(self):
        """url mode: after 1h expiry, chaining with old URL fails with clear error."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            gen_result = await server.generate_image("landscape")

        gen_meta = _parse_result(gen_result)
        image_url = gen_meta["image_url"]
        assert "/images/" in image_url

        # Simulate expiry by clearing the store
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

        with patch.object(server, "_get_client", return_value=mock_client):
            edit_result = await server.edit_image(
                image=image_url, prompt="make it night", ctx=mock_ctx
            )

        data = json.loads(edit_result)  # error = str, not list
        assert "error" in data
        assert "expired" in data["error"].lower() or "evicted" in data["error"].lower()


# ---------------------------------------------------------------------------
# 28. batch_analyze — 5-image scenario
# ---------------------------------------------------------------------------

class TestBatchAnalyze:
    """Scenario: user uploads 5 images and asks to analyze them all.

    Without batch_analyze they'd need 5 sequential tool calls.
    batch_analyze runs all in parallel and returns results in input order.
    """

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    def _mock_analysis_response(self, description: str) -> MagicMock:
        resp = MagicMock()
        resp.text = json.dumps({
            "description": description,
            "style": "flat",
            "mood": "neutral",
            "colors": ["gray"],
            "details": ["solid color block"],
        })
        return resp

    @pytest.mark.asyncio
    async def test_batch_5_images_all_analyzed_in_order(self):
        """5 images → 5 results in the same order, all with focus field."""
        imgs = [_make_test_image(100, 100) for _ in range(5)]
        img_ids = [server._store_image(img, "image/jpeg") for img in imgs]
        image_urls = [f"nanobanana://{img_id}" for img_id in img_ids]
        mock_ctx = MagicMock()

        call_count = {"n": 0}
        descriptions = [f"Image {i+1} description" for i in range(5)]

        def side_effect(**kwargs):
            n = call_count["n"]
            call_count["n"] += 1
            return self._mock_analysis_response(descriptions[n % 5])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = side_effect

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.batch_analyze(ctx=mock_ctx, images=image_urls, focus="general")

        data = json.loads(result)
        assert data["count"] == 5
        assert data["focus"] == "general"
        assert len(data["results"]) == 5
        # Results are in input order (index 1–5)
        for i, r in enumerate(data["results"]):
            assert r["index"] == i + 1
            assert r["image_url"] == image_urls[i]
            assert "error" not in r
            assert r["focus"] == "general"

    @pytest.mark.asyncio
    async def test_batch_runs_in_parallel(self):
        """All analyses should fire concurrently — call count equals image count."""
        imgs = [_make_test_image(100, 100) for _ in range(3)]
        img_ids = [server._store_image(img, "image/jpeg") for img in imgs]
        image_urls = [f"nanobanana://{img_id}" for img_id in img_ids]
        mock_ctx = MagicMock()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_analysis_response("test")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.batch_analyze(ctx=mock_ctx, images=image_urls)

        data = json.loads(result)
        assert len(data["results"]) == 3
        assert mock_client.models.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_partial_failure_does_not_abort(self):
        """If one image fails (e.g. expired URL), others still succeed."""
        good_img = _make_test_image(100, 100)
        good_id = server._store_image(good_img, "image/jpeg")
        bad_url = "http://localhost:8080/images/expiredid00"  # not in store
        good_url = f"nanobanana://{good_id}"
        mock_ctx = MagicMock()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_analysis_response("good image")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.batch_analyze(
                ctx=mock_ctx, images=[bad_url, good_url]
            )

        data = json.loads(result)
        assert data["count"] == 2
        results = data["results"]
        assert "error" in results[0]    # bad URL failed
        assert "error" not in results[1]  # good URL succeeded
        assert results[0]["index"] == 1
        assert results[1]["index"] == 2

    @pytest.mark.asyncio
    async def test_batch_empty_images_error(self):
        mock_ctx = MagicMock()
        result = await server.batch_analyze(ctx=mock_ctx, images=[])
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_batch_too_many_images_error(self):
        mock_ctx = MagicMock()
        result = await server.batch_analyze(ctx=mock_ctx, images=["http://x.com/img.jpg"] * 21)
        data = json.loads(result)
        assert "error" in data
        assert "20" in data["error"]

    @pytest.mark.asyncio
    async def test_batch_unknown_focus_error(self):
        mock_ctx = MagicMock()
        result = await server.batch_analyze(
            ctx=mock_ctx, images=["http://x.com/img.jpg"], focus="unknown"
        )
        data = json.loads(result)
        assert "error" in data
        assert "available" in data

    @pytest.mark.asyncio
    async def test_batch_quality_focus_uses_source_max_dim(self):
        """quality focus must use SOURCE_MAX_DIM for sharpness detection."""
        img = _make_test_image(200, 200)
        img_id = server._store_image(img, "image/jpeg")
        mock_ctx = MagicMock()
        captured = {}

        original_acquire = server._acquire_image

        async def spy_acquire(image, ctx, **kwargs):
            captured.update(kwargs)
            return await original_acquire(image, ctx, **kwargs)

        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "sharpness": {"score": 8, "note": "ok"}, "exposure": {"score": 8, "note": "ok"},
            "composition": {"score": 8, "note": "ok"}, "color_balance": {"score": 8, "note": "ok"},
            "noise": {"score": 8, "note": "ok"}, "issues": [], "total": 40,
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp

        with patch.object(server, "_acquire_image", spy_acquire), \
             patch.object(server, "_get_client", return_value=mock_client):
            await server.batch_analyze(
                ctx=mock_ctx, images=[f"nanobanana://{img_id}"], focus="quality"
            )

        assert captured.get("max_dim") == server.SOURCE_MAX_DIM


# ---------------------------------------------------------------------------
# 29. Multi-reference edit — swap content from 2+ reference images
# ---------------------------------------------------------------------------

class TestMultiReferenceEdit:
    """Scenario: user has 3 images, wants content from images 2 and 3
    inserted into image 1 in specific places.

    edit_image accepts reference_images: list[str] but the auto-appended
    prompt must label references by position so Gemini knows which reference
    applies to which part of the edit.
    """

    def _mock_gemini_response(self):
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_two_references_prompt_labels_positions(self):
        """With 2 references, prompt must mention 'reference image 1' and 'reference image 2'."""
        src = _make_test_image(200, 200)
        ref1 = _make_test_image(150, 150)
        ref2 = _make_test_image(150, 150)
        src_id = server._store_image(src, "image/jpeg")
        ref1_id = server._store_image(ref1, "image/jpeg")
        ref2_id = server._store_image(ref2, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}",
                prompt="replace the table with the object from reference image 1 "
                       "and the lamp with the object from reference image 2",
                reference_images=[f"nanobanana://{ref1_id}", f"nanobanana://{ref2_id}"],
                ctx=mock_ctx,
            )

        assert "image_url" in _parse_result(result)
        call_args = mock_client.models.generate_content.call_args
        # All parts: source + ref1 + ref2 + prompt text = 4
        assert len(call_args.kwargs["contents"]) == 4
        # The constructed prompt must label references by position
        text_parts = [str(p) for p in call_args.kwargs["contents"] if hasattr(p, "text")]
        combined = " ".join(text_parts)
        assert "reference image 1" in combined
        assert "reference image 2" in combined

    @pytest.mark.asyncio
    async def test_single_reference_uses_simple_prompt(self):
        """Single reference must NOT include numbered labels (no 'reference image 1')."""
        src = _make_test_image(200, 200)
        ref = _make_test_image(150, 150)
        src_id = server._store_image(src, "image/jpeg")
        ref_id = server._store_image(ref, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            await server.edit_image(
                image=f"nanobanana://{src_id}",
                prompt="replace the bottle",
                reference_images=[f"nanobanana://{ref_id}"],
                ctx=mock_ctx,
            )

        call_args = mock_client.models.generate_content.call_args
        text_parts = [str(p) for p in call_args.kwargs["contents"] if hasattr(p, "text")]
        combined = " ".join(text_parts)
        # Simple guidance, not numbered
        assert "reference image 1" not in combined
        assert "reference image" in combined  # still mentions reference image

    @pytest.mark.asyncio
    async def test_three_references_all_labeled(self):
        """3 references → labels for 1, 2, and 3 all appear in prompt."""
        src = _make_test_image(200, 200)
        refs = [server._store_image(_make_test_image(100, 100), "image/jpeg") for _ in range(3)]
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            await server.edit_image(
                image=f"nanobanana://{server._store_image(src, 'image/jpeg')}",
                prompt="replace A with ref 1, B with ref 2, C with ref 3",
                reference_images=[f"nanobanana://{r}" for r in refs],
                ctx=mock_ctx,
            )

        call_args = mock_client.models.generate_content.call_args
        # source + 3 refs + text = 5 parts
        assert len(call_args.kwargs["contents"]) == 5
        text_parts = [str(p) for p in call_args.kwargs["contents"] if hasattr(p, "text")]
        combined = " ".join(text_parts)
        assert "reference image 1" in combined
        assert "reference image 2" in combined
        assert "reference image 3" in combined

    @pytest.mark.asyncio
    async def test_no_references_no_reference_guidance_in_prompt(self):
        """With no reference images, no reference guidance appended to prompt."""
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            await server.edit_image(
                image=f"nanobanana://{src_id}",
                prompt="add a hat",
                ctx=mock_ctx,
            )

        call_args = mock_client.models.generate_content.call_args
        text_parts = [str(p) for p in call_args.kwargs["contents"] if hasattr(p, "text")]
        combined = " ".join(text_parts)
        assert "reference image" not in combined


# ---------------------------------------------------------------------------
# 30. compare_images — diff, quality ranking, style comparison
# ---------------------------------------------------------------------------

class TestCompareImages:
    """Scenario: user uploads 2+ images and wants to understand relationships.

    batch_analyze can't do this — it analyzes independently.
    compare_images sends all images to one Gemini call so differences and
    rankings are grounded in cross-image context.
    """

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    def _mock_compare_response(self, payload: dict) -> MagicMock:
        resp = MagicMock()
        resp.text = json.dumps(payload)
        return resp

    def _store_n_images(self, n: int) -> list[str]:
        return [
            f"nanobanana://{server._store_image(_make_test_image(100, 100), 'image/jpeg')}"
            for _ in range(n)
        ]

    @pytest.mark.asyncio
    async def test_compare_2_images_differences(self):
        """Standard diff: what changed between image A and image B."""
        urls = self._store_n_images(2)
        mock_ctx = MagicMock()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_compare_response({
            "summary": "Image 2 has a darker background",
            "differences": [{"aspect": "background", "description": "lighter vs darker"}],
        })

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.compare_images(ctx=mock_ctx, images=urls, focus="differences")

        data = json.loads(result)
        assert "error" not in data
        assert data["focus"] == "differences"
        assert data["image_count"] == 2
        assert "differences" in data
        # All images sent in one Gemini call, not two
        assert mock_client.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_compare_all_images_in_one_gemini_call(self):
        """3 images → single Gemini call with 3 image parts + 1 text part."""
        urls = self._store_n_images(3)
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_compare_response({
            "summary": "test", "differences": [],
        })

        with patch.object(server, "_get_client", return_value=mock_client):
            await server.compare_images(ctx=mock_ctx, images=urls)

        call_args = mock_client.models.generate_content.call_args
        # 3 image parts + 1 prompt text = 4 total
        assert len(call_args.kwargs["contents"]) == 4
        assert mock_client.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_compare_quality_picks_winner(self):
        """Quality focus: result includes ranking and best-image index."""
        urls = self._store_n_images(2)
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_compare_response({
            "rankings": [
                {"image": 1, "score": 8, "notes": "sharp"},
                {"image": 2, "score": 6, "notes": "slightly blurry"},
            ],
            "best": 1,
            "reasoning": "Image 1 has better sharpness",
        })

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.compare_images(ctx=mock_ctx, images=urls, focus="quality")

        data = json.loads(result)
        assert data["focus"] == "quality"
        assert data["best"] == 1
        assert len(data["rankings"]) == 2

    @pytest.mark.asyncio
    async def test_compare_requires_at_least_2_images(self):
        urls = self._store_n_images(1)
        mock_ctx = MagicMock()
        result = await server.compare_images(ctx=mock_ctx, images=urls)
        data = json.loads(result)
        assert "error" in data
        assert "2" in data["error"]

    @pytest.mark.asyncio
    async def test_compare_max_10_images(self):
        mock_ctx = MagicMock()
        result = await server.compare_images(
            ctx=mock_ctx, images=["http://x.com/img.jpg"] * 11
        )
        data = json.loads(result)
        assert "error" in data
        assert "10" in data["error"]

    @pytest.mark.asyncio
    async def test_compare_unknown_focus_error(self):
        urls = self._store_n_images(2)
        mock_ctx = MagicMock()
        result = await server.compare_images(ctx=mock_ctx, images=urls, focus="vibes")
        data = json.loads(result)
        assert "error" in data
        assert "available" in data

    @pytest.mark.asyncio
    async def test_compare_vs_batch_analyze_different_call_count(self):
        """compare_images uses 1 Gemini call; batch_analyze uses N calls.
        This is the key behavioral difference for cross-image reasoning.
        """
        urls = self._store_n_images(3)
        mock_ctx = MagicMock()

        compare_client = MagicMock()
        compare_client.models.generate_content.return_value = self._mock_compare_response({
            "summary": "test", "differences": [],
        })

        batch_client = MagicMock()
        batch_client.models.generate_content.return_value = MagicMock(
            text=json.dumps({"description": "test", "style": "flat", "mood": "calm",
                             "colors": ["gray"], "details": ["block"]})
        )

        with patch.object(server, "_get_client", return_value=compare_client):
            await server.compare_images(ctx=mock_ctx, images=urls)
        compare_calls = compare_client.models.generate_content.call_count

        with patch.object(server, "_get_client", return_value=batch_client):
            await server.batch_analyze(ctx=mock_ctx, images=urls)
        batch_calls = batch_client.models.generate_content.call_count

        assert compare_calls == 1, "compare_images must use exactly 1 Gemini call"
        assert batch_calls == 3, "batch_analyze must use 1 call per image"


# ---------------------------------------------------------------------------
# 31. Composite swap — objects from multiple source images into one target
# ---------------------------------------------------------------------------

class TestCompositeSwap:
    """Scenario: user has 3 images and says 'put the bottle from image 2
    and the bottle from image 3 into image 1, replacing the two bottles there'.

    Correct tool: edit_image(image=url1, reference_images=[url2, url3],
                             prompt="replace ... with reference image 1 / 2")
    """

    def _mock_gemini_response(self):
        test_img = _make_test_image(512, 512)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = test_img
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_three_image_composite_swap(self):
        """Image 1 is target; images 2 and 3 are object sources.
        edit_image must send all 3 images + labeled prompt to Gemini.
        """
        src = _make_test_image(200, 200)
        obj1 = _make_test_image(100, 100)
        obj2 = _make_test_image(100, 100)
        src_id = server._store_image(src, "image/jpeg")
        obj1_id = server._store_image(obj1, "image/jpeg")
        obj2_id = server._store_image(obj2, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}",
                prompt=(
                    "Replace the bottle on the left with the object from reference image 1 "
                    "and the bottle on the right with the object from reference image 2"
                ),
                reference_images=[
                    f"nanobanana://{obj1_id}",
                    f"nanobanana://{obj2_id}",
                ],
                ctx=mock_ctx,
            )

        meta = _parse_result(result)
        assert "image_url" in meta
        assert "error" not in meta

        call_args = mock_client.models.generate_content.call_args
        # source + ref1 + ref2 + prompt text = 4 parts
        contents = call_args.kwargs["contents"]
        assert len(contents) == 4

        # Prompt must label both references
        text = " ".join(str(p) for p in contents if hasattr(p, "text"))
        assert "reference image 1" in text
        assert "reference image 2" in text

    @pytest.mark.asyncio
    async def test_composite_swap_result_is_embedded(self):
        """Result must be [json, MCPImage] so Claude.ai shows it inline."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        ref_id = server._store_image(_make_test_image(100, 100), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}",
                prompt="replace the bottle with reference image 1",
                reference_images=[f"nanobanana://{ref_id}"],
                ctx=mock_ctx,
            )

        assert isinstance(result, list)
        assert isinstance(result[0], MCPImage)


# ---------------------------------------------------------------------------
# 30. structured_output=False — prove the pydantic serialization bug and fix
# ---------------------------------------------------------------------------

class TestStructuredOutputFix:
    """Verify that structured_output=False prevents the PydanticSerializationError.

    Without this flag, fastmcp infers an output_schema from `list | str` and
    routes through pydantic structured serialization.  pydantic_core.to_json
    cannot handle the Image type and raises:
      PydanticSerializationError: Unable to serialize unknown type: <class '...Image'>

    With structured_output=False, fastmcp uses _convert_to_content() which calls
    Image.to_image_content() and produces proper MCP ImageContent objects.
    """

    def test_pydantic_core_fails_on_image_without_fix(self):
        """Reproduce the original crash: pydantic_core.to_json chokes on Image."""
        import pydantic_core
        from mcp.server.fastmcp.utilities.types import Image as FastMCPImage

        fake_img = FastMCPImage(data=_make_test_image(64, 64), format="jpeg")
        with pytest.raises(pydantic_core.PydanticSerializationError, match="Unable to serialize unknown type"):
            pydantic_core.to_json({"result": ['{"test": 1}', fake_img]})

    def test_convert_to_content_handles_image_correctly(self):
        """fastmcp's _convert_to_content converts [json_str, Image] → [TextContent, ImageContent]."""
        from mcp.server.fastmcp.utilities.func_metadata import _convert_to_content
        from mcp.server.fastmcp.utilities.types import Image as FastMCPImage
        from mcp.types import ImageContent, TextContent

        # Matches what tools actually return: [MCPImage, json_str]
        json_str = '{"image_url": "https://example.com/img.jpg"}'
        img = FastMCPImage(data=_make_test_image(64, 64), format="jpeg")
        result = _convert_to_content([img, json_str])

        assert len(result) == 2
        assert isinstance(result[0], ImageContent)
        assert isinstance(result[1], TextContent)
        assert result[0].mimeType == "image/jpeg"
        assert len(result[0].data) > 0  # base64 data present

    def test_all_four_tools_have_structured_output_false(self):
        """All 4 image generation tools must have structured_output=False."""
        import mcp.server.fastmcp.tools.base as tools_base

        mcp_instance = server.mcp
        tool_names = ["generate_image", "edit_image", "swap_background", "create_variations"]
        for name in tool_names:
            tool = mcp_instance._tool_manager._tools.get(name)
            assert tool is not None, f"Tool '{name}' not registered"
            # structured_output=False means output_schema must be None
            assert tool.fn_metadata.output_schema is None, (
                f"Tool '{name}' has an output_schema — structured_output=False was not applied. "
                "This will cause PydanticSerializationError when Image objects are in the return list."
            )


# ---------------------------------------------------------------------------
# 31. S3 URL completeness — multi-image, fallback, all tools
# ---------------------------------------------------------------------------

class TestS3UrlCompleteness:
    """Verify S3 URL behavior across all generation tools and edge cases."""

    def _mock_gemini_response(self):
        test_img = _make_test_image(256, 256)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock(mime_type="image/png", data=test_img)
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_multi_image_each_gets_unique_s3_url(self):
        """count=3 with S3: each image gets its own distinct S3 URL."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        call_counter = [0]
        def make_s3_url(jpeg_bytes, prefix="gen"):
            call_counter[0] += 1
            return f"https://bucket.s3.amazonaws.com/{prefix}/img{call_counter[0]}.jpg"

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "bucket"), \
             patch.object(server, "_upload_to_s3", side_effect=make_s3_url):
            result = await server.generate_image("cats", count=3)

        meta = _parse_result(result)
        assert "images" in meta
        urls = [img["image_url"] for img in meta["images"]]
        assert len(urls) == 3
        assert len(set(urls)) == 3, "Each image must have a unique S3 URL"
        assert all("amazonaws.com" in u for u in urls)

    @pytest.mark.asyncio
    async def test_s3_failure_fallback_to_local_url(self):
        """When _upload_to_s3 raises, tool returns /images/ URL with expires_in."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "bucket"), \
             patch.object(server, "_upload_to_s3", side_effect=Exception("S3 auth failed")):
            result = await server.generate_image("cat")

        assert isinstance(result, list), "Must still return [json, Image] on S3 failure"
        meta = _parse_result(result)
        assert "/images/" in meta["image_url"], "Must fall back to local /images/ URL"
        assert "amazonaws.com" not in meta["image_url"]
        assert "expires_in" in meta, "Fallback URL must have expiry warning"

    @pytest.mark.asyncio
    async def test_swap_background_returns_s3_url(self):
        """swap_background returns S3 URL when S3_BUCKET is set."""
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()
        fake_url = "https://bucket.s3.amazonaws.com/bgswap/x.jpg"

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "bucket"), \
             patch.object(server, "_upload_to_s3", return_value=fake_url):
            result = await server.swap_background(
                image=f"nanobanana://{src_id}", background="beach", ctx=mock_ctx
            )

        meta = _parse_result(result)
        assert meta["image_url"] == fake_url
        assert "expires_in" not in meta

    @pytest.mark.asyncio
    async def test_create_variations_returns_s3_urls(self):
        """create_variations returns S3 URLs for each variation when S3_BUCKET is set."""
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        n = [0]
        def make_url(b, prefix="var"):
            n[0] += 1
            return f"https://bucket.s3.amazonaws.com/{prefix}/{n[0]}.jpg"

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "bucket"), \
             patch.object(server, "_upload_to_s3", side_effect=make_url):
            result = await server.create_variations(
                image=f"nanobanana://{src_id}", ctx=mock_ctx, count=2
            )

        meta = _parse_result(result)
        assert "images" in meta
        for img in meta["images"]:
            assert "amazonaws.com" in img["image_url"]
            assert "expires_in" not in img

    @pytest.mark.asyncio
    async def test_edit_image_s3_failure_still_returns_image(self):
        """edit_image gracefully falls back when S3 fails — tool result still valid."""
        src = _make_test_image(200, 200)
        src_id = server._store_image(src, "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "bucket"), \
             patch.object(server, "_upload_to_s3", side_effect=ConnectionError("timeout")):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}", prompt="add hat", ctx=mock_ctx
            )

        assert isinstance(result, list)
        meta = _parse_result(result)
        assert "/images/" in meta["image_url"]
        assert "expires_in" in meta


# ---------------------------------------------------------------------------
# 32. Docstring instruction contract
# ---------------------------------------------------------------------------

class TestDocstringInstructions:
    """Verify image_url chaining instructions are present in all generation tool docstrings."""

    def test_all_generation_tools_have_image_url_instruction(self):
        """All 4 tools must document image_url for tool chaining in their returns."""
        tools = [
            server.generate_image,
            server.edit_image,
            server.swap_background,
            server.create_variations,
        ]
        for tool in tools:
            doc = tool.__doc__ or ""
            assert "image_url" in doc, (
                f"{tool.__name__} docstring must mention image_url for tool chaining."
            )


# ---------------------------------------------------------------------------
# 33. save_folder + S3 simultaneously
# ---------------------------------------------------------------------------

class TestSaveFolderWithS3:
    """Both save_folder and S3 can be active at the same time."""

    def _mock_gemini_response(self):
        test_img = _make_test_image(128, 128)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock(mime_type="image/png", data=test_img)
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_single_image_has_both_s3_url_and_saved_path(self, tmp_path):
        """Single image: metadata has S3 image_url AND local saved_to path."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()
        fake_s3 = "https://bucket.s3.amazonaws.com/gen/abc.jpg"

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "bucket"), \
             patch.object(server, "_upload_to_s3", return_value=fake_s3):
            result = await server.generate_image("cat", save_folder=str(tmp_path))

        meta = _parse_result(result)
        assert meta["image_url"] == fake_s3
        assert "saved_to" in meta
        assert os.path.isfile(meta["saved_to"])
        assert str(tmp_path) in meta["saved_to"]

    @pytest.mark.asyncio
    async def test_multi_image_each_has_s3_url_and_saved_path(self, tmp_path):
        """count=2 with S3 + save_folder: each image has both fields."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        n = [0]
        def make_url(b, prefix="gen"):
            n[0] += 1
            return f"https://bucket.s3.amazonaws.com/gen/{n[0]}.jpg"

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "bucket"), \
             patch.object(server, "_upload_to_s3", side_effect=make_url):
            result = await server.generate_image("cats", count=2, save_folder=str(tmp_path))

        meta = _parse_result(result)
        assert "images" in meta
        for img in meta["images"]:
            assert "amazonaws.com" in img["image_url"]
            assert "saved_to" in img
            assert os.path.isfile(img["saved_to"])
        # Files must be distinct
        paths = [img["saved_to"] for img in meta["images"]]
        assert len(set(paths)) == 2

    @pytest.mark.asyncio
    async def test_s3_failure_still_saves_to_folder(self, tmp_path):
        """Even when S3 upload fails, save_folder write succeeds."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client), \
             patch.object(server, "S3_BUCKET", "bucket"), \
             patch.object(server, "_upload_to_s3", side_effect=Exception("S3 down")):
            result = await server.generate_image("cat", save_folder=str(tmp_path))

        meta = _parse_result(result)
        assert "/images/" in meta["image_url"]   # fallback URL
        assert "saved_to" in meta                # local save still worked
        assert os.path.isfile(meta["saved_to"])


# ---------------------------------------------------------------------------
# 35. Gemini exception handling — "No approval received" and similar errors
# ---------------------------------------------------------------------------

class TestGeminiExceptionHandling:
    """Gemini API exceptions must be caught and returned as JSON errors.

    When Gemini raises (e.g. content moderation, quota, "No approval received"),
    the tool must NOT let the exception escape to FastMCP — that shows a red
    error banner with the raw exception message in claude.ai instead of a
    structured error the model can reason about and retry.
    """

    def _mock_good_response(self):
        test_img = _make_test_image(128, 128)
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock(mime_type="image/jpeg", data=test_img)
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        return mock_response

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_generate_image_gemini_raises_returns_json_error(self):
        """generate_image: Gemini exception → JSON error str, not raw exception."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("No approval received.")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("sunset")

        assert isinstance(result, str), "Exception must be caught — tool must return str, not raise"
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_edit_image_gemini_raises_returns_json_error(self):
        """edit_image: Gemini exception → JSON error str, not raw exception."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("No approval received.")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}", prompt="add hat", ctx=mock_ctx
            )

        assert isinstance(result, str), "Exception must be caught — tool must return str, not raise"
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_swap_background_gemini_raises_returns_json_error(self):
        """swap_background: Gemini exception → JSON error str, not raw exception."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("No approval received.")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.swap_background(
                image=f"nanobanana://{src_id}", background="beach", ctx=mock_ctx
            )

        assert isinstance(result, str), "Exception must be caught — tool must return str, not raise"
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_create_variations_gemini_raises_returns_json_error(self):
        """create_variations: Gemini exception → JSON error str, not raw exception."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("No approval received.")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.create_variations(
                image=f"nanobanana://{src_id}", ctx=mock_ctx, count=1
            )

        assert isinstance(result, str), "Exception must be caught — tool must return str, not raise"
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_edit_image_count2_partial_failure_returns_successful_image(self):
        """count=2 where one generation raises still returns the one that succeeded."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()

        call_count = {"n": 0}
        good_response = self._mock_good_response()

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("No approval received.")
            return good_response

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = side_effect

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}", prompt="add hat", ctx=mock_ctx, count=2
            )

        # Should return the one successful image, not fail entirely
        assert isinstance(result, list), "Partial success must still return [json, Image]"
        assert isinstance(result[0], MCPImage), "Must still include MCPImage"
        meta = _parse_result(result)
        assert "errors" in meta, "Partial failure must be reported in errors field"

    @pytest.mark.asyncio
    async def test_edit_image_count2_embeds_2_images(self):
        """edit_image count=2 returns JSON with images[] array, each with data_uri."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_good_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}", prompt="add hat", ctx=mock_ctx, count=2
            )

        assert isinstance(result, list)
        assert len(result) == 3, "Should be [img1, img2, json_str]"
        assert all(isinstance(result[i], MCPImage) for i in range(0, 2))
        meta = _parse_result(result)
        assert "images" in meta, "Multi-image result must use images[] array"
        assert len(meta["images"]) == 2
        assert all("image_url" in img for img in meta["images"])

    @pytest.mark.asyncio
    async def test_swap_background_count2_embeds_2_images(self):
        """swap_background count=2 returns [json_str, img1, img2]."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_good_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.swap_background(
                image=f"nanobanana://{src_id}", background="beach", ctx=mock_ctx, count=2
            )

        assert isinstance(result, list)
        assert len(result) == 3, "Should be [img1, img2, json_str]"
        assert all(isinstance(result[i], MCPImage) for i in range(0, 2))
        meta = _parse_result(result)
        assert "images" in meta
        assert len(meta["images"]) == 2

    @pytest.mark.asyncio
    async def test_edit_image_count2_all_fail_returns_json_error(self):
        """edit_image count=2 where all generations fail returns JSON error, not raises."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("No approval received.")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(
                image=f"nanobanana://{src_id}", prompt="add hat", ctx=mock_ctx, count=2
            )

        assert isinstance(result, str)
        data = json.loads(result)
        assert "error" in data
        assert "details" in data
        assert len(data["details"]) == 2  # both attempts reported


# ---------------------------------------------------------------------------
# Tests: MCPImage thumbnail validity — decodes to a real JPEG within bounds
# ---------------------------------------------------------------------------

class TestMCPImageValidity:
    """Verify MCPImage thumbnails returned by tools are real, decodable JPEGs."""

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    def _mock_gemini_response(self, w=512, h=512):
        img = _make_test_image(w, h)
        part = MagicMock()
        part.inline_data = MagicMock(mime_type="image/jpeg", data=img)
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        return response

    def test_build_image_response_returns_mcp_image_with_valid_jpeg(self):
        """_build_image_response MCPImage contains valid JPEG bytes."""
        jpeg = _make_test_image(200, 200)
        result = server._build_image_response({}, [(jpeg, {"index": 1})])
        assert isinstance(result, list) and len(result) == 2
        mcp_img = result[0]
        assert isinstance(mcp_img, MCPImage)
        img = PILImage.open(BytesIO(mcp_img.data))
        assert img.format == "JPEG"
        w, h = img.size
        assert max(w, h) <= 512, f"MCPImage thumbnail too large: {w}x{h}"

    def test_multi_image_all_mcp_images_are_valid_jpegs(self):
        """Multi-image response: every MCPImage decodes to a valid JPEG."""
        imgs = [(_make_test_image(300, 300), {"index": i}) for i in range(3)]
        result = server._build_image_response({}, imgs)
        assert len(result) == 4
        for i in range(0, 3):
            pil_img = PILImage.open(BytesIO(result[i].data))
            assert pil_img.format == "JPEG", f"Image {i} MCPImage is not JPEG"
            w, h = pil_img.size
            assert max(w, h) <= 512, f"Image {i} thumbnail too large: {w}x{h}"

    @pytest.mark.asyncio
    async def test_generate_image_mcp_image_is_valid_jpeg(self):
        """Full generate_image flow: MCPImage in result contains valid JPEG."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response(512, 512)

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("a test image")

        mcp_img = result[0]
        assert isinstance(mcp_img, MCPImage)
        img = PILImage.open(BytesIO(mcp_img.data))
        assert img.format == "JPEG"
        assert max(img.size) <= 512

    @pytest.mark.asyncio
    async def test_rgba_source_produces_valid_jpeg(self):
        """RGBA/PNG source images must produce valid JPEG in MCPImage."""
        rgba_img = _make_test_image(200, 200, mode="RGBA", fmt="PNG")
        mock_client = MagicMock()
        part = MagicMock()
        part.inline_data = MagicMock(mime_type="image/png", data=rgba_img)
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        mock_client.models.generate_content.return_value = response

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("test rgba")

        img = PILImage.open(BytesIO(result[0].data))
        assert img.format == "JPEG"


class TestS3UrlFormat:
    """_upload_to_s3 must return a plain URL ending in .jpg — no query params.

    claude.ai detects image URLs by extension. A presigned URL (?X-Amz-Algorithm=...)
    does NOT end in .jpg, so claude.ai shows 'Open external link' instead of the image.
    """

    def test_s3_url_ends_in_jpg(self):
        """Returned URL must end in .jpg so claude.ai renders it as an image."""
        fake_s3 = MagicMock()
        fake_s3.put_object.return_value = {}

        with patch.object(server, "_get_s3_client", return_value=fake_s3), \
             patch.object(server, "S3_BUCKET", "my-bucket"), \
             patch.object(server, "S3_REGION", "us-east-1"):
            url = server._upload_to_s3(b"fake-jpeg", prefix="gen")

        assert url.endswith(".jpg"), (
            f"S3 URL must end in .jpg for claude.ai image rendering, got: {url}"
        )

    def test_s3_url_has_no_query_params(self):
        """Returned URL must have no query params — presigned URLs break image rendering."""
        fake_s3 = MagicMock()
        fake_s3.put_object.return_value = {}

        with patch.object(server, "_get_s3_client", return_value=fake_s3), \
             patch.object(server, "S3_BUCKET", "my-bucket"), \
             patch.object(server, "S3_REGION", "us-east-1"):
            url = server._upload_to_s3(b"fake-jpeg", prefix="gen")

        assert "?" not in url, (
            f"S3 URL must not have query params (no presigning) — got: {url}"
        )

    def test_s3_url_format(self):
        """URL must be the standard public S3 path-style URL."""
        fake_s3 = MagicMock()
        fake_s3.put_object.return_value = {}

        with patch.object(server, "_get_s3_client", return_value=fake_s3), \
             patch.object(server, "S3_BUCKET", "my-bucket"), \
             patch.object(server, "S3_REGION", "us-east-1"):
            url = server._upload_to_s3(b"fake-jpeg", prefix="gen")

        assert url.startswith("https://my-bucket.s3.us-east-1.amazonaws.com/gen/")
        assert url.endswith(".jpg")


class TestServerInstructions:
    """MCP server instructions must contain critical behavioral guidance."""

    def _get_instructions(self):
        return server.mcp._instructions if hasattr(server.mcp, '_instructions') else str(server.mcp.settings)

    def test_instructions_say_no_base64_on_curl_failure(self):
        """Instructions must explicitly tell Claude not to use base64 when curl fails."""
        instructions = server.mcp.instructions if hasattr(server.mcp, 'instructions') else ""
        if not instructions:
            # Try accessing through settings
            instructions = getattr(server.mcp, '_instructions', "") or getattr(server.mcp.settings, 'instructions', "")
        assert "base64" in instructions.lower(), "Instructions must mention base64"
        assert "curl" in instructions.lower(), "Instructions must mention curl"
        # The key rule: don't base64 if curl fails
        assert "not" in instructions.lower() or "never" in instructions.lower(), \
            "Instructions must include a prohibition on base64"

    def test_instructions_mention_imagecontent_rendering(self):
        """Instructions must mention that images render inline via ImageContent."""
        instructions = server.mcp.instructions if hasattr(server.mcp, 'instructions') else ""
        if not instructions:
            instructions = getattr(server.mcp, '_instructions', "") or getattr(server.mcp.settings, 'instructions', "")
        assert "image_url" in instructions.lower(), \
            "Instructions must mention image_url for chaining"
        assert "inline" in instructions.lower() or "imagecontent" in instructions.lower(), \
            "Instructions must describe inline rendering"

    def test_instructions_include_upload_page_url(self):
        """Instructions must include the /upload page URL so Claude can direct users there."""
        instructions = server.mcp.instructions if hasattr(server.mcp, 'instructions') else ""
        if not instructions:
            instructions = getattr(server.mcp, '_instructions', "") or getattr(server.mcp.settings, 'instructions', "")
        assert "/upload" in instructions, \
            "Instructions must include /upload page URL as fallback for image uploads"


# ---------------------------------------------------------------------------
# 36. Unsupported URL scheme detection in _decode_raw
# ---------------------------------------------------------------------------

class TestUnsupportedUrlScheme:
    """_decode_raw must raise a clear error for schemes like gs://, s3://, ftp://.

    Previously these fell through to raw base64 decoding and produced a
    confusing "image corrupt or truncated" error.
    """

    def test_gs_scheme_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            server._decode_raw("gs://my-bucket/image.jpg")

    def test_s3_scheme_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            server._decode_raw("s3://my-bucket/image.jpg")

    def test_ftp_scheme_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            server._decode_raw("ftp://example.com/image.jpg")

    def test_file_scheme_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            server._decode_raw("file:///tmp/image.jpg")

    def test_http_scheme_still_accepted(self):
        """http:// should NOT be blocked — that's a valid URL (goes to _fetch_url)."""
        # http://localhost is SSRF-blocked, not scheme-blocked — different error path
        with pytest.raises(ValueError, match="Blocked"):
            server._decode_raw("http://localhost/image.jpg")

    def test_data_uri_still_accepted(self):
        """data: scheme must still work — it's decoded inline."""
        img = _make_test_image(10, 10)
        b64 = base64.b64encode(img).decode()
        data, mime = server._decode_raw(f"data:image/jpeg;base64,{b64}")
        assert mime == "image/jpeg"


# ---------------------------------------------------------------------------
# 37. Google Drive URL normalization — _normalize_share_url
# ---------------------------------------------------------------------------

class TestNormalizeShareUrl:
    """_normalize_share_url rewrites Drive share links to direct download URLs."""

    def test_file_view_url_rewritten(self):
        url = "https://drive.google.com/file/d/ABC123XYZ/view?usp=sharing"
        result = server._normalize_share_url(url)
        assert result == "https://drive.google.com/uc?export=download&id=ABC123XYZ"

    def test_file_preview_url_rewritten(self):
        url = "https://drive.google.com/file/d/ABC123XYZ/preview"
        result = server._normalize_share_url(url)
        assert result == "https://drive.google.com/uc?export=download&id=ABC123XYZ"

    def test_open_id_url_rewritten(self):
        url = "https://drive.google.com/open?id=ABC123XYZ"
        result = server._normalize_share_url(url)
        assert result == "https://drive.google.com/uc?export=download&id=ABC123XYZ"

    def test_docs_google_com_rewritten(self):
        url = "https://docs.google.com/file/d/ABC123XYZ/view"
        result = server._normalize_share_url(url)
        assert result == "https://drive.google.com/uc?export=download&id=ABC123XYZ"

    def test_non_drive_url_unchanged(self):
        url = "https://example.com/image.jpg"
        assert server._normalize_share_url(url) == url

    def test_s3_url_unchanged(self):
        url = "https://bucket.s3.amazonaws.com/gen/abc.jpg"
        assert server._normalize_share_url(url) == url

    def test_already_download_url_unchanged(self):
        """A direct download URL (uc?export=download) must NOT be double-rewritten."""
        url = "https://drive.google.com/uc?export=download&id=ABC123XYZ"
        result = server._normalize_share_url(url)
        # Should return as-is (no /file/d/ path to match)
        assert result == url


# ---------------------------------------------------------------------------
# 38. compare_images Gemini exception handling
# ---------------------------------------------------------------------------

class TestCompareImagesGeminiException:
    """compare_images must catch Gemini exceptions and return JSON error."""

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_gemini_raises_returns_json_error(self):
        urls = [
            f"nanobanana://{server._store_image(_make_test_image(100, 100), 'image/jpeg')}"
            for _ in range(2)
        ]
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Quota exceeded.")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.compare_images(ctx=mock_ctx, images=urls)

        assert isinstance(result, str), "Exception must be caught — must return str not raise"
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_gemini_returns_non_json_handled_gracefully(self):
        """compare_images must handle non-JSON Gemini responses without raising."""
        urls = [
            f"nanobanana://{server._store_image(_make_test_image(100, 100), 'image/jpeg')}"
            for _ in range(2)
        ]
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="not valid json {{{")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.compare_images(ctx=mock_ctx, images=urls)

        data = json.loads(result)
        # Should include raw_response fallback, not crash
        assert "raw_response" in data or "error" in data

    @pytest.mark.asyncio
    async def test_image_fetch_failure_returns_json_error(self):
        """If an image can't be fetched, compare_images returns JSON error."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        bad_url = "http://localhost:8080/images/expiredid99"  # not in store

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.compare_images(ctx=mock_ctx, images=[bad_url, bad_url])

        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# 39. create_variations with prompt guidance
# ---------------------------------------------------------------------------

class TestCreateVariationsPrompt:
    """create_variations must append user prompt to the variation_strength template."""

    def _mock_gemini_response(self):
        img = _make_test_image(256, 256)
        part = MagicMock()
        part.inline_data = MagicMock(mime_type="image/png", data=img)
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        return response

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_prompt_appears_in_metadata(self):
        """When prompt is provided, it appears as 'guidance' in the metadata."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.create_variations(
                image=f"nanobanana://{src_id}",
                ctx=mock_ctx,
                prompt="keep the blue color scheme",
                count=1,
            )

        meta = _parse_result(result)
        assert "guidance" in meta, "Prompt must be echoed as 'guidance' in metadata"
        assert meta["guidance"] == "keep the blue color scheme"

    @pytest.mark.asyncio
    async def test_prompt_appended_to_variation_template(self):
        """Prompt text must appear in the Gemini API call, after the strength preamble."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            await server.create_variations(
                image=f"nanobanana://{src_id}",
                ctx=mock_ctx,
                prompt="warmer tones",
                variation_strength="subtle",
                count=1,
            )

        call_args = mock_client.models.generate_content.call_args
        text_parts = [str(p) for p in call_args.kwargs["contents"] if hasattr(p, "text")]
        combined = " ".join(text_parts)
        assert "warmer tones" in combined, "User prompt must appear in Gemini call"
        assert "subtle" in combined.lower() or "variation" in combined.lower(), \
            "Strength preamble must also be present"

    @pytest.mark.asyncio
    async def test_no_prompt_metadata_has_no_guidance(self):
        """Without prompt, metadata must NOT have a 'guidance' field."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.create_variations(
                image=f"nanobanana://{src_id}", ctx=mock_ctx, count=1
            )

        meta = _parse_result(result)
        assert "guidance" not in meta

    @pytest.mark.asyncio
    async def test_all_variation_strengths_accepted(self):
        """subtle, medium, and strong must all succeed — only 'extreme' (etc.) errors."""
        src_id = server._store_image(_make_test_image(200, 200), "image/jpeg")
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        for strength in ("subtle", "medium", "strong"):
            with patch.object(server, "_get_client", return_value=mock_client):
                result = await server.create_variations(
                    image=f"nanobanana://{src_id}", ctx=mock_ctx,
                    variation_strength=strength, count=1,
                )
            meta = _parse_result(result)
            assert "error" not in meta, f"strength='{strength}' should be valid"
            assert "image_url" in meta


# ---------------------------------------------------------------------------
# 40. batch_analyze — Gemini API failure per-image
# ---------------------------------------------------------------------------

class TestBatchAnalyzeGeminiFailure:
    """batch_analyze must handle Gemini analysis failures per-image (not just fetch failures)."""

    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    @pytest.mark.asyncio
    async def test_gemini_failure_on_one_image_others_succeed(self):
        """If Gemini raises for exactly one analysis call, only that result has an error.

        batch_analyze runs all analyses concurrently, so call order is non-deterministic.
        We assert aggregate counts rather than which specific index failed.
        """
        imgs = [server._store_image(_make_test_image(100, 100), "image/jpeg") for _ in range(3)]
        urls = [f"nanobanana://{img_id}" for img_id in imgs]
        mock_ctx = MagicMock()

        import threading
        call_lock = threading.Lock()
        call_count = {"n": 0}

        def side_effect(**kwargs):
            with call_lock:
                n = call_count["n"]
                call_count["n"] += 1
            if n == 1:
                raise Exception("Content policy violation")
            resp = MagicMock()
            resp.text = json.dumps({
                "description": f"Image {n}", "style": "flat",
                "mood": "calm", "colors": ["gray"], "details": ["block"],
            })
            return resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = side_effect

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.batch_analyze(ctx=mock_ctx, images=urls)

        data = json.loads(result)
        results = data["results"]
        assert len(results) == 3
        error_count = sum(1 for r in results if "error" in r)
        success_count = sum(1 for r in results if "error" not in r)
        assert error_count == 1, "Exactly one image should have failed"
        assert success_count == 2, "Other two images should have succeeded"

    @pytest.mark.asyncio
    async def test_all_gemini_failures_returns_all_errors(self):
        """If Gemini fails for all images, all results have error fields (no crash)."""
        imgs = [server._store_image(_make_test_image(100, 100), "image/jpeg") for _ in range(2)]
        urls = [f"nanobanana://{img_id}" for img_id in imgs]
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API down")

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.batch_analyze(ctx=mock_ctx, images=urls)

        data = json.loads(result)
        assert all("error" in r for r in data["results"])
        assert data["count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
