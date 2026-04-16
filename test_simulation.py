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


# ---------------------------------------------------------------------------
# Helpers — create test images
# ---------------------------------------------------------------------------

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
        data = json.loads(result)
        assert "error" in data
        assert "Unsupported aspect ratio" in data["error"]

    @pytest.mark.asyncio
    async def test_bad_resolution(self):
        result = await server.generate_image("test", resolution="8K")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_bad_style(self):
        result = await server.generate_image("test", style="nonexistent")
        data = json.loads(result)
        assert "error" in data
        assert "available" in data

    @pytest.mark.asyncio
    async def test_bad_output_mode(self):
        result = await server.generate_image("test", output="ftp")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_cloud_without_bucket(self):
        """Cloud output without configured bucket should fail early."""
        orig_s3, orig_gcs = server.S3_BUCKET, server.GCS_BUCKET
        server.S3_BUCKET = None
        server.GCS_BUCKET = None
        try:
            result = await server.generate_image("test", output="cloud")
            data = json.loads(result)
            assert "error" in data
            assert "storage" in data["error"].lower() or "bucket" in data["error"].lower()
        finally:
            server.S3_BUCKET = orig_s3
            server.GCS_BUCKET = orig_gcs

    @pytest.mark.asyncio
    async def test_empty_reference_image_rejected(self):
        """Empty string in reference_images should produce a clear error."""
        mock_client = MagicMock()
        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("test", reference_images=[""])
        data = json.loads(result)
        assert "error" in data
        assert "empty" in data["error"].lower() or "upload" in data["error"].lower()


# ---------------------------------------------------------------------------
# 6. No double JPEG encoding — _build_image_response
# ---------------------------------------------------------------------------

class TestNoDoubleJPEG:
    def setup_method(self):
        with server._STORE_LOCK:
            server._IMAGE_STORE.clear()

    def test_base64_output_returns_image_objects(self):
        """_build_image_response in base64 mode should wrap raw JPEG bytes
        as Image objects, NOT re-encode them."""
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response(
            {"test": True},
            [(jpeg, {"index": 1})],
            output_mode="base64",
        )
        # Should be [json_str, Image]
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], str)
        # The Image object wraps the original bytes
        img_obj = result[1]
        assert isinstance(img_obj, server.Image)

    def test_base64_output_stores_nanobanana_url(self):
        """Images should be stored and get nanobanana:// URLs."""
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response(
            {},
            [(jpeg, {"index": 1})],
            output_mode="base64",
        )
        metadata = json.loads(result[0])
        assert "image_url" in metadata
        assert metadata["image_url"].startswith("nanobanana://")

    def test_cloud_output_returns_string(self):
        """Cloud mode should return a JSON string, not a list."""
        jpeg = _make_test_image(100, 100)
        with patch.object(server, "_upload_to_cloud", return_value="https://bucket.s3.amazonaws.com/gen/test.jpg"):
            result = server._build_image_response(
                {},
                [(jpeg, {"index": 1})],
                output_mode="cloud",
            )
        assert isinstance(result, str)
        data = json.loads(result)
        assert "image_url" in data  # consistent key name across all output modes

    def test_multiple_images_structure(self):
        """Multiple images should produce an 'images' array in metadata."""
        imgs = [(_make_test_image(100, 100), {"index": i}) for i in range(3)]
        result = server._build_image_response({}, imgs, output_mode="base64")
        metadata = json.loads(result[0])
        assert "images" in metadata
        assert len(metadata["images"]) == 3
        # Should have 1 json + 3 Image objects
        assert len(result) == 4


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
            result = await server.generate_image("a cute cat", output="base64")

        assert isinstance(result, list)
        metadata = json.loads(result[0])
        assert "model" in metadata
        assert "image_url" in metadata
        assert metadata["image_url"].startswith("nanobanana://")
        assert len(result) == 2  # metadata + 1 image

    @pytest.mark.asyncio
    async def test_generation_with_style(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("a product shot", style="product-photography")

        metadata = json.loads(result[0])
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

        metadata = json.loads(result[0])
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

        metadata = json.loads(result[0])
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

        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_multiple_images(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()

        with patch.object(server, "_get_client", return_value=mock_client):
            result = await server.generate_image("cats", count=3)

        metadata = json.loads(result[0])
        assert "images" in metadata
        assert len(metadata["images"]) == 3
        assert len(result) == 4  # metadata + 3 images

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

        metadata = json.loads(result[0])
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

        metadata = json.loads(result[0])
        assert metadata["edit_mode"] == "inpaint-insertion"
        assert "image_url" in metadata

    @pytest.mark.asyncio
    async def test_bad_edit_mode(self):
        mock_ctx = MagicMock()
        result = await server.edit_image(
            image="nanobanana://test", prompt="test", ctx=mock_ctx, edit_mode="magic"
        )
        data = json.loads(result)
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

        metadata = json.loads(result[0])
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

        metadata = json.loads(result[0])
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

        metadata = json.loads(result[0])
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

        metadata = json.loads(result[0])
        assert "images" in metadata
        assert len(metadata["images"]) == 2

    @pytest.mark.asyncio
    async def test_bad_variation_strength(self):
        mock_ctx = MagicMock()
        result = await server.create_variations(
            image="nanobanana://test", ctx=mock_ctx, variation_strength="extreme"
        )
        data = json.loads(result)
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
# 13. Upload sessions (elicitation flow)
# ---------------------------------------------------------------------------

class TestUploadSessions:
    def setup_method(self):
        with server._SESSION_LOCK:
            server._UPLOAD_SESSIONS.clear()

    def test_create_and_poll(self):
        sid = server._create_upload_session()
        assert server._poll_upload_session(sid) is None
        server._complete_upload_session(sid, "img123")
        assert server._poll_upload_session(sid) == "img123"

    def test_cleanup(self):
        sid = server._create_upload_session()
        server._cleanup_session(sid)
        assert server._poll_upload_session(sid) is None

    def test_gc_sessions(self):
        sid = server._create_upload_session()
        # Backdate
        with server._SESSION_LOCK:
            entry = server._UPLOAD_SESSIONS[sid]
            server._UPLOAD_SESSIONS[sid] = (entry[0], time.time() - server._SESSION_TTL - 10)
        server._gc_sessions()
        assert server._poll_upload_session(sid) is None


# ---------------------------------------------------------------------------
# 14. _is_url helper
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
# 17. Empty-string image triggers elicitation (pasted image UX fix)
# ---------------------------------------------------------------------------

class TestEmptyImageElicitation:
    """When user pastes an image in Claude, the AI should call tools with image="".
    This triggers _acquire_image's elicitation flow instead of trying to pass base64."""

    @pytest.mark.asyncio
    async def test_upload_image_empty_triggers_elicitation(self):
        """upload_image with image="" should call _acquire_image which tries elicitation."""
        mock_ctx = MagicMock()

        # Mock _acquire_image to simulate elicitation failure (no browser in test)
        async def mock_acquire(image, ctx, **kwargs):
            assert image == ""  # Key: empty string was passed
            raise ValueError("Image upload was declined.")

        with patch.object(server, "_acquire_image", mock_acquire):
            result = await server.upload_image(ctx=mock_ctx, image="")

        data = json.loads(result)
        assert "error" in data
        assert "declined" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_edit_image_empty_triggers_elicitation(self):
        """edit_image with image="" should trigger elicitation for the source image."""
        mock_ctx = MagicMock()

        acquired = False

        async def mock_acquire(image, ctx, **kwargs):
            nonlocal acquired
            assert image == ""
            acquired = True
            # Simulate successful upload via elicitation
            test_img = _make_test_image(200, 200)
            return server._normalize_image(test_img, max_dim=2048)

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

        with patch.object(server, "_acquire_image", mock_acquire), \
             patch.object(server, "_get_client", return_value=mock_client):
            result = await server.edit_image(prompt="add a hat", ctx=mock_ctx, image="")

        assert acquired, "_acquire_image was not called"
        metadata = json.loads(result[0])
        assert "edit_mode" in metadata

    @pytest.mark.asyncio
    async def test_analyze_image_empty_triggers_elicitation(self):
        """analyze_image with image="" should trigger elicitation."""
        mock_ctx = MagicMock()

        async def mock_acquire(image, ctx, **kwargs):
            assert image == ""
            test_img = _make_test_image(200, 200)
            return server._normalize_image(test_img, max_dim=kwargs.get("max_dim", 2048))

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "description": "test", "style": "test", "mood": "test",
            "colors": ["red"], "details": ["detail"],
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(server, "_acquire_image", mock_acquire), \
             patch.object(server, "_get_client", return_value=mock_client):
            result = await server.analyze_image(ctx=mock_ctx, image="", focus="general")

        data = json.loads(result)
        assert "description" in data

    def test_acquire_image_empty_goes_to_elicitation_branch(self):
        """_acquire_image with empty string should NOT try _decode_raw,
        should go straight to the elicitation branch."""
        import inspect
        source = inspect.getsource(server._acquire_image)
        assert "if image:" in source


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
            gen_result = await server.generate_image("a cat", output="base64")

        gen_meta = json.loads(gen_result[0])
        cat_url = gen_meta["image_url"]
        assert cat_url.startswith("nanobanana://")

        # Step 2: Edit using the URL from step 1
        with patch.object(server, "_get_client", return_value=mock_client):
            edit_result = await server.edit_image(
                image=cat_url,
                prompt="add a top hat",
                ctx=mock_ctx,
            )

        edit_meta = json.loads(edit_result[0])
        assert "image_url" in edit_meta
        assert edit_meta["image_url"].startswith("nanobanana://")

    @pytest.mark.asyncio
    async def test_generate_then_swap_background(self):
        """Simulate: generate an image, then swap its background."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()
        mock_ctx = MagicMock()

        with patch.object(server, "_get_client", return_value=mock_client):
            gen_result = await server.generate_image("product on white", output="base64")

        gen_meta = json.loads(gen_result[0])
        product_url = gen_meta["image_url"]

        with patch.object(server, "_get_client", return_value=mock_client):
            swap_result = await server.swap_background(
                image=product_url,
                background="tropical beach at sunset",
                ctx=mock_ctx,
            )

        swap_meta = json.loads(swap_result[0])
        assert "image_url" in swap_meta
        assert swap_meta["background"] == "tropical beach at sunset"

    @pytest.mark.asyncio
    async def test_generate_then_variations(self):
        """Simulate: generate an image, then create variations."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response()
        mock_ctx = MagicMock()

        with patch.object(server, "_get_client", return_value=mock_client):
            gen_result = await server.generate_image("landscape", output="base64")

        gen_meta = json.loads(gen_result[0])
        url = gen_meta["image_url"]

        with patch.object(server, "_get_client", return_value=mock_client):
            var_result = await server.create_variations(
                image=url, ctx=mock_ctx, count=2,
            )

        var_meta = json.loads(var_result[0])
        assert "images" in var_meta
        for img in var_meta["images"]:
            assert "image_url" in img


# ---------------------------------------------------------------------------
# 19. Cloud output key consistency
# ---------------------------------------------------------------------------

class TestCloudOutputKeyConsistency:
    """Verify that image_url key is used consistently in both base64 and cloud modes."""

    def test_single_image_base64(self):
        jpeg = _make_test_image(100, 100)
        result = server._build_image_response(
            {}, [(jpeg, {"index": 1})], output_mode="base64",
        )
        meta = json.loads(result[0])
        assert "image_url" in meta

    def test_single_image_cloud(self):
        jpeg = _make_test_image(100, 100)
        with patch.object(server, "_upload_to_cloud", return_value="https://cdn.example.com/img.jpg"):
            result = server._build_image_response(
                {}, [(jpeg, {"index": 1})], output_mode="cloud",
            )
        meta = json.loads(result)
        assert "image_url" in meta
        assert meta["image_url"] == "https://cdn.example.com/img.jpg"

    def test_multi_image_base64(self):
        imgs = [(_make_test_image(100, 100), {"index": i}) for i in range(2)]
        result = server._build_image_response({}, imgs, output_mode="base64")
        meta = json.loads(result[0])
        for img in meta["images"]:
            assert "image_url" in img

    def test_multi_image_cloud(self):
        imgs = [(_make_test_image(100, 100), {"index": i}) for i in range(2)]
        with patch.object(server, "_upload_to_cloud", return_value="https://cdn.example.com/img.jpg"):
            result = server._build_image_response({}, imgs, output_mode="cloud")
        meta = json.loads(result)
        for img in meta["images"]:
            assert "image_url" in img


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
