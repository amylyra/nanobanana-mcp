"""
NanoBanana MCP Server — Image generation, editing, and variations via Gemini.

Tools:
  - generate_image:    Text-to-image with references, styles, prompt enhancement, QA.
  - edit_image:        Edit an existing image (inpaint, remove, outpaint).
  - swap_background:   Keep foreground subject, replace background.
  - create_variations: Generate variations of an existing image.
  - analyze_image:     Describe/tag an image using Gemini vision.
  - list_styles:       List available style presets.

Features:
  - Image QA: optional AI scoring of generated images (composition, clarity, etc.)
  - GCS output: optionally save images to Google Cloud Storage and return URLs
  - Background swap: one-step foreground preservation + background replacement

Deployment: Cloud Run with Streamable HTTP transport.
"""

import base64
import json
import os
import uuid
from io import BytesIO
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_FLASH = "gemini-3.1-flash-image-preview"   # NanoBanana 2 — fast
MODEL_PRO = "gemini-3-pro-image-preview"          # NanoBanana Pro — higher quality
MODEL_TEXT = "gemini-2.5-flash"                    # For prompt enhancement + QA + analysis

SUPPORTED_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5",
    "5:4", "9:16", "16:9", "21:9",
}
RESOLUTIONS = {"0.5K", "1K", "2K", "4K"}

# ---------------------------------------------------------------------------
# Image size limits (max dimension in pixels)
# ---------------------------------------------------------------------------
REF_MAX_DIM = 1024    # Reference images — preserves logo/label detail
SOURCE_MAX_DIM = 2048  # Source images for edit/variations — high quality but bounded

# ---------------------------------------------------------------------------
# GCS config (optional — set GCS_BUCKET env var to enable)
# ---------------------------------------------------------------------------
GCS_BUCKET = os.environ.get("GCS_BUCKET")  # e.g. "my-nanobanana-images"

# ---------------------------------------------------------------------------
# Style presets — tuned prompt prefixes
# ---------------------------------------------------------------------------
STYLE_PRESETS = {
    "cinematic": (
        "Cinematic film still, dramatic lighting, shallow depth of field, "
        "anamorphic lens flare, color graded with rich shadows and highlights. "
    ),
    "product-photography": (
        "Professional product photography, clean white background, studio lighting "
        "with soft shadows, high-end commercial look, sharp focus on product details. "
    ),
    "editorial": (
        "Editorial magazine photography, sophisticated composition, natural light, "
        "muted tones, fashion-forward styling, clean and minimal. "
    ),
    "watercolor": (
        "Delicate watercolor painting, soft translucent washes, visible paper texture, "
        "organic bleeding edges, gentle color gradients, fine artistic detail. "
    ),
    "flat-illustration": (
        "Modern flat illustration, clean vector style, bold geometric shapes, "
        "limited color palette, minimal shadows, graphic design aesthetic. "
    ),
    "neon-noir": (
        "Neon noir aesthetic, dark moody scene, vibrant neon reflections on wet surfaces, "
        "cyberpunk atmosphere, dramatic contrast between shadows and neon glow. "
    ),
    "minimalist": (
        "Minimalist composition, vast negative space, single focal point, "
        "muted desaturated palette, clean lines, zen-like simplicity. "
    ),
    "vintage-film": (
        "Vintage analog film photograph, warm color cast, visible grain, "
        "slightly faded highlights, nostalgic 1970s aesthetic, soft vignette. "
    ),
}

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "nanobanana",
    instructions=(
        "NanoBanana image generation server powered by Gemini. "
        "Tools: generate_image (text-to-image with references, styles, prompt enhancement, QA), "
        "edit_image (inpaint, remove objects, outpaint), "
        "swap_background (keep subject, replace background), "
        "create_variations (produce variations of an existing image), "
        "analyze_image (describe/tag an image). "
        "Default aspect ratio 4:5, resolution 1K. "
        "IMPORTANT: For best results, pass image URLs instead of base64 — "
        "the server fetches them directly, avoiding size/truncation issues with inline data. "
        "Set output='gcs' to return URLs instead of base64 (requires GCS_BUCKET env var)."
    ),
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", 8080)),
)


# ---------------------------------------------------------------------------
# Helpers — client
# ---------------------------------------------------------------------------
def _get_client():
    """Initialize Gemini client."""
    from google import genai

    api_key = os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set GOOGLE_AI_API_KEY or GEMINI_API_KEY env var."
        )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Helpers — image decode/encode
# ---------------------------------------------------------------------------
def _is_url(s: str) -> bool:
    try:
        parsed = urlparse(s)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def _fetch_url(url: str) -> tuple[bytes, str]:
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return resp.content, content_type


def _fix_base64_padding(s: str) -> str:
    s = s.rstrip()
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)
    return s


def _normalize_image(img_bytes: bytes, max_dim: int, quality: int = 85,
                     output_format: str = "JPEG") -> tuple[bytes, str]:
    from PIL import Image as PILImage

    try:
        img = PILImage.open(BytesIO(img_bytes))
    except Exception:
        raise ValueError(
            "Could not decode image data. The image may have been truncated in transit. "
            "Try passing an image URL instead of base64 for reliable results."
        )

    w, h = img.size
    if w < 1 or h < 1:
        raise ValueError(
            "Image has invalid dimensions. The data may be corrupted or truncated. "
            "Try passing an image URL instead of base64."
        )

    if output_format == "JPEG":
        if img.mode == "RGBA":
            bg = PILImage.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

    buf = BytesIO()
    if output_format == "PNG":
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"
    else:
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), "image/jpeg"


def _decode_raw(ref: str) -> tuple[bytes, str]:
    if _is_url(ref):
        return _fetch_url(ref)
    if ref.startswith("data:"):
        header, data = ref.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        return base64.b64decode(_fix_base64_padding(data)), mime
    return base64.b64decode(_fix_base64_padding(ref)), "image/jpeg"


def _decode_reference(ref: str) -> tuple[bytes, str]:
    raw, _ = _decode_raw(ref)
    return _normalize_image(raw, max_dim=REF_MAX_DIM)


def _decode_source(ref: str) -> tuple[bytes, str]:
    raw, _ = _decode_raw(ref)
    return _normalize_image(raw, max_dim=SOURCE_MAX_DIM, quality=92)


def _decode_mask(ref: str) -> tuple[bytes, str]:
    raw, _ = _decode_raw(ref)
    return _normalize_image(raw, max_dim=SOURCE_MAX_DIM, output_format="PNG")


def _to_jpeg(img_bytes: bytes) -> bytes:
    from PIL import Image as PILImage

    img = PILImage.open(BytesIO(img_bytes))
    if img.mode == "RGBA":
        bg = PILImage.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()


def _pick_model(quality: str) -> str:
    return MODEL_PRO if quality == "pro" else MODEL_FLASH


def _extract_image(response) -> bytes | None:
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                return part.inline_data.data
    return None


def _build_ref_parts(reference_images: list | None) -> list:
    from google.genai import types

    parts = []
    if not reference_images:
        return parts
    for ref in reference_images:
        img_bytes, mime = _decode_reference(ref)
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    return parts


# ---------------------------------------------------------------------------
# Helpers — GCS output
# ---------------------------------------------------------------------------
def _upload_to_gcs(jpeg_bytes: bytes, prefix: str = "gen") -> str:
    """Upload JPEG bytes to GCS and return a public URL.

    Requires GCS_BUCKET env var and appropriate service account permissions.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob_name = f"{prefix}/{uuid.uuid4().hex}.jpg"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(jpeg_bytes, content_type="image/jpeg")
    blob.make_public()
    return blob.public_url


def _format_image_output(jpeg_bytes: bytes, output_mode: str, prefix: str = "gen") -> dict:
    """Format a JPEG image as either base64 data URI or GCS URL.

    Expects pre-converted JPEG bytes — does NOT re-encode.
    """
    if output_mode == "gcs":
        if not GCS_BUCKET:
            raise ValueError(
                "GCS output requested but GCS_BUCKET env var is not set. "
                "Use output='base64' or configure GCS_BUCKET on the server."
            )
        url = _upload_to_gcs(jpeg_bytes, prefix=prefix)
        return {"url": url, "size_kb": len(jpeg_bytes) // 1024}
    else:
        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        return {
            "data_uri": f"data:image/jpeg;base64,{b64}",
            "size_kb": len(jpeg_bytes) // 1024,
        }


# ---------------------------------------------------------------------------
# Helpers — prompt enhancement
# ---------------------------------------------------------------------------
def _enhance_prompt(client, prompt: str) -> str:
    system = (
        "You are an expert image prompt engineer. Given a short description, "
        "expand it into a detailed, vivid image generation prompt. "
        "Include specifics about lighting, composition, mood, color palette, "
        "and technical camera details where appropriate. "
        "Keep it under 200 words. Return ONLY the enhanced prompt, no explanation."
    )
    resp = client.models.generate_content(
        model=MODEL_TEXT,
        contents=prompt,
        config={"system_instruction": system},
    )
    return resp.text.strip() if resp.text else prompt


# ---------------------------------------------------------------------------
# Helpers — image QA
# ---------------------------------------------------------------------------
QA_SYSTEM_PROMPT = (
    "You are an expert image quality analyst. Score this generated image on each "
    "criterion below from 1-10 and provide a one-sentence rationale for each.\n\n"
    "Criteria:\n"
    "- composition: framing, rule of thirds, visual balance, focal point\n"
    "- clarity: sharpness, absence of blur or artifacts, detail quality\n"
    "- lighting: natural/intentional lighting, exposure, shadow quality\n"
    "- color: palette harmony, saturation balance, color accuracy\n"
    "- prompt_adherence: how well the image matches the generation prompt\n\n"
    "Return valid JSON only, no markdown fences:\n"
    '{"composition": {"score": N, "note": "..."}, "clarity": {"score": N, "note": "..."}, '
    '"lighting": {"score": N, "note": "..."}, "color": {"score": N, "note": "..."}, '
    '"prompt_adherence": {"score": N, "note": "..."}, "total": N}'
)


def _score_image(client, img_bytes: bytes, prompt: str) -> dict:
    """Score a generated image using Gemini vision."""
    from google.genai import types

    try:
        resp = client.models.generate_content(
            model=MODEL_TEXT,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                types.Part.from_text(
                    text=f"The image was generated from this prompt: \"{prompt}\"\n\n"
                         "Score this image."
                ),
            ],
            config={
                "system_instruction": QA_SYSTEM_PROMPT,
                "response_mime_type": "application/json",
            },
        )
        scores = json.loads(resp.text)
        # Ensure total is computed
        if "total" not in scores or not isinstance(scores["total"], (int, float)):
            criteria = ["composition", "clarity", "lighting", "color", "prompt_adherence"]
            total = sum(scores.get(c, {}).get("score", 0) for c in criteria)
            scores["total"] = total
        return scores
    except Exception as e:
        return {"error": f"QA scoring failed: {e}", "total": 0}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def generate_image(
    prompt: str,
    reference_images: list[str] | None = None,
    style: str | None = None,
    enhance_prompt: bool = False,
    aspect_ratio: str = "4:5",
    resolution: str = "1K",
    quality: str = "default",
    count: int = 1,
    qa: bool = False,
    output: str = "base64",
) -> str:
    """Generate an image from a text prompt with optional reference images and style presets.

    Reference images guide the model on style, subject appearance, or composition.
    URLs are strongly recommended over base64 — the server fetches them directly,
    avoiding truncation issues with large inline data.

    Args:
        prompt: What to generate. Describe subject, style, lighting, mood, etc.
        reference_images: Optional list of reference images. Accepts:
                          - Image URLs (recommended — fetched server-side, most reliable)
                          - Base64 data URIs (data:image/jpeg;base64,...)
                          - Raw base64 strings
                          Multiple references supported for combining style + subject cues.
        style: Optional style preset. Available: cinematic, product-photography,
               editorial, watercolor, flat-illustration, neon-noir, minimalist, vintage-film.
        enhance_prompt: If true, AI expands your prompt into a detailed generation prompt.
                        Great for short prompts. Default: false
        aspect_ratio: Output aspect ratio. Default: 4:5
        resolution: Output resolution: 0.5K, 1K, 2K, 4K. Default: 1K
        quality: "default" (fast) or "pro" (higher quality). Default: default
        count: Number of images to generate (1–4). Default: 1
        qa: If true, each image is scored by AI on composition, clarity, lighting,
            color, and prompt adherence (1-10 each). When count > 1, images are
            ranked by total score with the best first. Default: false
        output: "base64" returns data URIs (default). "gcs" uploads to Google Cloud
                Storage and returns public URLs (requires GCS_BUCKET env var on server).

    Returns:
        JSON with image data (base64 or URL), metadata, QA scores (if qa=true).
    """
    from google.genai import types

    if aspect_ratio not in SUPPORTED_RATIOS:
        return json.dumps({"error": f"Unsupported aspect ratio '{aspect_ratio}'.", "supported": sorted(SUPPORTED_RATIOS)})
    if resolution not in RESOLUTIONS:
        return json.dumps({"error": f"Unsupported resolution '{resolution}'.", "supported": sorted(RESOLUTIONS)})
    if style and style not in STYLE_PRESETS:
        return json.dumps({"error": f"Unknown style '{style}'.", "available": sorted(STYLE_PRESETS.keys())})
    if output not in ("base64", "gcs"):
        return json.dumps({"error": f"Unknown output mode '{output}'.", "available": ["base64", "gcs"]})
    if output == "gcs" and not GCS_BUCKET:
        return json.dumps({"error": "GCS output requested but GCS_BUCKET env var is not set on the server. Use output='base64' instead."})

    count = max(1, min(count, 4))

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    # Build final prompt
    final_prompt = prompt
    if style:
        final_prompt = STYLE_PRESETS[style] + final_prompt
    if enhance_prompt:
        try:
            final_prompt = _enhance_prompt(client, final_prompt)
        except Exception as e:
            return json.dumps({"error": f"Prompt enhancement failed: {e}"})

    # Build content parts
    try:
        parts = _build_ref_parts(reference_images)
    except Exception as e:
        return json.dumps({"error": f"Failed to process reference image: {e}"})
    parts.append(types.Part.from_text(text=final_prompt))

    model_name = _pick_model(quality)
    images = []
    errors = []

    for i in range(count):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution,
                    ),
                ),
            )
            img_bytes = _extract_image(response)
            if img_bytes:
                jpeg_bytes = _to_jpeg(img_bytes)
                entry = _format_image_output(jpeg_bytes, output, prefix="gen")
                if qa:
                    entry["qa"] = _score_image(client, jpeg_bytes, final_prompt)
                entry["index"] = i + 1
                images.append(entry)
            else:
                errors.append(f"Image {i + 1}: no image in response")
        except Exception as e:
            errors.append(f"Image {i + 1}: {e}")

    if not images:
        return json.dumps({"error": "All generation attempts failed.", "details": errors})

    # Rank by QA score if scoring was enabled
    if qa and len(images) > 1:
        images.sort(key=lambda x: x.get("qa", {}).get("total", 0), reverse=True)
        for i, img in enumerate(images):
            img["rank"] = i + 1

    result = {
        "model": model_name,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "prompt_used": final_prompt[:200] + "..." if len(final_prompt) > 200 else final_prompt,
    }
    if enhance_prompt or style:
        result["original_prompt"] = prompt[:120] + "..." if len(prompt) > 120 else prompt
    if style:
        result["style"] = style
    if reference_images:
        result["reference_count"] = len(reference_images)

    if count == 1:
        result.update(images[0])
        result.pop("index", None)
    else:
        result["images"] = images
        if errors:
            result["errors"] = errors

    return json.dumps(result)


@mcp.tool()
def edit_image(
    image: str,
    prompt: str,
    mask: str | None = None,
    edit_mode: str = "inpaint-insertion",
    aspect_ratio: str | None = None,
    count: int = 1,
    output: str = "base64",
) -> str:
    """Edit an existing image — add objects, remove objects, or extend the canvas.

    Pass the source image as a URL (recommended) or base64.
    Provide a mask for precise control, or let the model infer the edit region.

    Args:
        image: The source image. URL (recommended), base64 data URI, or raw base64.
        prompt: Edit instruction. Be specific about what to change.
        mask: Optional mask image (URL or base64). White = edit region, black = preserve.
              If omitted, the model uses automatic segmentation based on your prompt.
        edit_mode: "inpaint-insertion" (add/replace), "inpaint-removal" (remove + fill),
                   "outpaint" (extend canvas). Default: inpaint-insertion
        aspect_ratio: Output aspect ratio (useful for outpaint). Default: same as input.
        count: Number of candidates (1–4). Default: 1
        output: "base64" (default) or "gcs" (upload to GCS, return URL).

    Returns:
        JSON with edited image data (base64 or URL).
    """
    from google.genai import types

    edit_modes = {
        "inpaint-insertion": "EDIT_MODE_INPAINT_INSERTION",
        "inpaint-removal": "EDIT_MODE_INPAINT_REMOVAL",
        "outpaint": "EDIT_MODE_OUTPAINT",
    }
    if edit_mode not in edit_modes:
        return json.dumps({"error": f"Unknown edit_mode '{edit_mode}'.", "available": list(edit_modes.keys())})
    if output not in ("base64", "gcs"):
        return json.dumps({"error": f"Unknown output mode '{output}'.", "available": ["base64", "gcs"]})
    if output == "gcs" and not GCS_BUCKET:
        return json.dumps({"error": "GCS output requested but GCS_BUCKET env var is not set on the server. Use output='base64' instead."})

    count = max(1, min(count, 4))

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    try:
        img_bytes, img_mime = _decode_source(image)
    except Exception as e:
        return json.dumps({"error": str(e)})

    ref_images = [
        types.RawReferenceImage(
            reference_id=1,
            reference_image=types.Image(image_bytes=img_bytes, mime_type=img_mime),
        )
    ]

    if mask:
        try:
            mask_bytes, mask_mime = _decode_mask(mask)
            ref_images.append(
                types.MaskReferenceImage(
                    reference_id=2,
                    config=types.MaskReferenceConfig(
                        mask_mode="MASK_MODE_USER_PROVIDED",
                        mask_image=types.Image(image_bytes=mask_bytes, mime_type=mask_mime),
                    ),
                )
            )
        except Exception as e:
            return json.dumps({"error": f"Failed to decode mask image: {e}"})
    elif edit_mode != "outpaint":
        ref_images.append(
            types.MaskReferenceImage(
                reference_id=2,
                config=types.MaskReferenceConfig(
                    mask_mode="MASK_MODE_SEMANTIC",
                    mask_dilation=0.03,
                ),
            )
        )

    config = types.EditImageConfig(
        edit_mode=edit_modes[edit_mode],
        number_of_images=count,
    )
    if aspect_ratio:
        config.aspect_ratio = aspect_ratio

    try:
        response = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=prompt,
            reference_images=ref_images,
            config=config,
        )
    except Exception as e:
        return json.dumps({"error": f"Edit failed: {e}"})

    images = []
    if response.generated_images:
        for i, gen_img in enumerate(response.generated_images):
            img_data = gen_img.image
            raw = img_data.image_bytes if img_data.image_bytes else None
            if raw:
                jpeg_bytes = _to_jpeg(raw)
                entry = _format_image_output(jpeg_bytes, output, prefix="edit")
                entry["index"] = i + 1
                images.append(entry)

    if not images:
        return json.dumps({"error": "Edit produced no output images."})

    result = {"edit_mode": edit_mode, "prompt": prompt}
    if len(images) == 1:
        result.update(images[0])
        result.pop("index", None)
    else:
        result["images"] = images

    return json.dumps(result)


@mcp.tool()
def swap_background(
    image: str,
    background: str,
    aspect_ratio: str | None = None,
    count: int = 1,
    output: str = "base64",
) -> str:
    """Replace the background of an image while keeping the foreground subject intact.

    Automatically segments the foreground (person, product, object) and generates
    a new background based on your description. Much simpler than manual edit_image
    with masks for this common use case. Uses the Imagen 3 editing model.

    Args:
        image: Source image with the subject to keep. URL (recommended) or base64.
        background: Description of the new background. Be specific.
                    E.g. "tropical beach at sunset with palm trees",
                    "clean white studio with soft shadows",
                    "busy Tokyo street at night with neon signs".
        aspect_ratio: Output aspect ratio. Default: same as input.
        count: Number of candidates (1–4). Default: 1
        output: "base64" (default) or "gcs".

    Returns:
        JSON with the composited image data (base64 or URL).
    """
    from google.genai import types

    if output not in ("base64", "gcs"):
        return json.dumps({"error": f"Unknown output mode '{output}'.", "available": ["base64", "gcs"]})
    if output == "gcs" and not GCS_BUCKET:
        return json.dumps({"error": "GCS output requested but GCS_BUCKET env var is not set on the server. Use output='base64' instead."})

    count = max(1, min(count, 4))

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    try:
        img_bytes, img_mime = _decode_source(image)
    except Exception as e:
        return json.dumps({"error": str(e)})

    # Use Imagen edit with background mask + inpaint-insertion for the new background
    ref_images = [
        types.RawReferenceImage(
            reference_id=1,
            reference_image=types.Image(image_bytes=img_bytes, mime_type=img_mime),
        ),
        types.MaskReferenceImage(
            reference_id=2,
            config=types.MaskReferenceConfig(
                mask_mode="MASK_MODE_BACKGROUND",
                mask_dilation=0.03,
            ),
        ),
    ]

    config = types.EditImageConfig(
        edit_mode="EDIT_MODE_INPAINT_INSERTION",
        number_of_images=count,
    )
    if aspect_ratio:
        config.aspect_ratio = aspect_ratio

    prompt = f"Replace the background with: {background}"

    try:
        response = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=prompt,
            reference_images=ref_images,
            config=config,
        )
    except Exception as e:
        return json.dumps({"error": f"Background swap failed: {e}"})

    images = []
    if response.generated_images:
        for i, gen_img in enumerate(response.generated_images):
            raw = gen_img.image.image_bytes if gen_img.image.image_bytes else None
            if raw:
                jpeg_bytes = _to_jpeg(raw)
                entry = _format_image_output(jpeg_bytes, output, prefix="bgswap")
                entry["index"] = i + 1
                images.append(entry)

    if not images:
        return json.dumps({"error": "Background swap produced no output images."})

    result = {"background": background}
    if len(images) == 1:
        result.update(images[0])
        result.pop("index", None)
    else:
        result["images"] = images

    return json.dumps(result)


@mcp.tool()
def create_variations(
    image: str,
    prompt: str | None = None,
    variation_strength: str = "medium",
    aspect_ratio: str = "4:5",
    resolution: str = "1K",
    quality: str = "default",
    count: int = 3,
    qa: bool = False,
    output: str = "base64",
) -> str:
    """Generate variations of an existing image.

    Preserves the core subject while exploring different compositions, lighting,
    or styling. Pass the source image as a URL (recommended) or base64.

    Args:
        image: Source image. URL (recommended), base64 data URI, or raw base64.
        prompt: Optional guidance. E.g. "same product but on a beach",
                "warmer color palette", "more dramatic lighting".
        variation_strength: "subtle", "medium", or "strong". Default: medium
        aspect_ratio: Output aspect ratio. Default: 4:5
        resolution: Output resolution: 0.5K, 1K, 2K, 4K. Default: 1K
        quality: "default" or "pro". Default: default
        count: Number of variations (1–4). Default: 3
        qa: Score each variation and rank by quality. Default: false
        output: "base64" (default) or "gcs".

    Returns:
        JSON with variation image data, ranked by QA score if qa=true.
    """
    from google.genai import types

    if aspect_ratio not in SUPPORTED_RATIOS:
        return json.dumps({"error": f"Unsupported aspect ratio '{aspect_ratio}'.", "supported": sorted(SUPPORTED_RATIOS)})
    if resolution not in RESOLUTIONS:
        return json.dumps({"error": f"Unsupported resolution '{resolution}'.", "supported": sorted(RESOLUTIONS)})
    if output not in ("base64", "gcs"):
        return json.dumps({"error": f"Unknown output mode '{output}'.", "available": ["base64", "gcs"]})
    if output == "gcs" and not GCS_BUCKET:
        return json.dumps({"error": "GCS output requested but GCS_BUCKET env var is not set on the server. Use output='base64' instead."})

    strength_prompts = {
        "subtle": "Create a subtle variation of this image, keeping the composition and style very close to the original with only minor differences in details. ",
        "medium": "Create a variation of this image, maintaining the core subject and mood but exploring different composition, angle, or lighting. ",
        "strong": "Create a creative reinterpretation of this image, keeping the same subject but with significantly different composition, style, lighting, or artistic treatment. ",
    }
    if variation_strength not in strength_prompts:
        return json.dumps({"error": f"Unknown variation_strength '{variation_strength}'.", "available": list(strength_prompts.keys())})

    count = max(1, min(count, 4))

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    try:
        img_bytes, img_mime = _decode_source(image)
    except Exception as e:
        return json.dumps({"error": str(e)})

    variation_prompt = strength_prompts[variation_strength]
    if prompt:
        variation_prompt += prompt

    parts = [
        types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
        types.Part.from_text(text=variation_prompt),
    ]

    model_name = _pick_model(quality)
    images = []
    errors = []

    for i in range(count):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution,
                    ),
                ),
            )
            raw = _extract_image(response)
            if raw:
                jpeg_bytes = _to_jpeg(raw)
                entry = _format_image_output(jpeg_bytes, output, prefix="var")
                if qa:
                    entry["qa"] = _score_image(client, jpeg_bytes, variation_prompt)
                entry["index"] = i + 1
                images.append(entry)
            else:
                errors.append(f"Variation {i + 1}: no image in response")
        except Exception as e:
            errors.append(f"Variation {i + 1}: {e}")

    if not images:
        return json.dumps({"error": "All variation attempts failed.", "details": errors})

    if qa and len(images) > 1:
        images.sort(key=lambda x: x.get("qa", {}).get("total", 0), reverse=True)
        for i, img in enumerate(images):
            img["rank"] = i + 1

    result = {
        "model": model_name,
        "variation_strength": variation_strength,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "count": len(images),
        "images": images,
    }
    if errors:
        result["errors"] = errors
    if prompt:
        result["guidance"] = prompt

    return json.dumps(result)


@mcp.tool()
def analyze_image(
    image: str,
    focus: str = "general",
) -> str:
    """Analyze an image using Gemini vision — describe, tag, or assess quality.

    Useful for understanding uploaded references, generating SEO alt text,
    verifying a generation matched intent, or extracting visual details.

    Args:
        image: Image to analyze. URL (recommended), base64 data URI, or raw base64.
        focus: Analysis focus. Options:
               - "general" (default): Comprehensive description of content, style, mood
               - "tags": Keyword tags for search/SEO (returns a list)
               - "alt-text": Concise accessible alt text (1-2 sentences)
               - "quality": Technical quality assessment (sharpness, lighting, composition)
               - "brand": Brand/marketing analysis (target audience, mood, messaging)

    Returns:
        JSON with the analysis result.
    """
    from google.genai import types

    focus_prompts = {
        "general": (
            "Describe this image comprehensively. Include: subject matter, style, "
            "composition, lighting, color palette, mood, and any notable details. "
            "Be specific and detailed. Return as JSON: "
            '{"description": "...", "style": "...", "mood": "...", "colors": ["..."], "details": ["..."]}'
        ),
        "tags": (
            "Generate keyword tags for this image suitable for search engines and "
            "asset management. Include tags for subject, style, mood, colors, setting, "
            "and technical aspects. Return as JSON: "
            '{"tags": ["tag1", "tag2", ...], "primary_subject": "..."}'
        ),
        "alt-text": (
            "Write concise, accessible alt text for this image (1-2 sentences). "
            "Describe what is shown, not the style. Be specific enough for someone "
            "who cannot see the image. Return as JSON: "
            '{"alt_text": "...", "short": "..."}'
        ),
        "quality": (
            "Assess the technical quality of this image. Score each from 1-10: "
            "sharpness, exposure, composition, color balance, noise level. "
            "Note any artifacts, blur, or quality issues. Return as JSON: "
            '{"sharpness": {"score": N, "note": "..."}, "exposure": {"score": N, "note": "..."}, '
            '"composition": {"score": N, "note": "..."}, "color_balance": {"score": N, "note": "..."}, '
            '"noise": {"score": N, "note": "..."}, "issues": ["..."], "total": N}'
        ),
        "brand": (
            "Analyze this image from a marketing/brand perspective. Identify: "
            "target audience, emotional tone, brand positioning, visual messaging, "
            "and suggested use cases (social media, hero banner, product page, etc). "
            "Return as JSON: "
            '{"target_audience": "...", "tone": "...", "positioning": "...", '
            '"messaging": "...", "use_cases": ["..."], "strengths": ["..."], "suggestions": ["..."]}'
        ),
    }

    if focus not in focus_prompts:
        return json.dumps({"error": f"Unknown focus '{focus}'.", "available": list(focus_prompts.keys())})

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    # Use higher resolution for quality analysis to detect real defects
    try:
        if focus == "quality":
            img_bytes, img_mime = _decode_source(image)
        else:
            img_bytes, img_mime = _decode_reference(image)
    except Exception as e:
        return json.dumps({"error": str(e)})

    try:
        resp = client.models.generate_content(
            model=MODEL_TEXT,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
                types.Part.from_text(text=focus_prompts[focus]),
            ],
            config={"response_mime_type": "application/json"},
        )
        analysis = json.loads(resp.text)
        analysis["focus"] = focus
        return json.dumps(analysis)
    except json.JSONDecodeError:
        return json.dumps({"focus": focus, "raw_response": resp.text if resp.text else "No response"})
    except Exception as e:
        return json.dumps({"error": f"Analysis failed: {e}"})


@mcp.tool()
def list_styles() -> str:
    """List available style presets for generate_image.

    Returns:
        JSON with style names and descriptions.
    """
    styles = []
    for name, prefix in sorted(STYLE_PRESETS.items()):
        styles.append({"name": name, "description": prefix.strip()})
    return json.dumps({"styles": styles})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
