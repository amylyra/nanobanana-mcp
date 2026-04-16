"""
NanoBanana MCP Server — Image generation, editing, and variations via Gemini.

Tools:
  - generate_image:    Text-to-image with optional reference images, style presets,
                       prompt enhancement, and multi-model routing.
  - edit_image:        Edit an existing image (inpaint, remove, outpaint).
  - create_variations: Generate variations of an existing image.
  - list_styles:       List available style presets.

Deployment: Cloud Run with Streamable HTTP transport.
"""

import base64
import json
import os
from io import BytesIO
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_FLASH = "gemini-3.1-flash-image-preview"   # NanoBanana 2 — fast
MODEL_PRO = "gemini-3-pro-image-preview"          # NanoBanana Pro — higher quality
MODEL_TEXT = "gemini-2.5-flash"                    # For prompt enhancement

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
        "Tools: generate_image (text-to-image with references, styles, prompt enhancement), "
        "edit_image (inpaint, remove objects, outpaint), "
        "create_variations (produce variations of an existing image). "
        "Default aspect ratio 4:5, resolution 1K. "
        "IMPORTANT: For best results, pass image URLs instead of base64 — "
        "the server fetches them directly, avoiding size/truncation issues with inline data. "
        "Base64 is supported but large images may be corrupted in transit."
    ),
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", 8080)),
)


# ---------------------------------------------------------------------------
# Helpers
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


def _is_url(s: str) -> bool:
    """Check if string looks like an HTTP(S) URL."""
    try:
        parsed = urlparse(s)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def _fetch_url(url: str) -> tuple[bytes, str]:
    """Fetch an image from a URL. Returns (bytes, mime_type)."""
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return resp.content, content_type


def _fix_base64_padding(s: str) -> str:
    """Fix missing base64 padding — strings often get truncated in transit."""
    s = s.rstrip()
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)
    return s


def _normalize_image(img_bytes: bytes, max_dim: int, quality: int = 85,
                     output_format: str = "JPEG") -> tuple[bytes, str]:
    """Normalize an image: convert color mode, cap dimensions, encode.

    Args:
        img_bytes: Raw image bytes.
        max_dim: Maximum dimension (width or height).
        quality: JPEG quality (ignored for PNG).
        output_format: "JPEG" or "PNG".

    Returns (encoded_bytes, mime_type).
    """
    from PIL import Image as PILImage

    try:
        img = PILImage.open(BytesIO(img_bytes))
    except Exception:
        raise ValueError(
            "Could not decode image data. The image may have been truncated in transit. "
            "Try passing an image URL instead of base64 for reliable results."
        )

    # Sanity check — a truncated JPEG can open but have tiny/zero dimensions
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
    # For PNG masks, preserve as-is (L or RGB)

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
    """Decode raw bytes from a base64 string, data URI, or URL.

    Returns (bytes, mime_type). Fixes base64 padding. No resizing.
    """
    if _is_url(ref):
        return _fetch_url(ref)
    if ref.startswith("data:"):
        header, data = ref.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        return base64.b64decode(_fix_base64_padding(data)), mime
    return base64.b64decode(_fix_base64_padding(ref)), "image/jpeg"


def _decode_reference(ref: str) -> tuple[bytes, str]:
    """Decode a reference image: fetch/decode, then downscale to REF_MAX_DIM.

    1024px preserves logo/label detail while keeping size manageable.
    """
    raw, _ = _decode_raw(ref)
    return _normalize_image(raw, max_dim=REF_MAX_DIM)


def _decode_source(ref: str) -> tuple[bytes, str]:
    """Decode a source image for editing/variations: fetch/decode, cap at SOURCE_MAX_DIM.

    2048px retains high quality for edits while bounding memory/payload size.
    """
    raw, _ = _decode_raw(ref)
    return _normalize_image(raw, max_dim=SOURCE_MAX_DIM, quality=92)


def _decode_mask(ref: str) -> tuple[bytes, str]:
    """Decode a mask image: fetch/decode, cap at SOURCE_MAX_DIM, output as PNG.

    Masks use PNG to avoid JPEG compression artifacts at black/white boundaries.
    """
    raw, _ = _decode_raw(ref)
    return _normalize_image(raw, max_dim=SOURCE_MAX_DIM, output_format="PNG")


def _to_jpeg(img_bytes: bytes) -> bytes:
    """Convert image bytes to high-quality JPEG for output."""
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
    """Select model based on quality setting."""
    return MODEL_PRO if quality == "pro" else MODEL_FLASH


def _enhance_prompt(client, prompt: str) -> str:
    """Use a text model to expand a short prompt into a detailed image generation prompt."""
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


def _image_to_data_uri(img_bytes: bytes) -> dict:
    """Convert raw image bytes to a JPEG data URI dict."""
    jpeg_bytes = _to_jpeg(img_bytes)
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return {
        "data_uri": f"data:image/jpeg;base64,{b64}",
        "size_kb": len(jpeg_bytes) // 1024,
    }


def _extract_image(response) -> bytes | None:
    """Extract the first image from a Gemini response."""
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                return part.inline_data.data
    return None


def _build_ref_parts(reference_images: list | None) -> list:
    """Build Gemini Part objects from reference images (base64, data URI, or URL)."""
    from google.genai import types

    parts = []
    if not reference_images:
        return parts
    for ref in reference_images:
        img_bytes, mime = _decode_reference(ref)
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    return parts


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
                          Multiple references are supported for combining style + subject cues.
        style: Optional style preset. Available: cinematic, product-photography,
               editorial, watercolor, flat-illustration, neon-noir, minimalist, vintage-film.
               Use list_styles to see descriptions.
        enhance_prompt: If true, uses AI to expand your prompt into a detailed, vivid
                        image generation prompt before generating. Great for short prompts
                        like "cat on a couch". Default: false
        aspect_ratio: Output aspect ratio. Supported: 1:1, 4:5, 9:16, 16:9, 3:4,
                      4:3, 2:3, 3:2, 5:4, 21:9. Default: 4:5
        resolution: Output resolution: 0.5K, 1K, 2K, 4K. Default: 1K
        quality: Model selection. "default" = NanoBanana 2 (fast);
                 "pro" = NanoBanana Pro (higher quality). Default: default
        count: Number of images to generate (1–4). Default: 1

    Returns:
        JSON with base64 JPEG data URI(s), metadata, and enhanced prompt (if used).
    """
    from google.genai import types

    if aspect_ratio not in SUPPORTED_RATIOS:
        return json.dumps({"error": f"Unsupported aspect ratio '{aspect_ratio}'.", "supported": sorted(SUPPORTED_RATIOS)})
    if resolution not in RESOLUTIONS:
        return json.dumps({"error": f"Unsupported resolution '{resolution}'.", "supported": sorted(RESOLUTIONS)})
    if style and style not in STYLE_PRESETS:
        return json.dumps({"error": f"Unknown style '{style}'.", "available": sorted(STYLE_PRESETS.keys())})

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
                entry = _image_to_data_uri(img_bytes)
                entry["index"] = i + 1
                images.append(entry)
            else:
                errors.append(f"Image {i + 1}: no image in response")
        except Exception as e:
            errors.append(f"Image {i + 1}: {e}")

    if not images:
        return json.dumps({"error": "All generation attempts failed.", "details": errors})

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
        del result["index"]
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
) -> str:
    """Edit an existing image — add objects, remove objects, or extend the canvas.

    Pass the source image as a URL (recommended) or base64.
    Provide a mask for precise control, or let the model infer the edit region.

    Note: For inpaint modes without a mask, the model uses automatic segmentation.
    For best results with precise edits, provide a black-and-white mask image.

    Args:
        image: The source image to edit. URL (recommended), base64 data URI, or raw base64.
        prompt: Edit instruction. Be specific about what to change.
                E.g. "Add a red hat on the person", "Remove the person in the background",
                "Extend the sky upward to make it taller".
        mask: Optional mask image (URL or base64). White pixels = area to edit,
              black pixels = area to preserve. Provides precise spatial control.
              If omitted, the model uses automatic segmentation based on your prompt.
        edit_mode: Edit operation type:
                   - "inpaint-insertion" (default): Add or replace content in the masked/detected area
                   - "inpaint-removal": Remove content and fill naturally
                   - "outpaint": Extend/expand the image canvas
        aspect_ratio: Output aspect ratio (mainly useful for outpaint). Default: same as input.
        count: Number of edit candidates (1–4). Default: 1

    Returns:
        JSON with base64 JPEG data URI(s) of the edited image.
    """
    from google.genai import types

    edit_modes = {
        "inpaint-insertion": "EDIT_MODE_INPAINT_INSERTION",
        "inpaint-removal": "EDIT_MODE_INPAINT_REMOVAL",
        "outpaint": "EDIT_MODE_OUTPAINT",
    }
    if edit_mode not in edit_modes:
        return json.dumps({"error": f"Unknown edit_mode '{edit_mode}'.", "available": list(edit_modes.keys())})

    count = max(1, min(count, 4))

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    # Decode source image — capped at 2048px for quality + manageability
    try:
        img_bytes, img_mime = _decode_source(image)
    except Exception as e:
        return json.dumps({"error": str(e)})

    # Build reference images for the edit API
    ref_images = [
        types.RawReferenceImage(
            reference_id=1,
            reference_image=types.Image(image_bytes=img_bytes, mime_type=img_mime),
        )
    ]

    # Add mask — user-provided or automatic segmentation
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
        # Use semantic segmentation — let the model figure out the region
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
                entry = _image_to_data_uri(raw)
                entry["index"] = i + 1
                images.append(entry)

    if not images:
        return json.dumps({"error": "Edit produced no output images."})

    result = {"edit_mode": edit_mode, "prompt": prompt}
    if len(images) == 1:
        result.update(images[0])
        del result["index"]
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
) -> str:
    """Generate variations of an existing image.

    Takes a source image and produces creative variations, preserving the core
    subject while exploring different compositions, lighting, or styling.
    Pass the source image as a URL (recommended) or base64.

    Args:
        image: Source image. URL (recommended), base64 data URI, or raw base64.
        prompt: Optional guidance for variations. E.g. "same product but on a beach",
                "warmer color palette", "more dramatic lighting". If omitted, the model
                generates free variations.
        variation_strength: How much to diverge from the original.
                            "subtle" = minor tweaks, "medium" = noticeable changes,
                            "strong" = significant creative departures. Default: medium
        aspect_ratio: Output aspect ratio. Default: 4:5
        resolution: Output resolution: 0.5K, 1K, 2K, 4K. Default: 1K
        quality: "default" (fast) or "pro" (higher quality). Default: default
        count: Number of variations (1–4). Default: 3

    Returns:
        JSON with base64 JPEG data URIs of all variations.
    """
    from google.genai import types

    if aspect_ratio not in SUPPORTED_RATIOS:
        return json.dumps({"error": f"Unsupported aspect ratio '{aspect_ratio}'.", "supported": sorted(SUPPORTED_RATIOS)})
    if resolution not in RESOLUTIONS:
        return json.dumps({"error": f"Unsupported resolution '{resolution}'.", "supported": sorted(RESOLUTIONS)})

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

    # Decode source image — capped at 2048px
    try:
        img_bytes, img_mime = _decode_source(image)
    except Exception as e:
        return json.dumps({"error": str(e)})

    # Build prompt
    variation_prompt = strength_prompts[variation_strength]
    if prompt:
        variation_prompt += prompt

    # Build parts: source image + variation prompt
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
                entry = _image_to_data_uri(raw)
                entry["index"] = i + 1
                images.append(entry)
            else:
                errors.append(f"Variation {i + 1}: no image in response")
        except Exception as e:
            errors.append(f"Variation {i + 1}: {e}")

    if not images:
        return json.dumps({"error": "All variation attempts failed.", "details": errors})

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
def list_styles() -> str:
    """List available style presets for generate_image.

    Returns:
        JSON with style names and their prompt prefixes.
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
