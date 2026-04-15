"""
NanoBanana MCP Server — General-purpose image generation via Gemini NanoBanana 2.

Exposes a generate_image tool for Claude.ai and Claude Code (cowork or CLI).
Users pass a prompt and optional reference images; the server calls the
NanoBanana 2 API (gemini-3.1-flash-image-preview) and returns base64 image data.

Default: 4:5 aspect ratio, 1K resolution.

Deployment: Cloud Run with Streamable HTTP transport.
"""

import base64
import json
import os
from io import BytesIO

from mcp.server.fastmcp import FastMCP

MODEL_DEFAULT = "gemini-3.1-flash-image-preview"  # NanoBanana 2 — fast, efficient
MODEL_PRO = "gemini-3-pro-image-preview"           # NanoBanana Pro — higher quality

SUPPORTED_RATIOS = {
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5",
    "5:4", "8:1", "9:16", "16:9", "21:9"
}
RESOLUTIONS = {"0.5K", "1K", "2K", "4K"}

mcp = FastMCP(
    "nanobanana",
    instructions=(
        "NanoBanana image generation server powered by Gemini NanoBanana 2. "
        "Generate any image by providing a prompt and optional reference images. "
        "Default aspect ratio is 4:5 (portrait); default resolution is 1K. "
        "Reference images guide style, subject, or composition — pass them as base64 data URIs."
    ),
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", 8080)),
)


def _get_client():
    """Initialize Gemini client from GOOGLE_AI_API_KEY or GEMINI_API_KEY env var."""
    from google import genai

    api_key = os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set GOOGLE_AI_API_KEY or GEMINI_API_KEY environment variable."
        )
    return genai.Client(api_key=api_key)


def _decode_reference(ref: str) -> tuple:
    """Decode a reference image from a base64 string or data URI.

    Returns (bytes, mime_type). Accepts:
      - data URI: data:image/jpeg;base64,<data>
      - raw base64 string (assumed JPEG)
    """
    if ref.startswith("data:"):
        header, data = ref.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        return base64.b64decode(data), mime
    return base64.b64decode(ref), "image/jpeg"


def _to_jpeg(img_bytes: bytes) -> bytes:
    """Convert image bytes to high-quality JPEG."""
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


@mcp.tool()
def generate_image(
    prompt: str,
    reference_images: list = None,
    aspect_ratio: str = "4:5",
    resolution: str = "1K",
    quality: str = "default",
    count: int = 1,
) -> str:
    """Generate an image using NanoBanana 2 (Gemini image generation API).

    Accepts an optional list of reference images to guide style, subject, or
    composition. Reference images should be passed as base64-encoded data URIs
    (data:image/jpeg;base64,...) or raw base64 strings.

    Args:
        prompt: Image generation instruction. Describe the subject, style,
                lighting, mood, and any other relevant details.
        reference_images: Optional list of reference images as base64 data URIs
                          or raw base64 strings. Reference images help the model
                          match a specific style, product appearance, or composition.
        aspect_ratio: Output aspect ratio. Supported: 1:1, 4:5, 9:16, 16:9, 3:4,
                      4:3, 2:3, 3:2, 5:4, 21:9. Default: 4:5 (portrait/social)
        resolution: Output resolution. Options: 0.5K, 1K, 2K, 4K. Default: 1K
        quality: Model quality. "default" = NanoBanana 2 (fast, gemini-3.1-flash-image-preview);
                 "pro" = NanoBanana Pro (higher quality, gemini-3-pro-image-preview). Default: default
        count: Number of images to generate (1–4). All returned as data URIs. Default: 1

    Returns:
        JSON object. Single image: {"data_uri": "data:image/jpeg;base64,...", "size_kb": ..., ...}
        Multiple images: {"images": [{"index": 1, "data_uri": ..., "size_kb": ...}, ...], ...}
        On error: {"error": "...", "details": [...]}
    """
    from google.genai import types

    # Validate
    if aspect_ratio not in SUPPORTED_RATIOS:
        return json.dumps({
            "error": f"Unsupported aspect ratio '{aspect_ratio}'.",
            "supported": sorted(SUPPORTED_RATIOS),
        })
    if resolution not in RESOLUTIONS:
        return json.dumps({
            "error": f"Unsupported resolution '{resolution}'.",
            "supported": sorted(RESOLUTIONS),
        })

    count = max(1, min(count, 4))
    model_name = MODEL_PRO if quality == "pro" else MODEL_DEFAULT

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    # Build content parts: reference images first, then prompt text
    parts = []
    if reference_images:
        for i, ref in enumerate(reference_images):
            try:
                img_bytes, mime = _decode_reference(ref)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
            except Exception as e:
                return json.dumps({"error": f"Failed to decode reference image {i + 1}: {e}"})

    parts.append(types.Part.from_text(text=prompt))

    # Generate image(s)
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
            img_bytes = None
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        img_bytes = part.inline_data.data
                        break

            if img_bytes:
                jpeg_bytes = _to_jpeg(img_bytes)
                b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                images.append({
                    "index": i + 1,
                    "data_uri": f"data:image/jpeg;base64,{b64}",
                    "size_kb": len(jpeg_bytes) // 1024,
                })
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
        "prompt_preview": prompt[:120] + "..." if len(prompt) > 120 else prompt,
    }

    if count == 1:
        result["data_uri"] = images[0]["data_uri"]
        result["size_kb"] = images[0]["size_kb"]
    else:
        result["images"] = images
        if errors:
            result["errors"] = errors

    return json.dumps(result)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
