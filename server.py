"""
NanoBanana MCP Server — Image generation, editing, and variations via Gemini.

Tools:
  - upload_image:      Upload an image to the server, get back a URL for other tools.
  - generate_image:    Text-to-image with references, styles, prompt enhancement, QA.
  - edit_image:        Edit an existing image (inpaint, remove, outpaint).
  - swap_background:   Keep foreground subject, replace background.
  - create_variations: Generate variations of an existing image.
  - analyze_image:     Describe/tag an image using Gemini vision.
  - list_styles:       List available style presets.

HTTP endpoints (outside MCP — for direct image upload):
  - POST /upload:      Upload an image directly, get back a URL to paste into Claude.
  - GET /images/{id}:  Retrieve a stored image by ID.

Deployment: Cloud Run with Streamable HTTP transport.
"""

import asyncio
import base64
import json
import os
import threading
import time
import uuid
from io import BytesIO
from urllib.parse import urlparse

# Load .env file if python-dotenv is installed (optional convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx
from mcp.server.fastmcp import FastMCP, Context, Image
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

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
RESOLUTIONS = {"1K", "2K", "4K"}

# ---------------------------------------------------------------------------
# Image size limits (max dimension in pixels)
# ---------------------------------------------------------------------------
REF_MAX_DIM = 1024    # Reference images — preserves logo/label detail
SOURCE_MAX_DIM = 2048  # Source images for edit/variations — high quality but bounded

# ---------------------------------------------------------------------------
# Cloud storage config (optional — set bucket env var to enable)
# ---------------------------------------------------------------------------
GCS_BUCKET = os.environ.get("GCS_BUCKET")  # e.g. "my-nanobanana-images"
S3_BUCKET = os.environ.get("S3_BUCKET")    # e.g. "claude-image-cache"
S3_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
DEFAULT_OUTPUT = os.environ.get("DEFAULT_OUTPUT", "base64")  # "base64", "cloud", "s3", "gcs"

# ---------------------------------------------------------------------------
# Server-side image store — upload once, reference by URL in all subsequent calls.
# Images expire after 1 hour to prevent unbounded memory growth.
# ---------------------------------------------------------------------------
_IMAGE_STORE: dict[str, tuple[bytes, str, float]] = {}  # id -> (bytes, mime, timestamp)
_STORE_LOCK = threading.Lock()
_STORE_TTL = int(os.environ.get("STORE_TTL", 3600))  # 1 hour default
_STORE_MAX_ITEMS = int(os.environ.get("STORE_MAX_ITEMS", 100))  # configurable


def _store_image(img_bytes: bytes, mime: str) -> str:
    """Store image bytes and return the image ID."""
    _gc_store()
    img_id = uuid.uuid4().hex[:12]
    with _STORE_LOCK:
        # Evict oldest entries if store is full
        while len(_IMAGE_STORE) >= _STORE_MAX_ITEMS:
            oldest_key = min(_IMAGE_STORE, key=lambda k: _IMAGE_STORE[k][2])
            del _IMAGE_STORE[oldest_key]
        _IMAGE_STORE[img_id] = (img_bytes, mime, time.time())
    return img_id


def _fetch_from_store(img_id: str) -> tuple[bytes, str]:
    """Retrieve image bytes from the store."""
    with _STORE_LOCK:
        entry = _IMAGE_STORE.get(img_id)
    if not entry:
        raise ValueError(
            f"Image '{img_id}' not found in server store. "
            "It may have expired (images are kept for 1 hour). Upload again."
        )
    return entry[0], entry[1]


def _gc_store():
    """Remove expired images from the store."""
    now = time.time()
    with _STORE_LOCK:
        expired = [k for k, (_, _, ts) in _IMAGE_STORE.items() if now - ts > _STORE_TTL]
        for k in expired:
            del _IMAGE_STORE[k]


# ---------------------------------------------------------------------------
# Session-based upload tracking — links elicitation sessions to uploaded images
# ---------------------------------------------------------------------------
_UPLOAD_SESSIONS: dict[str, str | None] = {}  # session_id -> image_id or None (pending)
_SESSION_LOCK = threading.Lock()
_SESSION_TTL = 300  # 5 min — user has this long to complete the upload


def _create_upload_session() -> str:
    """Create a pending upload session and return its ID."""
    _gc_sessions()
    session_id = uuid.uuid4().hex  # 128-bit — not guessable
    with _SESSION_LOCK:
        _UPLOAD_SESSIONS[session_id] = (None, time.time())  # (image_id, created_at)
    return session_id


def _complete_upload_session(session_id: str, img_id: str) -> None:
    """Mark a session as complete with the uploaded image ID."""
    with _SESSION_LOCK:
        entry = _UPLOAD_SESSIONS.get(session_id)
        if entry is not None:
            _UPLOAD_SESSIONS[session_id] = (img_id, entry[1])


def _poll_upload_session(session_id: str) -> str | None:
    """Check if a session has a completed upload. Returns image_id or None."""
    with _SESSION_LOCK:
        entry = _UPLOAD_SESSIONS.get(session_id)
        if entry is None:
            return None
        return entry[0]


def _cleanup_session(session_id: str) -> None:
    """Remove a completed or expired session."""
    with _SESSION_LOCK:
        _UPLOAD_SESSIONS.pop(session_id, None)


def _gc_sessions():
    """Remove expired upload sessions."""
    now = time.time()
    with _SESSION_LOCK:
        expired = [k for k, (_, ts) in _UPLOAD_SESSIONS.items() if now - ts > _SESSION_TTL]
        for k in expired:
            del _UPLOAD_SESSIONS[k]

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
# Helpers — URL construction
# ---------------------------------------------------------------------------
def _get_upload_base_url() -> str:
    """Get the server's external base URL for building upload links."""
    base_url = os.environ.get("PUBLIC_URL")
    if base_url:
        return base_url
    k_service = os.environ.get("K_SERVICE")
    k_region = os.environ.get("CLOUD_RUN_REGION") or os.environ.get("GOOGLE_CLOUD_REGION")
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if k_service and k_region:
        return (
            f"https://{k_service}-{gcp_project}.{k_region}.run.app"
            if gcp_project
            else f"https://{k_service}.run.app"
        )
    port = int(os.environ.get("PORT", 8080))
    return f"http://localhost:{port}"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "nanobanana",
    instructions=(
        "NanoBanana image generation server. Powered by Gemini.\n\n"
        "## RULE #1 — NEVER pass base64 image data to any tool.\n"
        "Do NOT encode, re-encode, resize, or convert images to base64 for tool parameters.\n"
        "Do NOT use data: URIs. Do NOT read image files and pass their contents.\n"
        "Base64 strings consume the entire context window and get truncated. It WILL fail.\n\n"
        "## How to handle images:\n"
        "- If the user provides a URL (http/https): pass it directly to the tool.\n"
        "- If a previous tool returned a nanobanana:// URL: pass that URL.\n"
        "- If the user pastes/uploads an image with NO URL: tell them to open\n"
        "  {upload_url}/upload in their browser, drop the image there, and paste\n"
        "  the returned URL back into the chat. Then pass that URL to the tool.\n"
        "  Do NOT attempt to extract, encode, or forward the image data yourself.\n\n"
        "## Tools:\n"
        "generate_image, edit_image, swap_background, create_variations, "
        "analyze_image, upload_image, list_styles.\n"
        "Default aspect ratio 4:5, resolution 1K."
    ).format(upload_url=_get_upload_base_url()),
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", 8080)),
)


# ---------------------------------------------------------------------------
# HTTP endpoints — direct image upload/serving (outside MCP protocol)
# ---------------------------------------------------------------------------
_UPLOAD_HTML = """<!DOCTYPE html>
<html><head><title>NanoBanana — Upload Image</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 1.4em; }
  .drop-zone { border: 2px dashed #ccc; border-radius: 8px; padding: 40px; text-align: center;
    cursor: pointer; transition: border-color 0.2s; margin: 20px 0; }
  .drop-zone:hover, .drop-zone.drag-over { border-color: #f5a623; background: #fffbf0; }
  input[type="file"] { display: none; }
  #result { margin-top: 20px; padding: 12px; background: #f0f0f0; border-radius: 6px;
    word-break: break-all; display: none; }
  #result.success { background: #e8f5e9; }
  #result.error { background: #fce4ec; }
  code { background: #e8e8e8; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
</style></head>
<body>
  <h1>NanoBanana — Upload Image</h1>
  <p id="instructions">Upload an image to get a URL you can paste into Claude.</p>
  <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    Drop an image here or click to browse
  </div>
  <input type="file" id="fileInput" accept="image/*">
  <div id="result"></div>
  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const result = document.getElementById('result');

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault(); dropZone.classList.remove('drag-over');
      if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length) upload(fileInput.files[0]); });

    async function upload(file) {
      dropZone.textContent = 'Uploading...';
      result.style.display = 'none';
      const form = new FormData();
      form.append('file', file);
      // Forward session param if present (for elicitation-triggered uploads)
      const params = new URLSearchParams(window.location.search);
      const session = params.get('session');
      const uploadUrl = session ? '/upload?session=' + session : '/upload';
      if (session) {
        document.getElementById('instructions').textContent =
          'Drop your image here — the tool waiting in Claude will pick it up automatically.';
      }
      try {
        const resp = await fetch(uploadUrl, { method: 'POST', body: form });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        result.className = 'success';
        if (session) {
          result.innerHTML = '<strong>Image received!</strong> You can close this tab — ' +
            'the tool in Claude will continue automatically.';
        } else {
          result.innerHTML = '<strong>Image URL (paste this into Claude):</strong><br><br>' +
            '<code>' + data.url + '</code>';
          navigator.clipboard.writeText(data.url).catch(() => {});
        }
        result.style.display = 'block';
        dropZone.textContent = 'Upload another image';
      } catch (err) {
        result.className = 'error';
        result.textContent = 'Upload failed: ' + err.message;
        result.style.display = 'block';
        dropZone.textContent = 'Drop an image here or click to browse';
      }
    }
  </script>
</body></html>"""


@mcp.custom_route("/upload", methods=["GET"])
async def upload_form(request: Request) -> Response:
    """Serve a simple HTML form for uploading images."""
    return HTMLResponse(_UPLOAD_HTML)


@mcp.custom_route("/upload", methods=["POST"])
async def http_upload(request: Request) -> Response:
    """Accept an image upload and return a URL for use in MCP tools."""
    from PIL import Image as PILImage

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file") or form.get("image")
        if file is None:
            return JSONResponse({"error": "No 'file' field in form data"}, status_code=400)
        raw = await file.read()
        await form.close()
    else:
        raw = await request.body()
        if not raw:
            return JSONResponse({"error": "Empty request body"}, status_code=400)

    try:
        normalized, mime = _normalize_image(raw, max_dim=SOURCE_MAX_DIM, quality=92)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    img = PILImage.open(BytesIO(normalized))
    w, h = img.size

    img_id = _store_image(normalized, mime)

    # Link to upload session if present (for elicitation flow)
    session_id = request.query_params.get("session")
    if session_id:
        _complete_upload_session(session_id, img_id)

    # Build the full URL using the request's Host header
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", request.url.netloc)
    image_url = f"{scheme}://{host}/images/{img_id}"

    return JSONResponse({
        "url": image_url,
        "width": w,
        "height": h,
        "size_kb": len(normalized) // 1024,
        "expires_in": "1 hour",
        "usage": "Paste this URL into Claude to use with any image tool",
    }, status_code=201)


@mcp.custom_route("/images/{img_id}", methods=["GET"])
async def http_get_image(request: Request) -> Response:
    """Serve a stored image by ID."""
    img_id = request.path_params["img_id"]
    try:
        img_bytes, mime = _fetch_from_store(img_id)
    except ValueError:
        return JSONResponse({"error": "Image not found or expired"}, status_code=404)
    return Response(content=img_bytes, media_type=mime)


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
    # Block requests to cloud metadata endpoints and private networks (SSRF protection)
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    _BLOCKED_HOSTS = {
        "metadata.google.internal",
        "169.254.169.254",  # AWS/GCP metadata
        "100.100.100.200",  # Alibaba metadata
    }
    if hostname in _BLOCKED_HOSTS or hostname.startswith("169.254."):
        raise ValueError(f"Blocked request to internal address: {hostname}")
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        raise ValueError(f"Blocked request to localhost: {hostname}")

    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (403, 404):
            raise ValueError(
                f"Image URL returned {e.response.status_code}. "
                "The image may have been deleted or expired. Please re-upload."
            )
        raise ValueError(f"Failed to fetch image from URL: {e}")
    except httpx.TimeoutException:
        raise ValueError(f"Timed out fetching image from URL: {url}")
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
            "Try uploading via the /upload page or passing an image URL."
        )

    w, h = img.size
    if w < 1 or h < 1:
        raise ValueError(
            "Image has invalid dimensions. The data may be corrupted or truncated. "
            "Try uploading via the /upload page or passing an image URL."
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
    # Internal store reference (nanobanana://id)
    if ref.startswith("nanobanana://"):
        img_id = ref.removeprefix("nanobanana://")
        return _fetch_from_store(img_id)
    # HTTP URL — check if it's our own /images/ endpoint first
    if _is_url(ref):
        parsed = urlparse(ref)
        if parsed.path.startswith("/images/"):
            img_id = parsed.path.removeprefix("/images/")
            try:
                return _fetch_from_store(img_id)
            except ValueError:
                pass  # not in store — fall through to HTTP fetch
        return _fetch_url(ref)
    # Base64 data URI
    if ref.startswith("data:"):
        header, data = ref.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        return base64.b64decode(_fix_base64_padding(data)), mime
    # Raw base64
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


async def _acquire_image(
    image: str | None,
    ctx: Context,
    max_dim: int = SOURCE_MAX_DIM,
    quality: int = 92,
    purpose: str = "image",
) -> tuple[bytes, str]:
    """Decode an image from a string ref, or raise with upload instructions.

    Returns (normalized_bytes, mime_type).
    Raises ValueError with a user-friendly message if decoding fails.
    """
    # Try direct decode if we have a non-empty image string
    if image:
        try:
            raw, _ = _decode_raw(image)
            return _normalize_image(raw, max_dim=max_dim, quality=quality)
        except Exception as decode_err:
            # If it looks like a URL or nanobanana ref, the error is real
            if _is_url(image) or image.startswith("nanobanana://"):
                raise ValueError(str(decode_err))
            # Likely truncated base64 — give upload instructions
            first_error = str(decode_err)
    else:
        first_error = f"No {purpose} provided"

    # No usable image — direct user to the upload page
    base_url = _get_upload_base_url()
    raise ValueError(
        f"{first_error}. "
        f"Please upload the image at {base_url}/upload and paste the returned URL here."
    )


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
    for i, ref in enumerate(reference_images):
        if not ref or not ref.strip():
            raise ValueError(
                f"Reference image {i + 1} is empty. "
                "Upload each reference image first using upload_image to get a "
                "nanobanana:// URL, then pass those URLs here."
            )
        img_bytes, mime = _decode_reference(ref)
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    return parts


# ---------------------------------------------------------------------------
# Helpers — cloud storage upload
# ---------------------------------------------------------------------------
def _upload_to_gcs(jpeg_bytes: bytes, prefix: str = "gen") -> str:
    """Upload JPEG bytes to GCS and return a public URL."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob_name = f"{prefix}/{uuid.uuid4().hex}.jpg"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(jpeg_bytes, content_type="image/jpeg")
    blob.make_public()
    return blob.public_url


_s3_client = None
_s3_client_lock = threading.Lock()


def _get_s3_client():
    """Get or create a cached boto3 S3 client."""
    global _s3_client
    if _s3_client is None:
        with _s3_client_lock:
            if _s3_client is None:
                import boto3
                _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def _upload_to_s3(jpeg_bytes: bytes, prefix: str = "gen") -> str:
    """Upload JPEG bytes to S3 and return a public URL."""
    s3 = _get_s3_client()
    key = f"{prefix}/{uuid.uuid4().hex}.jpg"
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=jpeg_bytes,
            ContentType="image/jpeg",
        )
    except Exception as e:
        raise ValueError(
            f"S3 upload failed: {e}. Check AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
            f"and that the IAM user has s3:PutObject on {S3_BUCKET}."
        )
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def _upload_to_cloud(jpeg_bytes: bytes, prefix: str = "gen") -> str:
    """Upload to whichever cloud storage is configured (S3 preferred, then GCS)."""
    if S3_BUCKET:
        return _upload_to_s3(jpeg_bytes, prefix=prefix)
    if GCS_BUCKET:
        return _upload_to_gcs(jpeg_bytes, prefix=prefix)
    raise ValueError(
        "No cloud storage configured. Set S3_BUCKET or GCS_BUCKET env var, "
        "or use output='base64' instead."
    )


# ---------------------------------------------------------------------------
# Helpers — image output formatting
# ---------------------------------------------------------------------------
def _build_image_response(
    result: dict,
    generated: list[tuple[bytes, dict]],
    output_mode: str,
    prefix: str = "gen",
) -> list | str:
    """Build a tool response with metadata + images.

    generated: list of (jpeg_bytes, per_image_metadata) tuples.

    base64 mode: returns [json_metadata, Image, Image, ...]
        Images render natively in Claude. Metadata includes stored URLs for re-use.
    cloud mode (gcs/s3): returns json_metadata string with public URLs.
    """
    if output_mode in ("gcs", "s3", "cloud"):
        for jpeg_bytes, meta in generated:
            cloud_url = _upload_to_cloud(jpeg_bytes, prefix=prefix)
            meta["image_url"] = cloud_url  # consistent key across all output modes
            meta["size_kb"] = len(jpeg_bytes) // 1024
        if len(generated) == 1:
            result.update(generated[0][1])
            result.pop("index", None)
        else:
            result["images"] = [meta for _, meta in generated]
        return json.dumps(result)
    else:
        # Store images server-side so they can be referenced by URL in later tool calls
        for jpeg_bytes, meta in generated:
            img_id = _store_image(jpeg_bytes, "image/jpeg")
            meta["image_url"] = f"nanobanana://{img_id}"
            meta["size_kb"] = len(jpeg_bytes) // 1024
        if len(generated) == 1:
            result.update(generated[0][1])
            result.pop("index", None)
        else:
            result["images"] = [meta for _, meta in generated]
        # Return metadata text + native Image content blocks
        return [json.dumps(result)] + [
            Image(data=jpeg_bytes, format="jpeg")
            for jpeg_bytes, _ in generated
        ]


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
async def upload_image(
    ctx: Context,
    image: str = "",
) -> str:
    """Store an image on the server and get back a URL for other tools.

    IMPORTANT: Only pass URLs (http/https or nanobanana://). NEVER pass base64 data —
    it will consume the entire context window and likely get truncated.

    If the user has no URL, direct them to the /upload page to get one.

    Args:
        image: Image URL (http/https) or nanobanana:// URL.

    Returns:
        JSON with a nanobanana:// URL to use in other tools, plus image dimensions.
    """
    from PIL import Image as PILImage

    try:
        normalized, norm_mime = await _acquire_image(image, ctx, purpose="image")
    except ValueError as e:
        return json.dumps({"error": str(e)})

    img = PILImage.open(BytesIO(normalized))
    w, h = img.size

    img_id = _store_image(normalized, norm_mime)

    return json.dumps({
        "url": f"nanobanana://{img_id}",
        "width": w,
        "height": h,
        "size_kb": len(normalized) // 1024,
        "expires_in": "1 hour",
        "usage": "Pass this URL to any other tool's image parameter",
    })


@mcp.tool()
async def generate_image(
    prompt: str,
    reference_images: list[str] | None = None,
    style: str | None = None,
    enhance_prompt: bool = False,
    aspect_ratio: str = "4:5",
    resolution: str = "1K",
    quality: str = "default",
    count: int = 1,
    qa: bool = False,
    output: str = DEFAULT_OUTPUT,
) -> list | str:
    """Generate an image from a text prompt with optional reference images and style presets.

    Reference images guide the model on style, subject appearance, or composition.
    ONLY pass URLs — never base64.

    Args:
        prompt: What to generate. Describe subject, style, lighting, mood, etc.
        reference_images: Optional list of image URLs (http/https or nanobanana://). NEVER base64.
        style: Optional style preset. Available: cinematic, product-photography,
               editorial, watercolor, flat-illustration, neon-noir, minimalist, vintage-film.
        enhance_prompt: If true, AI expands your prompt into a detailed generation prompt.
        aspect_ratio: Output aspect ratio. Default: 4:5
        resolution: Output resolution: 1K, 2K, 4K. Default: 1K
        quality: "default" (fast) or "pro" (higher quality). Default: default
        count: Number of images to generate (1–4). Default: 1
        qa: If true, AI-score each image. When count > 1, ranks by total score.
        output: "base64" returns inline images (default). "cloud"/"s3"/"gcs" uploads to cloud storage and returns URLs.

    Returns:
        Metadata JSON + inline images (base64 mode) or metadata with URLs (cloud mode).
    """
    from google.genai import types

    if aspect_ratio not in SUPPORTED_RATIOS:
        return json.dumps({"error": f"Unsupported aspect ratio '{aspect_ratio}'.", "supported": sorted(SUPPORTED_RATIOS)})
    if resolution not in RESOLUTIONS:
        return json.dumps({"error": f"Unsupported resolution '{resolution}'.", "supported": sorted(RESOLUTIONS)})
    if style and style not in STYLE_PRESETS:
        return json.dumps({"error": f"Unknown style '{style}'.", "available": sorted(STYLE_PRESETS.keys())})
    if output not in ("base64", "gcs", "s3", "cloud"):
        return json.dumps({"error": f"Unknown output mode '{output}'.", "available": ["base64", "gcs", "s3", "cloud"]})
    if output in ("gcs", "s3", "cloud") and not S3_BUCKET and not GCS_BUCKET:
        return json.dumps({"error": "Cloud output requested but no storage configured. Set S3_BUCKET or GCS_BUCKET env var, or use output='base64'."})

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

    # Build content parts (run in thread pool to avoid blocking on HTTP fetches)
    try:
        loop = asyncio.get_event_loop()
        parts = await loop.run_in_executor(None, _build_ref_parts, reference_images)
    except Exception as e:
        return json.dumps({"error": f"Failed to process reference image: {e}"})
    parts.append(types.Part.from_text(text=final_prompt))

    model_name = _pick_model(quality)
    generated = []  # (jpeg_bytes, metadata_dict)
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
                meta = {"index": i + 1}
                if qa:
                    meta["qa"] = _score_image(client, jpeg_bytes, final_prompt)
                generated.append((jpeg_bytes, meta))
            else:
                errors.append(f"Image {i + 1}: no image in response")
        except Exception as e:
            errors.append(f"Image {i + 1}: {e}")

    if not generated:
        return json.dumps({"error": "All generation attempts failed.", "details": errors})

    # Rank by QA score if scoring was enabled
    if qa and len(generated) > 1:
        generated.sort(key=lambda x: x[1].get("qa", {}).get("total", 0), reverse=True)
        for i, (_, meta) in enumerate(generated):
            meta["rank"] = i + 1

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
    if errors:
        result["errors"] = errors

    return _build_image_response(result, generated, output, prefix="gen")


@mcp.tool()
async def edit_image(
    prompt: str,
    ctx: Context,
    image: str = "",
    reference_images: list[str] | None = None,
    mask: str | None = None,
    edit_mode: str = "inpaint-insertion",
    aspect_ratio: str | None = None,
    count: int = 1,
    output: str = DEFAULT_OUTPUT,
) -> list | str:
    """Edit an existing image — add objects, remove objects, or extend the canvas.

    Pass the source image as a URL (http/https or nanobanana://). NEVER pass base64.
    If the user has no URL, direct them to the /upload page first.

    Args:
        image: Source image URL or nanobanana:// URL.
        prompt: Edit instruction. Be specific about what to change.
        reference_images: Optional list of reference image URLs (nanobanana:// or http).
            Use when the edit involves replacing or adding content based on another image
            (e.g. "replace bottle A with bottle B" — pass bottle B as a reference).
        mask: Optional mask image (URL). White = edit region, black = preserve.
        edit_mode: "inpaint-insertion" (add/replace), "inpaint-removal" (remove + fill),
                   "outpaint" (extend canvas). Default: inpaint-insertion
        aspect_ratio: Output aspect ratio (useful for outpaint). Default: same as input.
        count: Number of candidates (1–4). Default: 1
        output: "base64" (default), "s3", "gcs", or "cloud".

    Returns:
        Metadata JSON + inline images (base64 mode) or metadata with URLs (gcs mode).
    """
    from google.genai import types

    valid_edit_modes = {"inpaint-insertion", "inpaint-removal", "outpaint"}
    if edit_mode not in valid_edit_modes:
        return json.dumps({"error": f"Unknown edit_mode '{edit_mode}'.", "available": sorted(valid_edit_modes)})
    if output not in ("base64", "gcs", "s3", "cloud"):
        return json.dumps({"error": f"Unknown output mode '{output}'.", "available": ["base64", "gcs", "s3", "cloud"]})
    if output in ("gcs", "s3", "cloud") and not S3_BUCKET and not GCS_BUCKET:
        return json.dumps({"error": "Cloud output requested but no storage configured. Set S3_BUCKET or GCS_BUCKET env var, or use output='base64'."})

    count = max(1, min(count, 4))

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    try:
        img_bytes, img_mime = await _acquire_image(image, ctx, purpose="source image")
    except ValueError as e:
        return json.dumps({"error": str(e)})

    # Build edit prompt with mode context
    mode_instructions = {
        "inpaint-insertion": f"Edit this image: {prompt}",
        "inpaint-removal": f"Remove the following from this image and fill naturally: {prompt}",
        "outpaint": f"Extend this image beyond its current borders: {prompt}",
    }
    edit_prompt = mode_instructions[edit_mode]

    # Build content parts: source image + optional references + optional mask + prompt
    parts = [types.Part.from_bytes(data=img_bytes, mime_type=img_mime)]

    if reference_images:
        try:
            ref_parts = _build_ref_parts(reference_images)
            parts.extend(ref_parts)
            edit_prompt += (
                " Use the provided reference image(s) to guide what the result should look like."
            )
        except Exception as e:
            return json.dumps({"error": f"Failed to process reference image: {e}"})

    if mask:
        try:
            mask_bytes, mask_mime = _decode_mask(mask)
            parts.append(types.Part.from_bytes(data=mask_bytes, mime_type=mask_mime))
            edit_prompt += " Use the provided mask: white areas should be edited, black areas preserved."
        except Exception as e:
            return json.dumps({"error": f"Failed to decode mask image: {e}"})

    parts.append(types.Part.from_text(text=edit_prompt))

    generated = []
    errors = []
    for i in range(count):
        try:
            response = client.models.generate_content(
                model=MODEL_FLASH,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            raw = _extract_image(response)
            if raw:
                jpeg_bytes = _to_jpeg(raw)
                generated.append((jpeg_bytes, {"index": i + 1}))
            else:
                errors.append(f"Edit {i + 1}: no image in response")
        except Exception as e:
            errors.append(f"Edit {i + 1}: {e}")

    if not generated:
        return json.dumps({"error": "Edit produced no output images.", "details": errors})

    result = {"edit_mode": edit_mode, "prompt": prompt}
    if reference_images:
        result["reference_count"] = len(reference_images)
    if errors:
        result["errors"] = errors
    return _build_image_response(result, generated, output, prefix="edit")


@mcp.tool()
async def swap_background(
    background: str,
    ctx: Context,
    image: str = "",
    aspect_ratio: str | None = None,
    count: int = 1,
    output: str = DEFAULT_OUTPUT,
) -> list | str:
    """Replace the background of an image while keeping the foreground subject intact.

    Automatically segments the foreground and generates a new background.
    NEVER pass base64 — only URLs.
    If the user has no URL, direct them to the /upload page first.

    Args:
        image: Source image URL or nanobanana:// URL.
        background: Description of the new background. Be specific.
        aspect_ratio: Output aspect ratio. Default: same as input.
        count: Number of candidates (1–4). Default: 1
        output: "base64" (default), "s3", "gcs", or "cloud".

    Returns:
        Metadata JSON + inline images (base64 mode) or metadata with URLs (gcs mode).
    """
    from google.genai import types

    if output not in ("base64", "gcs", "s3", "cloud"):
        return json.dumps({"error": f"Unknown output mode '{output}'.", "available": ["base64", "gcs", "s3", "cloud"]})
    if output in ("gcs", "s3", "cloud") and not S3_BUCKET and not GCS_BUCKET:
        return json.dumps({"error": "Cloud output requested but no storage configured. Set S3_BUCKET or GCS_BUCKET env var, or use output='base64'."})

    count = max(1, min(count, 4))

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    try:
        img_bytes, img_mime = await _acquire_image(image, ctx, purpose="source image")
    except ValueError as e:
        return json.dumps({"error": str(e)})

    swap_prompt = (
        f"Keep the foreground subject exactly as it is, but completely replace the background with: {background}. "
        "The subject should look naturally placed in the new background with appropriate lighting and shadows."
    )

    parts = [
        types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
        types.Part.from_text(text=swap_prompt),
    ]

    generated = []
    errors = []
    for i in range(count):
        try:
            response = client.models.generate_content(
                model=MODEL_FLASH,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            raw = _extract_image(response)
            if raw:
                jpeg_bytes = _to_jpeg(raw)
                generated.append((jpeg_bytes, {"index": i + 1}))
            else:
                errors.append(f"Swap {i + 1}: no image in response")
        except Exception as e:
            errors.append(f"Swap {i + 1}: {e}")

    if not generated:
        return json.dumps({"error": "Background swap produced no output images.", "details": errors})

    result = {"background": background}
    if errors:
        result["errors"] = errors
    return _build_image_response(result, generated, output, prefix="bgswap")


@mcp.tool()
async def create_variations(
    ctx: Context,
    image: str = "",
    prompt: str | None = None,
    variation_strength: str = "medium",
    aspect_ratio: str = "4:5",
    resolution: str = "1K",
    quality: str = "default",
    count: int = 3,
    qa: bool = False,
    output: str = DEFAULT_OUTPUT,
) -> list | str:
    """Generate variations of an existing image.

    Preserves the core subject while exploring different compositions, lighting,
    or styling. Pass the source image as a URL.
    NEVER pass base64 — only URLs.
    If the user has no URL, direct them to the /upload page first.

    Args:
        image: Source image URL or nanobanana:// URL.
        prompt: Optional guidance for variations.
        variation_strength: "subtle", "medium", or "strong". Default: medium
        aspect_ratio: Output aspect ratio. Default: 4:5
        resolution: Output resolution: 1K, 2K, 4K. Default: 1K
        quality: "default" or "pro". Default: default
        count: Number of variations (1–4). Default: 3
        qa: Score each variation and rank by quality. Default: false
        output: "base64" (default), "s3", "gcs", or "cloud".

    Returns:
        Metadata JSON + inline images (base64 mode) or metadata with URLs (gcs mode).
    """
    from google.genai import types

    if aspect_ratio not in SUPPORTED_RATIOS:
        return json.dumps({"error": f"Unsupported aspect ratio '{aspect_ratio}'.", "supported": sorted(SUPPORTED_RATIOS)})
    if resolution not in RESOLUTIONS:
        return json.dumps({"error": f"Unsupported resolution '{resolution}'.", "supported": sorted(RESOLUTIONS)})
    if output not in ("base64", "gcs", "s3", "cloud"):
        return json.dumps({"error": f"Unknown output mode '{output}'.", "available": ["base64", "gcs", "s3", "cloud"]})
    if output in ("gcs", "s3", "cloud") and not S3_BUCKET and not GCS_BUCKET:
        return json.dumps({"error": "Cloud output requested but no storage configured. Set S3_BUCKET or GCS_BUCKET env var, or use output='base64'."})

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
        img_bytes, img_mime = await _acquire_image(image, ctx, purpose="source image")
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
    generated = []
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
                meta = {"index": i + 1}
                if qa:
                    meta["qa"] = _score_image(client, jpeg_bytes, variation_prompt)
                generated.append((jpeg_bytes, meta))
            else:
                errors.append(f"Variation {i + 1}: no image in response")
        except Exception as e:
            errors.append(f"Variation {i + 1}: {e}")

    if not generated:
        return json.dumps({"error": "All variation attempts failed.", "details": errors})

    if qa and len(generated) > 1:
        generated.sort(key=lambda x: x[1].get("qa", {}).get("total", 0), reverse=True)
        for i, (_, meta) in enumerate(generated):
            meta["rank"] = i + 1

    result = {
        "model": model_name,
        "variation_strength": variation_strength,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }
    if errors:
        result["errors"] = errors
    if prompt:
        result["guidance"] = prompt

    return _build_image_response(result, generated, output, prefix="var")


@mcp.tool()
async def analyze_image(
    ctx: Context,
    image: str = "",
    focus: str = "general",
) -> str:
    """Analyze an image using Gemini vision — describe, tag, or assess quality.

    NEVER pass base64 — only URLs.
    If the user has no URL, direct them to the /upload page first.

    Args:
        image: Image URL or nanobanana:// URL.
        focus: Analysis focus: "general", "tags", "alt-text", "quality", "brand"

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

    try:
        if focus == "quality":
            img_bytes, img_mime = await _acquire_image(image, ctx, purpose="image to analyze")
        else:
            img_bytes, img_mime = await _acquire_image(
                image, ctx, max_dim=REF_MAX_DIM, quality=85, purpose="image to analyze"
            )
    except ValueError as e:
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
    import sys

    # Validate API key at startup so users get a clear error immediately
    api_key = os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.stderr.write(
            "\n ERROR: No API key found.\n"
            "  Set GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable.\n"
            "  Get a key at: https://aistudio.google.com/apikey\n\n"
        )
        raise SystemExit(1)

    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")

    # In stdio mode, stdout is the MCP protocol channel — all logging must go to stderr
    log = sys.stderr.write

    if transport == "streamable-http":
        log(f"Starting NanoBanana MCP server on {mcp.settings.host}:{mcp.settings.port}\n")
        log(f"  Upload page: http://localhost:{mcp.settings.port}/upload\n")
    else:
        log(f"Starting NanoBanana MCP server ({transport} transport)\n")
    if GCS_BUCKET:
        log(f"  GCS output enabled: gs://{GCS_BUCKET}\n")
    else:
        log("  GCS output disabled (set GCS_BUCKET to enable)\n")
    log("\n")

    mcp.run(transport=transport)
