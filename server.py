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
import functools
import json
import os
import sys
import threading
import time
import uuid
from io import BytesIO
from typing import Annotated
from urllib.parse import urlparse

from pydantic import Field

log = sys.stderr.write

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
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 20 * 1024 * 1024))  # 20MB default

# ---------------------------------------------------------------------------
# Cloud storage config (optional — set bucket env var to enable)
# ---------------------------------------------------------------------------
GCS_BUCKET = os.environ.get("GCS_BUCKET")  # e.g. "my-nanobanana-images"
S3_BUCKET = os.environ.get("S3_BUCKET")    # e.g. "claude-image-cache"
S3_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
_req_durable_env = os.environ.get("REQUIRE_DURABLE_UPLOADS")
if _req_durable_env is None:
    # If cloud storage is configured, default to durable-only behavior so links are
    # always cloud URLs (S3/GCS) and never silently fall back to expiring local URLs.
    REQUIRE_DURABLE_UPLOADS = bool(S3_BUCKET or GCS_BUCKET)
else:
    REQUIRE_DURABLE_UPLOADS = _req_durable_env.lower() == "true"
FETCH_RETRIES = int(os.environ.get("FETCH_RETRIES", 1))
MAX_DATA_URI_CHARS = int(os.environ.get("MAX_DATA_URI_CHARS", 5_000))

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
            "It may have expired (images are kept for 1 hour) or been evicted under memory pressure. "
            "Upload the image again to get a fresh URL."
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
def _compute_base_url() -> str:
    """Compute the server's external base URL once at startup.

    Priority:
    1. PUBLIC_URL env var (always set this on Cloud Run — it's the only reliable source)
    2. Localhost fallback for local dev

    The Cloud Run auto-detection formula that was here previously
    (service-project.region.run.app) does not match the actual URL format
    Cloud Run assigns (service-hash.region.run.app). Removed to avoid silently
    generating broken URLs. Set PUBLIC_URL instead.
    """
    base_url = os.environ.get("PUBLIC_URL", "").rstrip("/")
    if base_url:
        return base_url
    port = int(os.environ.get("PORT", 8080))
    return f"http://localhost:{port}"


# Computed once at import time — avoids repeated env reads and ensures
# all tool responses use a consistent base URL throughout the process lifetime.
_BASE_URL = _compute_base_url()


def _get_upload_base_url() -> str:
    return _BASE_URL


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "nanobanana",
    instructions=(
        "NanoBanana image generation server. Powered by Gemini.\n\n"
        "## Getting images into tools\n\n"
        "All image tool parameters accept http/https URLs.\n\n"
        "1. **Direct URL (http/https, S3, CDN)** → pass straight to the tool.\n"
        "2. **Public Google Drive link** → pass straight to the tool (auto-rewritten).\n"
        "3. **Pasted or local file (claude.ai web)** → use Python urllib POST — this is the ONLY correct approach:\n\n"
        "```python\n"
        "import urllib.request, json, os\n"
        "uploads = '/mnt/user-data/uploads'\n"
        "files = sorted([f for f in os.listdir(uploads) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))],\n"
        "               key=lambda f: os.path.getmtime(os.path.join(uploads, f)), reverse=True)\n"
        "path = os.path.join(uploads, files[0])\n"
        "with open(path, 'rb') as f: data = f.read()\n"
        "req = urllib.request.Request('{upload_url}/upload', data=data, method='POST')\n"
        "result = json.loads(urllib.request.urlopen(req, timeout=30).read())\n"
        "print('url:', result['url'])\n"
        "```\n"
        "Pass the returned `url` directly to any tool. "
        "For multiple images, run once per image using `files[0]`, `files[1]`, etc.\n\n"
        "**NEVER encode images to base64/data URI** — passing data URIs through MCP parameters causes "
        "the transport to hang indefinitely regardless of size. There is no size threshold that makes it safe.\n\n"
        "4. **Local file path (Claude Code only):** `upload_image(image='/full/path/to/file.jpg')`\n"
        "5. **Manual fallback:** direct user to {upload_url}/upload to drag-and-drop.\n\n"
        "If urllib fails, do not retry — switch paths or direct the user to manual upload. "
        "Never fabricate URLs. Never start a local HTTP server.\n\n"
        "## Tools\n"
        "- upload_image — re-host an image URL or local file path to a server URL for use in other tools\n"
        "- generate_image — text-to-image, optional reference image URLs, style presets\n"
        "- edit_image — edit an image (inpaint, remove, outpaint); accepts multiple reference_images\n"
        "- swap_background — keep subject, replace background\n"
        "- create_variations — generate variations of an image\n"
        "- analyze_image — describe/tag/assess a single image\n"
        "- batch_analyze — analyze 2–20 images in parallel; use instead of repeated analyze_image calls\n"
        "- compare_images — compare 2–10 images side-by-side in one call (differences, quality, style)\n"
        "- list_styles — list available style presets\n\n"
        "## Common multi-image patterns\n"
        "**Analyze multiple images**: use batch_analyze, not repeated analyze_image calls.\n"
        "**Compare / diff two images**: use compare_images(focus='differences').\n"
        "**Pick the best of N images**: use compare_images(focus='quality').\n"
        "**Composite swap** (put object from image B into image A): "
        "call edit_image(image=urlA, reference_images=[urlB], "
        "prompt='replace X with the object from reference image 1').\n"
        "**Multi-object swap** (objects from images B and C into image A): "
        "call edit_image(image=urlA, reference_images=[urlB, urlC], "
        "prompt='replace X with reference image 1 and Y with reference image 2').\n\n"
        "Default aspect ratio 4:5, resolution 1K.\n\n"
        "## Rendering images — CRITICAL\n"
        "Every image tool result begins with a standalone markdown image link (e.g. `![](https://...)`).\n"
        "In your assistant reply, include BOTH:\n"
        "1) `chat_response_template` exactly as returned.\n"
        "2) If unavailable, include `display_markdown` and `display_text_urls` exactly as returned.\n"
        "Do not paraphrase, skip, or replace these with a text-only summary.\n"
        "To chain a generated image into another tool, use the `image_url` from the JSON metadata section."
    ).format(upload_url=_get_upload_base_url()),
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", 8080)),
    stateless_http=True,
)


# ---------------------------------------------------------------------------
# HTTP endpoints — direct image upload/serving (outside MCP protocol)
# ---------------------------------------------------------------------------
_UPLOAD_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><title>NanoBanana — Upload Images</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 1.4em; }
  .drop-zone { border: 2px dashed #ccc; border-radius: 8px; padding: 40px; text-align: center;
    cursor: pointer; transition: border-color 0.2s; margin: 20px 0; }
  .drop-zone:hover, .drop-zone.drag-over { border-color: #f5a623; background: #fffbf0; }
  input[type="file"] { display: none; }
  #urls { margin-top: 20px; }
  .url-row { display: flex; align-items: center; gap: 8px; margin: 8px 0; padding: 10px;
    background: #e8f5e9; border-radius: 6px; word-break: break-all; }
  .url-row code { flex: 1; background: #d0e8d0; padding: 4px 8px; border-radius: 3px; font-size: 0.85em; }
  .url-row button { flex-shrink: 0; padding: 4px 10px; border: none; background: #4caf50;
    color: white; border-radius: 4px; cursor: pointer; font-size: 0.85em; }
  .url-row button:hover { background: #388e3c; }
  #copyAll { display: none; margin-top: 12px; padding: 8px 16px; border: none; background: #f5a623;
    color: white; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 0.95em; }
  #copyAll:hover { background: #e09500; }
  .error { margin: 8px 0; padding: 10px; background: #fce4ec; border-radius: 6px; }
  .thumb { width: 56px; height: 56px; object-fit: cover; border-radius: 6px; border: 1px solid #c8e6c9; }
</style></head>
<body>
  <h1>NanoBanana — Upload Images</h1>
  <p>Drop one or more images to get URLs you can paste into Claude. Max file size: __MAX_UPLOAD_MB__MB each.</p>
  <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    Drop images here or click to browse
  </div>
  <input type="file" id="fileInput" accept="image/*" multiple>
  <div id="urls"></div>
  <button id="copyAll">Copy all URLs</button>
  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const urlsDiv = document.getElementById('urls');
    const copyAllBtn = document.getElementById('copyAll');
    const allUrls = [];
    const MAX_UPLOAD_BYTES = __MAX_UPLOAD_BYTES__;

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault(); dropZone.classList.remove('drag-over');
      uploadAll(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', () => { uploadAll(fileInput.files); fileInput.value = ''; });

    function uploadAll(files) {
      for (const file of files) {
        if (!file.type.startsWith('image/')) continue;
        if (file.size > MAX_UPLOAD_BYTES) {
          const row = document.createElement('div');
          row.className = 'error';
          row.textContent = file.name + ': file is too large. Max allowed is ' + Math.floor(MAX_UPLOAD_BYTES / (1024 * 1024)) + 'MB.';
          urlsDiv.appendChild(row);
          continue;
        }
        upload(file);
      }
    }

    async function upload(file) {
      const row = document.createElement('div');
      row.className = 'url-row';
      row.innerHTML = '<code>Uploading ' + file.name + '...</code>';
      urlsDiv.appendChild(row);
      const form = new FormData();
      form.append('file', file);
      try {
        const resp = await fetch('/upload', { method: 'POST', body: form });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        allUrls.push(data.url);
        row.innerHTML = '<img class=\"thumb\" src=\"' + data.url + '\" alt=\"upload preview\">' +
          '<code>' + data.url + '</code>' +
          '<button onclick="navigator.clipboard.writeText(\\'' + data.url + '\\')">Copy</button>';
        copyAllBtn.style.display = allUrls.length > 1 ? 'inline-block' : 'none';
        if (allUrls.length === 1) navigator.clipboard.writeText(data.url).catch(() => {});
      } catch (err) {
        row.className = 'error';
        row.textContent = file.name + ': ' + err.message;
      }
    }

    copyAllBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(allUrls.join('\\n')).catch(() => {});
      copyAllBtn.textContent = 'Copied!';
      setTimeout(() => { copyAllBtn.textContent = 'Copy all URLs'; }, 1500);
    });
  </script>
</body></html>"""

_UPLOAD_HTML = (
    _UPLOAD_HTML_TEMPLATE
    .replace("__MAX_UPLOAD_MB__", str(MAX_IMAGE_BYTES // (1024 * 1024)))
    .replace("__MAX_UPLOAD_BYTES__", str(MAX_IMAGE_BYTES))
)


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
        _check_image_size_limit(raw, source="Uploaded image")
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=413)

    try:
        normalized, mime = await _run_in_thread(_normalize_image, raw, max_dim=SOURCE_MAX_DIM, quality=92)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    img = PILImage.open(BytesIO(normalized))
    w, h = img.size

    if REQUIRE_DURABLE_UPLOADS and not (S3_BUCKET or GCS_BUCKET):
        return JSONResponse({
            "error": (
                "Durable uploads are required but no cloud storage is configured. "
                "Set S3_BUCKET or GCS_BUCKET, or disable REQUIRE_DURABLE_UPLOADS."
            )
        }, status_code=503)

    # Prefer cloud storage (durable URLs); fall back to in-memory store
    if S3_BUCKET or GCS_BUCKET:
        try:
            image_url = await _run_in_thread(_upload_to_cloud, normalized, "uploads")
            return JSONResponse({
                "url": image_url,
                "width": w,
                "height": h,
                "size_kb": len(normalized) // 1024,
                "usage": "Paste this URL into Claude to use with any image tool",
            }, status_code=201)
        except Exception as e:
            if REQUIRE_DURABLE_UPLOADS:
                return JSONResponse({
                    "error": (
                        f"Cloud upload failed and durable uploads are required: {e}"
                    )
                }, status_code=503)
            log(f"Cloud upload failed, falling back to in-memory store: {e}\n")

    img_id = _store_image(normalized, mime)

    # Use _BASE_URL (from PUBLIC_URL env var) so upload URLs match MCP tool URLs.
    # Previously used x-forwarded-proto + host headers, which produced a different
    # hostname than MCP tool responses when PUBLIC_URL is a custom domain.
    image_url = f"{_BASE_URL}/images/{img_id}"

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
_gemini_client = None
_gemini_client_lock = threading.Lock()


def _get_client():
    """Return a cached Gemini client, creating it on first call.

    Creating a new client per call wastes TLS handshakes and spawns redundant
    connection pools. The client holds no per-request state — safe to share.
    """
    global _gemini_client
    if _gemini_client is None:
        with _gemini_client_lock:
            if _gemini_client is None:
                api_key = os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "No API key found. Set GOOGLE_AI_API_KEY or GEMINI_API_KEY env var."
                    )
                from google import genai
                _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ---------------------------------------------------------------------------
# Helpers — image decode/encode
# ---------------------------------------------------------------------------
def _is_url(s: str) -> bool:
    try:
        parsed = urlparse(s)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def _normalize_share_url(url: str) -> str:
    """Rewrite known share-page URLs to direct download URLs."""
    import re
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if hostname in ("drive.google.com", "docs.google.com"):
        # /file/d/FILE_ID/view  or  /file/d/FILE_ID/preview
        m = re.search(r"/file/d/([^/?]+)", parsed.path)
        if m:
            return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
        # /open?id=FILE_ID  (common share link format)
        params = dict(p.split("=", 1) for p in parsed.query.split("&") if "=" in p)
        if "id" in params:
            return f"https://drive.google.com/uc?export=download&id={params['id']}"

    return url


def _fetch_url(url: str) -> tuple[bytes, str]:
    url = _normalize_share_url(url)
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

    for attempt in range(FETCH_RETRIES + 1):
        try:
            timeout = httpx.Timeout(connect=5.0, read=20.0, write=20.0, pool=20.0)
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()

                # Early fail before downloading large payloads when upstream advertises size.
                content_length = resp.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > MAX_IMAGE_BYTES:
                            raise ValueError(
                                f"Fetched image is too large ({int(content_length) / (1024 * 1024):.1f}MB). "
                                f"Maximum allowed size is {MAX_IMAGE_BYTES // (1024 * 1024)}MB."
                            )
                    except ValueError:
                        raise
                    except Exception:
                        # Ignore malformed Content-Length and fall back to streamed guard.
                        pass

                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        raise ValueError(
                            f"Fetched image is too large ({total / (1024 * 1024):.1f}MB). "
                            f"Maximum allowed size is {MAX_IMAGE_BYTES // (1024 * 1024)}MB."
                        )
                    chunks.append(chunk)
                body = b"".join(chunks)
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                hint = (
                    " If this is a Google Drive link, the file may not be publicly shared — "
                    "either share it with 'Anyone with the link' or upload it at the /upload page."
                ) if "drive.google.com" in url else " The server was denied access to this URL."
                raise ValueError(f"Image URL returned 403 (forbidden).{hint}")
            if e.response.status_code == 404:
                raise ValueError(
                    f"Image URL returned 404 — the image was not found. "
                    "It may have been deleted or the URL may be wrong."
                )
            raise ValueError(f"Failed to fetch image from URL: {e}")
        except ValueError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if attempt < FETCH_RETRIES:
                continue
            raise ValueError(f"Timed out or network error fetching image URL: {e}")

    if content_type == "text/html":
        if "drive.google.com" in url:
            raise ValueError(
                "Google Drive returned an HTML page instead of the image file. "
                "This happens with large files (>25MB) that require a virus-scan confirmation. "
                "To fix: open the link in your browser and click 'Download anyway', then use "
                "that download URL — or use the Google Drive MCP to read the file directly."
            )
        raise ValueError(
            "URL returned an HTML page instead of an image. "
            "Check that the URL points directly to an image file, not a web page."
        )
    _check_image_size_limit(body, source="Fetched image")
    return body, content_type


def _fix_base64_padding(s: str) -> str:
    s = s.rstrip()
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)
    return s


def _preview_text(text: str, max_chars: int = 180) -> str:
    """Shorten long text for metadata/UI surfaces while preserving meaning."""
    return text[:max_chars] + "..." if len(text) > max_chars else text


def _check_image_size_limit(img_bytes: bytes, source: str = "image") -> None:
    if len(img_bytes) > MAX_IMAGE_BYTES:
        max_mb = MAX_IMAGE_BYTES // (1024 * 1024)
        actual_mb = len(img_bytes) / (1024 * 1024)
        raise ValueError(
            f"{source} is too large ({actual_mb:.1f}MB). "
            f"Maximum allowed size is {max_mb}MB."
        )


def _normalize_image(img_bytes: bytes, max_dim: int, quality: int = 85,
                     output_format: str = "JPEG") -> tuple[bytes, str]:
    from PIL import Image as PILImage

    # Wrap all PIL operations — open() is lazy (reads header only), so truncated
    # images that pass open() can still raise OSError during resize/convert/save.
    try:
        img = PILImage.open(BytesIO(img_bytes))

        w, h = img.size
        if w < 1 or h < 1:
            raise ValueError(
                "Image has invalid dimensions. The data may be corrupted or truncated. "
                "Try using upload_image with a different image or passing an image URL."
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
    except ValueError:
        raise  # already a user-friendly message
    except Exception:
        raise ValueError(
            "Could not decode image data. The image may be corrupt or truncated. "
            "Try uploading the image again or using a different image URL."
        )


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
                # This is our own /images/ URL. Don't fall through to _fetch_url —
                # on localhost it hits the SSRF block; on Cloud Run multi-instance it
                # 404s with a confusing error. Give the real diagnosis instead.
                raise ValueError(
                    f"Image '{img_id}' has expired or was evicted from the server store. "
                    "Upload the image again to get a fresh URL."
                )
        return _fetch_url(ref)
    # Base64 data URI
    if ref.startswith("data:"):
        if len(ref) > MAX_DATA_URI_CHARS:
            raise ValueError(
                f"Data URI is too large ({len(ref):,} chars). "
                f"Maximum supported inline size is {MAX_DATA_URI_CHARS:,} chars. "
                "Use the /upload page to get a URL instead."
            )
        header, data = ref.split(",", 1)
        # Normalize common transport artifacts from shell/files/tool payloads.
        # Handles wrapped lines, escaped newlines, surrounding quotes, and URL-encoded base64.
        data = data.strip().strip('"').strip("'")
        data = data.replace("\\n", "").replace("\n", "").replace("\r", "").replace(" ", "")
        if "%" in data:
            from urllib.parse import unquote
            data = unquote(data)
        mime = header.split(";")[0].split(":")[1]
        try:
            decoded = base64.b64decode(_fix_base64_padding(data))
        except Exception:
            raise ValueError(
                "Could not decode data URI base64. The payload may be truncated or malformed. "
                "If this came from a text file, avoid printing/copying the full URI and use "
                "upload_image(image='/full/path/to/file.jpg') or /upload instead."
            )
        _check_image_size_limit(decoded, source="Data URI image")
        return decoded, mime
    # Catch unsupported URL schemes (gs://, s3://, ftp://, file://, etc.) before they
    # fall through to base64 decoding and produce a confusing "truncated" error.
    try:
        parsed = urlparse(ref)
        if parsed.scheme and parsed.scheme not in ("http", "https", "data") and len(parsed.scheme) > 1:
            raise ValueError(
                f"Unsupported URL scheme '{parsed.scheme}://'. "
                "Pass an http/https URL, a data URI (data:image/...;base64,...), "
                "or use upload_image to get a URL first."
            )
    except ValueError:
        raise
    except Exception:
        pass
    # Raw base64
    decoded = base64.b64decode(_fix_base64_padding(ref))
    _check_image_size_limit(decoded, source="Base64 image")
    return decoded, "image/jpeg"


def _decode_reference(ref: str) -> tuple[bytes, str]:
    raw, _ = _decode_raw(ref)
    return _normalize_image(raw, max_dim=REF_MAX_DIM)


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
            raw, _ = await _run_in_thread(_decode_raw, image)
            return await _run_in_thread(_normalize_image, raw, max_dim=max_dim, quality=quality)
        except Exception as decode_err:
            # If it looks like a URL or nanobanana ref, the error is real
            if _is_url(image) or image.startswith("nanobanana://"):
                raise ValueError(str(decode_err))
            # Likely truncated base64 — give upload instructions
            first_error = str(decode_err)
    else:
        first_error = f"No {purpose} provided"

    raise ValueError(
        f"{first_error}. "
        "Use the upload_image tool to upload the image first, then pass the returned URL here."
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


# ---------------------------------------------------------------------------
# Helpers — async wrappers for blocking operations
# ---------------------------------------------------------------------------
async def _call_gemini(client, **kwargs):
    """Run a Gemini API call in a thread so it doesn't block the event loop.

    This is critical on Cloud Run: blocking the event loop for 10-30s during
    image generation causes health check failures and dropped MCP connections.

    Uses _THREAD_POOL (32 workers) instead of asyncio.to_thread's default executor
    (~5 workers on Cloud Run 1-vCPU) so analyze_image handles concurrent load the
    same as generate_image and other tools.
    """
    return await _run_in_thread(client.models.generate_content, **kwargs)


import concurrent.futures as _cf
# Explicit pool sized for I/O-bound Gemini calls. Default pool (cpu_count+4 ≈ 5 on
# Cloud Run's 1-vCPU) would serialize concurrent requests even with asyncio.gather.
# 32 threads handle 8 concurrent users × count=4 with headroom to spare.
_THREAD_POOL = _cf.ThreadPoolExecutor(max_workers=32, thread_name_prefix="nb-worker")


async def _run_in_thread(fn, *args, **kwargs):
    """Run any blocking function (PIL, HTTP fetch, cloud upload) in a thread."""
    call = functools.partial(fn, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(_THREAD_POOL, call)


async def _generate_images(client, model, parts, config, count, label, qa_prompt=None):
    """Run count Gemini image generation calls concurrently, then score concurrently if qa_prompt given.

    All generation calls fire in parallel (each in its own thread), cutting wall time
    from count*T to ~T for a single model latency T. QA scoring also runs in parallel
    after all images are ready. This prevents Cloud Run timeout on count > 2 with qa=True.

    Returns (generated, errors) where generated is [(jpeg_bytes, meta_dict), ...].
    """
    async def _gen_one(index):
        try:
            response = await _run_in_thread(
                client.models.generate_content,
                model=model, contents=list(parts), config=config,
            )
            img_bytes = _extract_image(response)
            if img_bytes:
                jpeg_bytes = await _run_in_thread(_to_jpeg, img_bytes)
                return (jpeg_bytes, {"index": index}), None
            return None, f"{label} {index}: no image in response"
        except Exception as e:
            return None, f"{label} {index}: {e}"

    results = await asyncio.gather(*[_gen_one(i + 1) for i in range(count)])
    generated = [item for item, _ in results if item]
    errors = [err for _, err in results if err]

    if qa_prompt and generated:
        async def _score_one(item):
            jpeg_bytes, meta = item
            meta["qa"] = await _run_in_thread(_score_image, client, jpeg_bytes, qa_prompt)
            return jpeg_bytes, meta
        generated = list(await asyncio.gather(*[_score_one(item) for item in generated]))

    return generated, errors


def _build_ref_parts(reference_images: list | None) -> list:
    from google.genai import types

    if not reference_images:
        return []

    # Validate all refs before fetching
    for i, ref in enumerate(reference_images):
        if not ref or not ref.strip():
            raise ValueError(
                f"Reference image {i + 1} is empty. "
                "Upload each reference image first using upload_image to get a "
                "URL, then pass those URLs here."
            )

    # Fetch all reference images in parallel — sequential fetches degrade
    # linearly with ref count (500ms × N per image fetch)
    def _fetch_one(args):
        i, ref = args
        try:
            return i, _decode_reference(ref), None
        except Exception as e:
            return i, None, e

    with _cf.ThreadPoolExecutor(max_workers=min(len(reference_images), 8)) as pool:
        fetch_results = list(pool.map(_fetch_one, enumerate(reference_images)))

    # Re-raise first error encountered
    for i, result, err in fetch_results:
        if err is not None:
            raise ValueError(f"Reference image {i + 1}: {err}") from err

    # Reassemble in original order
    parts = [None] * len(reference_images)
    for i, result, _ in fetch_results:
        img_bytes, mime = result
        parts[i] = types.Part.from_bytes(data=img_bytes, mime_type=mime)
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
    """Upload JPEG bytes to S3 and return a plain public URL.

    The URL ends in .jpg so claude.ai's markdown renderer recognises it as an
    image and renders it inline. Presigned URLs (with ?X-Amz-... query params)
    do NOT end in .jpg — claude.ai falls back to "Open external link" instead
    of rendering the image. Requires the bucket to have a public-read policy.
    """
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
def _save_to_folder(jpeg_bytes: bytes, folder: str, prefix: str = "gen") -> str | None:
    """Save JPEG bytes to a local folder. Returns the saved file path, or None on failure."""
    try:
        os.makedirs(folder, exist_ok=True)
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(folder, filename)
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        return path
    except Exception as e:
        log(f"[nanobanana] Warning: failed to save image to {folder}: {e}\n")
        return None



def _build_image_response(
    result: dict,
    generated: list[tuple[bytes, dict]],
    save_folder: str | None = None,
    prefix: str = "gen",
) -> tuple[str, str, list[bytes]]:
    """Build a tool response.

    generated: list of (jpeg_bytes, per_image_metadata) tuples.

    Returns (render_md, json_str, thumbnails):
    - render_md: standalone markdown image link(s) — "![](url)" for one image,
      "![Image 1](url1)\\n\\n![Image 2](url2)" for multiple. This is the FIRST
      item returned by tools so Claude treats it as the primary result and
      includes it verbatim in its reply.
    - json_str: JSON metadata for tool chaining (image_url, size_kb, etc.)
    - thumbnails: 512px JPEG bytes for ImageContent blocks (tool pane previews)
    """
    base_url = _get_upload_base_url()
    thumbnails: list[bytes] = []
    for jpeg_bytes, meta in generated:
        img_id = _store_image(jpeg_bytes, "image/jpeg")
        local_url = f"{base_url}/images/{img_id}"
        meta["size_kb"] = len(jpeg_bytes) // 1024
        if S3_BUCKET:
            try:
                s3_url = _upload_to_s3(jpeg_bytes, prefix=prefix)
                meta["image_url"] = s3_url
                meta["storage_backend"] = "s3"
            except Exception as e:
                if REQUIRE_DURABLE_UPLOADS:
                    raise ValueError(
                        f"S3 upload failed and durable uploads are required: {e}"
                    )
                log(f"[nanobanana] S3 upload failed, falling back to local URL: {e}\n")
                meta["image_url"] = local_url
                meta["expires_in"] = "1 hour"
                meta["storage_backend"] = "local"
        else:
            meta["image_url"] = local_url
            meta["expires_in"] = "1 hour"
            meta["storage_backend"] = "local"
        # Build 512px thumbnail for ImageContent (tool pane preview)
        from PIL import Image as PILImage
        pil = PILImage.open(BytesIO(jpeg_bytes))
        if max(pil.size) > 512:
            pil.thumbnail((512, 512), PILImage.LANCZOS)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=75, optimize=True)
        thumbnails.append(buf.getvalue())
        if save_folder:
            saved_path = _save_to_folder(jpeg_bytes, save_folder, prefix)
            if saved_path:
                meta["saved_to"] = saved_path

    if len(generated) == 1:
        result.update(generated[0][1])
        result.pop("index", None)
        single_url = result.get("image_url", "")
        render_md = f"![]({single_url})\n\nURL: {single_url}"
        display_urls = [result.get("image_url", "")]
    else:
        result["images"] = [{k: v for k, v in meta.items() if k != "index"} for _, meta in generated]
        render_md = "\n\n".join(
            f"![Image {i + 1}]({img['image_url']})\nURL {i + 1}: {img['image_url']}"
            for i, img in enumerate(result["images"])
            if img.get("image_url")
        )
        display_urls = [img.get("image_url", "") for img in result["images"] if img.get("image_url")]

    # Extra fields to help Claude include visible links/images in the main chat reply.
    result["display_markdown"] = render_md
    result["display_urls"] = display_urls
    result["display_text_urls"] = "\n".join(display_urls)
    if display_urls:
        result["chat_response_template"] = (
            "Here is the image result:\n\n"
            f"{render_md}\n\n"
            "Direct URL(s):\n"
            f"{result['display_text_urls']}"
        )
        result["assistant_reply_hint"] = (
            "Use `chat_response_template` exactly in the assistant reply. "
            "If not possible, include both `display_markdown` and `display_text_urls`."
        )

    return render_md, json.dumps(result), thumbnails


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
        # Always recompute total from individual scores — Gemini sometimes returns
        # a normalized 1-10 overall score instead of the raw sum (max 50).
        criteria = ["composition", "clarity", "lighting", "color", "prompt_adherence"]
        scores["total"] = sum(scores.get(c, {}).get("score", 0) for c in criteria)
        return scores
    except Exception as e:
        return {"error": f"QA scoring failed: {e}", "total": 0}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def _upload_error(message: str, next_step: str | None = None) -> str:
    payload = {"error": message, "retryable": False}
    if next_step:
        payload["next_step"] = next_step
    return json.dumps(payload)


@mcp.tool()
async def upload_image(
    ctx: Context,
    image: Annotated[str, Field(pattern=r"^(https?://|/)")],
) -> str:
    """Re-host an image to a server URL for use in other tools.

    Accepts:
    - http/https URLs (including Google Drive share links — auto-rewritten)
    - local file paths (Claude Code only): `/full/path/to/image.jpg`

    **For pasted/dragged images in claude.ai web:** DO NOT pass a data URI here.
    Use the urllib POST snippet from server instructions instead — it POSTs the raw
    file bytes to /upload and returns a URL to pass to this or any other tool.
    Encoding to base64/data URI and passing it as a parameter causes the MCP
    transport to hang regardless of size. There is no workaround — use urllib POST.

    **For local files (Claude Code):** pass the file path directly:
        upload_image(image='/full/path/to/file.jpg')

    When cloud storage (S3/GCS) is configured, the returned URL is durable (no expiry).
    Otherwise the URL expires after 1 hour — the response includes "expires_in".

    Args:
        image: http/https URL or local file path.

    Returns:
        JSON with `url` (pass to any other tool), width, height, size_kb.
    """
    from PIL import Image as PILImage

    image = image.strip()
    if (image.startswith('"') and image.endswith('"')) or (image.startswith("'") and image.endswith("'")):
        image = image[1:-1]

    upload_url = f"{_BASE_URL}/upload"
    is_data_uri = image.startswith("data:")
    is_local_file = (
        not is_data_uri
        and not _is_url(image)
        and os.path.isfile(image)
    )
    if is_data_uri:
        return _upload_error(
            (
                "upload_image does NOT accept data URIs or base64-encoded images. "
                "Passing base64 through MCP parameters hangs the transport — there is no size that works. "
                "For pasted images in claude.ai, use the urllib POST snippet from server instructions: "
                "read the file from /mnt/user-data/uploads and POST it to /upload to get a URL. "
                "Do NOT compress and retry with a smaller data URI."
            ),
            next_step=f"Use urllib POST to {upload_url} (see server instructions for the snippet)",
        )
    if not _is_url(image) and not is_local_file:
        return _upload_error(
            (
                "upload_image accepts http/https URLs or local file paths. "
                "Do NOT pass data URIs — they hang the MCP transport. "
                "For pasted images in claude.ai web, use the urllib POST snippet "
                "from server instructions to upload from /mnt/user-data/uploads. "
                "For local files in Claude Code, pass the file path directly: "
                "upload_image(image='/full/path/to/file.jpg')."
            ),
            next_step=f"Or user can drag-and-drop at: {upload_url}",
        )

    try:
        if is_local_file:
            with open(image, "rb") as f:
                raw = f.read()
            _check_image_size_limit(raw, source="Local file image")
            normalized, norm_mime = await _run_in_thread(
                _normalize_image, raw, max_dim=SOURCE_MAX_DIM, quality=92
            )
        else:
            normalized, norm_mime = await _acquire_image(image, ctx, purpose="image")
    except ValueError as e:
        return _upload_error(str(e), next_step=f"Try /upload: {upload_url}")
    except Exception as e:
        return _upload_error(
            f"Failed to read local image file: {e}",
            next_step="Check that the path exists and points to an image file.",
        )

    img = PILImage.open(BytesIO(normalized))
    w, h = img.size

    if REQUIRE_DURABLE_UPLOADS and not (S3_BUCKET or GCS_BUCKET):
        return _upload_error(
            (
                "Durable uploads are required but no cloud storage is configured. "
                "Set S3_BUCKET or GCS_BUCKET, or disable REQUIRE_DURABLE_UPLOADS."
            )
        )

    # Prefer cloud storage (durable URLs); fall back to in-memory store
    if S3_BUCKET or GCS_BUCKET:
        try:
            url = await _run_in_thread(_upload_to_cloud, normalized, "uploads")
            return json.dumps({
                "url": url,
                "width": w,
                "height": h,
                "size_kb": len(normalized) // 1024,
                "usage": "Pass this URL to any other tool's image parameter",
            })
        except Exception as e:
            if REQUIRE_DURABLE_UPLOADS:
                return _upload_error(
                    f"Cloud upload failed and durable uploads are required: {e}"
                )
            log(f"Cloud upload failed, falling back to in-memory store: {e}\n")

    img_id = _store_image(normalized, norm_mime)
    # Return an HTTP URL so the image is accessible across Cloud Run instances.
    # nanobanana:// was instance-local with no HTTP fallback — requests routed to
    # a different instance would see "not found" even for freshly uploaded images.
    return json.dumps({
        "url": f"{_BASE_URL}/images/{img_id}",
        "width": w,
        "height": h,
        "size_kb": len(normalized) // 1024,
        "expires_in": "1 hour",
        "usage": "Pass this URL to any other tool's image parameter",
    })


@mcp.tool(structured_output=False)
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
    save_folder: str | None = None,
) -> list | str:
    """Generate an image from a text prompt with optional reference images and style presets.

    Reference images guide the model on style, subject appearance, or composition.
    Only pass URLs — use upload_image first if needed.

    If save_folder is provided, images are also saved as JPEG files in that directory.

    Args:
        prompt: What to generate. Describe subject, style, lighting, mood, etc.
        reference_images: Optional list of image URLs. Use upload_image first if needed.
        style: Optional style preset. Available: cinematic, product-photography,
               editorial, watercolor, flat-illustration, neon-noir, minimalist, vintage-film.
        enhance_prompt: If true, AI expands your prompt into a detailed generation prompt.
        aspect_ratio: Output aspect ratio. Default: 4:5
        resolution: Output resolution: 1K, 2K, 4K. Default: 1K
        quality: "default" (fast) or "pro" (higher quality). Default: default
        count: Number of images to generate (1–4). Default: 1
        qa: If true, AI-score each image. When count > 1, ranks by total score.
        save_folder: Optional local folder path to save generated JPEG files.

    Returns:
        JSON metadata with image_url for chaining, plus ImageContent objects shown inline
        by claude.ai. Use image_url to pass this image to edit_image, swap_background, etc.
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

    # Build prompt and fetch reference images — run in parallel when both are needed
    # since they're completely independent (prompt enhancement doesn't need the images,
    # image fetching doesn't need the enhanced prompt).
    final_prompt = prompt
    if style:
        final_prompt = STYLE_PRESETS[style] + final_prompt

    if enhance_prompt and reference_images:
        results = await asyncio.gather(
            _run_in_thread(_enhance_prompt, client, final_prompt),
            _run_in_thread(_build_ref_parts, reference_images),
            return_exceptions=True,
        )
        if isinstance(results[0], Exception):
            return json.dumps({"error": f"Prompt enhancement failed: {results[0]}"})
        if isinstance(results[1], Exception):
            return json.dumps({"error": f"Failed to process reference image: {results[1]}"})
        final_prompt, parts = results
    elif enhance_prompt:
        try:
            final_prompt = await _run_in_thread(_enhance_prompt, client, final_prompt)
        except Exception as e:
            return json.dumps({"error": f"Prompt enhancement failed: {e}"})
        parts = []
    else:
        try:
            parts = await _run_in_thread(_build_ref_parts, reference_images)
        except Exception as e:
            return json.dumps({"error": f"Failed to process reference image: {e}"})

    parts.append(types.Part.from_text(text=final_prompt))

    model_name = _pick_model(quality)
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=resolution,
        ),
    )

    generated, errors = await _generate_images(
        client, model_name, parts, config,
        count, "Image", qa_prompt=final_prompt if qa else None,
    )

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
        "prompt_used": _preview_text(final_prompt, 200),
    }
    if enhance_prompt or style:
        result["original_prompt"] = _preview_text(prompt, 120)
    if style:
        result["style"] = style
    if reference_images:
        result["reference_count"] = len(reference_images)
    if errors:
        result["errors"] = errors

    try:
        render_md, json_result, thumbnails = await _run_in_thread(_build_image_response, result, generated, save_folder, prefix="gen")
    except Exception as e:
        return json.dumps({"error": f"Failed to store/upload generated images: {e}"})
    return [render_md, json_result] + [Image(data=t, format="jpeg") for t in thumbnails]


@mcp.tool(structured_output=False)
async def edit_image(
    prompt: str,
    ctx: Context,
    image: str,
    reference_images: list[str] | None = None,
    mask: str | None = None,
    edit_mode: str = "inpaint-insertion",
    aspect_ratio: str | None = None,
    count: int = 1,
    save_folder: str | None = None,
) -> list | str:
    """Edit an existing image — add objects, remove objects, or extend the canvas.

    Only pass URLs — use upload_image first if needed.

    Edited images are always displayed inline in Claude. If save_folder is provided,
    images are also saved as JPEG files in that directory.

    Args:
        image: Source image URL. Use upload_image first if needed.
        prompt: Edit instruction. Be specific about what to change.
        reference_images: Optional list of reference image URLs.
            Use when the edit involves replacing or inserting content from other images.
            Single ref:   "replace the bottle with the one in the reference image"
            Multiple refs: name them by position in the prompt —
              "replace the bottle on the left with the object from reference image 1
               and the bottle on the right with the object from reference image 2"
            Composite swap (e.g. user has 3 images and wants objects from 2+3 placed
            into image 1): pass image 1 as `image`, images 2 and 3 as
            `reference_images`, and write the prompt describing which object goes where.
        mask: Optional mask image (URL). White = edit region, black = preserve.
        edit_mode: "inpaint-insertion" (add/replace), "inpaint-removal" (remove + fill),
                   "outpaint" (extend canvas). Default: inpaint-insertion
        aspect_ratio: Output aspect ratio (useful for outpaint). Default: same as input.
        count: Number of candidates (1–4). Default: 1
        save_folder: Optional local folder path to save edited JPEG files.

    Returns:
        JSON metadata with image_url for chaining, plus ImageContent objects shown inline
        by claude.ai. Use image_url to pass this image to edit_image, swap_background, etc.
    """
    from google.genai import types

    valid_edit_modes = {"inpaint-insertion", "inpaint-removal", "outpaint"}
    if edit_mode not in valid_edit_modes:
        return json.dumps({"error": f"Unknown edit_mode '{edit_mode}'.", "available": sorted(valid_edit_modes)})
    if aspect_ratio is not None and aspect_ratio not in SUPPORTED_RATIOS:
        return json.dumps({"error": f"Unsupported aspect ratio '{aspect_ratio}'.", "supported": sorted(SUPPORTED_RATIOS)})

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
            ref_parts = await _run_in_thread(_build_ref_parts, reference_images)
            parts.extend(ref_parts)
            if len(reference_images) == 1:
                edit_prompt += " Use the provided reference image to guide the edit."
            else:
                # Label references by position so Gemini can match prompt intent to
                # specific images — "first reference image", "second reference image", etc.
                labels = ", ".join(
                    f"reference image {i + 1}" for i in range(len(reference_images))
                )
                edit_prompt += (
                    f" {len(reference_images)} reference images are provided ({labels}). "
                    "Apply each reference to the specific part of the prompt it relates to "
                    "(e.g. 'replace X with the object from reference image 1')."
                )
        except Exception as e:
            return json.dumps({"error": f"Failed to process reference image: {e}"})

    if mask:
        try:
            mask_bytes, mask_mime = await _run_in_thread(_decode_mask, mask)
            parts.append(types.Part.from_bytes(data=mask_bytes, mime_type=mask_mime))
            edit_prompt += " Use the provided mask: white areas should be edited, black areas preserved."
        except Exception as e:
            return json.dumps({"error": f"Failed to decode mask image: {e}"})

    parts.append(types.Part.from_text(text=edit_prompt))

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio) if aspect_ratio else None,
    )
    generated, errors = await _generate_images(
        client, MODEL_FLASH, parts, config, count, "Edit",
    )

    if not generated:
        return json.dumps({"error": "Edit produced no output images.", "details": errors})

    result = {"edit_mode": edit_mode, "prompt_preview": _preview_text(prompt, 180)}
    if reference_images:
        result["reference_count"] = len(reference_images)
    if errors:
        result["errors"] = errors
    try:
        render_md, json_result, thumbnails = await _run_in_thread(_build_image_response, result, generated, save_folder, prefix="edit")
    except Exception as e:
        return json.dumps({"error": f"Failed to store/upload edited images: {e}"})
    return [render_md, json_result] + [Image(data=t, format="jpeg") for t in thumbnails]


@mcp.tool(structured_output=False)
async def swap_background(
    background: str,
    ctx: Context,
    image: str,
    aspect_ratio: str | None = None,
    count: int = 1,
    save_folder: str | None = None,
) -> list | str:
    """Replace the background of an image while keeping the foreground subject intact.

    Automatically segments the foreground and generates a new background.
    Only pass URLs — use upload_image first if needed.

    Result images are always displayed inline in Claude. If save_folder is provided,
    images are also saved as JPEG files in that directory.

    Args:
        image: Source image URL. Use upload_image first if needed.
        background: Description of the new background. Be specific.
        aspect_ratio: Output aspect ratio. Default: same as input.
        count: Number of candidates (1–4). Default: 1
        save_folder: Optional local folder path to save result JPEG files.

    Returns:
        JSON metadata with image_url for chaining, plus ImageContent objects shown inline
        by claude.ai. Use image_url to pass this image to edit_image, swap_background, etc.
    """
    from google.genai import types

    if aspect_ratio is not None and aspect_ratio not in SUPPORTED_RATIOS:
        return json.dumps({"error": f"Unsupported aspect ratio '{aspect_ratio}'.", "supported": sorted(SUPPORTED_RATIOS)})

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

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio) if aspect_ratio else None,
    )
    generated, errors = await _generate_images(
        client, MODEL_FLASH, parts, config, count, "Swap",
    )

    if not generated:
        return json.dumps({"error": "Background swap produced no output images.", "details": errors})

    result = {"background": background}
    if errors:
        result["errors"] = errors
    try:
        render_md, json_result, thumbnails = await _run_in_thread(_build_image_response, result, generated, save_folder, prefix="bgswap")
    except Exception as e:
        return json.dumps({"error": f"Failed to store/upload background-swapped images: {e}"})
    return [render_md, json_result] + [Image(data=t, format="jpeg") for t in thumbnails]


@mcp.tool(structured_output=False)
async def create_variations(
    ctx: Context,
    image: str,
    prompt: str | None = None,
    variation_strength: str = "medium",
    aspect_ratio: str | None = None,
    resolution: str = "1K",
    quality: str = "default",
    count: int = 3,
    qa: bool = False,
    save_folder: str | None = None,
) -> list | str:
    """Generate variations of an existing image.

    Preserves the core subject while exploring different compositions, lighting,
    or styling. Only pass URLs — use upload_image first if needed.

    Variations are always displayed inline in Claude. If save_folder is provided,
    images are also saved as JPEG files in that directory.

    Args:
        image: Source image URL. Use upload_image first if needed.
        prompt: Optional guidance for variations.
        variation_strength: "subtle", "medium", or "strong". Default: medium
        aspect_ratio: Output aspect ratio. Default: same as source image
        resolution: Output resolution: 1K, 2K, 4K. Default: 1K
        quality: "default" or "pro". Default: default
        count: Number of variations (1–4). Default: 3
        qa: Score each variation and rank by quality. Default: false
        save_folder: Optional local folder path to save variation JPEG files.

    Returns:
        JSON metadata with image_url for chaining, plus ImageContent objects shown inline
        by claude.ai. Use image_url to pass this image to edit_image, swap_background, etc.
    """
    from google.genai import types

    if aspect_ratio is not None and aspect_ratio not in SUPPORTED_RATIOS:
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
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=resolution,
        ) if aspect_ratio else types.ImageConfig(image_size=resolution),
    )
    generated, errors = await _generate_images(
        client, model_name, parts, config, count, "Variation",
        qa_prompt=variation_prompt if qa else None,
    )

    if not generated:
        return json.dumps({"error": "All variation attempts failed.", "details": errors})

    if qa and len(generated) > 1:
        generated.sort(key=lambda x: x[1].get("qa", {}).get("total", 0), reverse=True)
        for i, (_, meta) in enumerate(generated):
            meta["rank"] = i + 1

    result = {
        "model": model_name,
        "variation_strength": variation_strength,
        "resolution": resolution,
    }
    if errors:
        result["errors"] = errors
    if prompt:
        result["guidance"] = prompt

    try:
        render_md, json_result, thumbnails = await _run_in_thread(_build_image_response, result, generated, save_folder, prefix="var")
    except Exception as e:
        return json.dumps({"error": f"Failed to store/upload variation images: {e}"})

    return [render_md, json_result] + [Image(data=t, format="jpeg") for t in thumbnails]


# ---------------------------------------------------------------------------
# Analysis focus prompts — shared by analyze_image and batch_analyze
# ---------------------------------------------------------------------------
ANALYZE_FOCUS_PROMPTS: dict[str, str] = {
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


async def _analyze_one(client, img_bytes: bytes, img_mime: str, focus: str) -> dict:
    """Run a single image analysis and return the parsed result dict."""
    from google.genai import types

    resp = await _call_gemini(
        client,
        model=MODEL_TEXT,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
            types.Part.from_text(text=ANALYZE_FOCUS_PROMPTS[focus]),
        ],
        config={"response_mime_type": "application/json"},
    )
    try:
        analysis = json.loads(resp.text)
    except (json.JSONDecodeError, TypeError):
        return {"focus": focus, "raw_response": resp.text if resp.text else "No response"}

    # Recompute total from individual scores — Gemini sometimes returns a
    # normalized 1-10 overall score instead of the raw sum (max 50).
    if focus == "quality":
        criteria = ["sharpness", "exposure", "composition", "color_balance", "noise"]
        analysis["total"] = sum(analysis.get(c, {}).get("score", 0) for c in criteria)
    analysis["focus"] = focus
    return analysis


@mcp.tool()
async def analyze_image(
    ctx: Context,
    image: str,
    focus: str = "general",
) -> str:
    """Analyze a single image using Gemini vision — describe, tag, or assess quality.

    For multiple images, use batch_analyze instead (runs all analyses in parallel).
    Only pass URLs — use upload_image first if needed.

    Args:
        image: Image URL. Use upload_image first if needed.
        focus: Analysis focus: "general", "tags", "alt-text", "quality", "brand"

    Returns:
        JSON with the analysis result.
    """
    if focus not in ANALYZE_FOCUS_PROMPTS:
        return json.dumps({"error": f"Unknown focus '{focus}'.", "available": list(ANALYZE_FOCUS_PROMPTS.keys())})

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
        analysis = await _analyze_one(client, img_bytes, img_mime, focus)
        return json.dumps(analysis)
    except Exception as e:
        return json.dumps({"error": f"Analysis failed: {e}"})


@mcp.tool()
async def batch_analyze(
    ctx: Context,
    images: list[str],
    focus: str = "general",
) -> str:
    """Analyze multiple images in parallel using Gemini vision.

    Use this whenever you have 2 or more images to analyze — it runs all analyses
    concurrently so the total time is roughly the same as analyzing one image.
    For a single image use analyze_image instead.

    Only pass URLs — use upload_image first if needed.

    Args:
        images: List of image URLs (2–20). Use upload_image first if needed.
        focus: Analysis focus applied to every image: "general", "tags",
               "alt-text", "quality", "brand"

    Returns:
        JSON with a "results" array in the same order as the input images.
        Each result contains the analysis fields plus an "index" (1-based) and
        the original "image_url" for easy cross-referencing. Failed images get
        an "error" field instead of analysis fields.
    """
    if not images:
        return json.dumps({"error": "images list is empty."})
    if len(images) > 20:
        return json.dumps({"error": f"Too many images ({len(images)}). Maximum is 20 per call."})
    if focus not in ANALYZE_FOCUS_PROMPTS:
        return json.dumps({"error": f"Unknown focus '{focus}'.", "available": list(ANALYZE_FOCUS_PROMPTS.keys())})

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    max_dim = SOURCE_MAX_DIM if focus == "quality" else REF_MAX_DIM

    async def _analyze_indexed(index: int, image_ref: str) -> dict:
        result = {"index": index + 1, "image_url": image_ref}
        try:
            img_bytes, img_mime = await _acquire_image(
                image_ref, ctx, max_dim=max_dim, quality=85, purpose=f"image {index + 1}"
            )
            analysis = await _analyze_one(client, img_bytes, img_mime, focus)
            result.update(analysis)
        except Exception as e:
            result["error"] = str(e)
        return result

    results = await asyncio.gather(*[_analyze_indexed(i, img) for i, img in enumerate(images)])
    return json.dumps({"focus": focus, "count": len(images), "results": list(results)})


# ---------------------------------------------------------------------------
# Comparison focus prompts — all images are visible to the model in one call
# ---------------------------------------------------------------------------
COMPARE_FOCUS_PROMPTS: dict[str, str] = {
    "differences": (
        "You are shown {n} images (image 1, image 2, ...). "
        "Identify and describe every meaningful visual difference between them. "
        "Be specific: note changes in objects, colors, lighting, composition, text, etc. "
        "Return as JSON: "
        '{{"summary": "...", "differences": [{{"aspect": "...", "description": "..."}}]}}'
    ),
    "similarities": (
        "You are shown {n} images. "
        "Describe what they have in common visually: shared objects, style, palette, mood, composition. "
        "Return as JSON: "
        '{{"summary": "...", "similarities": [{{"aspect": "...", "description": "..."}}]}}'
    ),
    "quality": (
        "You are shown {n} images. Rank them by overall technical quality (sharpness, exposure, "
        "composition, color accuracy). State which is best and why. "
        "Score each image 1–10. "
        "Return as JSON: "
        '{{"rankings": [{{"image": N, "score": N, "notes": "..."}}], "best": N, "reasoning": "..."}}'
    ),
    "style": (
        "You are shown {n} images. Compare their visual style and aesthetic: "
        "mood, color palette, photographic/illustration style, intended audience. "
        "Return as JSON: "
        '{{"per_image": [{{"image": N, "style": "...", "mood": "...", "palette": ["..."]}}], '
        '"overall_comparison": "..."}}'
    ),
    "general": (
        "You are shown {n} images. Provide a comprehensive side-by-side comparison: "
        "content, style, quality, mood, notable differences and similarities. "
        "Return as JSON: "
        '{{"summary": "...", "per_image": [{{"image": N, "description": "..."}}], '
        '"key_differences": ["..."], "key_similarities": ["..."]}}'
    ),
}


@mcp.tool()
async def compare_images(
    ctx: Context,
    images: list[str],
    focus: str = "differences",
) -> str:
    """Compare 2–10 images side-by-side using Gemini vision in a single call.

    Unlike batch_analyze (which analyzes each image independently), compare_images
    sends all images to the model at once so it can reason about relationships,
    differences, and similarities across the whole set.

    Use cases:
    - "What changed between version A and version B?"
    - "Which of these 3 product photos looks best?"
    - "How does image 1's style compare to image 2?"

    Only pass URLs — use upload_image first if needed.

    Args:
        images: List of 2–10 image URLs.
        focus: What to compare.
            "differences" — what changed or differs between images (default)
            "similarities" — what they have in common
            "quality"      — which is highest quality and why; scores + ranking
            "style"        — aesthetic/style comparison
            "general"      — comprehensive side-by-side overview

    Returns:
        JSON with comparison results. Image references use 1-based index
        (image 1, image 2, …) matching the input order.
    """
    if len(images) < 2:
        return json.dumps({"error": "compare_images requires at least 2 images."})
    if len(images) > 10:
        return json.dumps({"error": f"Too many images ({len(images)}). Maximum is 10 per comparison."})
    if focus not in COMPARE_FOCUS_PROMPTS:
        return json.dumps({"error": f"Unknown focus '{focus}'.", "available": list(COMPARE_FOCUS_PROMPTS.keys())})

    try:
        client = _get_client()
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    max_dim = SOURCE_MAX_DIM if focus == "quality" else REF_MAX_DIM

    # Fetch all images concurrently
    async def _fetch_indexed(index: int, image_ref: str):
        return index, await _acquire_image(
            image_ref, ctx, max_dim=max_dim, quality=85, purpose=f"image {index + 1}"
        )

    try:
        fetched = await asyncio.gather(*[_fetch_indexed(i, img) for i, img in enumerate(images)])
    except Exception as e:
        return json.dumps({"error": f"Failed to load images: {e}"})

    from google.genai import types

    # Build parts: all images in order, then the comparison prompt
    parts = []
    for _, (img_bytes, img_mime) in sorted(fetched, key=lambda x: x[0]):
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=img_mime))
    prompt_text = COMPARE_FOCUS_PROMPTS[focus].format(n=len(images))
    parts.append(types.Part.from_text(text=prompt_text))

    try:
        resp = await _call_gemini(
            client,
            model=MODEL_TEXT,
            contents=parts,
            config={"response_mime_type": "application/json"},
        )
        comparison = json.loads(resp.text)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"focus": focus, "raw_response": resp.text if resp.text else "No response"})
    except Exception as e:
        return json.dumps({"error": f"Comparison failed: {e}"})

    comparison["focus"] = focus
    comparison["image_count"] = len(images)
    return json.dumps(comparison)


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
# MCP Prompts — invocable workflows for Claude.ai
# ---------------------------------------------------------------------------
@mcp.prompt()
def handle_image() -> str:
    """Guide the user through getting a pasted or local image into NanoBanana tools."""
    upload_url = _get_upload_base_url()
    return (
        "The user wants to use an image with NanoBanana tools but doesn't have a URL for it. "
        "Here's how to get one:\n\n"
        "**Pasted/local image (use urllib POST — do NOT encode to data URI)**\n"
        "In claude.ai, pasted images are saved to `/mnt/user-data/uploads`. "
        "Use the urllib POST snippet from server instructions to upload the raw file bytes "
        "to /upload and get back a URL. NEVER encode to base64/data URI — it hangs the transport.\n\n"
        "**Manual fallback**\n"
        f"Direct the user to upload manually:\n\n"
        f"> Please upload your image at: **{upload_url}/upload**\n"
        "> Drag it onto that page — it takes a second and gives you a URL to paste back here.\n\n"
        "**Google Drive link**\n"
        "Public Drive links work automatically — just pass them to the tool.\n\n"
        "**Once you have a URL**\n"
        "Pass it to the appropriate tool: `edit_image`, `swap_background`, "
        "`create_variations`, `analyze_image`, or as `reference_images` for `generate_image`."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
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

    if transport == "streamable-http":
        log(f"Starting NanoBanana MCP server on {mcp.settings.host}:{mcp.settings.port}\n")
        log(f"  Upload page: http://localhost:{mcp.settings.port}/upload\n")
    else:
        log(f"Starting NanoBanana MCP server ({transport} transport)\n")
    if S3_BUCKET:
        log(f"  S3 storage: s3://{S3_BUCKET} (region: {S3_REGION}) — durable URLs returned per generation\n")
    if GCS_BUCKET:
        log(f"  GCS storage: gs://{GCS_BUCKET}\n")
    if not S3_BUCKET and not GCS_BUCKET:
        log("  No cloud storage configured (set S3_BUCKET or GCS_BUCKET to enable)\n")
    log("\n")

    if transport == "streamable-http":
        # Wrap the Starlette app with a body-size middleware so oversized MCP
        # requests (e.g. large data URIs) are rejected with HTTP 413 immediately
        # rather than hanging the tool call indefinitely in the client.
        import uvicorn
        from starlette.middleware.base import BaseHTTPMiddleware

        _MAX_REQUEST_BODY = int(os.environ.get("MAX_REQUEST_BODY", 400_000))  # bytes

        class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                cl = request.headers.get("content-length")
                if cl and int(cl) > _MAX_REQUEST_BODY:
                    from starlette.responses import Response
                    return Response(
                        (
                            '{"jsonrpc":"2.0","error":{'
                            '"code":-32600,'
                            '"message":"Request body too large. Compress the image further '
                            f'(max ~{_MAX_REQUEST_BODY // 1000}KB) or use the /upload page."'
                            '},"id":null}'
                        ),
                        status_code=413,
                        media_type="application/json",
                    )
                return await call_next(request)

        app = mcp.streamable_http_app()
        app.add_middleware(_BodySizeLimitMiddleware)
        uvicorn.run(app, host=mcp.settings.host, port=mcp.settings.port, log_level="warning")
    else:
        mcp.run(transport=transport)
