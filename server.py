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
        "## STOP — read before ANY image tool call\n"
        "NEVER pass base64 strings or data: URIs to any tool parameter. "
        "They are too large for MCP transport and WILL hang the connection. "
        "Pick the upload path that matches your environment below.\n\n"
        "## Getting images into tools\n\n"
        "All image tool parameters accept http/https URLs.\n\n"
        "1. **Direct URL (http/https, S3, CDN)** → pass straight to the tool.\n"
        "2. **Public Google Drive link** → pass straight to the tool (auto-rewritten).\n\n"
        "### Pasted / dragged image — pick the path for YOUR environment\n\n"
        "**A. claude.ai web OR Claude Cowork — Python tool (auto-discovers uploads folder)**\n\n"
        "Works for both claude.ai web (`/mnt/user-data/uploads/`) and Cowork "
        "(`~/Library/Application Support/Claude/local-agent-mode-sessions/<session>/uploads/`). "
        "Run with the **Python** tool (NOT bash/shell — fails with 'Bad for loop variable'):\n\n"
        "```python\n"
        "import urllib.request, json, os, time, glob\n"
        "SERVER = '{upload_url}'\n"
        "candidates = ['/mnt/user-data/uploads'] + sorted(\n"
        "    glob.glob(os.path.expanduser('~/Library/Application Support/Claude/local-agent-mode-sessions/*/uploads')),\n"
        "    key=os.path.getmtime, reverse=True)\n"
        "uploads = next((p for p in candidates if os.path.isdir(p)), None)\n"
        "if not uploads:\n"
        "    print(f'No uploads folder found. Use the upload/paperclip button in Cowork to attach the image as a FILE (pasted/dragged-inline images are NOT written to disk), or upload manually at {{SERVER}}/upload')\n"
        "else:\n"
        "    files = sorted([f for f in os.listdir(uploads) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))],\n"
        "                   key=lambda f: os.path.getmtime(os.path.join(uploads, f)), reverse=True)[:4]\n"
        "    if not files:\n"
        "        print(f'Uploads folder is empty: {{uploads}}. Pasted/dragged-inline images are NOT written to disk in Cowork — re-attach using the upload/paperclip button (which writes files to disk), or upload manually at {{SERVER}}/upload')\n"
        "    for i, fname in enumerate(files):\n"
        "        with open(os.path.join(uploads, fname), 'rb') as fh: data = fh.read()\n"
        "        req = urllib.request.Request(f'{{SERVER}}/upload', data=data, method='POST')\n"
        "        for attempt in range(3):\n"
        "            try:\n"
        "                result = json.loads(urllib.request.urlopen(req, timeout=30).read())\n"
        "                print(f'image{{i}}: {{result[\"url\"]}}')\n"
        "                break\n"
        "            except Exception as e:\n"
        "                if attempt < 2: time.sleep(2 ** attempt)\n"
        "                else: print(f'image{{i}}: FAILED after 3 attempts: {{e}}. Upload manually at {{SERVER}}/upload')\n"
        "```\n\n"
        "**B. Claude Code (CLI) — Bash tool, real local filesystem**\n\n"
        "If the user gives you a real path on their machine, just upload it via curl:\n\n"
        "```bash\n"
        'curl -fsS -X POST --data-binary "@/full/path/to/image.jpg" \\\n'
        '  "{upload_url}/upload" \\\n'
        "  | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get('url') or d)\"\n"
        "```\n\n"
        "Faster path when this MCP is running locally as a stdio subprocess (the server can read "
        "the user's filesystem directly): call `upload_image(image='/full/path/to/file.jpg')` and "
        "the tool will read and re-host the file itself.\n\n"
        "**NEVER encode to base64/data URI** — the upload_image tool rejects them and MCP transport hangs.\n\n"
        "### Fallbacks (any environment)\n"
        "3. **Manual upload:** direct user to {upload_url}/upload to drag-and-drop.\n"
        "4. **Full web app (no MCP needed):** {upload_url}/app — upload, generate, edit, swap backgrounds, "
        "create variations, and download results. Best option for a smooth workflow.\n\n"
        "If a snippet fails (503, network error), do not retry blindly — direct the user to "
        "{upload_url}/upload or {upload_url}/app.\n\n"
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
        "## Intake before any image-output tool — REQUIRED\n"
        "Image-output tools: `generate_image`, `edit_image`, `swap_background`, `create_variations`. "
        "Before the first call to any of these, confirm aspect ratio (and resolution where supported) "
        "with the user. Ask once, in a single message:\n\n"
        "> Before I {{generate / edit / swap background / create variations}}, two quick choices:\n"
        "> • **Aspect ratio?** 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9\n"
        "> • **Resolution?** 1K, 2K, 4K\n"
        ">\n"
        "> Defaults: 4:5 / 1K for new generations; same shape as the source image for edits, "
        "background swaps, and variations. Reply with your picks, or say 'defaults'.\n\n"
        "Per-tool support:\n"
        "- `generate_image`: aspect_ratio + resolution. No source, so the user must pick a ratio.\n"
        "- `create_variations`: aspect_ratio + resolution. Default = same shape as source.\n"
        "- `edit_image`, `swap_background`: aspect_ratio only — Gemini's edit API doesn't accept "
        "a resolution param. Default = same shape as source. If the user picks a resolution for these, "
        "explain it'll match the source and proceed.\n\n"
        "Skip the intake when the user already named values in the conversation, when re-running at "
        "known settings, or when chaining from another tool with fixed values.\n\n"
        "## Presenting images to the user — CRITICAL\n"
        "The tool pane shows a small thumbnail. Your chat reply is the user-facing presentation.\n"
        "After every image tool call, your assistant reply MUST:\n"
        "1. Show a markdown image embed `![](image_url)` for each image — use the URL from `image_url` in the JSON metadata.\n"
        "2. Show a clickable `[Download image](image_url)` link below each embed.\n"
        "3. Carry the **Save to Google Drive?** nudge from the tool's render_md through to the user verbatim.\n"
        "4. Add a brief, friendly description of what was generated or edited.\n"
        "Do NOT dump raw JSON in your reply — extract just the URL.\n"
        "Do NOT say 'here is your image' without the actual embed and download link.\n"
        "The download link is the user's reliable fallback when the embed doesn't render.\n"
        "To chain a generated image into another tool, use the `image_url` from the JSON metadata.\n\n"
        "## Save to Google Drive — when the user replies 'save'\n"
        "Image-output tools end their render_md with a 'Save to Google Drive?' nudge. "
        "If the user replies 'save', 'save to drive', 'save to google drive', or similar:\n"
        "1. Fetch the image bytes from the most recent `image_url` (S3/GCS or `/images/<id>`) "
        "using the Python tool: `urllib.request.urlopen(url, timeout=30).read()`.\n"
        "2. Call the Google Drive MCP `create_file` tool with the bytes and a sensible filename "
        "(e.g. `nanobanana-<short-prompt-or-timestamp>.jpg`, MIME `image/jpeg`).\n"
        "3. Reply with the resulting Drive link so the user can open it.\n"
        "If the Google Drive MCP isn't installed, tell the user — do not try to fabricate a Drive URL."
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
# REST API — JSON endpoints for the web app (bypass MCP protocol)
# ---------------------------------------------------------------------------
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class _DummyCtx:
    """Minimal stand-in for mcp.Context — tool bodies never call ctx methods."""
    async def info(self, msg: str) -> None: pass  # noqa: E704
    async def report_progress(self, current: int, total: int) -> None: pass  # noqa: E704


_dummy_ctx = _DummyCtx()


def _api_json(tool_result) -> JSONResponse:
    """Convert a tool function's return value to a clean JSON API response."""
    if isinstance(tool_result, str):
        try:
            data = json.loads(tool_result)
        except json.JSONDecodeError:
            data = {"error": tool_result}
        status = 400 if "error" in data else 200
        return JSONResponse(data, status_code=status, headers=_CORS_HEADERS)
    # List: [Image(...), ..., render_md_str, json_str]
    json_str = tool_result[-1]
    data = json.loads(json_str)
    return JSONResponse(data, headers=_CORS_HEADERS)


@mcp.custom_route("/api/styles", methods=["GET", "OPTIONS"])
async def api_styles(request: Request) -> Response:
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=_CORS_HEADERS)
    return JSONResponse(sorted(STYLE_PRESETS.keys()), headers=_CORS_HEADERS)


@mcp.custom_route("/api/generate", methods=["POST", "OPTIONS"])
async def api_generate(request: Request) -> Response:
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=_CORS_HEADERS)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400, headers=_CORS_HEADERS)
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400, headers=_CORS_HEADERS)
    try:
        result = await generate_image(
            prompt=prompt,
            reference_images=body.get("reference_images"),
            style=body.get("style"),
            aspect_ratio=body.get("aspect_ratio", "4:5"),
            resolution=body.get("resolution", "1K"),
            model=body.get("model", "fast"),
            enhance_prompt=body.get("enhance_prompt", False),
            qa=body.get("qa", False),
            count=body.get("count", 1),
        )
        return _api_json(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS_HEADERS)


@mcp.custom_route("/api/edit", methods=["POST", "OPTIONS"])
async def api_edit(request: Request) -> Response:
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=_CORS_HEADERS)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400, headers=_CORS_HEADERS)
    prompt = body.get("prompt", "").strip()
    image = body.get("image", "").strip()
    if not prompt or not image:
        return JSONResponse({"error": "prompt and image are required"}, status_code=400, headers=_CORS_HEADERS)
    try:
        result = await edit_image(
            prompt=prompt,
            ctx=_dummy_ctx,
            image=image,
            reference_images=body.get("reference_images"),
            mask=body.get("mask"),
            edit_mode=body.get("edit_mode", "inpaint-insertion"),
            aspect_ratio=body.get("aspect_ratio"),
            count=body.get("count", 1),
        )
        return _api_json(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS_HEADERS)


@mcp.custom_route("/api/swap-background", methods=["POST", "OPTIONS"])
async def api_swap_background(request: Request) -> Response:
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=_CORS_HEADERS)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400, headers=_CORS_HEADERS)
    background = body.get("background", "").strip()
    image = body.get("image", "").strip()
    if not background or not image:
        return JSONResponse({"error": "background and image are required"}, status_code=400, headers=_CORS_HEADERS)
    try:
        result = await swap_background(
            background=background,
            ctx=_dummy_ctx,
            image=image,
            aspect_ratio=body.get("aspect_ratio"),
            count=body.get("count", 1),
        )
        return _api_json(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS_HEADERS)


@mcp.custom_route("/api/variations", methods=["POST", "OPTIONS"])
async def api_variations(request: Request) -> Response:
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=_CORS_HEADERS)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400, headers=_CORS_HEADERS)
    image = body.get("image", "").strip()
    if not image:
        return JSONResponse({"error": "image is required"}, status_code=400, headers=_CORS_HEADERS)
    try:
        result = await create_variations(
            ctx=_dummy_ctx,
            image=image,
            prompt=body.get("prompt"),
            variation_strength=body.get("variation_strength", "medium"),
            aspect_ratio=body.get("aspect_ratio"),
            resolution=body.get("resolution", "1K"),
            model=body.get("model", "fast"),
            qa=body.get("qa", False),
            count=body.get("count", 1),
        )
        return _api_json(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS_HEADERS)


# ---------------------------------------------------------------------------
# Web App — served at /app
# ---------------------------------------------------------------------------
_APP_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NanoBanana Studio</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f0f0f;color:#e8e8e8;min-height:100vh}
header{background:#1a1a1a;border-bottom:1px solid #333;padding:16px 24px;display:flex;align-items:center;gap:12px}
header h1{font-size:1.3em;font-weight:600;color:#f5a623}
header span{font-size:.85em;color:#888}
.container{max-width:960px;margin:0 auto;padding:24px}

/* Upload zone */
.upload-zone{border:2px dashed #444;border-radius:12px;padding:40px;text-align:center;cursor:pointer;
  transition:all .2s;margin-bottom:24px;background:#1a1a1a}
.upload-zone:hover,.upload-zone.drag-over{border-color:#f5a623;background:#1f1a10}
.upload-zone p{color:#888;margin-top:8px;font-size:.9em}
.upload-zone .icon{font-size:2em;margin-bottom:8px}
input[type="file"]{display:none}

/* Uploaded images strip */
.uploads-strip{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:24px;min-height:0}
.uploads-strip:empty{display:none}
.upload-thumb{position:relative;width:80px;height:80px;border-radius:8px;overflow:hidden;cursor:pointer;
  border:2px solid transparent;transition:border-color .2s;flex-shrink:0}
.upload-thumb.selected{border-color:#f5a623}
.upload-thumb img{width:100%;height:100%;object-fit:cover}
.upload-thumb .url-tag{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);
  font-size:.55em;padding:2px 4px;color:#aaa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* Tool tabs */
.tabs{display:flex;gap:2px;margin-bottom:20px;background:#1a1a1a;border-radius:10px;padding:4px;border:1px solid #333}
.tab{flex:1;padding:10px;text-align:center;border-radius:8px;cursor:pointer;font-size:.9em;
  color:#888;transition:all .2s;font-weight:500}
.tab:hover{color:#ccc}
.tab.active{background:#f5a623;color:#000;font-weight:600}

/* Tool forms */
.tool-form{background:#1a1a1a;border:1px solid #333;border-radius:12px;padding:20px;margin-bottom:24px}
.tool-form[hidden]{display:none}
.field{margin-bottom:14px}
.field label{display:block;font-size:.85em;color:#aaa;margin-bottom:4px;font-weight:500}
.field input,.field select,.field textarea{width:100%;padding:10px 12px;background:#0f0f0f;border:1px solid #444;
  border-radius:8px;color:#e8e8e8;font-size:.9em;font-family:inherit}
.field textarea{resize:vertical;min-height:70px}
.field input:focus,.field select:focus,.field textarea:focus{outline:none;border-color:#f5a623}
.field select{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center}
.row{display:flex;gap:12px}
.row .field{flex:1}
.selected-image{font-size:.8em;color:#f5a623;margin-bottom:10px;word-break:break-all}

.btn{display:inline-flex;align-items:center;gap:8px;padding:12px 28px;border:none;border-radius:8px;
  font-size:.95em;font-weight:600;cursor:pointer;transition:all .2s}
.btn-primary{background:#f5a623;color:#000}
.btn-primary:hover{background:#e09500}
.btn-primary:disabled{background:#555;color:#888;cursor:not-allowed}
.btn-sm{padding:8px 16px;font-size:.85em}
.btn-outline{background:transparent;border:1px solid #555;color:#aaa}
.btn-outline:hover{border-color:#f5a623;color:#f5a623}

/* Status / spinner */
.status{text-align:center;padding:30px;color:#888;font-size:.9em}
.spinner{display:inline-block;width:24px;height:24px;border:3px solid #333;border-top-color:#f5a623;
  border-radius:50%;animation:spin 1s linear infinite;margin-right:10px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}

/* Results */
.results{margin-top:8px}
.results h2{font-size:1.1em;color:#aaa;margin-bottom:16px;font-weight:500}
.result-card{background:#1a1a1a;border:1px solid #333;border-radius:12px;overflow:hidden;margin-bottom:16px}
.result-card img{width:100%;display:block;cursor:pointer}
.result-card img:hover{opacity:.92}
.result-actions{padding:12px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.result-actions a{text-decoration:none}
.result-url{font-size:.75em;color:#666;word-break:break-all;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.result-meta{font-size:.75em;color:#555;padding:0 16px 12px}

/* Lightbox */
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:100;display:flex;align-items:center;
  justify-content:center;cursor:pointer}
.lightbox[hidden]{display:none}
.lightbox img{max-width:95vw;max-height:95vh;border-radius:8px}
</style>
</head>
<body>
<header>
  <h1>NanoBanana Studio</h1>
  <span>Image generation &amp; editing</span>
</header>

<div class="container">
  <!-- Upload -->
  <div class="upload-zone" id="dropZone">
    <div class="icon">+</div>
    <strong>Drop images here or click to browse</strong>
    <p>Upload images to get URLs for editing, background swap, or variations</p>
  </div>
  <input type="file" id="fileInput" accept="image/*" multiple>
  <div class="uploads-strip" id="uploads"></div>

  <!-- Tabs -->
  <div class="tabs" id="tabs">
    <div class="tab active" data-tool="generate">Generate</div>
    <div class="tab" data-tool="edit">Edit</div>
    <div class="tab" data-tool="swap">Swap Background</div>
    <div class="tab" data-tool="variations">Variations</div>
  </div>

  <!-- Generate form -->
  <div class="tool-form" id="form-generate">
    <div class="field"><label>Prompt</label>
      <textarea id="gen-prompt" placeholder="Describe what to generate..."></textarea></div>
    <div class="row">
      <div class="field"><label>Style</label>
        <select id="gen-style"><option value="">None</option></select></div>
      <div class="field"><label>Aspect Ratio</label>
        <select id="gen-ratio">
          <option value="4:5" selected>4:5</option><option value="1:1">1:1</option>
          <option value="3:2">3:2</option><option value="2:3">2:3</option>
          <option value="16:9">16:9</option><option value="9:16">9:16</option>
          <option value="3:4">3:4</option><option value="4:3">4:3</option>
        </select></div>
      <div class="field"><label>Count</label>
        <select id="gen-count"><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
    </div>
    <div class="row">
      <div class="field"><label>Model</label>
        <select id="gen-model"><option value="fast">Fast</option><option value="pro">Pro (higher quality)</option></select></div>
      <div class="field"><label>Resolution</label>
        <select id="gen-res"><option>1K</option><option>2K</option><option>4K</option></select></div>
    </div>
    <button class="btn btn-primary" onclick="doGenerate()">Generate</button>
  </div>

  <!-- Edit form -->
  <div class="tool-form" id="form-edit" hidden>
    <div class="selected-image" id="edit-selected">Select an uploaded image above, or paste a URL:</div>
    <div class="field"><label>Image URL</label>
      <input id="edit-image" placeholder="https://... (or click an upload above)"></div>
    <div class="field"><label>Edit prompt</label>
      <textarea id="edit-prompt" placeholder="Describe what to change..."></textarea></div>
    <div class="row">
      <div class="field"><label>Edit Mode</label>
        <select id="edit-mode">
          <option value="inpaint-insertion">Inpaint (add/replace)</option>
          <option value="inpaint-removal">Remove object</option>
          <option value="outpaint">Outpaint (expand)</option>
        </select></div>
      <div class="field"><label>Count</label>
        <select id="edit-count"><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
    </div>
    <button class="btn btn-primary" onclick="doEdit()">Edit Image</button>
  </div>

  <!-- Swap background form -->
  <div class="tool-form" id="form-swap" hidden>
    <div class="selected-image" id="swap-selected">Select an uploaded image above, or paste a URL:</div>
    <div class="field"><label>Image URL</label>
      <input id="swap-image" placeholder="https://... (or click an upload above)"></div>
    <div class="field"><label>New background description</label>
      <textarea id="swap-bg" placeholder="Describe the new background..."></textarea></div>
    <div class="row">
      <div class="field"><label>Count</label>
        <select id="swap-count"><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
    </div>
    <button class="btn btn-primary" onclick="doSwap()">Swap Background</button>
  </div>

  <!-- Variations form -->
  <div class="tool-form" id="form-variations" hidden>
    <div class="selected-image" id="var-selected">Select an uploaded image above, or paste a URL:</div>
    <div class="field"><label>Image URL</label>
      <input id="var-image" placeholder="https://... (or click an upload above)"></div>
    <div class="field"><label>Guidance (optional)</label>
      <textarea id="var-prompt" placeholder="Optional direction for variations..."></textarea></div>
    <div class="row">
      <div class="field"><label>Variation Strength</label>
        <select id="var-strength">
          <option value="subtle">Subtle</option><option value="medium" selected>Medium</option>
          <option value="strong">Strong</option>
        </select></div>
      <div class="field"><label>Count</label>
        <select id="var-count"><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
    </div>
    <button class="btn btn-primary" onclick="doVariations()">Create Variations</button>
  </div>

  <!-- Status -->
  <div class="status" id="status" hidden></div>

  <!-- Results -->
  <div class="results" id="results"></div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" hidden onclick="this.hidden=true">
  <img id="lightbox-img" src="">
</div>

<script>
const API = window.location.origin;
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadsDiv = document.getElementById('uploads');
const statusDiv = document.getElementById('status');
const resultsDiv = document.getElementById('results');
let uploadedUrls = [];
let selectedUrl = '';

// -- Upload --
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); uploadFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', () => { uploadFiles(fileInput.files); fileInput.value = ''; });

async function uploadFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith('image/')) continue;
    const data = await file.arrayBuffer();
    showStatus('Uploading ' + file.name + '...');
    try {
      const resp = await fetch(API + '/upload', { method: 'POST', body: data });
      const json = await resp.json();
      if (json.url) {
        uploadedUrls.push(json.url);
        addUploadThumb(json.url);
      } else {
        showStatus('Upload failed: ' + (json.error || 'unknown'), true);
        return;
      }
    } catch (e) {
      showStatus('Upload failed: ' + e.message, true);
      return;
    }
  }
  hideStatus();
}

function addUploadThumb(url) {
  const div = document.createElement('div');
  div.className = 'upload-thumb';
  div.innerHTML = '<img src="' + url + '"><div class="url-tag">' + url.split('/').pop() + '</div>';
  div.onclick = () => selectUpload(url, div);
  uploadsDiv.appendChild(div);
}

function selectUpload(url, el) {
  selectedUrl = url;
  document.querySelectorAll('.upload-thumb').forEach(t => t.classList.remove('selected'));
  el.classList.add('selected');
  // Fill in URL fields for current tool
  ['edit-image','swap-image','var-image'].forEach(id => { document.getElementById(id).value = url; });
  document.querySelectorAll('.selected-image').forEach(s => { s.textContent = 'Selected: ' + url; });
}

// -- Tabs --
document.getElementById('tabs').addEventListener('click', e => {
  const tab = e.target.closest('.tab');
  if (!tab) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  tab.classList.add('active');
  document.querySelectorAll('.tool-form').forEach(f => f.hidden = true);
  document.getElementById('form-' + tab.dataset.tool).hidden = false;
});

// -- Styles --
fetch(API + '/api/styles').then(r => r.json()).then(styles => {
  const sel = document.getElementById('gen-style');
  styles.forEach(s => { const o = document.createElement('option'); o.value = s; o.textContent = s; sel.appendChild(o); });
});

// -- API calls --
function showStatus(msg, isError) {
  statusDiv.hidden = false;
  statusDiv.innerHTML = (isError ? '' : '<span class="spinner"></span>') +
    '<span style="color:' + (isError ? '#e57373' : '#888') + '">' + msg + '</span>';
}
function hideStatus() { statusDiv.hidden = true; }

function disableButtons(yes) {
  document.querySelectorAll('.btn-primary').forEach(b => b.disabled = yes);
}

async function callAPI(endpoint, body) {
  disableButtons(true);
  showStatus('Working... this may take 10\u201330 seconds');
  try {
    const resp = await fetch(API + endpoint, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const data = await resp.json();
    hideStatus();
    disableButtons(false);
    if (data.error) { showStatus('Error: ' + data.error, true); disableButtons(false); return null; }
    return data;
  } catch (e) {
    showStatus('Request failed: ' + e.message, true);
    disableButtons(false);
    return null;
  }
}

function showResults(data) {
  // data has image_url (single) or images (array)
  const images = data.images ? data.images : [data];
  const cards = images.map((img, i) => {
    const url = img.image_url || '';
    if (!url) return '';
    const sizeKb = img.size_kb ? img.size_kb + ' KB' : '';
    return '<div class="result-card">' +
      '<img src="' + url + '" onclick="openLightbox(\\'' + url + '\\')" alt="Result ' + (i+1) + '">' +
      '<div class="result-actions">' +
        '<a href="' + url + '" download class="btn btn-sm btn-primary">Download</a>' +
        '<button class="btn btn-sm btn-outline" onclick="copyUrl(\\'' + url + '\\')">Copy URL</button>' +
        '<button class="btn btn-sm btn-outline" onclick="useAsInput(\\'' + url + '\\')">Use as input</button>' +
        '<span class="result-url">' + url + '</span>' +
      '</div>' +
      (sizeKb ? '<div class="result-meta">' + sizeKb + '</div>' : '') +
    '</div>';
  }).join('');
  resultsDiv.innerHTML = '<h2>Results</h2>' + cards + resultsDiv.innerHTML.replace(/^<h2>Results<\\/h2>/, '');
}

async function doGenerate() {
  const data = await callAPI('/api/generate', {
    prompt: document.getElementById('gen-prompt').value,
    style: document.getElementById('gen-style').value || undefined,
    aspect_ratio: document.getElementById('gen-ratio').value,
    count: parseInt(document.getElementById('gen-count').value),
    model: document.getElementById('gen-model').value,
    resolution: document.getElementById('gen-res').value,
    reference_images: selectedUrl ? [selectedUrl] : undefined,
  });
  if (data) showResults(data);
}

async function doEdit() {
  const image = document.getElementById('edit-image').value.trim();
  if (!image) { showStatus('Select or paste an image URL first', true); return; }
  const data = await callAPI('/api/edit', {
    prompt: document.getElementById('edit-prompt').value,
    image: image,
    edit_mode: document.getElementById('edit-mode').value,
    count: parseInt(document.getElementById('edit-count').value),
  });
  if (data) showResults(data);
}

async function doSwap() {
  const image = document.getElementById('swap-image').value.trim();
  if (!image) { showStatus('Select or paste an image URL first', true); return; }
  const data = await callAPI('/api/swap-background', {
    image: image,
    background: document.getElementById('swap-bg').value,
    count: parseInt(document.getElementById('swap-count').value),
  });
  if (data) showResults(data);
}

async function doVariations() {
  const image = document.getElementById('var-image').value.trim();
  if (!image) { showStatus('Select or paste an image URL first', true); return; }
  const data = await callAPI('/api/variations', {
    image: image,
    prompt: document.getElementById('var-prompt').value || undefined,
    variation_strength: document.getElementById('var-strength').value,
    count: parseInt(document.getElementById('var-count').value),
  });
  if (data) showResults(data);
}

function openLightbox(url) {
  document.getElementById('lightbox-img').src = url;
  document.getElementById('lightbox').hidden = false;
}

function copyUrl(url) {
  navigator.clipboard.writeText(url).then(() => {
    const btn = event.target; const orig = btn.textContent; btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = orig, 1500);
  });
}

function useAsInput(url) {
  selectedUrl = url;
  ['edit-image','swap-image','var-image'].forEach(id => { document.getElementById(id).value = url; });
  document.querySelectorAll('.selected-image').forEach(s => { s.textContent = 'Selected: ' + url; });
}
</script>
</body></html>"""


@mcp.custom_route("/app", methods=["GET"])
async def http_app(request: Request) -> HTMLResponse:
    """Serve the NanoBanana Studio web app."""
    return HTMLResponse(_APP_HTML)


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
def _build_image_response(
    result: dict,
    generated: list[tuple[bytes, dict]],
    prefix: str = "gen",
) -> tuple[str, str, list[bytes]]:
    """Build a tool response.

    generated: list of (jpeg_bytes, per_image_metadata) tuples.

    Returns (render_md, json_str, thumbnails):
    - render_md: markdown image embed(s) + download link(s)
    - json_str: JSON metadata for tool chaining (image_url, size_kb, etc.)
    - thumbnails: 1024px JPEG bytes for ImageContent blocks (tool pane previews)
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
            except Exception as e:
                if REQUIRE_DURABLE_UPLOADS:
                    raise ValueError(
                        f"S3 upload failed and durable uploads are required: {e}"
                    )
                log(f"[nanobanana] S3 upload failed, falling back to local URL: {e}\n")
                meta["image_url"] = local_url
                meta["expires_in"] = "1 hour"
        else:
            meta["image_url"] = local_url
            meta["expires_in"] = "1 hour"
        # Build 1024px thumbnail for ImageContent (tool pane preview)
        from PIL import Image as PILImage
        pil = PILImage.open(BytesIO(jpeg_bytes))
        if max(pil.size) > 1024:
            pil.thumbnail((1024, 1024), PILImage.LANCZOS)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=85, optimize=True)
        thumbnails.append(buf.getvalue())

    save_nudge = (
        "**Save to Google Drive?** Reply *save* and I'll upload "
        "via the Google Drive MCP."
    )

    if len(generated) == 1:
        result.update(generated[0][1])
        result.pop("index", None)
        single_url = result.get("image_url", "")
        render_md = (
            f"![]({single_url})\n\n"
            f"[Download image]({single_url})\n\n"
            f"{save_nudge}"
        )
    else:
        result["images"] = [{k: v for k, v in meta.items() if k != "index"} for _, meta in generated]
        parts = []
        for i, img in enumerate(result["images"]):
            url = img.get("image_url", "")
            if url:
                parts.append(
                    f"![Image {i + 1}]({url})\n"
                    f"[Download image {i + 1}]({url})"
                )
        render_md = "\n\n".join(parts) + f"\n\n{save_nudge}"

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
def _urllib_snippet() -> str:
    """Upload snippet for **claude.ai web AND Claude Cowork** (Python tool).

    Auto-discovers the uploads folder by trying known paths in order:
      1. /mnt/user-data/uploads — claude.ai web sandbox
      2. ~/Library/Application Support/Claude/local-agent-mode-sessions/<session>/uploads
         — Claude Cowork (Mac); picks the most recently modified session.

    Falls back to a manual-upload message if neither is found. Has no access to
    arbitrary user filesystem paths — that's Claude Code's job (see
    `_claude_code_snippet`).
    """
    server_url = _BASE_URL
    return (
        "import urllib.request, json, os, time, glob\n"
        f"SERVER = '{server_url}'\n"
        "candidates = ['/mnt/user-data/uploads'] + sorted(\n"
        "    glob.glob(os.path.expanduser('~/Library/Application Support/Claude/local-agent-mode-sessions/*/uploads')),\n"
        "    key=os.path.getmtime, reverse=True)\n"
        "uploads = next((p for p in candidates if os.path.isdir(p)), None)\n"
        "if not uploads:\n"
        "    print(f'No uploads folder found. Use the upload/paperclip button in Cowork to attach the image as a FILE (pasted/dragged-inline images are NOT written to disk), or upload manually at {SERVER}/upload')\n"
        "else:\n"
        "    files = sorted([f for f in os.listdir(uploads) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))],\n"
        "                   key=lambda f: os.path.getmtime(os.path.join(uploads, f)), reverse=True)[:4]\n"
        "    if not files:\n"
        "        print(f'Uploads folder is empty: {uploads}. Pasted/dragged-inline images are NOT written to disk in Cowork — re-attach using the upload/paperclip button (which writes files to disk), or upload manually at {SERVER}/upload')\n"
        "    for i, fname in enumerate(files):\n"
        "        with open(os.path.join(uploads, fname), 'rb') as fh: data = fh.read()\n"
        "        req = urllib.request.Request(f'{SERVER}/upload', data=data, method='POST')\n"
        "        for attempt in range(3):\n"
        "            try:\n"
        "                result = json.loads(urllib.request.urlopen(req, timeout=30).read())\n"
        "                print(f'image{i}: {result[\"url\"]}')\n"
        "                break\n"
        "            except Exception as e:\n"
        "                if attempt < 2: time.sleep(2 ** attempt)\n"
        "                else: print(f'image{i}: FAILED after 3 attempts: {e}. Upload manually at {SERVER}/upload')\n"
    )


def _claude_code_snippet(path: str | None = None) -> str:
    """Upload snippet for **Claude Code** (Bash tool, real local filesystem).

    Claude Code has shell access and can read the user's actual files, so curl
    with --data-binary is the simplest path. When `path` is provided (e.g. the
    agent already tried `upload_image(image='/Users/me/img.jpg')` and the file
    doesn't exist on the remote server), it's substituted into the command;
    otherwise the placeholder /full/path/to/image.jpg is used.
    """
    server_url = _BASE_URL
    target = path if path else "/full/path/to/image.jpg"
    return (
        f'curl -fsS -X POST --data-binary "@{target}" \\\n'
        f'  "{server_url}/upload" \\\n'
        '  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get(\'url\') or d)"'
    )


def _upload_error(message: str, next_step: str | None = None) -> str:
    payload = {"error": message, "retryable": False}
    if next_step:
        payload["next_step"] = next_step
    return json.dumps(payload)


@mcp.tool()
async def upload_image(
    ctx: Context,
    image: str,
) -> str:
    """Re-host an image to a server URL for use in other tools.

    Accepts:
    - http/https URLs (including Google Drive share links — auto-rewritten)
    - local file paths — works only when this MCP runs locally as a stdio
      subprocess so the server can read the user's filesystem. For a remote
      MCP (Cloud Run), use the curl snippet below instead.

    DO NOT pass a data URI here. MCP transport hangs on large parameters
    regardless of size. Use one of the upload paths below instead.

    Pick the path for your environment:
    - **claude.ai web OR Claude Cowork:** Python tool + the auto-discovering urllib
      snippet from server instructions (handles both /mnt/user-data/uploads/ and
      ~/Library/Application Support/Claude/local-agent-mode-sessions/<session>/uploads/).
    - **Claude Code (CLI):** Bash tool + curl --data-binary "@/path" to /upload.
    - **Local stdio MCP:** call this tool with the file path directly:
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

    upload_endpoint = f"{_BASE_URL}/upload"
    webapp_url = f"{_BASE_URL}/app"
    is_data_uri = image.startswith("data:")
    is_local_file = (
        not is_data_uri
        and not _is_url(image)
        and os.path.isfile(image)
    )
    # Pass the user-provided path into the Claude Code snippet when present so
    # the agent can run the curl command verbatim instead of editing a placeholder.
    cc_path = image if (not is_data_uri and not _is_url(image)) else None
    if is_data_uri:
        return _upload_error(
            "upload_image does NOT accept data URIs. Do NOT re-encode at lower quality and retry.\n\n"
            "Pick the option for your environment:\n\n"
            "**A. claude.ai web OR Claude Cowork — run this with the Python tool (NOT bash):**\n\n"
            f"{_urllib_snippet()}\n"
            "**B. Claude Code (CLI) — run this with the Bash tool:**\n\n"
            f"{_claude_code_snippet(cc_path)}\n\n"
            f"**C. Manual upload:** {upload_endpoint}  •  **Full web app:** {webapp_url}",
            next_step=(
                "claude.ai web / Cowork → run snippet A with the Python tool. "
                "Claude Code → run snippet B with the Bash tool. "
                f"Or send the user to {upload_endpoint} (drag-drop) or {webapp_url} (full UI)."
            ),
        )
    if not _is_url(image) and not is_local_file:
        return _upload_error(
            "upload_image accepts http/https URLs or local file paths only. "
            "The path you provided does not exist on this server "
            "(this is normal when the MCP runs remotely — the server can't see your local filesystem).\n\n"
            "Pick the option for your environment:\n\n"
            "**A. claude.ai web OR Claude Cowork — run this with the Python tool (NOT bash):**\n\n"
            f"{_urllib_snippet()}\n"
            "**B. Claude Code (CLI) — run this with the Bash tool:**\n\n"
            f"{_claude_code_snippet(cc_path)}\n\n"
            f"**C. Manual upload:** {upload_endpoint}  •  **Full web app:** {webapp_url}",
            next_step=(
                "claude.ai web / Cowork → run snippet A with the Python tool. "
                "Claude Code → run snippet B with the Bash tool. "
                f"Or send the user to {upload_endpoint} (drag-drop) or {webapp_url} (full UI)."
            ),
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
        return _upload_error(str(e), next_step=f"Try /upload: {upload_endpoint}")
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
) -> list | str:
    """Generate an image from a text prompt with optional reference images and style presets.

    Reference images guide the model on style, subject appearance, or composition.
    Only pass URLs — use upload_image first if needed.

    INTAKE REQUIRED: Before calling this tool, confirm aspect_ratio and resolution
    with the user. Defaults: 4:5 / 1K. Ask both in a single message — see the
    "Intake before any image-output tool" section in server instructions. Skip
    when the user already named a ratio/resolution, when re-generating at known
    settings, or when chaining from a fixed-settings tool.

    Args:
        prompt: What to generate. Describe subject, style, lighting, mood, etc.
        reference_images: Optional list of image URLs. Use upload_image first if needed.
        style: Optional style preset. Available: cinematic, product-photography,
               editorial, watercolor, flat-illustration, neon-noir, minimalist, vintage-film.
        enhance_prompt: If true, AI expands your prompt into a detailed generation prompt.
        aspect_ratio: One of 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9.
                      Default 4:5. Confirm with the user before calling.
        resolution: 1K, 2K, or 4K. Default 1K. Confirm with the user before calling.
        quality: "default" (fast) or "pro" (higher quality). Default: default
        count: Number of images to generate (1–4). Default: 1
        qa: If true, AI-score each image. When count > 1, ranks by total score.

    Returns:
        Inline image previews, markdown download links, and JSON metadata with
        image_url for tool chaining.
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
        render_md, json_result, thumbnails = await _run_in_thread(_build_image_response, result, generated, prefix="gen")
    except Exception as e:
        return json.dumps({"error": f"Failed to store/upload generated images: {e}"})
    return [Image(data=t, format="jpeg") for t in thumbnails] + [render_md, json_result]


@mcp.tool(structured_output=False)
async def edit_image(
    prompt: str,
    ctx: Context,
    image: Annotated[str, Field(max_length=2048)],
    reference_images: list[str] | None = None,
    mask: str | None = None,
    edit_mode: str = "inpaint-insertion",
    aspect_ratio: str | None = None,
    count: int = 1,
) -> list | str:
    """Edit an existing image — add objects, remove objects, or extend the canvas.

    Only pass URLs — use upload_image first if needed.

    INTAKE REQUIRED: Before calling this tool, confirm aspect_ratio with the user
    (default = same shape as source). Resolution is NOT configurable for edits —
    the output matches the source. See "Intake before any image-output tool" in
    server instructions. Skip when the user already specified, when re-running at
    known settings, or when chaining from a fixed-settings tool.

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
        aspect_ratio: One of 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9.
                      Default = same as source. Confirm with the user (especially
                      for outpaint, where reshaping is the whole point).
        count: Number of candidates (1–4). Default: 1

    Returns:
        Inline image previews, markdown download links, and JSON metadata with
        image_url for tool chaining.
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
        render_md, json_result, thumbnails = await _run_in_thread(_build_image_response, result, generated, prefix="edit")
    except Exception as e:
        return json.dumps({"error": f"Failed to store/upload edited images: {e}"})
    return [Image(data=t, format="jpeg") for t in thumbnails] + [render_md, json_result]


@mcp.tool(structured_output=False)
async def swap_background(
    background: str,
    ctx: Context,
    image: Annotated[str, Field(max_length=2048)],
    aspect_ratio: str | None = None,
    count: int = 1,
) -> list | str:
    """Replace the background of an image while keeping the foreground subject intact.

    Only pass URLs — use upload_image first if needed.

    INTAKE REQUIRED: Before calling this tool, confirm aspect_ratio with the user
    (default = same shape as source). Resolution is NOT configurable for swaps —
    the output matches the source. See "Intake before any image-output tool" in
    server instructions. Skip when the user already specified, when re-running at
    known settings, or when chaining from a fixed-settings tool.

    Args:
        image: Source image URL. Use upload_image first if needed.
        background: Description of the new background. Be specific.
        aspect_ratio: One of 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9.
                      Default = same as source. Confirm with the user.
        count: Number of candidates (1–4). Default: 1

    Returns:
        Inline image previews, markdown download links, and JSON metadata with
        image_url for tool chaining.
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
        render_md, json_result, thumbnails = await _run_in_thread(_build_image_response, result, generated, prefix="bgswap")
    except Exception as e:
        return json.dumps({"error": f"Failed to store/upload background-swapped images: {e}"})
    return [Image(data=t, format="jpeg") for t in thumbnails] + [render_md, json_result]


@mcp.tool(structured_output=False)
async def create_variations(
    ctx: Context,
    image: Annotated[str, Field(max_length=2048)],
    prompt: str | None = None,
    variation_strength: str = "medium",
    aspect_ratio: str | None = None,
    resolution: str = "1K",
    quality: str = "default",
    count: int = 3,
    qa: bool = False,
) -> list | str:
    """Generate variations of an existing image.

    Preserves the core subject while exploring different compositions, lighting,
    or styling. Only pass URLs — use upload_image first if needed.

    INTAKE REQUIRED: Before calling this tool, confirm aspect_ratio and resolution
    with the user. Defaults: aspect_ratio = same as source, resolution = 1K. See
    "Intake before any image-output tool" in server instructions. Skip when the
    user already specified, when re-running at known settings, or when chaining
    from a fixed-settings tool.

    Args:
        image: Source image URL. Use upload_image first if needed.
        prompt: Optional guidance for variations.
        variation_strength: "subtle", "medium", or "strong". Default: medium
        aspect_ratio: One of 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9.
                      Default = same as source. Confirm with the user.
        resolution: 1K, 2K, or 4K. Default 1K. Confirm with the user.
        quality: "default" or "pro". Default: default
        count: Number of variations (1–4). Default: 3
        qa: Score each variation and rank by quality. Default: false

    Returns:
        Inline image previews, markdown download links, and JSON metadata with
        image_url for tool chaining.
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
    except ValueError as e:
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
        render_md, json_result, thumbnails = await _run_in_thread(_build_image_response, result, generated, prefix="var")
    except Exception as e:
        return json.dumps({"error": f"Failed to store/upload variation images: {e}"})

    return [Image(data=t, format="jpeg") for t in thumbnails] + [render_md, json_result]


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
    image: Annotated[str, Field(max_length=2048)],
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
        "Pasted images are saved to `/mnt/user-data/uploads/` (claude.ai web) or "
        "`~/Library/Application Support/Claude/local-agent-mode-sessions/<session>/uploads/` (Cowork). "
        "The urllib POST snippet from server instructions auto-discovers either location and uploads "
        "the raw bytes to /upload to get back a URL. NEVER encode to base64/data URI — it hangs the transport.\n\n"
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
