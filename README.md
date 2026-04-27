# NanoBanana MCP

A production MCP server for image generation, editing, and analysis in Claude, powered by Gemini.

**Live:** `https://nanobanana-739905005785.us-central1.run.app`

---

## What works

| Feature | Status |
|---|---|
| Generate images from text | Working |
| Edit images (inpaint, outpaint, remove objects) | Working |
| Swap backgrounds | Working |
| Create variations | Working |
| Analyze / compare images | Working |
| Upload pasted images (urllib POST flow) | Working |
| Upload via drag-drop at `/upload` | Working |
| Image thumbnails in tool pane (1024px) | Working |
| S3 durable image storage | Working |
| Web app at `/app` (no MCP needed) | Working |
| REST API (`/api/generate`, `/api/edit`, etc.) | Working |

## Known limitations

| Issue | Detail |
|---|---|
| Claude chat inline images | claude.ai shows "Show Image" click-to-reveal boxes instead of true inline rendering. Tool pane previews are reliable; chat reply images require a click. This is a claude.ai client-side consent gate — not fixable from the server. |
| Python snippet run as bash | When Claude runs the urllib upload snippet via the bash tool (instead of the Python code tool), it fails with `/bin/sh: Bad for loop variable`. Server instructions explicitly say "use the Python tool, NOT bash" and explain the symptom. Claude usually recovers on second try. |
| Sandbox 503s | claude.ai's sandbox sometimes cannot reach Cloud Run (transient). The urllib snippet retries 3 times with exponential backoff. If all fail, the user is directed to `/upload` or `/app`. |

---

## How image input works

All tool parameters accept `http/https` URLs only. Paths to get images in:

### Path 1 — You already have a URL
Pass it directly to any tool. Google Drive share links are auto-rewritten.

### Path 2 — Pasted image in claude.ai web (primary path)

Claude runs this Python snippet using the Python code tool (not bash):

```python
import urllib.request, json, os, time

SERVER = 'https://nanobanana-739905005785.us-central1.run.app'
uploads = '/mnt/user-data/uploads'

files = sorted(
    [f for f in os.listdir(uploads) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))],
    key=lambda f: os.path.getmtime(os.path.join(uploads, f)), reverse=True
)[:4]

for i, fname in enumerate(files):
    with open(os.path.join(uploads, fname), 'rb') as fh:
        data = fh.read()
    req = urllib.request.Request(f'{SERVER}/upload', data=data, method='POST')
    for attempt in range(3):
        try:
            result = json.loads(urllib.request.urlopen(req, timeout=30).read())
            print(f'image{i}: {result["url"]}')
            break
        except Exception as e:
            if attempt < 2: time.sleep(2 ** attempt)
            else: print(f'image{i}: FAILED after 3 attempts: {e}. Upload manually at {SERVER}/upload')
```

This POSTs raw bytes to `/upload`, which normalizes via PIL, uploads to S3, and returns a durable URL. The snippet retries up to 3 times with exponential backoff for transient 503s.

**Never encode to base64/data URI.** Passing large data URIs as MCP parameters hangs the transport layer before the server can reject them. `upload_image` has no Pydantic constraints so its error handler can fire and return the urllib snippet inline.

### Path 3 — Claude Code (CLI) with the remote MCP

The remote Cloud Run server can't read your local filesystem, so Claude Code uploads the file via curl using the Bash tool:

```bash
curl -fsS -X POST --data-binary "@/full/path/to/image.jpg" \
  "https://nanobanana-739905005785.us-central1.run.app/upload" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('url') or d)"
```

The returned URL goes straight into any other tool. `upload_image` returns this snippet inline whenever a path doesn't resolve on the server, so the agent can copy-run it.

### Path 4 — Local stdio MCP (any client)

When the MCP runs locally as a stdio subprocess (not Cloud Run), the server *can* read the user's filesystem, so the direct path works:

```python
upload_image(image='/full/path/to/file.jpg')
```

### Path 5 — Manual upload page

Open `https://nanobanana-739905005785.us-central1.run.app/upload`, drag-and-drop, paste returned URL.

### Path 6 — Web app (no MCP needed)

Open `https://nanobanana-739905005785.us-central1.run.app/app` for a full UI with upload, generate, edit, swap background, variations, and inline results with download buttons.

---

## Why Claude sometimes needs two tries to upload

When a pasted image is in `/mnt/user-data/uploads/`:

1. Claude calls `upload_image(image="data:image/...")` → function body detects data URI, returns error with urllib snippet inline
2. Claude runs the urllib snippet → `/upload` receives raw bytes → S3 URL returned → success

The data URI error response includes the full urllib snippet directly so Claude doesn't have to search server instructions for it. The `upload_image` parameter has no Pydantic constraints (no `pattern`, no `max_length`) specifically so the error handler can fire and guide recovery.

---

## Tools

| Tool | Description |
|---|---|
| `upload_image(image)` | Re-host any image to a server URL for use in other tools |
| `generate_image(prompt, ...)` | Text-to-image with optional reference images and style presets |
| `edit_image(image, prompt, ...)` | Inpaint, remove objects, outpaint; accepts multiple reference images |
| `swap_background(image, background, ...)` | Keep subject, replace background |
| `create_variations(image, ...)` | Generate style/composition variations |
| `analyze_image(image, focus?)` | Describe, tag, or assess a single image |
| `batch_analyze(images, focus?)` | Analyze 2–20 images in parallel |
| `compare_images(images, focus?)` | Compare 2–10 images (differences, quality, style) |
| `list_styles()` | List available style presets |

### Common patterns

```
# Analyze multiple images — use batch_analyze, not repeated analyze_image
# Compare two images — compare_images(focus='differences')
# Pick best of N — compare_images(focus='quality')
# Put object from image B into image A:
  edit_image(image=urlA, reference_images=[urlB], prompt='replace X with object from reference image 1')
```

---

## Output format

Image tools return content blocks in this order:

1. `ImageContent` — 1024px JPEG thumbnails visible in the tool pane
2. `render_md` — markdown image embeds (`![](url)`) + `[Download image](url)` links
3. `json_str` — JSON metadata (`image_url`, `size_kb`) for tool chaining

To chain a generated image into another tool, use `image_url` from the JSON metadata.

---

## REST API

The server also exposes JSON REST endpoints (with CORS) for the web app and programmatic use:

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload raw image bytes, get back a URL |
| `/api/generate` | POST | Generate images from a prompt |
| `/api/edit` | POST | Edit an image with a prompt |
| `/api/swap-background` | POST | Swap an image's background |
| `/api/variations` | POST | Create variations of an image |
| `/api/styles` | GET | List available style presets |
| `/app` | GET | Full web UI (NanoBanana Studio) |

---

## Setup

### Prerequisites

- Python 3.10+
- `GEMINI_API_KEY` (or `GOOGLE_AI_API_KEY`)

### Local run

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
python server.py
```

Endpoints: `http://localhost:8080/mcp` (MCP), `http://localhost:8080/upload` (upload page), `http://localhost:8080/app` (web UI)

### Cloud Run deployment

```bash
gcloud run deploy nanobanana \
  --source . \
  --region us-central1 \
  --min-instances=1 \
  --set-env-vars GEMINI_API_KEY="your-key",PUBLIC_URL="https://YOUR-SERVICE.run.app" \
  --allow-unauthenticated
```

`--min-instances=1` is required — without it, cold-start 503s break the urllib upload path.

### Connect to Claude

Add remote MCP connector: `https://YOUR-SERVICE.run.app/mcp`

---

## Cloud storage (S3 / GCS)

With `S3_BUCKET` or `GCS_BUCKET` set, all images are uploaded to cloud storage and URLs are durable (no expiry). Without cloud storage, images expire after 1 hour (`STORE_TTL`).

To require cloud storage (never fall back to in-memory):
```bash
REQUIRE_DURABLE_UPLOADS=true
```

**S3:** `pip install boto3>=1.34.0`, set `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`

**GCS:** `pip install google-cloud-storage>=2.0.0`, set `GCS_BUCKET`

---

## Testing

```bash
python -m pytest test_simulation.py -x -q
```

225 tests. Run before every deploy.

---

## License

MIT
