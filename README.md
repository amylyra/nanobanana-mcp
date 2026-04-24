# NanoBanana MCP

A production MCP server for image generation, editing, and analysis in Claude, powered by Gemini.

**Live:** `https://nanobanana-739905005785.us-central1.run.app`

---

## What works ✓

| Feature | Status |
|---|---|
| Generate images from text | ✓ Working |
| Edit images (inpaint, outpaint, remove objects) | ✓ Working |
| Swap backgrounds | ✓ Working |
| Create variations | ✓ Working |
| Analyze / compare images | ✓ Working |
| Upload pasted images (urllib POST flow) | ✓ Working |
| Upload via drag-drop at `/upload` | ✓ Working |
| Image thumbnails in tool pane | ✓ Working |
| S3 durable image storage | ✓ Working |

## Known limitations ✗

| Issue | Detail |
|---|---|
| Claude chat inline images | claude.ai shows **"Show Image"** click-to-reveal boxes instead of true inline rendering. Tool pane previews are reliable; chat reply images require a click. This is a claude.ai client-side consent gate — not fixable from the server. |
| Python snippet run as bash | When Claude runs the urllib upload snippet via the bash tool (instead of the Python code tool), it fails with an ImageMagick error. Claude recovers by retrying as `python3` — two steps instead of one. |

---

## How image input works

All tool parameters accept `http://https://` URLs only. Three paths to get images in:

### Path 1 — You already have a URL
Pass it directly to any tool. Google Drive share links are auto-rewritten.

### Path 2 — Pasted image in claude.ai web (primary path)

Claude runs this Python snippet automatically using the Python code tool:

```python
import urllib.request, json, os

SERVER = 'https://nanobanana-739905005785.us-central1.run.app'
uploads = '/mnt/user-data/uploads'

files = sorted(
    [f for f in os.listdir(uploads) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))],
    key=lambda f: os.path.getmtime(os.path.join(uploads, f)), reverse=True
)[:4]

for i, fname in enumerate(files):
    with open(os.path.join(uploads, fname), 'rb') as f:
        data = f.read()
    req = urllib.request.Request(f'{SERVER}/upload', data=data, method='POST')
    result = json.loads(urllib.request.urlopen(req, timeout=30).read())
    print(f'image{i}: {result["url"]}')
```

This POSTs raw bytes to `/upload`, which normalizes via PIL, uploads to S3, and returns a durable URL.

**Never encode to base64/data URI.** Passing large data URIs as MCP parameters hangs the transport layer before the server can reject them. The server's `Field(pattern=r"^(https?://|/)")` constraint blocks data URIs client-side.

### Path 3 — Local file (Claude Code)

```python
upload_image(image='/full/path/to/file.jpg')
```

### Path 4 — Manual upload page

Open `https://nanobanana-739905005785.us-central1.run.app/upload`, drag-and-drop, paste returned URL.

---

## Why Claude sometimes needs two tries to upload

When a pasted image is in `/mnt/user-data/uploads/`:

1. Claude calls `upload_image(image="data:image/...")` → pattern blocks it (Pydantic, cryptic error)
2. Claude tries `upload_image(image="/mnt/user-data/uploads/file.jpg")` → pattern accepts, but path doesn't exist on Cloud Run → error response includes **full urllib snippet inline**
3. Claude runs the urllib snippet → `/upload` receives raw bytes → S3 URL returned → success

The "not a local file" error now includes the urllib snippet directly so Claude doesn't have to search server instructions for it.

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

Image tools return three things in every response:

- `chat_response_template` — markdown image links for Claude's reply (`![](https://...)`)
- `json_str` — structured metadata (`image_url`, `size_kb`, `width`, `height`) for tool chaining
- `ImageContent` — inline thumbnail previews for the tool pane

To chain a generated image into another tool, use `image_url` from the JSON section.

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

Endpoints: `http://localhost:8080/mcp` (MCP), `http://localhost:8080/upload` (upload page)

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

212 tests. Run before every deploy.

---

## Key learnings about claude.ai + MCP

- **Data URI transport hang**: Large JSON bodies in MCP HTTP POST requests cause the claude.ai transport to hang indefinitely. The server never sees the request — it dies in transit. Fix: `Field(pattern=r"^(https?://|/)")` rejects data URIs client-side (Pydantic) before the body is sent.
- **Error-driven recovery is more reliable than pre-flight instructions**: Claude ignores upfront instructions but reliably reads tool error responses. Putting the urllib snippet in the error message that Claude actually receives is what makes the flow work.
- **Python code tool vs bash**: Claude sometimes picks the bash tool to run Python snippets. Bash interprets `import` as the ImageMagick `import` binary. Claude recovers by retrying as `python3`. No fix yet — acceptable two-step behavior.
- **`/mnt/user-data/uploads/`**: Where claude.ai stores pasted/dropped files, accessible via the Python code tool.
- **`min-instances=1`**: Required for Cloud Run — cold starts cause 503 on the urllib upload path.

---

## License

MIT
