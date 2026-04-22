# NanoBanana MCP

A production-focused MCP server for image generation, editing, and analysis in Claude, powered by Gemini image models.

> **Current status (April 22, 2026):**
> - Image upload can appear "stuck" in Claude when users try unsupported upload flows.
> - Images can render in the tool result pane reliably, but Claude chat replies may still show a **"Show Image"** consent gate.
>
> See [Troubleshooting](#troubleshooting-upload--display-issues) and [Optimization roadmap](OPTIMIZATION_PLAN.md).

## What this server does

- Generate images (`generate_image`)
- Edit existing images (`edit_image`)
- Replace backgrounds (`swap_background`)
- Create variations (`create_variations`)
- Analyze images (`analyze_image`, `batch_analyze`, `compare_images`)
- Re-host user images as tool-ready URLs (`upload_image`)

## How image input works (important)

All image parameters are text-only. Use either:

1. `http://` or `https://` URL
2. Data URI (`data:image/...;base64,...`) passed to `upload_image`

### Recommended upload flows

#### Flow A — already have an image URL
Pass URL directly to the tool.

#### Flow B — pasted/local file in Claude.ai
Use Python to build a data URI, then call `upload_image(image=uri)`.
This should be attempted automatically first; only fall back to manual `/upload` when automatic upload fails.

```python
import os, base64
from io import BytesIO
from PIL import Image

uploads = '/mnt/user-data/uploads'
files = sorted(
    [f for f in os.listdir(uploads) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))],
    key=lambda f: os.path.getmtime(os.path.join(uploads, f)),
    reverse=True,
)
img = Image.open(os.path.join(uploads, files[0]))
if max(img.size) > 1024:
    img.thumbnail((1024, 1024), Image.LANCZOS)
img = img.convert('RGB')

buf = BytesIO()
img.save(buf, format='JPEG', quality=70, optimize=True)
uri = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
print('uri_chars=', len(uri), uri[:80] + '...')
```

Then call:

```json
{"image": "data:image/jpeg;base64,..."}
```

If `uri_chars` is large (for example >300k), use `{PUBLIC_URL}/upload` instead of passing a huge data URI through tool parameters.  
Server-side guard: `MAX_DATA_URI_CHARS` (default `300000`).

#### Flow C — manual upload page
Open:

```
{PUBLIC_URL}/upload
```

Drag-and-drop an image and paste returned URL into tool calls.

## Output behavior in Claude

For image generation tools, the server returns:

```text
[render_md, json_str, ImageContent thumbnails...]
```

- `render_md`: markdown links (`![](https://...)`) meant for Claude reply reuse
- `json_str`: structured metadata for tool chaining (`image_url`, `size_kb`, etc.)
- `ImageContent`: inline previews in tool pane

### Important limitation

Even when Claude includes markdown image links in chat replies, claude.ai may show **"Show Image"** boxes (click-to-reveal) instead of immediate inline display. This is a client-side behavior. Tool pane previews are currently the most reliable inline rendering path.

## Setup

### Prerequisites

- Python 3.10+
- Google AI key (`GEMINI_API_KEY` or `GOOGLE_AI_API_KEY`)

### Local run

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
python server.py
```

Optional `.env`:

```bash
echo 'GEMINI_API_KEY=your-key' > .env
python server.py
```

Default local URLs:

- Base: `http://localhost:8080`
- MCP endpoint: `/mcp`
- Upload page: `/upload`

### Cloud Run deployment

```bash
gcloud run deploy nanobanana \
  --source . \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY="your-key",PUBLIC_URL="https://YOUR-SERVICE.run.app" \
  --allow-unauthenticated
```

`PUBLIC_URL` must match your externally reachable service URL.

### Connect to Claude

Add remote connector:

```
https://YOUR-SERVICE.run.app/mcp
```

## Cloud storage

If `S3_BUCKET` or `GCS_BUCKET` is configured, generated images are uploaded to cloud storage and `image_url` is durable.

Without cloud storage, images are served from in-memory `/images/{id}` store and expire after 1 hour (`STORE_TTL`).

If you want `/upload` and `upload_image` to use cloud storage only (never in-memory fallback), set:

```bash
REQUIRE_DURABLE_UPLOADS=true
```

With this enabled, upload requests return an error when cloud storage is unavailable.
When cloud storage is configured, durable mode defaults to enabled unless you explicitly set `REQUIRE_DURABLE_UPLOADS=false`.

### S3

```bash
pip install boto3>=1.34.0
```

Env vars:

```bash
S3_BUCKET=my-nanobanana-images
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

### GCS

```bash
pip install google-cloud-storage>=2.0.0
```

```bash
gcloud run services update nanobanana \
  --set-env-vars GCS_BUCKET="my-nanobanana-images"
```

If both are set, S3 is preferred.

## Tools

- `upload_image(image)`
- `generate_image(prompt, reference_images?, style?, enhance_prompt?, aspect_ratio?, resolution?, quality?, count?, qa?, save_folder?)`
- `edit_image(image, prompt, reference_images?, mask?, edit_mode?, aspect_ratio?, count?, save_folder?)`
- `swap_background(image, background, aspect_ratio?, count?, save_folder?)`
- `create_variations(image, prompt?, variation_strength?, aspect_ratio?, resolution?, quality?, count?, qa?, save_folder?)`
- `analyze_image(image, focus?)`
- `batch_analyze(images, focus?)`
- `compare_images(images, focus?)`
- `list_styles()`

## Troubleshooting upload & display issues

See full runbook: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

Quick checks:

1. Ensure `PUBLIC_URL` is set to your real Cloud Run URL.
2. Verify `/upload` works in browser and returns JSON URL.
3. Prefer `upload_image` data URI flow for pasted/local files.
4. Validate returned `image_url` is publicly reachable.
5. Expect possible "Show Image" in Claude chat response; tool pane previews should still render.

## Optimization roadmap

See [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) for a prioritized cleanup and reliability plan.

## Testing

```bash
python -m pytest test_simulation.py -x -q
```

## License

MIT
