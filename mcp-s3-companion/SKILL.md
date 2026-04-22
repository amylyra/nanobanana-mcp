---
name: mcp-s3-companion
description: Handles user image uploads for NanoBanana MCP image tools. Use this skill whenever a user shares, pastes, drags, or uploads an image and wants to edit it, swap its background, create variations, analyze it, or use it as a reference — any workflow involving NanoBanana tools and a user-provided image. Also trigger this when a user asks "why can't you see my image" or a tool call fails with an invalid/missing URL.
---

# MCP S3 Companion

## How to get an image into NanoBanana

MCP tool parameters are plain text — image bytes cannot be passed directly. Every image needs to arrive as a URL. There are five paths depending on where the image comes from:

---

### Path 1 — User gives a direct URL

Any `http://` or `https://` URL that points directly at an image file (including S3 URLs, CDN links, etc.) — pass it straight to the tool. No upload needed.

---

### Path 2 — User shares a Google Drive link

The NanoBanana server automatically rewrites public Google Drive share links to direct download URLs, so this just works for **publicly shared** files:

```
https://drive.google.com/file/d/FILE_ID/view  →  server fetches it automatically
```

For **private Drive files** (the user owns them but they aren't publicly shared):

1. Use the Google Drive MCP to read the file. It returns the image as a base64 blob along with a MIME type.
2. Construct a data URI using the returned MIME type: `data:<mime_type>;base64,<blob>`
3. Pass the data URI to `upload_image` — the server decodes, normalizes, and stores it to S3.

> Note: this works for small-to-medium images (roughly under 1MB). For large files, the data URI exceeds MCP parameter limits — ask the user to share the file publicly (Path 1) or use Path 4.

---

### Path 3 — User already has an S3 or cloud storage URL

Pass it directly to the tool — same as Path 1. The server fetches any `https://` URL.

---

### Path 4 — User pasted/uploaded image (preferred: Python data URI + `upload_image`)

When a user drops/pastes an image and does not have a URL, use Python to convert the image into a data URI, then call `upload_image(image=uri)`.

```python
import os, base64
from io import BytesIO
from PIL import Image

uploads = '/mnt/user-data/uploads'
files = sorted([f for f in os.listdir(uploads) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))],
               key=lambda f: os.path.getmtime(os.path.join(uploads, f)), reverse=True)
img = Image.open(os.path.join(uploads, files[0]))
if max(img.size) > 1024:
    img.thumbnail((1024, 1024), Image.LANCZOS)
img = img.convert('RGB')
buf = BytesIO()
img.save(buf, format='JPEG', quality=70, optimize=True)
uri = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
print('uri_chars=', len(uri), uri[:80] + '...')
```

Then pass `uri` to `upload_image`. Use returned `url` in downstream tools.

If `uri_chars` is large (for example >300k), do **not** pass it through MCP tool parameters.  
Use `[SERVER_URL]/upload` and paste the returned URL instead (especially when uploading 2+ images).

---

### Path 5 — No URL and Python path unavailable

Direct the user to the upload page:

> I can see your image. To use it with NanoBanana tools, please upload it at **[SERVER_URL]/upload** and paste back the returned URL.

`[SERVER_URL]` is the base URL from NanoBanana instructions.

---

## After you have a URL

Pass it to whichever NanoBanana tool the user asked for:

| Tool | What it does | Key params |
|---|---|---|
| `edit_image` | Inpaint, remove objects, outpaint | `image` = source; `prompt` = what to change |
| `swap_background` | Keep subject, replace background | `image` = source; `prompt` = new background |
| `create_variations` | Style/composition variations | `image` = source |
| `analyze_image` | Describe, tag, assess quality | `image` = source |
| `generate_image` | Text-to-image with references | `reference_images` = list of URLs |

Default aspect ratio 4:5, resolution 1K.

---

## What not to do

- Don't rely on curl/wget upload flows in restricted sandboxes; prefer Python data URI -> `upload_image` or the `/upload` page.
- Don't start a local HTTP server to serve the image.
- Don't fabricate or guess S3/GCS/CDN URLs — if you don't have a real URL, use one of the paths above.
- Don't pass API endpoints, web page URLs, or MCP service URLs as image parameters.
