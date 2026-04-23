---
name: mcp-s3-companion
description: Handles user image uploads for NanoBanana MCP image tools. Use this skill whenever a user shares, pastes, drags, or uploads an image and wants to edit it, swap its background, create variations, analyze it, or use it as a reference — any workflow involving NanoBanana tools and a user-provided image. Also trigger this when a user asks "why can't you see my image" or a tool call fails with an invalid/missing URL.
---

# MCP S3 Companion

## Step 1 — Run this code RIGHT NOW. No size checks. No assessment. Just run it.

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

This uploads all recently pasted images and prints their URLs. **Do not skip this step. Do not check image sizes first. Do not encode to base64. Just run it.**

The server handles any image size. The returned URLs are S3 links — pass them directly to NanoBanana tools.

If urllib raises an error, direct the user to upload manually at: `https://nanobanana-739905005785.us-central1.run.app/upload`

---

## Step 2 — Pass the URLs to the right tool

| Tool | What it does | Key params |
|---|---|---|
| `edit_image` | Inpaint, remove objects, outpaint | `image` = source URL; `prompt` = what to change |
| `swap_background` | Keep subject, replace background | `image` = source URL; `prompt` = new background |
| `create_variations` | Style/composition variations | `image` = source URL |
| `analyze_image` | Describe, tag, assess quality | `image` = source URL |
| `generate_image` | Text-to-image with references | `reference_images` = list of URLs |

---

## Other image sources (no upload needed)

| Source | What to do |
|---|---|
| `http://` or `https://` URL | Pass directly to the tool |
| Public Google Drive link | Pass directly — server rewrites it automatically |
| S3 / CDN URL | Pass directly |
| Claude Code local file | `upload_image(image='/full/path/to/file.jpg')` |

---

## What not to do

- **Never encode images to base64 or data URI** — causes MCP transport to hang, and upload_image rejects them
- Don't use curl or wget — blocked in the claude.ai sandbox
- Don't check image sizes before uploading — the server handles any size
- Don't start a local HTTP server
