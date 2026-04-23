---
name: mcp-s3-companion
description: Handles user image uploads for NanoBanana MCP image tools. Use this skill whenever a user shares, pastes, drags, or uploads an image and wants to edit it, swap its background, create variations, analyze it, or use it as a reference — any workflow involving NanoBanana tools and a user-provided image. Also trigger this when a user asks "why can't you see my image" or a tool call fails with an invalid/missing URL.
---

# MCP S3 Companion

## FIRST: How to upload a pasted/dropped image

**When a user pastes or uploads an image, do this immediately — before anything else:**

```python
import urllib.request, json, os
uploads = '/mnt/user-data/uploads'
files = sorted([f for f in os.listdir(uploads) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))],
               key=lambda f: os.path.getmtime(os.path.join(uploads, f)), reverse=True)

# Upload most recent file (files[0]). For 2nd image use files[1], etc.
with open(os.path.join(uploads, files[0]), 'rb') as f: data = f.read()
req = urllib.request.Request('[SERVER_URL]/upload', data=data, method='POST')
result = json.loads(urllib.request.urlopen(req, timeout=30).read())
print('url:', result['url'])
```

Replace `[SERVER_URL]` with the base URL from NanoBanana instructions (e.g. `https://nanobanana-....run.app`).

The returned `url` is an S3 link — pass it directly to any NanoBanana tool. **Do not encode to base64. Do not use curl. Do not use upload_image with image bytes.**

For multiple images, run the snippet once per image using `files[0]`, `files[1]`, etc.

---

## Other image sources

| Source | What to do |
|---|---|
| `http://` or `https://` URL | Pass directly to the tool — no upload needed |
| Public Google Drive link | Pass directly — server rewrites it automatically |
| S3 / CDN URL | Pass directly |
| Claude Code local file | `upload_image(image='/full/path/to/file.jpg')` |
| urllib fails / no uploads dir | Direct user to `[SERVER_URL]/upload` to drag-and-drop |

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

- **Never encode images to base64/data URI to pass through MCP parameters** — even small URIs cause the tool call to hang or fail.
- Don't use curl or wget — they are blocked in the claude.ai sandbox.
- Don't start a local HTTP server to serve the image.
- Don't fabricate or guess S3/GCS/CDN URLs — if you don't have a real URL, use one of the paths above.
