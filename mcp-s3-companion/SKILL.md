---
name: mcp-s3-companion
description: Handles user image uploads for NanoBanana MCP image tools. Use this skill whenever a user shares, pastes, drags, or uploads an image and wants to edit it, swap its background, create variations, analyze it, or use it as a reference — any workflow involving NanoBanana tools and a user-provided image. Also trigger this when a user asks "why can't you see my image" or a tool call fails with an invalid/missing URL.
---

# MCP S3 Companion

## Step 1 — Run this code RIGHT NOW using the **Python** tool (NOT bash/shell)

```python
import urllib.request, json, os, time, glob

SERVER = 'https://nanobanana-739905005785.us-central1.run.app'

# Auto-discover the uploads folder:
#   - claude.ai web sandbox: /mnt/user-data/uploads
#   - Claude Cowork (Mac): ~/Library/Application Support/Claude/local-agent-mode-sessions/<session>/uploads
candidates = ['/mnt/user-data/uploads'] + sorted(
    glob.glob(os.path.expanduser('~/Library/Application Support/Claude/local-agent-mode-sessions/*/uploads')),
    key=os.path.getmtime, reverse=True
)
uploads = next((p for p in candidates if os.path.isdir(p)), None)

if not uploads:
    print(f'No uploads folder found. Upload manually at {SERVER}/upload')
else:
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

This uploads all recently pasted images and prints their URLs. Works in both claude.ai web AND Claude Cowork — the snippet auto-discovers whichever uploads folder exists. **Do not skip this step. Do not check image sizes first. Do not encode to base64. Just run it.**

IMPORTANT: Execute this with the **Python code execution tool**. Running it in bash/shell causes "Bad for loop variable" errors.

The server handles any image size. The returned URLs are S3 links — pass them directly to NanoBanana tools.

If all attempts fail, direct the user to upload manually at: `https://nanobanana-739905005785.us-central1.run.app/upload`

Or use the full web app (no MCP needed): `https://nanobanana-739905005785.us-central1.run.app/app`

---

## Step 2 — Confirm aspect ratio BEFORE any image-output tool — REQUIRED

Image-output tools: `generate_image`, `edit_image`, `swap_background`, `create_variations`.
Before calling any of them for the first time in a conversation, ask the user in a single message:

> Before I {generate / edit / swap background / create variations}, two quick choices:
> • **Aspect ratio?** 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
> • **Resolution?** 1K, 2K, 4K
>
> Defaults: 4:5 / 1K for new generations; same shape as the source image for edits, background swaps, and variations. Reply with your picks, or say 'defaults'.

Per-tool support:
- `generate_image`, `create_variations`: aspect_ratio + resolution.
- `edit_image`, `swap_background`: aspect_ratio only — Gemini's edit API ignores resolution; output matches the source.

Skip the intake only when the user already named values, when re-running at known settings, or when chaining from a fixed-settings tool. **Tools that don't output images** (`analyze_image`, `batch_analyze`, `compare_images`, `list_styles`, `upload_image`) **don't need intake — call them directly.**

---

## Step 3 — Pass the URLs to the right tool

| Tool | What it does | Key params |
|---|---|---|
| `edit_image` | Inpaint, remove objects, outpaint | `image` = source URL; `prompt` = what to change |
| `swap_background` | Keep subject, replace background | `image` = source URL; `background` = new background description |
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
- Don't run the Python snippet in bash/shell — use the Python code tool
