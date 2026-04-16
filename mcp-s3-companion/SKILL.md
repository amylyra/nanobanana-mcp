---
name: mcp-s3-companion
description: Bridges pasted/uploaded images to NanoBanana MCP image tools via S3. Use this skill whenever a user shares, pastes, drags, or uploads an image in the conversation and then wants to edit it, swap its background, create variations, analyze it, or use it as a reference for generation — any workflow involving NanoBanana image tools and a user-provided image. Also use this if a user asks "why can't you see my image" or an image tool returns an error about missing/invalid URLs.
---

# MCP S3 Companion

## Why this exists

MCP tool parameters are JSON strings with a size limit — passing large images as base64 data URIs fails. Instead, this skill uses the Analysis tool to POST image bytes directly to the NanoBanana server's `/upload` HTTP endpoint, which returns a short S3 URL. Only the URL flows back into the conversation, keeping everything well within limits.

## When you don't need this

- The user already gave you an **http/https URL** — pass it straight to the tool.
- A previous NanoBanana tool already returned a URL — reuse it, no re-upload needed.
- The user only wants text-to-image generation with no input image.

## The pipeline

### Step 1 — Find the server URL

Look at the NanoBanana MCP server instructions — they contain the upload page URL in the form `https://.../upload`. The base URL is everything before `/upload`.

### Step 2 — Upload each image via HTTP

Claude.ai stores uploaded files at `/mnt/user-data/uploads/`. Use the code execution tool (Analysis) to POST each image directly to the server's `/upload` endpoint. Run this as a **single code block**:

```python
import os, requests
from io import BytesIO

SERVER_URL = "https://REPLACE-WITH-SERVER-URL-FROM-MCP-INSTRUCTIONS"
UPLOAD_URL = f"{SERVER_URL}/upload"

uploads = "/mnt/user-data/uploads"
image_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")

if not os.path.isdir(uploads):
    print("ERROR: uploads directory not found — see fallback instructions below")
else:
    files = sorted(
        [f for f in os.listdir(uploads) if f.lower().endswith(image_exts)],
        key=lambda f: os.path.getmtime(os.path.join(uploads, f)),
        reverse=True,
    )

    def upload_file(filepath):
        """Optionally compress, then POST raw bytes to /upload. Returns S3 URL."""
        with open(filepath, "rb") as f:
            raw = f.read()

        # Compress if PIL is available and image is large
        try:
            from PIL import Image
            img = Image.open(BytesIO(raw))
            w, h = img.size
            if max(w, h) > 1536:
                scale = 1536 / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
            buf = BytesIO()
            if has_alpha:
                img.convert("RGBA").save(buf, format="PNG", optimize=True)
            else:
                img.convert("RGB").save(buf, format="JPEG", quality=85)
            raw = buf.getvalue()
        except Exception:
            pass  # send as-is if PIL unavailable or fails

        resp = requests.post(UPLOAD_URL, data=raw, timeout=30)
        resp.raise_for_status()
        url = resp.json()["url"]
        print(f"  {os.path.basename(filepath)} -> {url}")
        return url

    if not files:
        print("No image files found — see fallback instructions below")
    else:
        s3_urls = []
        for fname in files:
            url = upload_file(os.path.join(uploads, fname))
            s3_urls.append(url)

        print(f"\n{len(s3_urls)} image(s) uploaded.")
        for i, url in enumerate(s3_urls):
            print(f"  [{i}] {files[i]}: {url}")
```

### Step 3 — Use the S3 URLs

The Analysis output contains the S3 URLs. Pass them to whichever NanoBanana tool the user asked for:

| Tool | What it does | Which URL goes where |
|---|---|---|
| `edit_image` | Inpaint, remove objects, outpaint | `image` = source to edit; `reference_images` = optional guides |
| `swap_background` | Keep subject, replace background | `image` = source; describe new background in prompt |
| `create_variations` | Style/composition variations | `image` = source |
| `analyze_image` | Describe, tag, assess quality | `image` = source |
| `generate_image` | Text-to-image with references | `reference_images` = style/subject guides |

Default aspect ratio is 4:5, default resolution is 1K.

### Fallback — if the sandbox can't reach the server

If `requests.post` fails with a connection error, the Analysis sandbox doesn't have outbound network access. Tell the user:

> The sandbox can't reach the server directly. Please upload your image at:
> **[upload page URL from server instructions]**
> That page returns an S3 URL — paste it back here and I'll continue.

## Common failure modes

- **Fabricated URLs** — never invent S3 or GCS paths. If you don't have a URL from an actual upload, you don't have a URL.
- **Wrong SERVER_URL** — replace the placeholder with the actual URL from the MCP server instructions before running the code.
- **Passing non-image URLs** (API endpoints, web pages, MCP service URLs) to image parameters will fail.
