---
name: mcp-s3-companion
description: Bridges pasted/uploaded images to NanoBanana MCP image tools via S3. Use this skill whenever a user shares, pastes, drags, or uploads an image in the conversation and then wants to edit it, swap its background, create variations, analyze it, or use it as a reference for generation — any workflow involving NanoBanana image tools and a user-provided image. Also use this if a user asks "why can't you see my image" or an image tool returns an error about missing/invalid URLs.
---

# MCP S3 Companion

## Why this exists

MCP tools receive parameters as JSON strings — they cannot accept raw image bytes. When a user pastes or uploads an image in Claude.ai, it appears as vision content (you can see and describe it), but there is no way to serialize those pixels into a tool parameter directly.

This skill solves that by reading the image from Claude.ai's sandbox filesystem, compressing it, encoding it as a base64 data URI, and uploading it through the NanoBanana `upload_image` tool. The result is a durable S3 URL that any NanoBanana tool accepts.

## When you don't need this

- The user already gave you an **http/https URL** — pass it straight to the tool.
- A previous NanoBanana tool already returned a URL — reuse it, no re-upload needed.
- The user only wants text-to-image generation with no input image.

## The pipeline

### Step 1 — Read, compress, and encode the image

Claude.ai stores uploaded files at `/mnt/user-data/uploads/`. Use the code execution tool (Analysis) to read, compress, and base64-encode the image. Run this as a **single code block**:

```python
import os, base64
from io import BytesIO

uploads = "/mnt/user-data/uploads"
image_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")
MAX_DIM = 1536
MAX_BYTES = 400_000  # keeps base64 under ~530 KB
RAW_MAX_BYTES = 500_000  # skip files larger than this if PIL unavailable

if not os.path.isdir(uploads):
    print(f"Directory not found: {uploads}")
    print("The user's image may not be accessible from the sandbox.")
    print("FALLBACK: ask the user to upload via the server's /upload page.")
else:
    files = sorted(
        [f for f in os.listdir(uploads) if f.lower().endswith(image_exts)],
        key=lambda f: os.path.getmtime(os.path.join(uploads, f)),
        reverse=True,
    )

    def compress_image(filepath):
        try:
            from PIL import Image
            img = Image.open(filepath)
            has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
            orig_w, orig_h = img.size

            if max(orig_w, orig_h) > MAX_DIM:
                scale = MAX_DIM / max(orig_w, orig_h)
                img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)

            if has_alpha:
                img = img.convert("RGBA")
                buf = BytesIO()
                img.save(buf, format="PNG", optimize=True)
                while buf.tell() > MAX_BYTES:
                    w, h = img.size
                    if w < 200 or h < 200:
                        break
                    img = img.resize((int(w * 0.75), int(h * 0.75)), Image.LANCZOS)
                    buf = BytesIO()
                    img.save(buf, format="PNG", optimize=True)
                mime = "image/png"
                label = f"PNG {img.size[0]}x{img.size[1]}"
            else:
                img = img.convert("RGB")
                quality = 85
                buf = BytesIO()
                while quality >= 40:
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=quality)
                    if buf.tell() <= MAX_BYTES:
                        break
                    quality -= 10
                mime = "image/jpeg"
                label = f"JPEG q={quality} {img.size[0]}x{img.size[1]}"

            raw = buf.getvalue()
            print(f"  {os.path.basename(filepath)}: {orig_w}x{orig_h} -> {label}, {len(raw)//1024} KB")
            return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

        except Exception:
            # PIL not available or can't decode this format — send raw if small enough
            with open(filepath, "rb") as f:
                raw = f.read()
            if len(raw) > RAW_MAX_BYTES:
                print(f"  {os.path.basename(filepath)}: {len(raw)//1024} KB — too large without PIL")
                return None
            ext = filepath.rsplit(".", 1)[-1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/png")
            print(f"  {os.path.basename(filepath)}: {len(raw)//1024} KB (raw, no compression)")
            return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

    if not files:
        print("No image files found in uploads/")
        print("FALLBACK: ask the user to upload via the server's /upload page.")
    else:
        data_uris = []
        for fname in files:
            uri = compress_image(os.path.join(uploads, fname))
            if uri:
                data_uris.append(uri)

        print(f"\n{len(data_uris)} image(s) ready.")
        # Print each data URI so it can be used in the next step
        for i, uri in enumerate(data_uris):
            print(f"\n=== IMAGE {i} DATA URI ({len(uri)//1024} KB) ===")
            print(uri)
```

### Step 2 — Upload to S3

The Analysis output above contains the data URI strings (the long `data:image/...;base64,...` lines). Take each one and pass it to the NanoBanana `upload_image` tool:

```
upload_image(image="<paste the data URI from the Analysis output>")
```

It returns JSON with a durable S3 `url`. For multiple images, call `upload_image` once per data URI.

The data URI is the entire string starting with `data:image/` — copy it exactly from the Analysis output.

### Step 3 — Use the S3 URL

Pass the S3 URL from Step 2 to whichever NanoBanana tool the user asked for:

| Tool | What it does |
|---|---|
| `edit_image` | Inpaint, remove objects, outpaint |
| `swap_background` | Keep the foreground subject, replace the background |
| `create_variations` | Generate style/composition variations |
| `analyze_image` | Describe, tag, or assess quality |
| `generate_image` | Pass as `reference_images` for style/subject guidance |

Default aspect ratio is 4:5, default resolution is 1K.

### Fallback — if the sandbox doesn't work

When the uploads directory doesn't exist or has no images, the user's image exists only as vision content with no file path. Check the NanoBanana MCP server's instructions for its upload page URL, and tell the user:

> I can see your image but can't access the file from the sandbox. Please upload it at:
> **[upload page URL from server instructions]**
> That page will give you an S3 URL — paste it back here and I'll continue.

## Common failure modes

The biggest source of errors is **fabricated URLs** — inventing a plausible-looking S3 or GCS path instead of going through the upload pipeline. If you don't have a real URL from either `upload_image` or a previous tool response, you don't have a URL. Go through the pipeline.

Similarly, passing non-image URLs (API endpoints, MCP service URLs, web pages) to image parameters will fail. The tools expect actual image content behind the URL.
