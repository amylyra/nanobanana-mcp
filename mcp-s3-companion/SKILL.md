---
name: mcp-s3-companion
description: Handles user image uploads for NanoBanana MCP image tools. Use this skill whenever a user shares, pastes, drags, or uploads an image and wants to edit it, swap its background, create variations, analyze it, or use it as a reference — any workflow involving NanoBanana tools and a user-provided image. Also trigger this when a user asks "why can't you see my image" or a tool call fails with an invalid/missing URL.
---

# MCP S3 Companion

## How to get an image into NanoBanana

MCP tool parameters are plain text — image bytes cannot be passed directly. Every image needs to arrive as a URL. There are four paths depending on where the image comes from:

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

1. Use the Google Drive MCP to read the file. It returns the image as a base64 blob.
2. Construct a data URI: `data:image/jpeg;base64,<blob>`
3. Pass the data URI to `upload_image` — the server decodes, normalizes, and stores it to S3.

> Note: this works for small-to-medium images. For very large files (>2MB), the data URI may exceed MCP parameter limits. If that happens, ask the user to share the file publicly and use Path 1, or use Path 4.

---

### Path 3 — User already has an S3 or cloud storage URL

Pass it directly to the tool — same as Path 1. The server fetches any `https://` URL.

---

### Path 4 — User pasted or uploaded an image (no URL)

This is the fallback when there's no URL available (Ctrl+V paste, drag into chat, etc.). Direct the user to the upload page:

> I can see your image! To use it with the NanoBanana tools I need a URL for it.
> Please upload it at: **[SERVER_URL]/upload**
> Drag your image onto that page — it takes a second and gives you a URL to paste back here.

`[SERVER_URL]` is the base URL from the NanoBanana MCP server instructions (everything before `/upload`).

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

- Don't fabricate or guess S3/GCS/CDN URLs — if you don't have a real URL, use one of the paths above.
- Don't pass API endpoints, web page URLs, or MCP service URLs as image parameters.
- Don't try to read sandbox files and POST them to the server — that network path is blocked.
