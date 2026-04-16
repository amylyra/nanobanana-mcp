---
name: mcp-s3-companion
description: Handles user image uploads for NanoBanana MCP image tools. Use this skill whenever a user shares, pastes, drags, or uploads an image and wants to edit it, swap its background, create variations, analyze it, or use it as a reference — any workflow involving NanoBanana tools and a user-provided image. Also trigger this if a user asks "why can't you see my image" or a tool call fails with an invalid/missing URL.
---

# MCP S3 Companion

## The situation

Claude.ai can see pasted images visually, but MCP tool parameters only accept text (URLs, strings). There is no way to pass image bytes directly into a tool call — this is a known gap in the MCP protocol with no client-side workaround. The reliable solution is the NanoBanana upload page, which accepts the image and returns a durable S3 URL.

## When you don't need this

- User gave you an **http/https URL** → pass it directly to the tool.
- A previous NanoBanana tool already returned a URL → reuse it.
- User wants text-to-image with no input image.

## When a user shares an image

Tell them immediately — don't attempt any other approach first:

> I can see your image! To use it with the NanoBanana tools, I need a URL for it.
> Please upload it at: **[SERVER_URL]/upload**
> It only takes a second — drag your image onto that page and you'll get a URL to paste back here.

Replace `[SERVER_URL]` with the base URL from the NanoBanana MCP server instructions (the URL that appears before `/upload` in the instructions).

**For multiple images:** ask the user to upload each one and collect all the URLs before proceeding.

## After the user pastes the URL(s)

Use the S3 URL(s) with whichever tool fits the request:

| Tool | What it does | Key params |
|---|---|---|
| `edit_image` | Inpaint, remove objects, outpaint | `image` = source; `prompt` = what to change |
| `swap_background` | Keep subject, replace background | `image` = source; `prompt` = new background description |
| `create_variations` | Style/composition variations | `image` = source |
| `analyze_image` | Describe, tag, assess quality | `image` = source |
| `generate_image` | Text-to-image with style/subject guidance | `reference_images` = list of URLs |

Default aspect ratio 4:5, resolution 1K.

## What not to do

- Don't try to read the image from the Analysis sandbox and POST it to the server — the sandbox network is blocked and this doesn't work reliably.
- Don't try to pass a base64 data URI as a tool parameter — too large, will fail.
- Don't fabricate or guess S3/GCS URLs.
- Don't pass non-image URLs (API endpoints, web pages, MCP service URLs) to image parameters.
