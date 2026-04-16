# NanoBanana MCP

General-purpose image generation server for [Claude](https://claude.ai) via the [Model Context Protocol](https://modelcontextprotocol.io). Powered by Google Gemini's image generation models.

## What it does

Exposes a single `generate_image` tool that Claude can call to create images from text prompts, with optional reference images to guide style, subject, or composition.

**Features:**
- Text-to-image generation via Gemini (NanoBanana 2 / NanoBanana Pro)
- Reference image support — pass base64 data URIs to match a style, product, or composition
- Configurable aspect ratio, resolution, quality, and batch count
- Returns base64 JPEG data URIs ready for inline display
- Deploys to Cloud Run with Streamable HTTP transport

## Tool: `generate_image`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | yes | — | Image generation instruction |
| `reference_images` | array | no | null | Base64 data URIs or raw base64 strings for style/subject guidance |
| `aspect_ratio` | string | no | `4:5` | Output ratio: `1:1`, `4:5`, `9:16`, `16:9`, `3:4`, `4:3`, `2:3`, `3:2`, `5:4`, `21:9` |
| `resolution` | string | no | `1K` | Output size: `0.5K`, `1K`, `2K`, `4K` |
| `quality` | string | no | `default` | `default` (fast) or `pro` (higher quality) |
| `count` | integer | no | `1` | Number of images to generate (1–4) |

## Setup

### Prerequisites

- Python 3.12+
- A [Google AI API key](https://aistudio.google.com/apikey) with Gemini image generation access

### Local

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
python server.py
```

The server starts on `http://localhost:8080/mcp`.

### Deploy to Cloud Run

```bash
gcloud run deploy nanobanana \
  --source . \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY="your-key-here" \
  --allow-unauthenticated
```

### Connect to Claude

Add the deployed URL as a [remote MCP connector](https://modelcontextprotocol.io/docs/concepts/transports#streamable-http) in Claude settings:

```
https://your-service-url.run.app/mcp
```

## How reference images work

Pass one or more base64-encoded images to guide generation:

```json
{
  "prompt": "Product bottle on a marble countertop, soft morning light",
  "reference_images": [
    "data:image/jpeg;base64,/9j/4AAQ..."
  ]
}
```

The model uses reference images to match visual characteristics like shape, color, texture, and style — useful for product photography, brand consistency, or style transfer.

## License

MIT
