# NanoBanana MCP

General-purpose image generation, editing, and variation server for [Claude](https://claude.ai) via the [Model Context Protocol](https://modelcontextprotocol.io). Powered by Google Gemini.

> **Tip:** Pass image URLs instead of base64 whenever possible — the server fetches them directly, avoiding size and truncation issues with inline data.

## Tools

### `generate_image`

Text-to-image generation with optional reference images, style presets, and AI prompt enhancement.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | yes | — | Image generation instruction |
| `reference_images` | string[] | no | null | Image URLs (recommended), base64 data URIs, or raw base64. Multiple supported. |
| `style` | string | no | null | Style preset — see [Style presets](#style-presets) |
| `enhance_prompt` | bool | no | false | AI-expand a short prompt into a detailed generation prompt |
| `aspect_ratio` | string | no | `4:5` | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| `resolution` | string | no | `1K` | `0.5K`, `1K`, `2K`, `4K` |
| `quality` | string | no | `default` | `default` (fast) or `pro` (higher quality) |
| `count` | int | no | `1` | Number of images (1–4) |

Reference images are downscaled to 1024px server-side to preserve detail (logos, labels) while keeping payloads manageable.

### `edit_image`

Edit an existing image — add objects, remove objects, or extend the canvas. Uses the Imagen 3 editing model.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image — URL (recommended), base64 data URI, or raw base64 |
| `prompt` | string | yes | — | Edit instruction — be specific about what to change |
| `mask` | string | no | null | Mask image (URL or base64). White = edit region, black = preserve. PNG recommended for clean edges. |
| `edit_mode` | string | no | `inpaint-insertion` | `inpaint-insertion`, `inpaint-removal`, `outpaint` |
| `aspect_ratio` | string | no | null | Output ratio (useful for outpaint) |
| `count` | int | no | `1` | Number of candidates (1–4) |

When no mask is provided, the model uses automatic semantic segmentation based on your prompt. For precise edits, provide a black-and-white mask. Source images are capped at 2048px.

### `create_variations`

Generate creative variations of an existing image with controllable divergence.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image — URL (recommended), base64 data URI, or raw base64 |
| `prompt` | string | no | null | Guidance for variations (e.g. "warmer lighting", "on a beach") |
| `variation_strength` | string | no | `medium` | `subtle`, `medium`, or `strong` |
| `aspect_ratio` | string | no | `4:5` | Output ratio |
| `resolution` | string | no | `1K` | `0.5K`, `1K`, `2K`, `4K` |
| `quality` | string | no | `default` | `default` or `pro` |
| `count` | int | no | `3` | Number of variations (1–4) |

Source images are capped at 2048px.

### `list_styles`

Returns all available style presets with descriptions.

## Style presets

Use the `style` parameter in `generate_image` to apply a consistent aesthetic:

| Preset | Effect |
|---|---|
| `cinematic` | Dramatic lighting, shallow DOF, film color grading |
| `product-photography` | Clean studio setup, white background, commercial look |
| `editorial` | Magazine-style, natural light, muted tones |
| `watercolor` | Translucent washes, paper texture, organic edges |
| `flat-illustration` | Bold vectors, geometric shapes, limited palette |
| `neon-noir` | Dark scene, neon reflections, cyberpunk mood |
| `minimalist` | Vast negative space, single focal point, zen simplicity |
| `vintage-film` | Warm cast, visible grain, 1970s nostalgia |

## Setup

### Prerequisites

- Python 3.12+
- A [Google AI API key](https://aistudio.google.com/apikey) with access to:
  - Gemini image generation (for `generate_image` and `create_variations`)
  - Imagen 3 editing (for `edit_image`)

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

## Examples

**Simple generation:**
```json
{
  "prompt": "a coffee cup on a wooden table"
}
```

**With style preset and prompt enhancement:**
```json
{
  "prompt": "a coffee cup on a wooden table",
  "style": "cinematic",
  "enhance_prompt": true
}
```

**With URL reference images:**
```json
{
  "prompt": "Same bottle but on a beach at sunset",
  "reference_images": [
    "https://example.com/my-product.jpg",
    "https://example.com/mood-board.jpg"
  ]
}
```

**Edit — remove an object:**
```json
{
  "image": "https://example.com/photo.jpg",
  "prompt": "Remove the person in the background",
  "edit_mode": "inpaint-removal"
}
```

**Edit — with a mask for precision:**
```json
{
  "image": "https://example.com/photo.jpg",
  "prompt": "Replace the sky with a dramatic sunset",
  "mask": "https://example.com/sky-mask.png",
  "edit_mode": "inpaint-insertion"
}
```

**Create variations:**
```json
{
  "image": "https://example.com/product-shot.jpg",
  "prompt": "warmer lighting, golden hour",
  "variation_strength": "medium",
  "count": 3
}
```

## Image handling

The server normalizes all input images to keep things reliable:

| Context | Max dimension | Format | Quality |
|---|---|---|---|
| Reference images (`generate_image`) | 1024px | JPEG | 85 |
| Source images (`edit_image`, `create_variations`) | 2048px | JPEG | 92 |
| Mask images (`edit_image`) | 2048px | PNG | lossless |
| Output images | Full resolution | JPEG | 92 |

Base64 padding is auto-fixed. Corrupt/truncated images return a clear error suggesting URL usage.

## License

MIT
