# NanoBanana MCP

General-purpose image generation, editing, and analysis server for [Claude](https://claude.ai) via the [Model Context Protocol](https://modelcontextprotocol.io). Powered by Google Gemini.

> **Tip:** When working with uploaded images, call `upload_image` first to store the image server-side. It returns a `nanobanana://` URL you can pass to any other tool — no base64 truncation issues.

## Tools

### `upload_image`

Upload an image to the server and get a `nanobanana://` URL. **Call this first** when a user provides/uploads an image, then pass the returned URL to other tools.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Base64 data URI, raw base64, or URL |

Returns a `nanobanana://` URL (valid for 1 hour), image dimensions, and size. Use the URL in any other tool's `image` or `reference_images` parameter.

### `generate_image`

Text-to-image generation with optional reference images, style presets, AI prompt enhancement, and QA scoring.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | yes | — | Image generation instruction |
| `reference_images` | string[] | no | null | Image URLs (recommended), base64 data URIs, or raw base64 |
| `style` | string | no | null | Style preset — see [Style presets](#style-presets) |
| `enhance_prompt` | bool | no | false | AI-expand a short prompt into a detailed generation prompt |
| `aspect_ratio` | string | no | `4:5` | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| `resolution` | string | no | `1K` | `0.5K`, `1K`, `2K`, `4K` |
| `quality` | string | no | `default` | `default` (fast) or `pro` (higher quality) |
| `count` | int | no | `1` | Number of images (1–4) |
| `qa` | bool | no | false | AI-score each image on composition, clarity, lighting, color, prompt adherence. Ranks results when count > 1. |
| `output` | string | no | `base64` | `base64` (data URI) or `gcs` (upload to GCS, return URL) |

### `edit_image`

Edit an existing image — add objects, remove objects, or extend the canvas. Uses Imagen 3.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image — URL (recommended) or base64 |
| `prompt` | string | yes | — | Edit instruction — be specific |
| `mask` | string | no | null | Mask image. White = edit, black = preserve. PNG recommended. |
| `edit_mode` | string | no | `inpaint-insertion` | `inpaint-insertion`, `inpaint-removal`, `outpaint` |
| `aspect_ratio` | string | no | null | Output ratio (useful for outpaint) |
| `count` | int | no | `1` | Number of candidates (1–4) |
| `output` | string | no | `base64` | `base64` or `gcs` |

### `swap_background`

One-step background replacement — keeps the foreground subject, generates a new background.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image with the subject to keep |
| `background` | string | yes | — | Description of the new background |
| `aspect_ratio` | string | no | null | Output ratio |
| `count` | int | no | `1` | Number of candidates (1–4) |
| `output` | string | no | `base64` | `base64` or `gcs` |

### `create_variations`

Generate creative variations of an existing image with controllable divergence and optional QA.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image — URL (recommended) or base64 |
| `prompt` | string | no | null | Guidance for variations |
| `variation_strength` | string | no | `medium` | `subtle`, `medium`, or `strong` |
| `aspect_ratio` | string | no | `4:5` | Output ratio |
| `resolution` | string | no | `1K` | `0.5K`, `1K`, `2K`, `4K` |
| `quality` | string | no | `default` | `default` or `pro` |
| `count` | int | no | `3` | Number of variations (1–4) |
| `qa` | bool | no | false | Score and rank variations by quality |
| `output` | string | no | `base64` | `base64` or `gcs` |

### `analyze_image`

Describe, tag, or assess an image using Gemini vision.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Image to analyze — URL (recommended) or base64 |
| `focus` | string | no | `general` | `general`, `tags`, `alt-text`, `quality`, `brand` |

Focus modes:
- **general** — comprehensive description (subject, style, mood, colors)
- **tags** — keyword tags for search/SEO/asset management
- **alt-text** — concise accessible alt text (1-2 sentences)
- **quality** — technical quality scores (sharpness, exposure, composition)
- **brand** — marketing analysis (target audience, tone, use cases)

### `list_styles`

Returns all available style presets with descriptions.

## Style presets

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
  - Gemini image generation (`generate_image`, `create_variations`)
  - Imagen 3 editing (`edit_image`, `swap_background`)

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

### GCS output (optional)

To return image URLs instead of base64, set up a GCS bucket:

```bash
# Create bucket
gsutil mb gs://my-nanobanana-images

# Set env var on Cloud Run
gcloud run services update nanobanana \
  --set-env-vars GEMINI_API_KEY="...",GCS_BUCKET="my-nanobanana-images"
```

Then use `output: "gcs"` in any tool call. Images are uploaded and a public URL is returned.

## Examples

**Upload an image, then edit it (recommended workflow):**
```json
// Step 1: Upload
upload_image({ "image": "data:image/jpeg;base64,/9j/4AAQ..." })
// Returns: { "url": "nanobanana://a1b2c3d4e5f6", ... }

// Step 2: Edit using the returned URL
edit_image({
  "image": "nanobanana://a1b2c3d4e5f6",
  "prompt": "Replace the bottle with a blue one",
  "edit_mode": "inpaint-insertion"
})
```

**Generate with style + prompt enhancement:**
```json
{
  "prompt": "a coffee cup on a wooden table",
  "style": "cinematic",
  "enhance_prompt": true
}
```

**Generate with QA scoring (pick the best of 3):**
```json
{
  "prompt": "luxury skincare bottle on marble",
  "style": "product-photography",
  "count": 3,
  "qa": true
}
```

**Swap background:**
```json
{
  "image": "https://example.com/product.jpg",
  "background": "tropical beach at sunset with palm trees"
}
```

**Analyze an image for SEO tags:**
```json
{
  "image": "https://example.com/photo.jpg",
  "focus": "tags"
}
```

**Generate and save to GCS:**
```json
{
  "prompt": "minimalist product shot",
  "output": "gcs"
}
```

## Image handling

| Context | Max dimension | Format | Quality |
|---|---|---|---|
| Reference images (`generate_image`) | 1024px | JPEG | 85 |
| Source images (`edit_image`, `swap_background`, `create_variations`) | 2048px | JPEG | 92 |
| Mask images (`edit_image`) | 2048px | PNG | lossless |
| Output images | Full resolution | JPEG | 92 |

## License

MIT
