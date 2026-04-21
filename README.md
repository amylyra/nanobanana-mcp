# NanoBanana MCP

General-purpose image generation, editing, and analysis server for [Claude](https://claude.ai) via the [Model Context Protocol](https://modelcontextprotocol.io). Powered by Google Gemini.

## How images work

MCP tool inputs are text-only — Claude **cannot** reliably forward pasted image bytes to tools (they get truncated). NanoBanana handles this automatically:

**With clients that support MCP elicitation (e.g. Claude Code):**
- Just paste your image and call any tool — if the image data is truncated, the server automatically opens an upload page in your browser. Drop the image there and the tool continues on its own. No manual URL copying needed.

**With other clients (e.g. Claude.ai web):**
1. **Open `{server_url}/upload`** in your browser (drag-and-drop form).
2. **Drop or select** an image — the server stores it and returns a URL.
3. **Paste the URL** into Claude and reference it in any tool call.

The URL is a direct HTTP link (e.g. `https://your-server.run.app/images/a1b2c3d4e5f6`) that the server resolves locally — no extra network round-trip. Images expire after 1 hour; re-upload if needed.

> **Tip:** You can also pass any public image URL directly — no upload needed. The upload/elicitation flow is only required for local images that don't have a URL.

## Tools

### `upload_image`

Store an image on the server and get a URL back. Pass an image URL — the server fetches it directly. For local images with no URL, use the HTTP upload endpoint above first.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Image URL (http/https or nanobanana://) |

Returns image URL, dimensions, and size.

### `generate_image`

Text-to-image generation with optional reference images, style presets, AI prompt enhancement, and QA scoring. Returns images inline (viewable directly in Claude) plus metadata.

> If your Claude.ai chat response does not visibly include the image, instruct Claude to include the `markdown` field returned by the tool result (or `markdown_gallery` for multiple images) in its reply.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | yes | — | Image generation instruction |
| `reference_images` | string[] | no | null | Image URLs (http/https or `nanobanana://`) |
| `style` | string | no | null | Style preset — see [Style presets](#style-presets) |
| `enhance_prompt` | bool | no | false | AI-expand a short prompt into a detailed generation prompt |
| `aspect_ratio` | string | no | `4:5` | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| `resolution` | string | no | `1K` | `1K`, `2K`, `4K` |
| `quality` | string | no | `default` | `default` (fast) or `pro` (higher quality) |
| `count` | int | no | `1` | Number of images (1–4) |
| `qa` | bool | no | false | AI-score each image on composition, clarity, lighting, color, prompt adherence. Ranks results when count > 1. |
| `output` | string | no | `base64` | `base64` (inline images) or `cloud`/`s3`/`gcs` (upload to cloud storage, return URL) |

### `edit_image`

Edit an existing image — add objects, remove objects, or extend the canvas. Uses Imagen 3.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image — URL or `nanobanana://` URL |
| `prompt` | string | yes | — | Edit instruction — be specific |
| `mask` | string | no | null | Mask image. White = edit, black = preserve. PNG recommended. |
| `edit_mode` | string | no | `inpaint-insertion` | `inpaint-insertion`, `inpaint-removal`, `outpaint` |
| `aspect_ratio` | string | no | null | Output ratio (useful for outpaint) |
| `count` | int | no | `1` | Number of candidates (1–4) |
| `output` | string | no | `base64` | `base64` or `cloud`/`s3`/`gcs` |

### `swap_background`

One-step background replacement — keeps the foreground subject, generates a new background.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image with the subject to keep |
| `background` | string | yes | — | Description of the new background |
| `aspect_ratio` | string | no | null | Output ratio |
| `count` | int | no | `1` | Number of candidates (1–4) |
| `output` | string | no | `base64` | `base64` or `cloud`/`s3`/`gcs` |

### `create_variations`

Generate creative variations of an existing image with controllable divergence and optional QA.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image — URL or `nanobanana://` URL |
| `prompt` | string | no | null | Guidance for variations |
| `variation_strength` | string | no | `medium` | `subtle`, `medium`, or `strong` |
| `aspect_ratio` | string | no | `4:5` | Output ratio |
| `resolution` | string | no | `1K` | `1K`, `2K`, `4K` |
| `quality` | string | no | `default` | `default` or `pro` |
| `count` | int | no | `3` | Number of variations (1–4) |
| `qa` | bool | no | false | Score and rank variations by quality |
| `output` | string | no | `base64` | `base64` or `cloud`/`s3`/`gcs` |

### `analyze_image`

Describe, tag, or assess an image using Gemini vision.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Image to analyze — URL or `nanobanana://` URL |
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

- Python 3.10+
- A [Google AI API key](https://aistudio.google.com/apikey) with access to:
  - Gemini image generation (`generate_image`, `create_variations`)
  - Imagen 3 editing (`edit_image`, `swap_background`)

### Local

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # or GOOGLE_AI_API_KEY
python server.py
```

Alternatively, use a `.env` file for convenience:

```bash
pip install python-dotenv
cp .env.example .env   # then edit .env with your key
python server.py
```

The server validates the API key at startup and prints a clear error if it's missing. It starts on `http://localhost:8080/mcp`.

The upload page is available at `http://localhost:8080/upload`.

### Deploy to Cloud Run

```bash
gcloud run deploy nanobanana \
  --source . \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY="your-key-here",PUBLIC_URL="https://nanobanana-xxx.run.app" \
  --allow-unauthenticated
```

Set `PUBLIC_URL` to your Cloud Run service URL so that the elicitation upload links work correctly.

### Connect to Claude

Add the deployed URL as a [remote MCP connector](https://modelcontextprotocol.io/docs/concepts/transports#streamable-http) in Claude settings:

```
https://your-service-url.run.app/mcp
```

### Cloud storage output (optional)

To return image URLs instead of inline images, configure S3 or GCS. Use `output: "cloud"` in any tool call.

#### AWS S3

```bash
pip install boto3>=1.34.0
```

Set env vars on Cloud Run (or in `.env` for local dev):

```
S3_BUCKET=claude-image-cache
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-2
```

Bucket needs a public-read policy so generated URLs are accessible. Set a lifecycle rule to auto-delete after 1 day if you don't need permanent storage.

#### GCS (alternative)

```bash
pip install google-cloud-storage>=2.0.0
```

```bash
gsutil mb gs://my-nanobanana-images

gcloud run services update nanobanana \
  --set-env-vars GEMINI_API_KEY="...",GCS_BUCKET="my-nanobanana-images"
```

If both `S3_BUCKET` and `GCS_BUCKET` are set, S3 is used.

## HTTP endpoints

These endpoints run alongside the MCP transport and handle image upload/serving outside the MCP protocol:

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | GET | Drag-and-drop upload form |
| `/upload` | POST | Accept image upload (multipart or raw bytes), return JSON with URL |
| `/images/{id}` | GET | Serve a stored image by ID |
| `/mcp` | — | MCP Streamable HTTP transport |

## Examples

**Upload an image via the browser, then edit it (recommended workflow):**

1. Open `https://your-server.run.app/upload` in your browser
2. Drop an image — you get a URL like `https://your-server.run.app/images/a1b2c3d4e5f6`
3. Paste the URL into Claude and ask it to edit the image:

```json
{
  "image": "https://your-server.run.app/images/a1b2c3d4e5f6",
  "prompt": "Replace the bottle with a blue one",
  "edit_mode": "inpaint-insertion"
}
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

**Generate and save to cloud storage (S3 or GCS):**
```json
{
  "prompt": "minimalist product shot",
  "output": "cloud"
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
