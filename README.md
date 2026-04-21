# NanoBanana MCP

General-purpose image generation, editing, and analysis server for [Claude](https://claude.ai) via the [Model Context Protocol](https://modelcontextprotocol.io). Powered by Google Gemini.

## How images work

### Getting an image into a tool

All image tool parameters accept **http/https URLs only**. Three paths:

1. **Direct URL** — pass it straight to the tool. Public S3, CDN, and Google Drive share links all work.
2. **Public Google Drive link** — pass it straight to the tool. Share links are auto-rewritten to direct download URLs.
3. **Pasted or local image (no URL)** — use the Python code tool to encode it, then call `upload_image`:

```python
import os, base64
from io import BytesIO
from PIL import Image
uploads = '/mnt/user-data/uploads'
files = sorted(
    [f for f in os.listdir(uploads) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))],
    key=lambda f: os.path.getmtime(os.path.join(uploads, f)), reverse=True,
)
img = Image.open(os.path.join(uploads, files[0]))
if max(img.size) > 1536: img.thumbnail((1536, 1536), Image.LANCZOS)
img = img.convert('RGB')
buf = BytesIO()
img.save(buf, format='JPEG', quality=85)
uri = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
print(uri[:80], '...')
```

Then call `upload_image(image=uri)` — it returns a URL you can pass to any other tool.

Alternatively, open `{server_url}/upload` in your browser and drag-and-drop any image to get a URL.

### How generated images are returned

Every image-generating tool returns two text blocks:

```
![generated image](https://...)

{"image_url": "...", ...}
```

The first block is a standalone markdown image link. Claude includes it in its reply, which claude.ai renders as a "Show Image" clickable box. The second block is JSON metadata — use `image_url` from it to pass the image to another tool.

## Tools

### `upload_image`

Re-host an image URL to a server URL for use in other tools. Accepts http/https URLs (including Google Drive share links) and data URIs.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | http/https URL or data URI (`data:image/...;base64,...`) |

Returns JSON with `url`, dimensions, and size. When cloud storage is configured, the URL is durable (no expiry). Otherwise it expires after 1 hour.

### `generate_image`

Text-to-image generation with optional reference images, style presets, AI prompt enhancement, and QA scoring.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | yes | — | What to generate |
| `reference_images` | string[] | no | null | Reference image URLs — guide style, subject, or composition |
| `style` | string | no | null | Style preset — see [Style presets](#style-presets) |
| `enhance_prompt` | bool | no | false | AI-expand a short prompt into a detailed generation prompt |
| `aspect_ratio` | string | no | `4:5` | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| `resolution` | string | no | `1K` | `1K`, `2K`, `4K` |
| `quality` | string | no | `default` | `default` (fast) or `pro` (higher quality) |
| `count` | int | no | `1` | Number of images to generate (1–4) |
| `qa` | bool | no | false | AI-score each image; rank by total score when count > 1 |
| `save_folder` | string | no | null | Local folder path to also save JPEG files |

### `edit_image`

Edit an existing image — add objects, remove objects, or extend the canvas.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image URL |
| `prompt` | string | yes | — | Edit instruction — be specific |
| `reference_images` | string[] | no | null | Reference image URLs. Use when replacing or inserting content from other images. For multiple refs, name them in the prompt: "replace X with reference image 1, Y with reference image 2" |
| `mask` | string | no | null | Mask image URL. White = edit region, black = preserve. |
| `edit_mode` | string | no | `inpaint-insertion` | `inpaint-insertion` (add/replace), `inpaint-removal` (remove + fill), `outpaint` (extend canvas) |
| `aspect_ratio` | string | no | null | Output ratio (useful for outpaint) |
| `count` | int | no | `1` | Number of candidates (1–4) |
| `save_folder` | string | no | null | Local folder path to also save JPEG files |

### `swap_background`

Keep the foreground subject, replace the background.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image URL |
| `background` | string | yes | — | Description of the new background |
| `aspect_ratio` | string | no | null | Output ratio |
| `count` | int | no | `1` | Number of candidates (1–4) |
| `save_folder` | string | no | null | Local folder path to also save JPEG files |

### `create_variations`

Generate variations of an existing image with controllable divergence.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Source image URL |
| `prompt` | string | no | null | Guidance for variations (echoed as `guidance` in metadata) |
| `variation_strength` | string | no | `medium` | `subtle`, `medium`, or `strong` |
| `aspect_ratio` | string | no | null | Output ratio |
| `resolution` | string | no | `1K` | `1K`, `2K`, `4K` |
| `quality` | string | no | `default` | `default` or `pro` |
| `count` | int | no | `3` | Number of variations (1–4) |
| `qa` | bool | no | false | Score and rank variations by quality |
| `save_folder` | string | no | null | Local folder path to also save JPEG files |

### `analyze_image`

Describe, tag, or assess a single image using Gemini vision. For multiple images use `batch_analyze`.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | yes | — | Image URL |
| `focus` | string | no | `general` | `general`, `tags`, `alt-text`, `quality`, `brand` |

Focus modes:
- **general** — comprehensive description (subject, style, mood, colors)
- **tags** — keyword tags for search/SEO/asset management
- **alt-text** — concise accessible alt text (1–2 sentences)
- **quality** — technical quality scores (sharpness, exposure, composition, color, noise)
- **brand** — marketing analysis (audience, tone, positioning, use cases)

### `batch_analyze`

Analyze 2–20 images in parallel using Gemini vision. Runs all analyses concurrently — same wall time as analyzing one image. Results are returned in the same order as the input list.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `images` | string[] | yes | — | List of 2–20 image URLs |
| `focus` | string | no | `general` | Same focus modes as `analyze_image` |

Returns a `results` array where each entry includes `index` (1-based), `image_url`, and the analysis fields (or an `error` field if that image failed).

### `compare_images`

Compare 2–10 images side-by-side in a single Gemini call. Unlike `batch_analyze` (which analyzes each image independently), all images are sent together so the model can reason about differences, rankings, and relationships across the whole set.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `images` | string[] | yes | — | List of 2–10 image URLs |
| `focus` | string | no | `differences` | `differences`, `similarities`, `quality`, `style`, `general` |

Focus modes:
- **differences** — every meaningful visual difference between images
- **similarities** — what the images have in common
- **quality** — rank images by technical quality with scores; identifies the best
- **style** — aesthetic/style comparison (mood, palette, photographic style)
- **general** — comprehensive side-by-side overview

### `list_styles`

Returns all available style presets with descriptions. No parameters.

## Style presets

| Preset | Effect |
|---|---|
| `cinematic` | Dramatic lighting, shallow DOF, anamorphic lens flare, film color grading |
| `product-photography` | Clean studio setup, soft shadows, white background, commercial look |
| `editorial` | Magazine-style, natural light, muted tones, fashion-forward |
| `watercolor` | Translucent washes, paper texture, organic bleeding edges |
| `flat-illustration` | Bold vectors, geometric shapes, limited palette, graphic design |
| `neon-noir` | Dark scene, vibrant neon reflections on wet surfaces, cyberpunk mood |
| `minimalist` | Vast negative space, single focal point, muted palette, zen simplicity |
| `vintage-film` | Warm cast, visible grain, faded highlights, 1970s nostalgia |

## Setup

### Prerequisites

- Python 3.10+
- A [Google AI API key](https://aistudio.google.com/apikey)

### Local

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # or GOOGLE_AI_API_KEY
python server.py
```

Optionally use a `.env` file:

```bash
pip install python-dotenv
echo 'GEMINI_API_KEY=your-key-here' > .env
python server.py
```

Server starts on `http://localhost:8080`. MCP endpoint: `/mcp`. Upload page: `/upload`.

### Deploy to Cloud Run

```bash
gcloud run deploy nanobanana \
  --source . \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY="your-key-here",PUBLIC_URL="https://nanobanana-xxx.run.app" \
  --allow-unauthenticated
```

`PUBLIC_URL` must be set to your Cloud Run service URL so that `/images/` URLs in tool responses are externally accessible.

### Connect to Claude

Add the deployed URL as a remote MCP connector in Claude settings:

```
https://your-service-url.run.app/mcp
```

### Cloud storage (optional)

When `S3_BUCKET` or `GCS_BUCKET` is set, generated images are uploaded to cloud storage and `image_url` in tool responses is a durable URL (no expiry). Without cloud storage, images are served from the server's in-memory store and expire after 1 hour.

#### AWS S3

```bash
pip install boto3>=1.34.0
```

Set env vars (Cloud Run or `.env`):

```
S3_BUCKET=my-nanobanana-images
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

The bucket needs a public-read policy. The returned URL is a plain `https://bucket.s3.region.amazonaws.com/prefix/uuid.jpg` — no query params — so claude.ai renders it inline.

#### GCS

```bash
pip install google-cloud-storage>=2.0.0
```

```bash
gsutil mb gs://my-nanobanana-images
gcloud run services update nanobanana \
  --set-env-vars GCS_BUCKET="my-nanobanana-images"
```

If both `S3_BUCKET` and `GCS_BUCKET` are set, S3 is used.

## HTTP endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | GET | Drag-and-drop upload form |
| `/upload` | POST | Accept image upload (multipart or raw bytes), return JSON with `url` |
| `/images/{id}` | GET | Serve a stored image by ID (expires after 1 hour) |
| `/mcp` | — | MCP Streamable HTTP transport |

## Examples

**Generate with style + prompt enhancement:**
```json
{
  "prompt": "a coffee cup on a wooden table",
  "style": "cinematic",
  "enhance_prompt": true
}
```

**Generate 3 images, score and rank them:**
```json
{
  "prompt": "luxury skincare bottle on marble",
  "style": "product-photography",
  "count": 3,
  "qa": true
}
```

**Edit an image — add an object:**
```json
{
  "image": "https://example.com/product.jpg",
  "prompt": "add a sprig of lavender next to the bottle",
  "edit_mode": "inpaint-insertion"
}
```

**Composite swap — put the object from image B into image A:**
```json
{
  "image": "https://example.com/scene.jpg",
  "prompt": "replace the bottle with the object from reference image 1",
  "reference_images": ["https://example.com/new-bottle.jpg"]
}
```

**Swap background:**
```json
{
  "image": "https://example.com/product.jpg",
  "background": "tropical beach at sunset with palm trees"
}
```

**Analyze 5 images for SEO tags in one call:**
```json
{
  "images": ["https://example.com/a.jpg", "https://example.com/b.jpg", "..."],
  "focus": "tags"
}
```

**Pick the best image from a set:**
```json
{
  "images": ["https://example.com/v1.jpg", "https://example.com/v2.jpg"],
  "focus": "quality"
}
```

**Save generated images to a local folder:**
```json
{
  "prompt": "minimalist product shot",
  "count": 2,
  "save_folder": "/tmp/nanobanana-output"
}
```

## Image handling

| Context | Max dimension | Format | Quality |
|---|---|---|---|
| Reference images (`generate_image`) | 1024px | JPEG | 85 |
| Source images (`edit_image`, `swap_background`, `create_variations`) | 2048px | JPEG | 92 |
| Mask images (`edit_image`) | 2048px | PNG | lossless |
| Inline thumbnails (ImageContent in tool result) | 512px | JPEG | 75 |
| `image_url` / `saved_to` | Full resolution | JPEG | 92 |

## License

MIT
