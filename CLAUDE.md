# NanoBanana MCP — Claude Instructions

## Deployment

**Always deploy to the `nanobanana` service, not `nanobanana-mcp`.**

```
gcloud run deploy nanobanana --source . --region us-central1
```

- Service name: `nanobanana`
- Region: `us-central1`
- Project: `stellar-builder-492016-s8`
- URL: `https://nanobanana-739905005785.us-central1.run.app`

The service already has `GEMINI_API_KEY`, `S3_BUCKET`, and other env vars configured in Cloud Run — do not pass them on the command line.

Before deploying, always run `gcloud run services list` to confirm the target service exists and the name matches.

## Testing

Run the full test suite before deploying:

```
python -m pytest test_simulation.py -x -q
```

All 176 tests must pass. `conftest.py` provides async test support without requiring `pytest-asyncio`.

## Architecture

- `server.py` — FastMCP server (single file)
- `test_simulation.py` — unit/integration tests (mock Gemini, real PIL/store logic)
- `conftest.py` — pytest hook that runs `@pytest.mark.asyncio` tests without the `pytest-asyncio` plugin
- Cloud Run serves both the MCP endpoint (`/mcp`) and a direct upload page (`/upload`)

## Tool output contract

Image-generating tools (`generate_image`, `edit_image`, `swap_background`, `create_variations`) return a list with a single string:

```
["![generated image](url)\n\n{json}"]
```

The string is a **combined text block**: markdown image link(s) first, then machine-readable JSON. Format:

```
![generated image](https://bucket.s3.amazonaws.com/gen/uuid.jpg)

{"response_mode": "deterministic_markdown", "image_url": "...", "expires_in": "1 hour", ...}
```

- **Markdown first** — Claude sees the `![](url)` as the primary content and includes it verbatim in its reply, making the image visible in the chat response.
- **`response_mode: "deterministic_markdown"`** — signals to clients that the text block is in this format.
- **`image_url`** — full-quality S3 or `/images/` URL for passing to other tools.
- **`expires_in`** — present when using in-memory store (no cloud storage configured).
- **Multi-image**: markdown section has N lines (`![image 1](url1)\n![image 2](url2)\n...`), JSON has an `images` array.

## Key design decisions

- **`structured_output=False`** on all image tools — kept for safety; prevents FastMCP from inferring an output schema that could break if the return type changes.
- **Single text block (no `ImageContent`)** — prior approach returned `[json_str, Image(...)]` (mixed content). The theory is that Claude sees `ImageContent` already rendered in the tool pane and doesn't include the image in its reply. With only a text block, Claude must emit the markdown to show the image at all.
- **Full-size in `image_url`** — the `/images/` store or S3 URL always points to the full-resolution JPEG.
- **`stateless_http=True`** — required for Cloud Run (no sticky sessions across instances).
- **`S3_BUCKET` env var** — when set, images are uploaded to S3 synchronously during `_build_image_response`. URLs are plain `https://bucket.s3.region.amazonaws.com/prefix/uuid.jpg` (no presigning) so claude.ai detects them as images.
