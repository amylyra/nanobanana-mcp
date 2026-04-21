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

All 177 tests must pass. `conftest.py` provides async test support without requiring `pytest-asyncio`.

## Architecture

- `server.py` — FastMCP server (single file)
- `test_simulation.py` — unit/integration tests (mock Gemini, real PIL/store logic)
- `conftest.py` — pytest hook that runs `@pytest.mark.asyncio` tests without the `pytest-asyncio` plugin
- Cloud Run serves both the MCP endpoint (`/mcp`) and a direct upload page (`/upload`)

## Tool output contract

Image-generating tools (`generate_image`, `edit_image`, `swap_background`, `create_variations`) return a mixed list (Attempt 10):

```
[render_md, json_str, Image(thumb1), Image(thumb2), ...]
```

- **`render_md`** — standalone markdown image link(s): `"![](url)"` (single) or `"![Image 1](url1)\n\n![Image 2](url2)"` (multi). First item so Claude sees it as the primary result.
- **`json_str`** — JSON metadata for tool chaining: `image_url`, `size_kb`, `expires_in`, etc.
- **`Image` objects** — 512px JPEG `ImageContent` blocks for tool pane previews (render inline in the tool result pane, no consent gate).

**Known limitation:** When Claude includes `render_md` in its reply, claude.ai shows clickable "Show Image" boxes rather than rendering inline. `ImageContent` renders inline in the tool pane without a consent gate. See `IMAGE_DISPLAY_ATTEMPTS.md` for the full history.

## Key design decisions

- **`structured_output=False`** on all image tools — required to prevent `PydanticSerializationError` when FastMCP serialises the mixed `[str, Image, ...]` return list.
- **JSON-first ordering** — `json_str` is the first list element so clients that only read the first content block still get usable metadata.
- **Thumbnails in ImageContent** — 512px max, quality 75; keeps payload small while being visible inline.
- **Full-size in `image_url`** — the `/images/` store or S3 URL always points to the full-resolution JPEG.
- **`stateless_http=True`** — required for Cloud Run (no sticky sessions across instances).
- **`S3_BUCKET` env var** — when set, images are uploaded to S3 synchronously during `_build_image_response`. URLs are plain `https://bucket.s3.region.amazonaws.com/prefix/uuid.jpg` (no presigning) so claude.ai detects them as images.
