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

Image-generating tools (`generate_image`, `edit_image`, `swap_background`, `create_variations`) return a two-item list (Attempt 11):

```
[render_md, json_str]
```

- **`render_md`** — standalone markdown image link(s): `"![](url)"` (single) or `"![Image 1](url1)\n\n![Image 2](url2)"` (multi). This is its own `TextContent` block — no JSON mixed in. Claude sees it as the primary result and includes it in its reply.
- **`json_str`** — JSON metadata for tool chaining: `image_url`, `size_kb`, `expires_in`, etc.

No `ImageContent` objects are returned (see `IMAGE_DISPLAY_ATTEMPTS.md` for the full history).

**Known limitation:** When Claude includes the markdown in its reply, claude.ai shows a "Show Image" click-to-load gate rather than rendering inline. `ImageContent` bypasses the gate but Claude doesn't reliably repeat those in its reply.

## Key design decisions

- **`structured_output=False`** on all image tools — required to prevent `PydanticSerializationError` when FastMCP serialises the `[str, str]` list return type.
- **render_md first** — render_md is `result[0]` so Claude sees it as the primary content; json_str is `result[1]` for tool chaining.
- **`_build_image_response` returns a tuple** — `(render_md, json_str)`. Tool call sites unpack and return `[render_md, json_result]` list.
- **`stateless_http=True`** — required for Cloud Run (no sticky sessions across instances).
- **`S3_BUCKET` env var** — when set, images are uploaded to S3 synchronously during `_build_image_response`. URLs are plain `https://bucket.s3.region.amazonaws.com/prefix/uuid.jpg` (no presigning) so claude.ai detects them as images.
