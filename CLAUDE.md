# NanoBanana MCP — Operator Notes

## Deploy target

Always deploy to service `nanobanana` in `us-central1`.

```bash
gcloud run deploy nanobanana --source . --region us-central1
```

Project: `stellar-builder-492016-s8`

Before deploy, validate service exists:

```bash
gcloud run services list
```

## Current behavior summary

- Image generation tools return mixed content:
  - `render_md` (standalone markdown links)
  - `json_str` metadata
  - `ImageContent` thumbnails for tool pane rendering
- Tool pane inline rendering is reliable.
- Claude chat reply inline rendering is not guaranteed and can show **"Show Image"** consent gating.

## Why uploads look "stuck"

Most failures are from unsupported upload paths in client sandboxes (for example, trying curl/wget to inaccessible endpoints).

Preferred sequence:

1. URL already available → pass directly.
2. Local/pasted image → encode to data URI with Python, then call `upload_image`.
3. Manual fallback → open `{PUBLIC_URL}/upload` and drag-drop.

## Runtime requirements

- `PUBLIC_URL` must be set in Cloud Run for externally reachable image URLs.
- Use `S3_BUCKET` or `GCS_BUCKET` for durable image links.
- Without cloud storage, `/images/{id}` entries expire after `STORE_TTL` (default 1 hour).

## Testing

```bash
python -m pytest test_simulation.py -x -q
```

`conftest.py` handles async test execution.

## Code map

- `server.py`: FastMCP server and HTTP upload/image endpoints.
- `test_simulation.py`: unit/integration tests with mocked Gemini and real PIL/store behavior.
- `TROUBLESHOOTING.md`: incident-style debugging guide.
- `OPTIMIZATION_PLAN.md`: prioritized cleanup and architecture roadmap.
