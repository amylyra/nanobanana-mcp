# NanoBanana MCP — Operator Notes

## Deploy target

Always deploy to service `nanobanana` in `us-central1`.

```bash
gcloud run deploy nanobanana --source . --region us-central1 --min-instances=1
```

`--min-instances=1` is required. Without it, cold-start 503s break the urllib upload path.

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

1. URL already available → pass directly to the tool.
2. Pasted/local image in claude.ai web → run the urllib POST snippet (reads `/mnt/user-data/uploads/`, POSTs raw bytes to `/upload`, gets back a URL).
3. Local file in Claude Code → `upload_image(image='/full/path/to/file.jpg')`.
4. Manual fallback → open `{PUBLIC_URL}/upload` and drag-drop.

**Never encode to base64/data URI** — passing large data URIs as MCP parameters hangs the transport before the server can reject them. The `Field(pattern=r"^(https?://|/)")` constraint blocks data URIs client-side.

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
