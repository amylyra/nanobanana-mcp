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

- Image generation tools return content blocks in this order:
  1. `ImageContent` thumbnails (1024px) — visible in tool pane
  2. `render_md` — markdown image embeds + `[Download image](url)` links
  3. `json_str` — metadata for tool chaining (`image_url`, `size_kb`, etc.)
- Tool pane inline rendering is reliable.
- Claude chat reply inline rendering is not guaranteed (consent gating). Download links are the reliable fallback.

## Upload constraints

- `upload_image` has **no** Pydantic constraints (no `pattern`, no `max_length`) — this is intentional so the data URI and invalid-path error handlers can fire and return the urllib recovery snippet.
- All other image params (`edit_image`, `swap_background`, `create_variations`, `analyze_image`) use `Field(max_length=2048)` to block oversized payloads before the function body.
- **Never encode to base64/data URI** — passing large data URIs as MCP parameters hangs the transport before the server can reject them.

## Why uploads look "stuck"

Most failures are from unsupported upload paths in client sandboxes (for example, trying curl/wget to inaccessible endpoints).

Preferred sequence:

1. URL already available → pass directly to the tool.
2. Pasted/local image in claude.ai web → run the urllib POST snippet (reads `/mnt/user-data/uploads/`, POSTs raw bytes to `/upload`, gets back a URL).
3. Local file in Claude Code → `upload_image(image='/full/path/to/file.jpg')`.
4. Manual fallback → open `{PUBLIC_URL}/upload` and drag-drop.
5. Full web app → `{PUBLIC_URL}/app` for upload + generate/edit/swap/variations in one UI.

## Runtime requirements

- `PUBLIC_URL` must be set in Cloud Run for externally reachable image URLs.
- Use `S3_BUCKET` or `GCS_BUCKET` for durable image links.
- Without cloud storage, `/images/{id}` entries expire after `STORE_TTL` (default 1 hour).

## Testing

```bash
python -m pytest test_simulation.py -x -q
```

`conftest.py` handles async test execution.

## After any code change

Always update these together:
- `README.md` — keep output format, tool list, known limitations, and snippets in sync with code.
- `CLAUDE.md` — keep behavior summary, constraints, and code map accurate.
- `test_simulation.py` — patch existing tests for changed behavior; add new tests for new features.

## Code map

- `server.py`: FastMCP server, HTTP upload/image endpoints, REST API (`/api/*`), and web app (`/app`).
- `test_simulation.py`: unit/integration tests with mocked Gemini and real PIL/store behavior.
- `TROUBLESHOOTING.md`: incident-style debugging guide.
- `mcp-s3-companion/SKILL.md`: Claude skill for automatic image upload handling.
