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

## Intake before any image-output tool

Image-output tools: `generate_image`, `edit_image`, `swap_background`, `create_variations`. The agent must confirm aspect ratio (and resolution where supported) with the user before the first call to any of them.

Single-message intake:
- Aspect ratio: `1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9`
- Resolution: `1K, 2K, 4K`
- Defaults: `4:5 / 1K` for new generations; `same shape as source` for edits/swaps/variations

Per-tool support (mirror in docstrings + server instructions):

| Tool | aspect_ratio | resolution | Default |
|---|---|---|---|
| `generate_image` | configurable | configurable | 4:5 / 1K |
| `create_variations` | configurable | configurable | source shape / 1K |
| `edit_image` | configurable | **not supported** (Gemini API limitation) | source shape |
| `swap_background` | configurable | **not supported** (Gemini API limitation) | source shape |

Lives in four places — all must stay aligned:
- Server `instructions` block (`## Intake before any image-output tool — REQUIRED`)
- Each tool's docstring (`INTAKE REQUIRED:` paragraph + per-arg notes)
- `mcp-s3-companion/SKILL.md` (`Step 2 — Confirm aspect ratio BEFORE any image-output tool`)
- This section

The skill copy is critical: claude.ai web loads the skill before the MCP, and the skill's directive workflow can override MCP intake guidance if it doesn't mention intake itself.

Skip cases (no intake needed): user already named values; re-running at known settings; chaining from a fixed-settings tool.

## Upload constraints

- `upload_image` has **no** Pydantic constraints (no `pattern`, no `max_length`) — this is intentional so the data URI and invalid-path error handlers can fire and return the urllib recovery snippet.
- All other image params (`edit_image`, `swap_background`, `create_variations`, `analyze_image`) use `Field(max_length=2048)` to block oversized payloads before the function body.
- **Never encode to base64/data URI** — passing large data URIs as MCP parameters hangs the transport before the server can reject them.

## Upload paths by environment

The upload path depends on **where the MCP is running** and **where the file lives**. The remote Cloud Run server can't read the user's local filesystem, so each client has its own snippet.

| Environment | Tool | File source | Path |
|---|---|---|---|
| claude.ai web (co-work) | Python | `/mnt/user-data/uploads/` | urllib POST snippet → `/upload` |
| Claude Code (CLI) — remote MCP | Bash | user's real filesystem | `curl --data-binary "@/path"` → `/upload` |
| Claude Code (CLI) — local stdio MCP | n/a | user's real filesystem | `upload_image(image='/path')` reads it server-side |
| Any | n/a | n/a | `{PUBLIC_URL}/upload` (manual) or `{PUBLIC_URL}/app` (full UI) |

Helpers in `server.py`:

- `_urllib_snippet()` — claude.ai web Python snippet. Reads most recent files from `/mnt/user-data/uploads/`.
- `_claude_code_snippet(path=...)` — Claude Code Bash curl one-liner. Substitutes the user-provided path when available so the agent runs it verbatim.

Both are returned together in `upload_image` data-URI / not-found errors so the agent picks the one matching its environment.

Most "stuck" uploads come from picking the wrong path (e.g. running the Python snippet under Bash, or expecting Cloud Run to see a `/Users/...` path).

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
