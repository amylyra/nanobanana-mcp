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

All 134 tests must pass.

## Architecture

- `server.py` — FastMCP server (single file)
- `test_simulation.py` — unit/integration tests (mock Gemini, real PIL/store logic)
- Cloud Run serves both the MCP endpoint (`/mcp`) and a direct upload page (`/upload`)
