# NanoBanana MCP Troubleshooting

## Scope

Use this runbook when users report:
- "Upload is stuck forever"
- "Image generated but not visible in Claude reply"
- "Tool returned URL but chain step fails"

---

## 1) Upload appears stuck forever

### Typical causes

1. Client tried unsupported upload path (for example curl/wget in blocked sandbox).
2. `PUBLIC_URL` misconfigured (wrong service URL or old deployment URL).
3. `/upload` endpoint reachable internally but not publicly.
4. Returned URL points to expiring in-memory store and object expired before reuse.

### Fast checks

1. Open `{PUBLIC_URL}/upload` in browser.
   - If page does not load: service/URL config issue.
2. Upload one image manually.
   - Verify JSON response includes `url`.
3. Open returned `url` in browser.
   - If not reachable, inspect Cloud Run ingress and URL mapping.
4. If using in-memory store, confirm reuse happens within TTL (`STORE_TTL`, default 3600s).

### Fixes

- Set exact Cloud Run URL in `PUBLIC_URL`.
- Prefer data URI -> `upload_image` flow for pasted/local images.
- Configure `S3_BUCKET` or `GCS_BUCKET` for durable URLs.

---

## 2) Claude response does not show inline image

### Expected behavior today

- Tool pane should show image previews (from `ImageContent`).
- Chat reply may show markdown links with **"Show Image"** gate instead of immediate inline display.

This is currently client behavior; server can strongly encourage but not force chat-surface rendering.

### Fast checks

1. Confirm tool output contains `render_md` + `image_url`.
2. Confirm URL is publicly accessible and returns actual image bytes.
3. Confirm at least one `ImageContent` preview is present in tool result pane.

### Mitigation strategy

- Keep returning `ImageContent` previews for guaranteed visual result.
- Also return `render_md` first so Claude can copy into reply when possible.
- Set user expectation in docs/UI that chat may require click-to-reveal.

---

## 3) Generated URL works once, then fails in follow-up

### Cause

Using local `/images/{id}` storage without cloud backing can expire or be evicted.

### Fix

- Configure S3 or GCS for durable URLs.
- Increase `STORE_TTL` and `STORE_MAX_ITEMS` only if memory budget allows.

---

## 4) Recommended production baseline

1. `PUBLIC_URL` set correctly.
2. `S3_BUCKET` configured (preferred) for durable assets.
3. Keep current mixed output contract: `[render_md, json_str, ImageContent...]`.
4. Document Claude chat rendering limitation prominently.

