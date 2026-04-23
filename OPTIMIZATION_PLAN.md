# NanoBanana MCP Optimization & Cleanup Plan

This plan prioritizes reliability first, then performance, then maintainability.

## Priority 0 — Reliability (do first)

1. **Clarify supported upload flows everywhere**
   - Keep one canonical path: URL direct OR Python data URI -> `upload_image` OR `/upload` page.
   - Remove/avoid instructions that recommend curl/wget in restricted environments.

2. **Strengthen URL durability defaults**
   - Prefer S3/GCS in production.
   - Make expiration behavior explicit in all responses when local store is used.

3. **Add response-level diagnostics**
   - Include `storage_backend` field (`local`, `s3`, `gcs`).
   - Include `expires_at` when local store is used.

## Priority 1 — User experience

4. **Improve upload endpoint UX**
   - Return richer error JSON with actionable messages.
   - Add max file size + accepted MIME hints in `/upload` page.

5. **Add deterministic response helper text**
   - Keep `render_md` first.
   - Add short `display_note` field: "If Claude shows Show Image, click to reveal." 

6. **Publish explicit client limitations**
   - Track chat-surface image behavior as client limitation, not server bug.

## Priority 2 — Performance

7. **Parallelize cloud upload + thumbnail generation**
   - For multi-image outputs, do thumbnail + upload concurrently with bounded workers.

8. **Cache common conversions**
   - Reuse decoded image object when generating thumbnail + upload payload.

9. **Metrics and timing**
   - Capture per-stage latency: fetch/decode/model/upload/serialize.

## Priority 3 — Code quality and maintainability

10. **Split `server.py` into modules**
    - `storage.py`, `image_io.py`, `tools_generation.py`, `tools_analysis.py`, `transport.py`.

11. **Centralize schema contracts**
    - Typed response model for all image tools.

12. **Expand tests**
    - Add regression tests for:
      - upload failure messaging
      - storage fallback behavior
      - response contract ordering

## Suggested execution order (2-week sprint)

- **Days 1–2:** Docs and instruction cleanup (done in this change set).
- **Days 3–5:** Add response diagnostics + upload error improvements.
- **Days 6–8:** Add storage durability defaults + env validation checks.
- **Days 9–10:** Performance instrumentation + concurrency for multi-image post-processing.

## Success criteria

- Upload-related support tickets reduced by >=50%.
- Zero ambiguity in docs about supported upload paths.
- >=95% of generated `image_url` values durable in production deployments.
- Median tool end-to-end latency improved by >=20% for `count > 1` image calls.

