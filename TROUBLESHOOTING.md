# NanoBanana MCP Troubleshooting

## Scope

Use this runbook when users report:
- "Upload is stuck forever"
- "Image generated but not visible in Claude reply"
- "Tool returned URL but chain step fails"
- "Bad for loop variable" errors

---

## 1) Upload appears stuck forever

### Typical causes

1. Client tried unsupported upload path (for example curl/wget in blocked sandbox).
2. Claude passed a data URI which hangs MCP transport before the server sees it.
3. Claude ran the urllib snippet in bash instead of the Python tool.
4. `PUBLIC_URL` misconfigured (wrong service URL or old deployment URL).
5. Returned URL points to expiring in-memory store and object expired before reuse.

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
- **Never use data URIs** — they hang MCP transport. Use the urllib POST snippet or `/upload` page.
- Configure `S3_BUCKET` or `GCS_BUCKET` for durable URLs.
- Direct users to `{PUBLIC_URL}/app` for a web UI that bypasses MCP entirely.

---

## 2) "Bad for loop variable" / bash errors

### Cause

Claude ran the Python urllib snippet using the bash/shell tool instead of the Python code execution tool. Bash interprets `import` as the ImageMagick binary and fails on Python `for` syntax.

### Fix

Server instructions explicitly say "use the **Python** tool (NOT bash, NOT shell — Python only, or it will fail with 'Bad for loop variable')". Claude usually recovers on the next attempt by using `python3`.

If it keeps failing, direct the user to `{PUBLIC_URL}/upload` or `{PUBLIC_URL}/app`.

---

## 3) Claude response does not show inline image

### Expected behavior

- Tool pane shows image previews (1024px `ImageContent` thumbnails).
- Chat reply may show markdown links with "Show Image" gate instead of immediate inline display.
- Claude creates downloadable image artifacts (JPG) in the Artifacts panel.

This is claude.ai client behavior; the server can encourage but not force chat-surface rendering.

### Fast checks

1. Confirm tool output contains `render_md` + `image_url`.
2. Confirm URL is publicly accessible and returns actual image bytes.
3. Confirm at least one `ImageContent` preview is present in tool result.

### Mitigation

- `ImageContent` thumbnails (first in response) provide guaranteed visual result in tool pane.
- `render_md` with `![](url)` and `[Download image](url)` follows for Claude's chat reply.
- Server instructions tell Claude to always include both embed and download link in chat.

---

## 4) Generated URL works once, then fails in follow-up

### Cause

Using local `/images/{id}` storage without cloud backing — images can expire or be evicted.

### Fix

- Configure S3 or GCS for durable URLs.
- Increase `STORE_TTL` and `STORE_MAX_ITEMS` only if memory budget allows.

---

## 5) "Save to Drive" reply doesn't save anything

### Cause

The Save-to-Drive flow is markdown-link UX, not a clickable button. When the user replies "save", the agent must:
1. Fetch bytes from the most recent `image_url` (Python `urllib.request.urlopen`).
2. Call the Google Drive MCP's `create_file` tool with the bytes.

Common failure modes:

- **Google Drive MCP not installed.** The agent has no `create_file` tool to call. It must tell the user, not fabricate a Drive URL.
- **`urlopen` blocked / 503 on the S3 URL.** Same transient Cloud Run scaling issues as the upload path. Retry with backoff.
- **Agent skipped the carry-through step.** Server instructions tell the agent to surface the nudge in chat; without it, the user never sees the prompt and never replies "save".
- **`image_url` missing from JSON metadata.** The agent should never strip it. Confirm `_build_image_response()` populates `image_url` (S3 or `/images/<id>`).

### Fast checks

1. Confirm the user is in claude.ai (web or Cowork) — Claude Code CLI doesn't have the Google Drive MCP by default.
2. Confirm the agent's last assistant message included the **Save to Google Drive?** nudge verbatim. If not, the nudge isn't being rendered or the agent dropped it.
3. Confirm `image_url` in the most recent `json_str` resolves to fetchable bytes (`curl -I {url}`).

### Mitigation

- If Drive MCP is unavailable, fall back to the `[Download image](url)` link — this always works.
- For frequent failures, ensure `S3_BUCKET` is set so URLs are durable.
- The workflow is taught in `server.py` instructions, `mcp-s3-companion/SKILL.md` Step 4, and `CLAUDE.md` — keep all three aligned (see CLAUDE.md "Save to Google Drive" section).

---

## 6) Cowork: agent says "uploads folder empty, please re-share"

### Cause

Cowork delivers **pasted/dragged-inline images** as multimodal data in the conversation — they are NOT written to `~/Library/Application Support/Claude/local-agent-mode-sessions/<session>/uploads/`. Only files attached via the **upload/paperclip button** land in that folder.

The auto-discovery snippet (`_urllib_snippet()`) lists `*.jpg/png/...` in the uploads folder and uploads each one. If pasted inline images are the only content, the folder is empty and the snippet has nothing to send.

Pre-fix symptom: silent no-op — agent improvises an explanation about multimodal data and asks the user to re-share, which feels random.

### Fix (server side)

The snippet now prints explicit guidance when `uploads` exists but `files == []`:

> Uploads folder is empty: <path>. Pasted/dragged-inline images are NOT written to disk in Cowork — re-attach using the upload/paperclip button (which writes files to disk), or upload manually at <SERVER>/upload

### Fast checks

1. Confirm the user is in Cowork (path mentions `local-agent-mode-sessions`).
2. Confirm whether they pasted vs uploaded — only uploads land in the folder.
3. Direct them to the paperclip/upload button, or `{PUBLIC_URL}/upload` for drag-drop.

### Mitigation

- Skill copy: SKILL.md Step 1 tells the agent to repeat the empty-folder guidance verbatim instead of improvising.
- For frequent users, `{PUBLIC_URL}/app` (full web app) avoids the paste/upload distinction entirely.

---

## 7) Recommended production baseline

1. `PUBLIC_URL` set correctly.
2. `S3_BUCKET` configured (preferred) for durable assets.
3. `--min-instances=1` on Cloud Run to prevent cold-start 503s.
4. Tool response order: `[ImageContent..., render_md, json_str]`.
5. `render_md` ends with the **Save to Google Drive?** nudge.
6. Document Claude chat rendering limitation prominently.
