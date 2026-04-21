# Attempts to display generated images in Claude's chat response

## The problem

When NanoBanana generates an image, there are two places the image could appear in claude.ai:

1. **Tool result pane** — the collapsible block that shows what a tool returned (below the tool call)
2. **Assistant chat response** — Claude's actual reply text, which the user reads

The goal is **#2**: the image appears in Claude's visible reply, not just buried in the tool result pane. Every attempt below is trying to solve this.

---

## Attempt 1 — Instruct Claude via docstring/system prompt to output markdown

**Commit:** `e04520e` — *"Instruct Claude to render images inline via markdown in its reply"*

**What we tried:**  
Added text to tool docstrings and MCP server instructions telling Claude to render the `image_url` as a markdown image in its reply: `![](url)`.

**Result:** Unreliable. Claude sometimes included the markdown, sometimes just described the image in text. No enforcement mechanism — Claude treats it as a soft suggestion.

---

## Attempt 2 — `render_markdown` field in JSON + TextContent as first return item

**Commits:** `470c647`, `ff9897a`, `e7370d9`

**What we tried:**  
Added a `render_markdown` field to the JSON metadata (`![generated image](url)`). Then changed the tool return to a list where the **first item** was a `TextContent` string containing the pre-formatted markdown — `[render_md_str, json_str]` — hoping Claude would echo it.

**Result:** Claude ignored the `render_md` TextContent item. MCP tool result `TextContent` blocks are not automatically echoed into the assistant response; Claude decides what to say independently.

---

## Attempt 3 — `data_uri` field in JSON (base64 thumbnail embedded in metadata)

**Commits:** `6009f6b`, related work in `test_simulation.py`

**What we tried:**  
Generated a 512px JPEG thumbnail of the image, base64-encoded it, and embedded it in the JSON metadata as a `data_uri` field. Instructed Claude to render it as `![](data:image/jpeg;base64,...)`.

**Result:** Two failures:
- Claude naturally preferred `image_url` (the HTTP URL) over `data_uri` and used that for its markdown instead.
- Even when Claude used the data URI, claude.ai **does not render `data:` URLs** in markdown — it outputs the raw base64 text.

---

## Attempt 4 — `inline_image` field (pre-formatted markdown with data URI)

**Related to Attempt 3**

**What we tried:**  
Renamed `data_uri` to `inline_image` and pre-formatted it as `![](data:image/jpeg;base64,...)` so Claude could paste it verbatim without constructing the markdown itself.

**Result:** Same two failures as Attempt 3 — data URIs don't render in claude.ai, and Claude truncated the base64 string mid-output on long images (thousands of tokens).

---

## Attempt 5 — Return `MCPImage` / `ImageContent` objects from the tool

**Commit:** `b9edd0c` — *"Return MCPImage objects for reliable inline image display in claude.ai"*

**What we tried:**  
Changed the tool return type to include FastMCP `Image` objects directly: `[json_str, Image(data=thumb, format="jpeg"), ...]`. FastMCP converts these to MCP `ImageContent` protocol blocks. These render inline in **the tool result pane** in claude.ai.

**Result:** Partial success. Images are visible in the tool result pane — but **not** in Claude's chat response. Users see them only if they expand the tool result block. Claude's text reply still just describes the image in words.

This approach was kept as a baseline because it's the only reliable way to show images at all.

---

## Attempt 6 — Images-first ordering `[Image, json_str]`

**Commit:** `6f93db5` (PR #4) — *"Fix MCP image block ordering for Claude inline rendering"*

**What we tried:**  
Reversed the list order to put `ImageContent` blocks first: `[Image(thumb), json_str]`. Hypothesis: clients that prioritise the first content block might render the image.

**Result:** No visible difference in claude.ai behaviour. The image still appeared only in the tool result pane, not in Claude's reply.

---

## Attempt 7 — Full-size images + `render_markdown` / `markdown` fields

**Commit:** `5e1347a` (PR #6) — *"Use full-size image previews and add explicit render markdown metadata"*

**What we tried:**  
- Switched from 512px thumbnails to **full-size JPEG bytes** in `ImageContent` (hypothesis: higher quality might improve inline rendering).
- Added per-image `markdown` field to JSON: `![generated image](url)`.
- Added `render_hint` ("Include the markdown below in your reply...") and `render_markdown` (the joined markdown string) to JSON.
- Updated MCP instructions to tell Claude to include those markdown snippets in its reply.

**Result:** Unverified in production. The approach was superseded before thorough testing. Full-size images also significantly increase the payload per tool call.

---

## Attempt 8 — `display_markdown` + `assistant_response_template` + deterministic instructions

**Commit:** `6cc46d6` — *"Apply display_markdown patch: thumbnails + JSON-first + deterministic render instructions"*

**What we tried:**  
- Reverted to 512px thumbnails.
- Reverted to JSON-first ordering `[json_str, Image, ...]`.
- Replaced `render_markdown`/`render_hint`/`markdown` with two new fields:
  - `display_markdown`: `![generated image](image_url)` — the exact string Claude should paste.
  - `assistant_response_template`: full suggested reply ("Done. Here is the generated image:\n![...]").
- Updated MCP instructions with a numbered, deterministic format:
  > "After any image tool call, your assistant response MUST include a Markdown image using the returned `image_url`. Use this format: 1) One short sentence summary. 2) The `display_markdown` value exactly as returned."

**Result:** ❌ Failed. Claude still did not reliably include the image in its chat response despite the stricter numbered format instructions and the pre-formatted `display_markdown` field.

---

## Attempt 9 — Single text block: markdown first, JSON second (`response_mode: "deterministic_markdown"`)

**What we tried:**
Replaced the mixed `[json_str, Image(thumbnail), ...]` return with a **single text block**:

```
![generated image](https://bucket.s3.amazonaws.com/gen/uuid.jpg)

{"response_mode": "deterministic_markdown", "image_url": "...", ...}
```

Key differences from all prior attempts:
- No `ImageContent` objects at all — the hypothesis is that Claude sees `ImageContent` rendered in the tool pane and considers the image "already shown", so it doesn't include it in its reply. Removing `ImageContent` forces Claude to emit the markdown to show the image at all.
- The markdown image link is the **first thing in the string** (not buried in a JSON field like `display_markdown` was), making it the obvious primary content of the tool result.
- A single, unambiguous text block removes the "mixed content" ambiguity of prior attempts.
- Added `response_mode: "deterministic_markdown"` to signal the format to clients.

**Result:** ⏳ Not yet verified in production.

---

## Root cause analysis

The fundamental issue is a **two-surface rendering problem**:

| Surface | How images appear | Controllable? |
|---|---|---|
| Tool result pane | `ImageContent` blocks render inline automatically | Yes — return `Image` objects |
| Assistant chat response | Only if Claude outputs `![](url)` in its text | Indirectly — depends on Claude following instructions |

The tool result pane is solved (Attempt 5). The chat response surface requires Claude to voluntarily emit markdown. We cannot force this — we can only instruct Claude via:
- MCP server `instructions` (loaded once at session start)
- Tool docstrings (visible to Claude when it decides to call the tool)
- Fields in the JSON metadata (visible after the tool returns)

So far no combination has been reliably effective. The open question is whether `display_markdown` + `assistant_response_template` (Attempt 8) with a stricter instruction format will hold.

---

## Things not yet tried

- **MCP Prompts** — define an MCP prompt that Claude runs after every image generation, forcing a structured reply. The server already has a `handle_image` prompt but it's for upload UX, not display.
- **Structured output with a `reply` field** — return a top-level `reply` string from the tool that MCP surfaces as a suggested assistant turn (not currently part of MCP spec).
- **Client-side rendering** — build a Claude Code extension or custom client that intercepts `ImageContent` and injects it into the visible response.
- **Webhook / side-channel** — have the server push a rendered HTML page to a browser tab when an image is generated, sidestepping the claude.ai response surface entirely.
