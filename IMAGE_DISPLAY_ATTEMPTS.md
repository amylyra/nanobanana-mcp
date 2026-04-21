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

**Result:** ❌ Failed — and worse than prior attempts in a new way. The tool result pane rendered the raw markdown text literally (`![image 1](url)`) instead of as a rendered image. Claude then wrote "Two candidates above" pointing at the unrendered text in the pane. Two regressions at once:
1. The tool pane, which previously showed `ImageContent` previews, now showed raw unstyled text.
2. Claude's chat response still contained no images.

Root cause: claude.ai's tool result pane does not render markdown in text content blocks. `ImageContent` was the only way to get a rendered preview there; removing it made the pane worse without improving the chat response.

---

## Root cause analysis

The fundamental issue is a **two-surface rendering problem**:

| Surface | How images appear | Controllable? |
|---|---|---|
| Tool result pane | `ImageContent` blocks render inline automatically | Yes — return `Image` objects |
| Assistant chat response | Only if Claude outputs `![](url)` in its text | No — Claude decides independently |

**New insight from Attempt 9:** claude.ai's tool result pane does **not** render markdown in text content blocks — it displays them as raw monospaced text. `ImageContent` is the *only* mechanism that produces rendered image previews in the tool pane. Removing `ImageContent` made the pane worse without any benefit to the chat response.

**New insight from Attempt 10:** claude.ai applies a **"Show Image" consent gate** to externally-linked images when they appear in Claude's text (`![](url)` markdown). This gate does NOT apply to `ImageContent` protocol blocks, which render inline automatically. So even when Claude successfully includes `![](url)` in its reply, the user still has to click "Show Image" rather than seeing the image inline. True inline rendering in the chat response may only be possible via `ImageContent`, which Claude cannot emit directly — only tools can return `ImageContent`.

**Current state (Attempts 1–10 exhausted):**
- Tool result pane: ✅ solved — `ImageContent` thumbnails render reliably (no consent gate)
- Chat response inline: ❌ fundamentally blocked — `![](url)` text triggers consent gate; `ImageContent` in chat response is not a feature Claude can emit
- Chat response text link: ✅ achievable — Claude can include the URL or markdown, but user must click "Show Image"

The chat response surface requires Claude to voluntarily emit markdown. We can instruct but not enforce via:
- MCP server `instructions` (loaded once at session start)
- Tool docstrings (visible when Claude decides to call the tool)
- Fields in the JSON metadata (`display_markdown`, `assistant_response_template`)

None of these have been reliable across conversations.

---

## Attempt 10 — Revive `ff9897a`: standalone render_md first + ImageContent

**What we tried:**  
Returned to the `ff9897a` approach, which the user confirmed worked in production at the time. Key structure:

```
[render_md, json_str, Image(thumb1), Image(thumb2), ...]
```

- `render_md` = `"![](url)"` (single) or `"![Image 1](url1)\n\n![Image 2](url2)"` (multi) — a short, standalone TextContent block containing *only* the markdown image link(s), nothing else.
- `json_str` = JSON metadata for tool chaining.
- `Image` objects = 512px JPEG thumbnails for the tool pane (ImageContent).

Key differences from prior attempts:
- Unlike Attempt 8 (`display_markdown` buried in JSON): render_md is its own separate TextContent block, not a field inside the JSON.
- Unlike Attempt 9 (combined string, no ImageContent): ImageContent is still present for the tool pane, and render_md is a clean, short, standalone string.
- Unlike Attempt 2 (`[render_md, json]` no Images): ImageContent objects are also returned alongside render_md.

Instructions updated to: "Every image tool result begins with a standalone markdown image link. Copy that value verbatim into your reply."

**Result:** ❌ Failed — but with a new insight. Claude DID include the markdown image link in its chat response (visible in the conversation), but claude.ai rendered it as **"Show Image" placeholder boxes** rather than inline images. Clicking "Show Image" loads the image behind a user-consent gate. This is better than Claude ignoring the link, but worse than true inline rendering.

**New insight from Attempt 10:** claude.ai gating may be tied to the S3 URL domain. When `ImageContent` is returned, claude.ai renders the image directly (no consent gate). When the URL appears in Claude's text as `![](url)`, claude.ai shows "Show Image" and requires a click. This suggests the consent gate is applied to externally-linked images in text but not to `ImageContent` protocol blocks.

**User preference:** Attempt 9 (raw markdown text in tool result pane, URLs visible and copyable) is preferred over Attempt 10's "Show Image" boxes, which require an extra click and don't auto-render.

---

## Attempt 11 — Two separate TextContent blocks: `[render_md, json_str]`

**What we tried:**
Reverted from Attempt 9's combined single string to two *separate* TextContent blocks:

```
[render_md, json_str]
```

- `render_md` = `"![](url)"` (single) or `"![Image 1](url1)\n\n![Image 2](url2)"` (multi) — a clean, standalone TextContent block containing *only* markdown image links.
- `json_str` = JSON metadata for tool chaining.

Key differences from Attempt 9 (combined string `"![](url)\n\n{json}"`):
- render_md and json_str are two separate blocks, not merged into one.
- The tool result pane shows both as separate text boxes.
- The MCP instruction tells Claude: "Every image tool result begins with a standalone markdown image link. Copy that value verbatim into your reply."

This matches what commit `09a18bb` produced — the version the user confirmed showed "Show Image" boxes in Claude's chat response (preferred over the text-box-only Attempt 9 result).

**Result:** ❌ Failed — same outcome as prior attempts. Claude did not reliably include the markdown in its reply. Reverted to Attempt 9 (single combined text block) as the stable baseline.

---

## Things not yet tried

- **MCP Prompts** — define an MCP prompt that Claude runs after every image generation, forcing a structured reply. Distinct from `instructions`; a named prompt the client actively invokes.
- **Structured output with a `reply` field** — return a top-level `reply` string from the tool that MCP surfaces as a suggested assistant turn (not currently part of MCP spec).
- **Client-side rendering** — build a Claude Code extension or custom client that intercepts `ImageContent` and injects it into the visible response.
- **Webhook / side-channel** — have the server push a rendered HTML page to a browser tab when an image is generated, sidestepping the claude.ai response surface entirely.
