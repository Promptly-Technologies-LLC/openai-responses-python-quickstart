# Chat Event Loop Orchestration & HTMX OOB Swap System

This document describes how the synchronous chat event loop, SSE streaming, and HTMX out-of-band (OOB) swap system work together to deliver a real-time chat experience with tool calling support.

## Architecture Overview

The system has three layers:

1. **FastAPI backend** (`routers/chat.py`) — receives user messages, calls the OpenAI Responses API with streaming, processes events, and emits Server-Sent Events (SSE)
2. **HTMX SSE extension** (`static/sse.js`) — subscribes to the SSE stream and performs DOM swaps
3. **Client-side JavaScript** (`static/stream-md.js`) — intercepts specific SSE events for custom rendering (markdown, tool arguments, text replacements)

```
User submits message
       │
       ▼
POST /chat/{id}/send
  ├─ Upload images → OpenAI Files API
  ├─ Create user message in conversation
  └─ Return HTML: user-message + assistant-run container
       │
       ▼
HTMX appends HTML to #messages
assistant-run connects SSE → GET /chat/{id}/receive
       │
       ▼
Backend streams OpenAI events as SSE
       │
       ├─ HTMX default swap (messageCreated, toolCallCreated, toolOutput, etc.)
       ├─ JS-intercepted events (textDelta, toolDelta, textReplacement, networkError)
       └─ Control events (endStream → closes SSE, re-enables send button)
```

## Phase 1: Message Submission

**Route:** `POST /chat/{conversation_id}/send` ([chat.py:59](routers/chat.py#L59))

1. User input text and optional images arrive as form data
2. Images are uploaded to OpenAI via `client.files.create(purpose="vision")`
3. A user message item is created in the conversation via `client.conversations.items.create()`
4. The response returns two HTML fragments concatenated:
   - `components/user-message.html` — renders the user's message bubble
   - `components/assistant-run.html` — the SSE container that triggers streaming

The form's `hx-swap="beforeend"` appends both fragments to `#messages`.

## Phase 2: SSE Connection & Stream Setup

**Template:** `components/assistant-run.html` ([assistant-run.html:1](templates/components/assistant-run.html#L1))

```html
<div class="assistant-run"
     hx-target="find .dots"
     hx-swap="beforebegin"
     hx-ext="sse"
     sse-connect="/chat/{id}/receive"
     sse-swap="messageCreated,toolCallCreated,toolOutput,..."
     hx-on:htmx:sse-before-message="handleCustomSseEvents(event)"
     sse-close="endStream">
  {% include "components/loading-dots.html" %}
  {% include "components/network-error.html" %}
</div>
```

Key attributes:
- **`sse-connect`** — opens an EventSource to `GET /chat/{id}/receive`
- **`sse-swap`** — lists event names HTMX will handle; for each matching event, HTMX inserts the event's `data` HTML into the DOM
- **`hx-target="find .dots"` + `hx-swap="beforebegin"`** — default swap target is the loading dots element; new content is inserted *before* it (keeping dots at the bottom until removed)
- **`hx-on:htmx:sse-before-message`** — JavaScript hook that fires *before* HTMX processes each SSE event; allows `evt.preventDefault()` to suppress HTMX's default swap
- **`sse-close="endStream"`** — when an event named `endStream` arrives, HTMX closes the EventSource

## Phase 3: Backend Event Processing

**Route:** `GET /chat/{conversation_id}/receive` ([chat.py:110](routers/chat.py#L110))

Returns a `StreamingResponse` with `media_type="text/event-stream"`.

### Initial API Call

```python
stream = await client.responses.create(
    input="",
    conversation=conversation_id,
    model=model,
    tools=tools,
    parallel_tool_calls=False,  # ← currently serialized
    stream=True
)
```

The `input=""` with `conversation=conversation_id` tells the API to generate a response based on the full conversation history. `parallel_tool_calls=False` forces the model to make one tool call at a time.

### The `iterate_stream()` Generator

The core loop is `iterate_stream(s, response_id)` ([chat.py:218](routers/chat.py#L218)), an async generator that processes OpenAI streaming events and yields formatted SSE strings.

It uses Python's `match`/`case` to dispatch on event type:

| OpenAI Event | SSE Event Emitted | Swap Behavior |
|---|---|---|
| `ResponseCreatedEvent` | *(none)* | Records `response_id` |
| `ResponseOutputItemAddedEvent` (message) | `messageCreated` | HTMX default swap — inserts `assistant-step.html` |
| `ResponseOutputItemAddedEvent` (function_call) | `toolCallCreated` | HTMX default swap — inserts `<details>` step |
| `ResponseOutputItemAddedEvent` (mcp_call) | `toolCallCreated` | HTMX default swap |
| `ResponseOutputItemAddedEvent` (computer_call) | `toolCallCreated` | HTMX default swap |
| `ResponseOutputItemAddedEvent` (McpApprovalRequest) | `mcpApprovalRequest` | HTMX default swap — inserts approval UI card |
| `ResponseWebSearchCallSearchingEvent` | `toolCallCreated` | HTMX default swap |
| `ResponseFileSearchCallSearchingEvent` | `toolCallCreated` | HTMX default swap |
| `ResponseCodeInterpreterCallInProgressEvent` | `toolCallCreated` | HTMX default swap |
| `ResponseImageGenCallInProgressEvent` | `toolCallCreated` | HTMX default swap |
| `ResponseTextDeltaEvent` | `textDelta` | **JS-intercepted** — accumulated markdown |
| `ResponseRefusalDeltaEvent` | `textDelta` | **JS-intercepted** — same as text delta |
| `ResponseCodeInterpreterCallCodeDeltaEvent` | `toolDelta` | **JS-intercepted** — appended to `<pre>` |
| `ResponseFunctionCallArgumentsDeltaEvent` | `toolDelta` | **JS-intercepted** (if `SHOW_TOOL_CALL_DETAIL`) |
| `ResponseMcpCallArgumentsDeltaEvent` | `toolDelta` | **JS-intercepted** (if `SHOW_TOOL_CALL_DETAIL`) |
| `ResponseOutputTextAnnotationAddedEvent` | `textDelta` or `textReplacement` or `imageOutput` | Varies by annotation type |
| `ResponseOutputItemDoneEvent` (code interpreter) | `fileOutput` | HTMX default swap (OOB to `#file-carousel`) |
| `ResponseOutputItemDoneEvent` (function call) | `toolOutput` + recurse | HTMX default swap, then loop continues |
| `ResponseOutputItemDoneEvent` (image generation) | `imageOutput` | HTMX default swap |
| `ResponseOutputItemDoneEvent` (computer call) | `imageOutput` + recurse | HTMX default swap, then loop continues |
| `ResponseCompletedEvent` | `runCompleted` + `endStream` | OOB removes dots; closes SSE |

### SSE Message Format

All events use the `sse_format()` helper ([sse.py:1](utils/sse.py#L1)):

```
event: textDelta
data: <span hx-swap-oob="beforeend:#step-abc123">Hello</span>

```

Each SSE message has an `event:` line (the event name) and one or more `data:` lines (the HTML payload), terminated by a blank line.

## Phase 4: The OOB Swap Pattern

### How OOB Swaps Work

HTMX's out-of-band swap mechanism allows a single SSE event to update an element *other than* the default swap target. The pattern:

1. Backend wraps content with `wrap_for_oob_swap(step_id, content)` ([chat.py:55](routers/chat.py#L55)):
   ```python
   f'<span hx-swap-oob="beforeend:#step-{step_id}">{content}</span>'
   ```

2. When HTMX processes this HTML:
   - It detects `hx-swap-oob` on the `<span>`
   - Instead of swapping at the default target (`.dots`), it finds `#step-{step_id}` in the DOM
   - It appends the span's inner content to that element (`beforeend` mode)

3. This allows text deltas and tool argument deltas to target *specific* step containers by their unique IDs

### Two Swap Paths

**Default HTMX swaps** — Events like `messageCreated` and `toolCallCreated` carry HTML *without* `hx-swap-oob`. HTMX inserts this HTML before `.dots` (the loading indicator), creating new step containers in the conversation.

**OOB swaps** — Events like `textDelta` carry HTML *with* `hx-swap-oob="beforeend:#step-{id}"`. HTMX (or intercepting JS) routes the content to the specific step container.

### Step Container Structure

`components/assistant-step.html` ([assistant-step.html:1](templates/components/assistant-step.html#L1)) renders two variants:

**For tool calls** (`step_type='toolCall'`):
```html
<details class="toolCall" id="step-outer-{step_id}">
  <summary>Calling {tool_name} tool...</summary>
  <div id="step-{step_id}" class="toolCallDetails"></div>
</details>
```
The inner `div#step-{step_id}` is the OOB target for streaming tool arguments.

**For messages** (`step_type='assistantMessage'`):
```html
<div class="assistantMessage" id="step-{step_id}"></div>
```
This `div#step-{step_id}` is the OOB target for streaming text deltas.

## Phase 5: Client-Side JavaScript Event Handling

**File:** `static/stream-md.js` ([stream-md.js:1](static/stream-md.js#L1))

The `handleCustomSseEvents()` function ([stream-md.js:6](static/stream-md.js#L6)) fires for every SSE event via the `hx-on:htmx:sse-before-message` attribute. It selectively intercepts events:

### `textDelta` — Progressive Markdown Rendering

1. `evt.preventDefault()` suppresses HTMX's default swap
2. `parseOobSwap()` extracts the target element ID and markdown chunk from the OOB HTML
3. The chunk is appended to a `WeakMap` (`window._streamingMarkdown`) keyed by the target DOM element
4. `renderMarkdown()` re-renders the full accumulated markdown using `marked.parse()` + `DOMPurify.sanitize()`
5. Links are post-processed to open in new tabs
6. Auto-scroll fires if the user is already at the bottom

### `toolDelta` — Streaming Tool Arguments

1. `evt.preventDefault()` suppresses HTMX's default swap
2. If the payload contains `data-tool-delta="replace"`, the target's children are replaced entirely (used for final complete arguments)
3. If `data-tool-delta="stream"`, content is appended to a `<pre>` element (used for streaming code interpreter code)

### `textReplacement` — Sandbox Path Resolution

1. `evt.preventDefault()` suppresses HTMX's default swap
2. Payload format: `sandbox:/path/to/file|/actual/download/url`
3. The sandbox path in the accumulated markdown is replaced with the real URL
4. Markdown is re-rendered to activate the new link

### Events NOT intercepted

`messageCreated`, `toolCallCreated`, `toolOutput`, `mcpApprovalRequest`, `imageOutput`, `fileOutput`, `runCompleted` — these pass through to HTMX's default swap, which inserts their HTML before `.dots`.

### `endStream` — Stream Termination

Not intercepted by JS. The `sse-close="endStream"` attribute on the assistant-run container tells HTMX to close the EventSource. The `hx-on::sse-close` attribute then calls `reEnableSendButton()`.

## Phase 6: The Tool Call Continuation Loop

This is the key synchronous orchestration pattern. When a tool call completes, the backend submits the result and *restarts the stream* within the same SSE connection:

### For Custom Functions ([chat.py:474-539](routers/chat.py#L474-L539)):

```
Stream event: ResponseOutputItemDoneEvent (function_call)
  │
  ├─ Execute function via FUNCTION_REGISTRY.call()
  ├─ Yield toolOutput SSE event (renders result in UI)
  ├─ Fetch conversation items to find call_id
  ├─ Create function_call_output item in conversation
  ├─ Call client.responses.create() again (new stream)
  └─ Recursively yield from iterate_stream(next_stream)
       │
       ├─ Model may generate text → textDelta events
       ├─ Model may call another tool → repeat loop
       └─ Model completes → ResponseCompletedEvent → endStream
```

### For Computer Use ([chat.py:553-609](routers/chat.py#L553-L609)):

Same pattern: execute actions → capture screenshot → submit `computer_call_output` → restart stream → recurse.

### Why Recursive?

The recursive `iterate_stream()` call means the outer SSE connection stays open while multiple OpenAI API calls happen sequentially. From the client's perspective, it's one continuous stream. The model can chain multiple tool calls (function → function → text response) without the client needing to reconnect.

### Current Limitation: Sequential Tool Calls

`parallel_tool_calls=False` ([chat.py:208](routers/chat.py#L208)) forces the model to emit one tool call at a time. The continuation loop handles exactly one `ResponseOutputItemDoneEvent` (function/computer call) before restarting the stream. To support parallel tool calls, this loop would need to:

1. Collect *all* tool call results from a single response
2. Submit all outputs together
3. Then restart the stream once

## SSE Event Summary

| SSE Event Name | Source | Client Handling | Purpose |
|---|---|---|---|
| `messageCreated` | New assistant message | HTMX default swap | Creates message container |
| `toolCallCreated` | Tool call started | HTMX default swap | Creates `<details>` element |
| `textDelta` | Text token | JS `processTextDelta()` | Streams markdown into message |
| `toolDelta` | Tool arguments/code | JS `processToolDelta()` | Streams into tool details |
| `textReplacement` | File path resolved | JS `processTextReplacement()` | Fixes sandbox URLs |
| `toolOutput` | Function result | HTMX default swap | Shows tool result HTML |
| `imageOutput` | Generated/captured image | HTMX default swap | Shows inline image |
| `fileOutput` | Code interpreter files | HTMX default swap (OOB) | Updates file carousel |
| `mcpApprovalRequest` | MCP needs approval | HTMX default swap | Shows approve/reject UI |
| `networkError` | Stream/API error | JS `showNetworkError()` | Shows error banner |
| `runCompleted` | Response done | HTMX default swap (OOB) | Removes loading dots |
| `endStream` | All done | `sse-close` attribute | Closes EventSource |

## Key Files

| File | Role |
|---|---|
| [routers/chat.py](routers/chat.py) | Chat routes: send, receive (SSE stream), approve |
| [utils/sse.py](utils/sse.py) | `sse_format()` — formats event name + data into SSE wire format |
| [utils/function_calling.py](utils/function_calling.py) | `ToolRegistry` — registers and dispatches custom functions |
| [utils/function_definitions.py](utils/function_definitions.py) | Introspects function signatures to build OpenAI tool schemas |
| [utils/computer_use.py](utils/computer_use.py) | Browser session management and action execution |
| [static/stream-md.js](static/stream-md.js) | Client-side SSE event handling and markdown rendering |
| [static/sse.js](static/sse.js) | HTMX SSE extension (EventSource management) |
| [templates/components/assistant-run.html](templates/components/assistant-run.html) | SSE container with swap configuration |
| [templates/components/assistant-step.html](templates/components/assistant-step.html) | Step container (message or tool call) |
