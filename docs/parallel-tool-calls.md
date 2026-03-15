# Parallel Tool Calls Implementation Plan

**Issue:** [#8 — Allow parallel function calls](https://github.com/Promptly-Technologies-LLC/openai-responses-python-quickstart/issues/8)

**Goal:** Allow the model to emit multiple tool calls in a single response, execute them concurrently, submit all outputs together, and restart the stream once.

## Current Behavior

Today, `parallel_tool_calls=False` forces the model to emit one tool call per response. When a tool call completes, the handler immediately submits the output and recursively calls `iterate_stream()` with a new stream. This creates a sequential chain:

```
Response 1: function_call A → execute A → submit A → Response 2: function_call B → execute B → submit B → Response 3: text
```

See [chat-orchestration.md](chat-orchestration.md) for full details on the current event loop.

## Design

### Concurrency Model: Spawn-Early, Emit-Late

Spawn tool execution tasks as soon as each `ResponseOutputItemDoneEvent` arrives. Gather all results when `ResponseCompletedEvent` fires. Emit SSE events and submit outputs after gathering.

This is the simplest model that still achieves real parallelism between tool executions:

```
Response 1: function_call A, function_call B, function_call C
  │
  ├─ OutputItemDone(A) → spawn task_A
  ├─ OutputItemDone(B) → spawn task_B
  ├─ OutputItemDone(C) → spawn task_C
  │
  └─ ResponseCompleted →
       gather(task_A, task_B, task_C)
       emit SSE events for A, B, C
       submit all outputs in one API call
       start Response 2
```

**Tradeoff:** Tool output SSE events are delayed until *all* tasks complete. If one tool is slow and others are fast, all outputs are held until the slowest finishes. The UI shows loading dots during this time, so it doesn't appear broken, but the user won't see partial results. This is acceptable for the initial implementation because:
- Tool executions are typically fast (custom functions are local; computer use is bounded)
- The loading dots remain visible during execution, so the UI doesn't appear frozen
- An alternative queue-based model that emits results as each task completes would add significant complexity (merging two async sources into a single generator) for marginal UX benefit

If slow tools become common in practice, the queue-based model described in the original proposal can be revisited as a follow-on enhancement.

### Data Structures

```python
@dataclass
class ToolTaskResult:
    """Result of a single tool execution task."""
    sse_events: list[tuple[str, str]]  # [(event_name, html_payload), ...]
    output_item: dict                   # payload for conversations.items.create
```

Within `iterate_stream()`, use separate, explicitly-typed collections for function tasks (concurrent) and computer coroutines (sequential):

```python
pending_fn_tasks: dict[str, asyncio.Task[ToolTaskResult]] = {}    # keyed by item_id; run concurrently
pending_computer_coros: dict[str, Coroutine] = {}                  # keyed by item_id; awaited sequentially
result_call_ids: dict[str, str] = {}                               # item_id → call_id (for error output submission)
has_approval_request: bool = False
```

Keeping function tasks and computer coroutines in separate, explicitly-typed collections avoids a heterogeneous collection and makes the `ResponseCompletedEvent` handler straightforward. The `result_call_ids` mapping is needed to submit failure output items when a task raises an exception (see "Error Handling" below).

### Event Handler Changes

#### `ResponseOutputItemDoneEvent` — Function Calls

Replace the current inline execute-submit-recurse block with task spawning:

```python
elif isinstance(event.item, ResponseFunctionToolCall):
    current_item_id = event.item.id
    call_id = event.item.call_id  # ← use directly, no API round-trip
    function_name = event.item.name
    arguments_json = json.loads(event.item.arguments)

    # Emit complete arguments into the collapsible details (immediate)
    yield sse_format("toolDelta", wrap_for_oob_swap(...))

    # Spawn execution task
    async def run_function(name, args, cid, fn_registry, tpl_registry):
        result = await fn_registry.call(name, args, context=Context())
        # Build SSE events
        sse_events = []
        try:
            if name in tpl_registry:
                tpl = tpl_registry[name]
                if isinstance(tpl, tuple):
                    tpl_name, context_builder = tpl
                    html = templates.get_template(tpl_name).render(**context_builder(result))
                else:
                    html = templates.get_template(tpl).render(tool=result)
                sse_events.append(("toolOutput", html))
            else:
                payload = result.model_dump(exclude_none=True)
                sse_events.append(("toolOutput", f"<pre>{json.dumps(payload, indent=2)}</pre>"))
        except Exception as e:
            logger.error(f"Error rendering tool output for '{name}': {e}")
            sse_events.append(("toolOutput", f"<pre>{json.dumps(result.model_dump(exclude_none=True))}</pre>"))
        # Build output item
        output_item = {
            "type": "function_call_output",
            "call_id": cid,
            "output": json.dumps(result.model_dump(exclude_none=True))
        }
        return ToolTaskResult(sse_events=sse_events, output_item=output_item)

    task = asyncio.create_task(
        run_function(function_name, arguments_json, call_id,
                     FUNCTION_REGISTRY, TEMPLATE_REGISTRY)
    )
    pending_fn_tasks[current_item_id] = task
    result_call_ids[current_item_id] = call_id
```

#### `ResponseOutputItemDoneEvent` — Computer Calls

Store a coroutine (unawaited) that executes actions, captures the screenshot, and returns a `ToolTaskResult` with `imageOutput` SSE event and `computer_call_output` item. Do **not** call `asyncio.create_task()` — these are awaited sequentially during `ResponseCompletedEvent` to avoid browser session race conditions (see "Computer Use" section below).

```python
elif isinstance(event.item, ResponseComputerToolCall):
    current_item_id = event.item.id
    call_id = event.item.call_id
    # ... emit toolDelta with action description (immediate) ...

    # Store coroutine for sequential execution later
    pending_computer_coros[current_item_id] = run_computer(
        event.item.actions, call_id,
        event.item.pending_safety_checks, conversation_id
    )
    result_call_ids[current_item_id] = call_id
```

#### `ResponseOutputItemAddedEvent` — MCP Approval Requests

Set `has_approval_request = True` when an `McpApprovalRequest` is detected (see "MCP Approval" section below).

#### `ResponseCompletedEvent` — Gather, Emit, Submit, Continue

Replace the current simple "emit runCompleted + endStream" with:

```python
case ResponseCompletedEvent():
    has_pending = bool(pending_fn_tasks or pending_computer_coros)

    if has_pending:
        # Track original insertion order for UI consistency
        all_item_ids = list(pending_fn_tasks.keys()) + list(pending_computer_coros.keys())

        # Run function tasks concurrently
        fn_results: dict[str, ToolTaskResult | Exception] = {}
        if pending_fn_tasks:
            gathered = await asyncio.gather(
                *pending_fn_tasks.values(),
                return_exceptions=True
            )
            fn_results = dict(zip(pending_fn_tasks.keys(), gathered))
        pending_fn_tasks.clear()

        # Run computer coroutines sequentially
        computer_results: dict[str, ToolTaskResult | Exception] = {}
        for item_id, coro in pending_computer_coros.items():
            try:
                computer_results[item_id] = await coro
            except Exception as e:
                computer_results[item_id] = e
        pending_computer_coros.clear()

        # Merge results and emit/collect in original call order
        all_results = {**fn_results, **computer_results}
        output_items = []
        for item_id in all_item_ids:
            result = all_results[item_id]
            if isinstance(result, Exception):
                logger.error(f"Tool task {item_id} failed: {result}")
                # Emit visible error in the tool's UI step
                yield sse_format("toolOutput",
                    f'<pre>Error: {escape(str(result))}</pre>')
                # Submit failure output so the model knows the call failed
                # (use the call_id stored when the task/coro was created)
                output_items.append({
                    "type": "function_call_output",
                    "call_id": result_call_ids[item_id],
                    "output": json.dumps({"error": str(result)})
                })
                continue
            for event_name, html in result.sse_events:
                yield sse_format(event_name, html)
            output_items.append(result.output_item)

        # Submit all outputs in one call
        if output_items:
            await client.conversations.items.create(
                conversation_id=conversation_id,
                items=output_items
            )

            if has_approval_request:
                # Don't restart stream — approval flow handles continuation
                yield sse_format("runCompleted", '<span hx-swap-oob="outerHTML:.dots"></span>')
                yield sse_format("endStream", "DONE")
                return
            else:
                # Restart stream once
                next_stream = await client.responses.create(
                    input="",
                    conversation=conversation_id,
                    model=model,
                    tools=tools or NOT_GIVEN,
                    instructions=instructions,
                    parallel_tool_calls=True,
                    stream=True
                )
                async for out in iterate_stream(next_stream, response_id):
                    yield out
    else:
        yield sse_format("runCompleted", '<span hx-swap-oob="outerHTML:.dots"></span>')
        yield sse_format("endStream", "DONE")
```

**Ordering:** Results are emitted in the order the model originally produced the tool calls (`all_item_ids` preserves insertion order), not in task completion order. This keeps the UI consistent with the order the user saw tool call steps appear.

#### Initial `responses.create()` Call

Change `parallel_tool_calls=False` to `parallel_tool_calls=True` at [chat.py:208](../routers/chat.py#L208) and in the continuation call.

## Edge Cases and Resolutions

### 1. `call_id` on `ResponseFunctionToolCall`

**Problem:** The current code does an unnecessary `conversations.items.list()` API call to find the `call_id` after function execution ([chat.py:515-520](../routers/chat.py#L515-L520)).

**Resolution:** `ResponseFunctionToolCall.call_id` is a required field in the OpenAI SDK. Use `event.item.call_id` directly, matching what the computer call handler already does at [chat.py:555](../routers/chat.py#L555). Remove the `conversations.items.list()` call entirely.

### 2. Computer Use: Browser Session Race Conditions

**Problem:** Multiple `ResponseComputerToolCall` items would execute against the same Playwright browser session concurrently, causing race conditions (clicks, keypresses, and screenshots interleaving).

**Resolution:** Store computer call work as unawaited coroutines in `pending_computer_coros` (not as `asyncio.Task`s). During `ResponseCompletedEvent`, await them one at a time while function tasks run concurrently via `asyncio.gather()`. See the `ResponseCompletedEvent` pseudocode above — function tasks are gathered first, then computer coroutines are awaited sequentially in a loop.

In practice, the model is unlikely to emit parallel computer calls, but this safeguard prevents subtle bugs if it does.

### 3. MCP Approval Requests

**Problem:** If the model emits a function call *and* an MCP call requiring approval in the same response, the stream pauses waiting for user approval while function tasks may already be running. After approval, a new `assistant-run.html` container opens a fresh SSE connection — but the original SSE connection's generator still has pending tasks.

**Resolution: Submit outputs, skip stream restart.** When `has_approval_request` is true:

1. Tool tasks still spawn and execute normally during the stream.
2. On `ResponseCompletedEvent`, gather all results, emit SSE events, and submit all tool outputs to the conversation — exactly as in the non-approval case.
3. **Skip the stream restart.** Instead, emit `runCompleted` + `endStream` to close the SSE connection.
4. The user then clicks approve/reject. The `POST /chat/{id}/approve` handler ([chat.py:643-679](../routers/chat.py#L643-L679)) submits the `mcp_approval_response` item and returns a new `assistant-run.html` fragment, which opens a fresh SSE connection via `sse-connect`. That new stream picks up the full conversation state — including the tool outputs we already submitted.

This works because conversation state is server-side (managed by the OpenAI API). The tool outputs are durably committed via `conversations.items.create()` before the stream ends, so the approval handler's new stream sees them.

The `has_approval_request` flag is set in the `ResponseOutputItemAddedEvent` handler when an `McpApprovalRequest` item is detected. See the `ResponseCompletedEvent` pseudocode above for the branching logic.

### 4. Error Handling in Concurrent Tasks

**Problem:** If one tool task fails, should we still submit the others? And what happens to the failed call from the model's perspective?

**Resolution:** Yes, submit the others. For the failed task:

1. **Emit a visible error** in the UI via a `toolOutput` SSE event so the user sees what happened.
2. **Submit a failure output item** to the conversation so the model knows the call failed and can respond appropriately (retry, report the error, or continue without it). Without this, the model would be waiting for an output that never arrives, causing desync.

To submit a failure output, we need the `call_id` for the failed item. Store a mapping of `item_id → call_id` when spawning tasks/coroutines (via a `result_call_ids: dict[str, str]` alongside the task collections).

See the `ResponseCompletedEvent` pseudocode above for the full error handling flow.

### 5. Cancellation and Cleanup

**Problem:** If the SSE connection drops (user navigates away, network error) or `iterate_stream()` is cancelled, any spawned `asyncio.Task`s in `pending_fn_tasks` continue running in the background. Their results are never consumed, and side effects (function calls, browser actions) execute without anyone watching.

**Resolution:** Catch `asyncio.CancelledError` in `iterate_stream()` and cancel all pending tasks before re-raising:

```python
except asyncio.CancelledError:
    # Cancel any in-flight function tasks
    for task in pending_fn_tasks.values():
        task.cancel()
    # Await cancellation to ensure cleanup
    await asyncio.gather(*pending_fn_tasks.values(), return_exceptions=True)
    pending_fn_tasks.clear()
    pending_computer_coros.clear()  # unawaited coroutines are just discarded
    raise
```

Computer coroutines in `pending_computer_coros` are unawaited and have no running state, so they can simply be discarded.

### 6. `current_item_id` Tracking

**Problem:** The current code uses a single `current_item_id` variable to track which step is "active" for OOB targeting. With parallel tool calls, multiple items are in-flight.

**Resolution:** This is already mostly fine. `current_item_id` is set on each `ResponseOutputItemAddedEvent` and `ResponseOutputItemDoneEvent`, and the OpenAI stream delivers these events sequentially (one item at a time within the stream). The parallel execution happens *after* the stream is fully consumed. The only place `current_item_id` matters during parallel execution is inside the task functions, and those receive their own `item_id` as a parameter.

### 7. Annotations After Tool Calls

**Problem:** `ResponseOutputTextAnnotationAddedEvent` uses `event.item_id` (not `current_item_id`) to target the correct step. With parallel tool calls, annotations from a prior text message could arrive interleaved with tool call events.

**Resolution:** No change needed. The annotation handler already uses `event.item_id` as the primary target ([chat.py:400](../routers/chat.py#L400)), falling back to `current_item_id` only if `event.item_id` is absent. This is correct for parallel tool calls.

## Implementation Checklist

1. Define `ToolTaskResult` dataclass (in `routers/chat.py` or a new `utils/tool_tasks.py`)
2. Add `pending_fn_tasks`, `pending_computer_coros`, `result_call_ids`, `has_approval_request` tracking to `iterate_stream()`
3. Refactor `ResponseOutputItemDoneEvent` function call handler to spawn tasks and store `call_id`
4. Refactor `ResponseOutputItemDoneEvent` computer call handler to store coroutines and `call_id`
5. Refactor `ResponseCompletedEvent` handler to gather/emit/submit/continue (preserving original call order)
6. Handle MCP approval + tool call coexistence (submit outputs, skip stream restart)
7. Use `event.item.call_id` directly for function calls (remove `conversations.items.list()` round-trip)
8. Change `parallel_tool_calls=False` to `True` in both initial and continuation `responses.create()` calls
9. Add error handling: emit visible error SSE + submit failure output item per failed task
10. Add cancellation cleanup: cancel pending tasks on `asyncio.CancelledError`
11. Write tests:
    - Single function call still works (regression)
    - Multiple function calls execute concurrently and results are submitted together
    - Results are emitted in original call order, not completion order
    - Computer calls execute sequentially
    - Mixed function + computer calls: functions run in parallel, computer calls serialize
    - MCP approval + function call: outputs submitted, stream ends without restart
    - One task fails, others succeed: failure output submitted, successful outputs submitted
    - SSE connection dropped: pending tasks are cancelled
    - No tool calls: unchanged behavior (text-only response)

## Files Changed

| File | Change |
|---|---|
| `routers/chat.py` | Main refactor: task spawning, gather, submit, `call_id` fix |
| `utils/tool_tasks.py` (new) | `ToolTaskResult` dataclass |
| `tests/test_parallel_tools.py` (new) | Test cases per checklist above |

## What Stays the Same

- SSE event names and HTML payloads (no frontend changes)
- HTMX OOB swap mechanics
- `assistant-step.html` template structure
- Built-in tool handling (web search, file search, code interpreter, image generation)
- MCP approval POST handler (`/chat/{id}/approve`)
- `stream-md.js` client-side event handling
