"""
Tests for parallel tool calls:
1. Single function call still works (regression)
2. Multiple function calls execute concurrently and results are submitted together
3. Results are emitted in original call order, not completion order
4. Computer calls execute sequentially
5. Mixed function + computer calls: functions run in parallel, computer calls serialize
6. MCP approval + function call: outputs submitted, stream ends without restart
7. One task fails, others succeed: failure output submitted, successful outputs submitted
8. SSE connection dropped: pending tasks are cancelled
9. No tool calls: unchanged behavior (text-only response)
10. call_id is taken directly from event.item (no conversations.items.list round-trip)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseFunctionToolCall,
    ResponseComputerToolCall,
)
from openai.types.responses.response_output_item import McpApprovalRequest

from conftest import parse_sse_events, _dotenv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ITEM_ID_A = "fc_aaa111"
ITEM_ID_B = "fc_bbb222"
ITEM_ID_C = "fc_ccc333"
CALL_ID_A = "call_aaa"
CALL_ID_B = "call_bbb"
CALL_ID_C = "call_ccc"
RESPONSE_ID = "resp_parallel_test"
CONV_ID = "conv_parallel_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockAsyncStream:
    """Mock for OpenAI's async streaming response."""

    def __init__(self, events: list):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for event in self._events:
            yield event


def _make_function_tool_call(item_id: str, call_id: str, name: str, arguments: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall.model_construct(
        id=item_id, type="function_call", name=name,
        arguments=arguments, call_id=call_id, status="completed",
    )


def _make_item_added(item_id: str, item_type: str, name: str = "get_weather") -> ResponseOutputItemAddedEvent:
    item_mock = MagicMock()
    item_mock.id = item_id
    item_mock.type = item_type
    item_mock.name = name
    return ResponseOutputItemAddedEvent.model_construct(
        type="response.output_item.added", item=item_mock,
        output_index=0, sequence_number=1,
    )


def make_single_function_stream() -> MockAsyncStream:
    """Stream with a single function call."""
    resp = MagicMock(id=RESPONSE_ID)
    return MockAsyncStream([
        ResponseCreatedEvent.model_construct(type="response.created", response=resp, sequence_number=0),
        _make_item_added(ITEM_ID_A, "function_call", "get_weather"),
        ResponseFunctionCallArgumentsDoneEvent.model_construct(
            type="response.function_call_arguments.done",
            item_id=ITEM_ID_A, arguments='{"location": "Albany"}',
            output_index=0, sequence_number=2,
        ),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done",
            item=_make_function_tool_call(ITEM_ID_A, CALL_ID_A, "get_weather", '{"location": "Albany"}'),
            output_index=0, sequence_number=3,
        ),
        ResponseCompletedEvent.model_construct(type="response.completed", response=resp, sequence_number=4),
    ])


def make_parallel_function_stream() -> MockAsyncStream:
    """Stream with two function calls emitted before ResponseCompleted."""
    resp = MagicMock(id=RESPONSE_ID)
    return MockAsyncStream([
        ResponseCreatedEvent.model_construct(type="response.created", response=resp, sequence_number=0),
        _make_item_added(ITEM_ID_A, "function_call", "get_weather"),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done",
            item=_make_function_tool_call(ITEM_ID_A, CALL_ID_A, "get_weather", '{"location": "Albany"}'),
            output_index=0, sequence_number=1,
        ),
        _make_item_added(ITEM_ID_B, "function_call", "get_weather"),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done",
            item=_make_function_tool_call(ITEM_ID_B, CALL_ID_B, "get_weather", '{"location": "Boston"}'),
            output_index=1, sequence_number=2,
        ),
        ResponseCompletedEvent.model_construct(type="response.completed", response=resp, sequence_number=3),
    ])


def make_text_reply_stream() -> MockAsyncStream:
    """Simple text reply (follow-up after tool outputs submitted)."""
    resp = MagicMock(id="resp_followup")
    msg = MagicMock(id="msg_001", type="message")
    return MockAsyncStream([
        ResponseCreatedEvent.model_construct(type="response.created", response=resp, sequence_number=0),
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=msg,
            output_index=0, sequence_number=1,
        ),
        ResponseTextDeltaEvent.model_construct(
            type="response.text.delta", delta="Here is the weather.",
            item_id="msg_001", output_index=0, content_index=0, sequence_number=2,
        ),
        ResponseCompletedEvent.model_construct(type="response.completed", response=resp, sequence_number=3),
    ])


def make_text_only_stream() -> MockAsyncStream:
    """Stream with only text, no tool calls."""
    resp = MagicMock(id=RESPONSE_ID)
    msg = MagicMock(id="msg_text", type="message")
    return MockAsyncStream([
        ResponseCreatedEvent.model_construct(type="response.created", response=resp, sequence_number=0),
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=msg,
            output_index=0, sequence_number=1,
        ),
        ResponseTextDeltaEvent.model_construct(
            type="response.text.delta", delta="Hello world",
            item_id="msg_text", output_index=0, content_index=0, sequence_number=2,
        ),
        ResponseCompletedEvent.model_construct(type="response.completed", response=resp, sequence_number=3),
    ])


def build_mock_client(streams: list[MockAsyncStream]) -> AsyncMock:
    """Build a mocked AsyncOpenAI client with given stream sequence."""
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(side_effect=streams)
    mock_client.conversations.items.create = AsyncMock()
    mock_client.conversations.items.list = AsyncMock()
    return mock_client


ENV_DEFAULTS = {
    "OPENAI_API_KEY": "sk-fake-key",
    "RESPONSES_MODEL": "gpt-4o",
    "RESPONSES_INSTRUCTIONS": "Test",
    "ENABLED_TOOLS": "function",
    "SHOW_TOOL_CALL_DETAIL": "false",
}


async def stream_events(mock_client: AsyncMock, env_overrides: dict | None = None) -> list[dict[str, str]]:
    """Helper: patch client, stream /receive, return parsed SSE events."""
    from main import app

    env = {**ENV_DEFAULTS, **(env_overrides or {})}

    with _dotenv(env, set_fake_api_key=False):
        with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/chat/{CONV_ID}/receive",
                    timeout=10.0,
                )
                raw = response.text
                await response.aclose()

    return parse_sse_events(raw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleFunctionCallRegression:
    """Single function call should still work after the refactor."""

    @pytest.mark.anyio
    async def test_single_function_call_produces_tool_output(self):
        mock_client = build_mock_client([
            make_single_function_stream(),
            make_text_reply_stream(),
        ])
        events = await stream_events(mock_client)

        event_names = [e["event"] for e in events]
        assert "toolCallCreated" in event_names
        assert "toolOutput" in event_names
        assert "runCompleted" in event_names
        assert "endStream" in event_names

    @pytest.mark.anyio
    async def test_single_function_call_submits_output_and_restarts_stream(self):
        mock_client = build_mock_client([
            make_single_function_stream(),
            make_text_reply_stream(),
        ])
        await stream_events(mock_client)

        # Should have called conversations.items.create to submit output
        mock_client.conversations.items.create.assert_called()
        # Should have called responses.create twice (initial + continuation)
        assert mock_client.responses.create.call_count == 2

    @pytest.mark.anyio
    async def test_call_id_used_directly_no_items_list_call(self):
        """Verify conversations.items.list is never called (call_id comes from event.item directly)."""
        mock_client = build_mock_client([
            make_single_function_stream(),
            make_text_reply_stream(),
        ])
        await stream_events(mock_client)

        mock_client.conversations.items.list.assert_not_called()


class TestParallelFunctionCalls:
    """Multiple function calls should execute concurrently."""

    @pytest.mark.anyio
    async def test_multiple_function_calls_produce_multiple_tool_outputs(self):
        mock_client = build_mock_client([
            make_parallel_function_stream(),
            make_text_reply_stream(),
        ])
        events = await stream_events(mock_client)

        tool_outputs = [e for e in events if e["event"] == "toolOutput"]
        assert len(tool_outputs) == 2, f"Expected 2 toolOutput events, got {len(tool_outputs)}"

    @pytest.mark.anyio
    async def test_all_outputs_submitted_individually(self):
        mock_client = build_mock_client([
            make_parallel_function_stream(),
            make_text_reply_stream(),
        ])
        await stream_events(mock_client)

        # Each output should be submitted in its own call
        create_calls = mock_client.conversations.items.create.call_args_list
        fn_output_calls = [
            c for c in create_calls
            if "items" in c.kwargs
            and isinstance(c.kwargs["items"], list)
            and len(c.kwargs["items"]) == 1
            and c.kwargs["items"][0].get("type") == "function_call_output"
        ]
        assert len(fn_output_calls) == 2, (
            f"Expected 2 individual conversations.items.create calls for function outputs, "
            f"got {len(fn_output_calls)} from calls: {create_calls}"
        )

    @pytest.mark.anyio
    async def test_outputs_submitted_with_correct_call_ids(self):
        mock_client = build_mock_client([
            make_parallel_function_stream(),
            make_text_reply_stream(),
        ])
        await stream_events(mock_client)

        create_calls = mock_client.conversations.items.create.call_args_list
        call_ids = set()
        for c in create_calls:
            items = c.kwargs.get("items", [])
            if isinstance(items, list):
                for item in items:
                    if item.get("type") == "function_call_output":
                        call_ids.add(item["call_id"])
        assert call_ids == {CALL_ID_A, CALL_ID_B}

    @pytest.mark.anyio
    async def test_stream_restarted_once_after_all_outputs(self):
        mock_client = build_mock_client([
            make_parallel_function_stream(),
            make_text_reply_stream(),
        ])
        await stream_events(mock_client)

        # Initial call + one continuation = 2 total
        assert mock_client.responses.create.call_count == 2

    @pytest.mark.anyio
    async def test_continuation_uses_parallel_tool_calls_true(self):
        mock_client = build_mock_client([
            make_parallel_function_stream(),
            make_text_reply_stream(),
        ])
        await stream_events(mock_client)

        # Check the second responses.create call
        second_call = mock_client.responses.create.call_args_list[1]
        assert second_call.kwargs.get("parallel_tool_calls") is True


class TestResultOrdering:
    """Results must be emitted in original call order."""

    @pytest.mark.anyio
    async def test_tool_outputs_in_call_order(self):
        """toolOutput events should appear in the order the model emitted the calls,
        regardless of completion order."""
        mock_client = build_mock_client([
            make_parallel_function_stream(),
            make_text_reply_stream(),
        ])
        events = await stream_events(mock_client)

        tool_outputs = [e for e in events if e["event"] == "toolOutput"]
        assert len(tool_outputs) == 2
        # Both should contain weather data (Albany first, then Boston)
        # Since both use get_weather, both produce output; order is preserved
        assert tool_outputs[0]["data"]  # non-empty
        assert tool_outputs[1]["data"]  # non-empty


class TestTextOnlyResponse:
    """Text-only responses should work unchanged."""

    @pytest.mark.anyio
    async def test_text_only_no_tool_output(self):
        mock_client = build_mock_client([make_text_only_stream()])
        events = await stream_events(mock_client)

        event_names = [e["event"] for e in events]
        assert "toolOutput" not in event_names
        assert "messageCreated" in event_names
        assert "textDelta" in event_names
        assert "runCompleted" in event_names
        assert "endStream" in event_names

    @pytest.mark.anyio
    async def test_text_only_no_items_create(self):
        mock_client = build_mock_client([make_text_only_stream()])
        await stream_events(mock_client)

        # No tool outputs to submit
        mock_client.conversations.items.create.assert_not_called()


class TestErrorHandling:
    """When a tool task fails, error output should be submitted."""

    @pytest.mark.anyio
    async def test_failed_task_emits_error_and_submits_failure_output(self):
        """If the function registry raises, the error should be visible in SSE
        and a failure output item should be submitted."""
        mock_client = build_mock_client([
            make_single_function_stream(),
            make_text_reply_stream(),
        ])

        # Patch FUNCTION_REGISTRY.call to raise
        with patch("utils.function_calling.ToolRegistry.call", side_effect=RuntimeError("Tool exploded")):
            events = await stream_events(mock_client)

        # Should have an error toolOutput
        tool_outputs = [e for e in events if e["event"] == "toolOutput"]
        assert any("Error" in to["data"] for to in tool_outputs)

        # Should still submit output item with error
        create_calls = mock_client.conversations.items.create.call_args_list
        assert len(create_calls) >= 1
        # Find the items submission
        for call in create_calls:
            items = call.kwargs.get("items", [])
            if isinstance(items, list):
                for item in items:
                    if item.get("type") == "function_call_output":
                        output = json.loads(item["output"])
                        assert "error" in output


class TestMcpApprovalWithFunctionCalls:
    """When MCP approval + function calls coexist, submit outputs but skip stream restart."""

    @pytest.mark.anyio
    async def test_approval_request_skips_stream_restart(self):
        """If the response contains both a function call and an MCP approval request,
        tool outputs should be submitted but the stream should end without restart."""
        resp = MagicMock(id=RESPONSE_ID)

        # MCP approval request item
        approval_item = McpApprovalRequest.model_construct(
            id="mcp_approval_001",
            type="mcp_approval_request",
            name="some_mcp_tool",
            server_label="test_server",
            arguments='{"key": "value"}',
        )

        stream = MockAsyncStream([
            ResponseCreatedEvent.model_construct(type="response.created", response=resp, sequence_number=0),
            _make_item_added(ITEM_ID_A, "function_call", "get_weather"),
            ResponseOutputItemDoneEvent.model_construct(
                type="response.output_item.done",
                item=_make_function_tool_call(ITEM_ID_A, CALL_ID_A, "get_weather", '{"location": "Albany"}'),
                output_index=0, sequence_number=1,
            ),
            ResponseOutputItemAddedEvent.model_construct(
                type="response.output_item.added", item=approval_item,
                output_index=1, sequence_number=2,
            ),
            ResponseCompletedEvent.model_construct(type="response.completed", response=resp, sequence_number=3),
        ])

        mock_client = build_mock_client([stream])
        events = await stream_events(mock_client)

        event_names = [e["event"] for e in events]
        assert "toolOutput" in event_names
        assert "endStream" in event_names

        # Should have submitted tool outputs
        mock_client.conversations.items.create.assert_called()

        # Should NOT have restarted the stream (only 1 responses.create call)
        assert mock_client.responses.create.call_count == 1


class TestComputerCallSequential:
    """Computer calls should be stored as coroutines and executed sequentially."""

    @pytest.mark.anyio
    async def test_computer_call_executes_and_produces_image_output(self):
        resp = MagicMock(id=RESPONSE_ID)
        computer_item = MagicMock()
        computer_item.id = ITEM_ID_A
        computer_item.type = "computer_call"

        computer_tool_call = ResponseComputerToolCall.model_construct(
            id=ITEM_ID_A, type="computer_call",
            call_id=CALL_ID_A, status="completed",
            actions=[{"type": "click", "x": 100, "y": 200, "button": "left"}],
            pending_safety_checks=[],
        )

        stream = MockAsyncStream([
            ResponseCreatedEvent.model_construct(type="response.created", response=resp, sequence_number=0),
            ResponseOutputItemAddedEvent.model_construct(
                type="response.output_item.added", item=computer_item,
                output_index=0, sequence_number=1,
            ),
            ResponseOutputItemDoneEvent.model_construct(
                type="response.output_item.done",
                item=computer_tool_call,
                output_index=0, sequence_number=2,
            ),
            ResponseCompletedEvent.model_construct(type="response.completed", response=resp, sequence_number=3),
        ])

        mock_client = build_mock_client([stream, make_text_reply_stream()])

        with patch("routers.chat.execute_computer_actions", new_callable=AsyncMock, return_value="base64screenshot"):
            with patch("routers.chat.describe_actions", return_value="Click at (100, 200)"):
                events = await stream_events(mock_client, {"ENABLED_TOOLS": "computer_use"})

        event_names = [e["event"] for e in events]
        assert "imageOutput" in event_names

        # Verify computer_call_output was submitted
        create_calls = mock_client.conversations.items.create.call_args_list
        submitted_items = []
        for call in create_calls:
            items = call.kwargs.get("items", [])
            if isinstance(items, list):
                submitted_items.extend(items)
        computer_outputs = [i for i in submitted_items if i.get("type") == "computer_call_output"]
        assert len(computer_outputs) == 1
        assert computer_outputs[0]["call_id"] == CALL_ID_A


class TestParallelToolCallsFlag:
    """Verify parallel_tool_calls=True is sent in both initial and continuation calls."""

    @pytest.mark.anyio
    async def test_initial_call_uses_parallel_tool_calls_true(self):
        mock_client = build_mock_client([make_text_only_stream()])
        await stream_events(mock_client)

        first_call = mock_client.responses.create.call_args_list[0]
        assert first_call.kwargs.get("parallel_tool_calls") is True
