"""
Tests for tool output rendering:
1. Template path in tool.config.json is valid
2. Weather widget template renders correctly
3. Tool output SSE events are NOT wrapped with OOB swap
4. Tool call arguments are emitted into the collapsible <details> via OOB swap
5. JavaScript handles toolDelta events for proper OOB insertion
6. Integration: SSE stream emits correct events for function tool calls
"""

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from jinja2 import Environment, FileSystemLoader
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseFunctionToolCall,
)

from utils.config import ToolConfig
from utils.sse import sse_format
from routers.chat import wrap_for_oob_swap


PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

ITEM_ID = "fc_test_abc123"
RESPONSE_ID = "resp_test_xyz789"
CALL_ID = "call_test_456"
ARGUMENTS_JSON = '{"location": "Albany"}'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_sse_events(raw: str) -> list[dict[str, str]]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data_lines: list[str] = []

    for line in raw.split("\n"):
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            current_data_lines.append(line[len("data: "):])
        elif line == "" and current_event is not None:
            events.append({
                "event": current_event,
                "data": "\n".join(current_data_lines),
            })
            current_event = None
            current_data_lines = []

    return events


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


def make_function_call_stream() -> MockAsyncStream:
    """Create a mock stream that emits a get_weather function call."""
    resp_mock = MagicMock()
    resp_mock.id = RESPONSE_ID

    item_mock = MagicMock()
    item_mock.id = ITEM_ID
    item_mock.type = "function_call"
    item_mock.name = "get_weather"

    tool_call = ResponseFunctionToolCall.model_construct(
        id=ITEM_ID, type="function_call", name="get_weather",
        arguments=ARGUMENTS_JSON, call_id=CALL_ID, status="completed",
    )

    return MockAsyncStream([
        ResponseCreatedEvent.model_construct(
            type="response.created", response=resp_mock, sequence_number=0,
        ),
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=item_mock,
            output_index=0, sequence_number=1,
        ),
        ResponseFunctionCallArgumentsDeltaEvent.model_construct(
            type="response.function_call_arguments.delta",
            item_id=ITEM_ID, delta='{"locat', output_index=0, sequence_number=2,
        ),
        ResponseFunctionCallArgumentsDeltaEvent.model_construct(
            type="response.function_call_arguments.delta",
            item_id=ITEM_ID, delta='ion": "Albany"}', output_index=0, sequence_number=3,
        ),
        ResponseFunctionCallArgumentsDoneEvent.model_construct(
            type="response.function_call_arguments.done",
            item_id=ITEM_ID, arguments=ARGUMENTS_JSON,
            output_index=0, sequence_number=4,
        ),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done", item=tool_call,
            output_index=0, sequence_number=5,
        ),
        ResponseCompletedEvent.model_construct(
            type="response.completed", response=resp_mock, sequence_number=6,
        ),
    ])


def make_text_reply_stream() -> MockAsyncStream:
    """Create a mock stream that just completes (the follow-up after function output)."""
    resp_mock = MagicMock()
    resp_mock.id = "resp_followup"

    return MockAsyncStream([
        ResponseCreatedEvent.model_construct(
            type="response.created", response=resp_mock, sequence_number=0,
        ),
        ResponseCompletedEvent.model_construct(
            type="response.completed", response=resp_mock, sequence_number=1,
        ),
    ])


def build_mock_openai_client() -> AsyncMock:
    """Build a fully mocked AsyncOpenAI client."""
    mock_client = AsyncMock()

    # responses.create returns the function call stream first, then text reply
    mock_client.responses.create = AsyncMock(
        side_effect=[make_function_call_stream(), make_text_reply_stream()]
    )

    # conversations.items.list returns a list with our function call item
    items_mock = MagicMock()
    item_obj = MagicMock()
    item_obj.id = ITEM_ID
    item_obj.call_id = CALL_ID
    items_mock.data = [item_obj]
    mock_client.conversations.items.list = AsyncMock(return_value=items_mock)

    # conversations.items.create succeeds
    mock_client.conversations.items.create = AsyncMock()

    return mock_client


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestToolConfigTemplatePath:
    """Verify that template_path values in tool.config.json point to real template files."""

    def test_tool_config_template_paths_exist(self):
        config_path = PROJECT_ROOT / "tool.config.json"
        config = ToolConfig.model_validate_json(config_path.read_text())
        for func in config.custom_functions:
            if func.template_path:
                full_path = TEMPLATES_DIR / func.template_path
                assert full_path.exists(), (
                    f"template_path '{func.template_path}' for function '{func.name}' "
                    f"does not exist at {full_path}"
                )

    def test_template_path_is_not_python_file(self):
        config_path = PROJECT_ROOT / "tool.config.json"
        config = ToolConfig.model_validate_json(config_path.read_text())
        for func in config.custom_functions:
            if func.template_path:
                assert not func.template_path.endswith(".py"), (
                    f"template_path '{func.template_path}' for function '{func.name}' "
                    f"points to a Python file instead of a template"
                )


class TestWeatherWidgetTemplate:
    """Verify the weather widget template renders correctly."""

    def setup_method(self):
        self.env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
        self.template = self.env.get_template("components/weather-widget.html")

    def test_renders_weather_data(self):
        tool = MagicMock()
        tool.error = None
        tool.warning = None
        tool.result = [
            MagicMock(location="Albany", date="2026-03-09",
                      temperature=67, unit="F", conditions="Snowy")
        ]
        html = self.template.render(tool=tool)
        assert "Albany" in html
        assert "67" in html
        assert "Snowy" in html

    def test_renders_error(self):
        tool = MagicMock()
        tool.error = "API unavailable"
        tool.warning = None
        tool.result = None
        html = self.template.render(tool=tool)
        assert "API unavailable" in html
        assert "toolError" in html

    def test_renders_no_data(self):
        tool = MagicMock()
        tool.error = None
        tool.warning = None
        tool.result = []
        html = self.template.render(tool=tool)
        assert "No weather data" in html


class TestToolOutputSseFormat:
    """Verify tool output SSE events are NOT wrapped with OOB swap."""

    def test_tool_output_not_oob_wrapped(self):
        html = '<div class="toolOutput"><p>Weather data</p></div>'
        sse = sse_format("toolOutput", html)
        assert "hx-swap-oob" not in sse

    def test_tool_delta_is_oob_wrapped(self):
        step_id = "fc_abc123"
        delta = '{"location": "Albany"}'
        wrapped = wrap_for_oob_swap(step_id, delta)
        sse = sse_format("toolDelta", wrapped)
        assert "hx-swap-oob" in sse
        assert f"#step-{step_id}" in sse

    def test_generic_tool_output_is_pre_json(self):
        payload = {"result": [{"location": "Albany"}]}
        html = f"<pre>{json.dumps(payload, indent=2)}</pre>"
        sse = sse_format("toolOutput", html)
        assert "<pre>" in sse
        assert "Albany" in sse
        assert "hx-swap-oob" not in sse


class TestDefaultToolConfigInMain:

    def test_main_default_template_path(self):
        main_py = (PROJECT_ROOT / "main.py").read_text()
        assert "components/weather-widget.html" in main_py
        assert '"template_path": "utils/custom_functions.py"' not in main_py


class TestToolDeltaOobSwapIntegration:
    """Verify OOB swap targets match the IDs in the assistant-step template."""

    def setup_method(self):
        self.env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
        self.template = self.env.get_template("components/assistant-step.html")

    def test_toolcall_template_has_oob_target_div(self):
        step_id = "fc_abc123"
        html = self.template.render(step_type="toolCall", step_id=step_id,
                                     content="Calling test tool...")
        assert f'id="step-{step_id}"' in html

    def test_oob_swap_target_matches_template_id(self):
        step_id = "fc_abc123"
        html = self.template.render(step_type="toolCall", step_id=step_id,
                                     content="Calling test tool...")
        wrapped = wrap_for_oob_swap(step_id, "test delta")
        match = re.search(r'hx-swap-oob="[^:]+:(#[^"]+)"', wrapped)
        assert match
        target_selector = match.group(1)
        expected_id = f"step-{step_id}"
        assert target_selector == f"#{expected_id}"
        assert f'id="{expected_id}"' in html

    def test_oob_target_is_inside_details_not_summary(self):
        step_id = "fc_abc123"
        html = self.template.render(step_type="toolCall", step_id=step_id,
                                     content="Calling test tool...")
        summary_end = html.index("</summary>")
        details_end = html.index("</details>")
        target_pos = html.index(f'id="step-{step_id}"')
        assert summary_end < target_pos < details_end


class TestToolDeltaJsHandling:
    """Verify stream-md.js properly handles toolDelta SSE events."""

    def setup_method(self):
        self.js_content = (STATIC_DIR / "stream-md.js").read_text()

    def test_tool_delta_prevented_in_js(self):
        assert re.search(
            r"originalSSEEvent\.type\s*===?\s*['\"]toolDelta['\"]",
            self.js_content,
        ), "stream-md.js must check for toolDelta event type"

        assert re.search(
            r"toolDelta.*?preventDefault|preventDefault.*?toolDelta",
            self.js_content,
            re.DOTALL,
        ), "stream-md.js must call evt.preventDefault() for toolDelta events"

    def test_tool_delta_content_processed_via_dedicated_handler(self):
        assert re.search(
            r"(processToolDelta|parseOobSwap).*toolDelta|toolDelta.*?(processToolDelta|parseOobSwap)",
            self.js_content,
            re.DOTALL,
        ), "stream-md.js must process toolDelta via parseOobSwap or processToolDelta"

    def test_tool_delta_streaming_uses_pre_node(self):
        assert 'data-tool-delta="stream"' in self.js_content
        assert "document.createElement('pre')" in self.js_content

    def test_tool_delta_final_replaces_streaming_content(self):
        assert "dataset.toolDelta === 'replace'" in self.js_content
        assert "replaceChildren(replacementNode.cloneNode(true))" in self.js_content


# ---------------------------------------------------------------------------
# Integration tests: SSE stream for function tool calls
# ---------------------------------------------------------------------------

@pytest.mark.anyio
class TestFunctionCallSseIntegration:
    """Integration test: hit the /receive endpoint with a mocked OpenAI client
    and verify SSE events for a function tool call contain correctly structured
    toolCallCreated, toolDelta, and toolOutput events."""

    async def _stream_events(self, env_overrides: dict | None = None) -> list[dict[str, str]]:
        """Helper: set env, patch OpenAI client, stream /receive, return parsed SSE events."""
        from main import app

        env_defaults = {
            "OPENAI_API_KEY": "sk-fake-key",
            "RESPONSES_MODEL": "gpt-4o",
            "RESPONSES_INSTRUCTIONS": "Test",
            "ENABLED_TOOLS": "function",
            "SHOW_TOOL_CALL_DETAIL": "false",
        }
        env_defaults.update(env_overrides or {})

        mock_client = build_mock_openai_client()

        try:
            with open(PROJECT_ROOT / ".env") as f:
                original_env = f.read()
        except FileNotFoundError:
            original_env = None

        # Write test env
        (PROJECT_ROOT / ".env").write_text(
            "\n".join(f"{k}={v}" for k, v in env_defaults.items()) + "\n"
        )

        try:
            # Patch AsyncOpenAI constructor so Depends(lambda: AsyncOpenAI()) returns our mock
            with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get(
                        "/chat/conv_test123/receive",
                        timeout=10.0,
                    )
                    raw = response.text
        finally:
            # Restore env
            if original_env is not None:
                (PROJECT_ROOT / ".env").write_text(original_env)
            else:
                (PROJECT_ROOT / ".env").unlink(missing_ok=True)

        return parse_sse_events(raw)

    async def test_tool_call_created_has_details_with_target_div(self):
        """toolCallCreated SSE event must contain a <details> element with an
        inner div whose id matches the OOB swap target pattern."""
        events = await self._stream_events()

        tool_call_created = [e for e in events if e["event"] == "toolCallCreated"]
        assert len(tool_call_created) >= 1, "Expected at least one toolCallCreated event"

        html = tool_call_created[0]["data"]
        assert "<details" in html, "toolCallCreated must emit a <details> element"
        assert f'id="step-{ITEM_ID}"' in html, (
            f"toolCallCreated must contain a div with id='step-{ITEM_ID}' for OOB targeting"
        )

    async def test_arguments_json_emitted_into_details(self):
        """After function call is complete, the arguments JSON must be emitted
        via a toolDelta event with OOB swap targeting the toolCallDetails div.
        This must work even when SHOW_TOOL_CALL_DETAIL is false (the default)."""
        events = await self._stream_events({"SHOW_TOOL_CALL_DETAIL": "false"})

        tool_deltas = [e for e in events if e["event"] == "toolDelta"]
        assert len(tool_deltas) >= 1, (
            "Expected at least one toolDelta event with complete arguments JSON "
            "(even when SHOW_TOOL_CALL_DETAIL is false)"
        )

        # At least one toolDelta must target the right element and contain the args
        args_emitted = False
        for td in tool_deltas:
            data = td["data"]
            if f"#step-{ITEM_ID}" in data and "Albany" in data:
                args_emitted = True
                break

        assert args_emitted, (
            f"No toolDelta event found targeting #step-{ITEM_ID} with arguments "
            f"containing 'Albany'. Got toolDeltas: {tool_deltas}"
        )

        final_payload = next(
            (td["data"] for td in tool_deltas if 'data-tool-delta="replace"' in td["data"]),
            None,
        )
        assert final_payload is not None, "Expected a final replacement payload for tool arguments"
        assert 'class="toolCallArgs"' in final_payload
        assert "<pre" in final_payload

    async def test_tool_output_not_oob_wrapped(self):
        """toolOutput SSE event must NOT contain hx-swap-oob (it goes to default swap target)."""
        events = await self._stream_events()

        tool_outputs = [e for e in events if e["event"] == "toolOutput"]
        assert len(tool_outputs) >= 1, "Expected at least one toolOutput event"

        for to in tool_outputs:
            assert "hx-swap-oob" not in to["data"], (
                "toolOutput must not be OOB-wrapped; it should appear as a sibling "
                "of <details>, not inside it"
            )

    async def test_streaming_deltas_when_show_detail_enabled(self):
        """When SHOW_TOOL_CALL_DETAIL=true, streaming argument deltas must be
        emitted as toolDelta events with OOB swap targeting the toolCallDetails div."""
        events = await self._stream_events({"SHOW_TOOL_CALL_DETAIL": "true"})

        tool_deltas = [e for e in events if e["event"] == "toolDelta"]
        # With show_detail enabled, we expect streaming deltas
        assert len(tool_deltas) >= 2, (
            f"Expected at least 2 toolDelta events (streaming deltas), got {len(tool_deltas)}"
        )

        # Each delta should target the correct element
        for td in tool_deltas:
            assert f"#step-{ITEM_ID}" in td["data"], (
                f"toolDelta must target #step-{ITEM_ID}, got: {td['data'][:100]}"
            )
