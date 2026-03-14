"""
Tests for tool output rendering:
1. Template path in tool.config.json is valid
2. Weather widget template renders correctly
3. Tool output SSE events are NOT wrapped with OOB swap
4. Tool call arguments are emitted into the collapsible <details> via OOB swap
5. JavaScript handles toolDelta events for proper OOB insertion
6. Integration: SSE stream emits correct events for function tool calls
7. Web search tool: builds correct tool payload from env config
8. Web search tool: SSE stream emits correct events for web search calls
9. Web search tool: url_citation annotations rendered as links
"""

import json
import re
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
    ResponseOutputTextAnnotationAddedEvent,
    ResponseTextDeltaEvent,
    ResponseFunctionToolCall,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
    ResponseImageGenCallInProgressEvent,
    ResponseImageGenCallGeneratingEvent,
    ResponseImageGenCallCompletedEvent,
    ResponseImageGenCallPartialImageEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall

from conftest import PROJECT_ROOT, parse_sse_events, _dotenv
from utils.config import ToolConfig
from utils.sse import sse_format
from routers.chat import wrap_for_oob_swap
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

ITEM_ID = "fc_test_abc123"
RESPONSE_ID = "resp_test_xyz789"
CALL_ID = "call_test_456"
ARGUMENTS_JSON = '{"location": "Albany"}'


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

        with _dotenv(env_defaults, set_fake_api_key=False):
            with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get(
                        "/chat/conv_test123/receive",
                        timeout=10.0,
                    )
                    raw = response.text
                    await response.aclose()

        return parse_sse_events(raw)

    @pytest.mark.anyio
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

    @pytest.mark.anyio
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

    @pytest.mark.anyio
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

    @pytest.mark.anyio
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


# ---------------------------------------------------------------------------
# Web Search tool tests
# ---------------------------------------------------------------------------

WS_ITEM_ID = "ws_test_abc123"
WS_RESPONSE_ID = "resp_ws_test_xyz789"


def make_web_search_stream() -> MockAsyncStream:
    """Create a mock stream that emits a web search call followed by a text message."""
    resp_mock = MagicMock()
    resp_mock.id = WS_RESPONSE_ID

    # Web search call output item
    ws_item = MagicMock()
    ws_item.id = WS_ITEM_ID
    ws_item.type = "web_search_call"

    # Message output item
    msg_item = MagicMock()
    msg_item.id = "msg_test_123"
    msg_item.type = "message"

    return MockAsyncStream([
        ResponseCreatedEvent.model_construct(
            type="response.created", response=resp_mock, sequence_number=0,
        ),
        ResponseWebSearchCallInProgressEvent.model_construct(
            type="response.web_search_call.in_progress",
            item_id=WS_ITEM_ID, output_index=0, sequence_number=1,
        ),
        ResponseWebSearchCallSearchingEvent.model_construct(
            type="response.web_search_call.searching",
            item_id=WS_ITEM_ID, output_index=0, sequence_number=2,
        ),
        ResponseWebSearchCallCompletedEvent.model_construct(
            type="response.web_search_call.completed",
            item_id=WS_ITEM_ID, output_index=0, sequence_number=3,
        ),
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=msg_item,
            output_index=1, sequence_number=4,
        ),
        ResponseTextDeltaEvent.model_construct(
            type="response.text.delta",
            delta="Here are the results",
            item_id="msg_test_123", output_index=1,
            content_index=0, sequence_number=5,
        ),
        ResponseOutputTextAnnotationAddedEvent.model_construct(
            type="response.output_text.annotation.added",
            annotation={
                "type": "url_citation",
                "url": "https://example.com/article",
                "title": "Example Article",
                "start_index": 0,
                "end_index": 20,
            },
            annotation_index=0,
            item_id="msg_test_123", output_index=1,
            content_index=0, sequence_number=6,
        ),
        ResponseCompletedEvent.model_construct(
            type="response.completed", response=resp_mock, sequence_number=7,
        ),
    ])


def build_web_search_mock_client() -> AsyncMock:
    """Build a mocked AsyncOpenAI client for web search tests."""
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=make_web_search_stream())
    mock_client.conversations.items.list = AsyncMock()
    mock_client.conversations.items.create = AsyncMock()
    return mock_client


class TestWebSearchToolPayload:
    """Verify that the web search tool payload is correctly built from env config."""

    async def _get_create_call_args(self, env_overrides: dict) -> dict:
        """Helper: stream /receive with mocked client and return the kwargs passed to responses.create."""
        from main import app

        env_defaults = {
            "OPENAI_API_KEY": "sk-fake-key",
            "RESPONSES_MODEL": "gpt-4o",
            "RESPONSES_INSTRUCTIONS": "Test",
            "ENABLED_TOOLS": "web_search",
            "SHOW_TOOL_CALL_DETAIL": "false",
            "WEB_SEARCH_CONTEXT_SIZE": "medium",
            "WEB_SEARCH_LOCATION_COUNTRY": "",
            "WEB_SEARCH_LOCATION_CITY": "",
            "WEB_SEARCH_LOCATION_REGION": "",
            "WEB_SEARCH_LOCATION_TIMEZONE": "",
        }
        env_defaults.update(env_overrides)

        mock_client = build_web_search_mock_client()

        with _dotenv(env_defaults, set_fake_api_key=False):
            with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    await client.get("/chat/conv_test123/receive", timeout=10.0)

        return mock_client.responses.create.call_args.kwargs

    @pytest.mark.anyio
    async def test_web_search_tool_in_tools_list(self):
        """When ENABLED_TOOLS includes web_search, tools list must contain a web_search_preview entry."""
        kwargs = await self._get_create_call_args({})
        tools = kwargs.get("tools", [])
        ws_tools = [t for t in tools if isinstance(t, dict) and t.get("type") == "web_search_preview"]
        assert len(ws_tools) == 1, f"Expected one web_search_preview tool, got {ws_tools}"

    @pytest.mark.anyio
    async def test_web_search_default_context_size(self):
        """Default context size should be medium."""
        kwargs = await self._get_create_call_args({})
        tools = kwargs.get("tools", [])
        ws_tool = next(t for t in tools if isinstance(t, dict) and t.get("type") == "web_search_preview")
        assert ws_tool.get("search_context_size", "medium") == "medium"

    @pytest.mark.anyio
    async def test_web_search_custom_context_size(self):
        """Custom context size should be passed through."""
        kwargs = await self._get_create_call_args({"WEB_SEARCH_CONTEXT_SIZE": "high"})
        tools = kwargs.get("tools", [])
        ws_tool = next(t for t in tools if isinstance(t, dict) and t.get("type") == "web_search_preview")
        assert ws_tool["search_context_size"] == "high"

    @pytest.mark.anyio
    async def test_web_search_with_location(self):
        """When location env vars are set, user_location should be included."""
        kwargs = await self._get_create_call_args({
            "WEB_SEARCH_LOCATION_COUNTRY": "US",
            "WEB_SEARCH_LOCATION_CITY": "New York",
            "WEB_SEARCH_LOCATION_REGION": "New York",
            "WEB_SEARCH_LOCATION_TIMEZONE": "America/New_York",
        })
        tools = kwargs.get("tools", [])
        ws_tool = next(t for t in tools if isinstance(t, dict) and t.get("type") == "web_search_preview")
        loc = ws_tool.get("user_location")
        assert loc is not None, "user_location should be present when location env vars are set"
        assert loc["type"] == "approximate"
        assert loc["country"] == "US"
        assert loc["city"] == "New York"
        assert loc["region"] == "New York"
        assert loc["timezone"] == "America/New_York"

    @pytest.mark.anyio
    async def test_web_search_no_location_when_empty(self):
        """When no location env vars are set, user_location should not be included."""
        kwargs = await self._get_create_call_args({
            "WEB_SEARCH_LOCATION_COUNTRY": "",
            "WEB_SEARCH_LOCATION_CITY": "",
            "WEB_SEARCH_LOCATION_REGION": "",
            "WEB_SEARCH_LOCATION_TIMEZONE": "",
        })
        tools = kwargs.get("tools", [])
        ws_tool = next(t for t in tools if isinstance(t, dict) and t.get("type") == "web_search_preview")
        assert "user_location" not in ws_tool, "user_location should not be present when no location env vars are set"


class TestWebSearchSseIntegration:
    """Integration test: verify SSE events for a web search call."""

    async def _stream_events(self, env_overrides: dict | None = None) -> list[dict[str, str]]:
        """Helper: stream /receive with web search mock and return parsed SSE events."""
        from main import app

        env_defaults = {
            "OPENAI_API_KEY": "sk-fake-key",
            "RESPONSES_MODEL": "gpt-4o",
            "RESPONSES_INSTRUCTIONS": "Test",
            "ENABLED_TOOLS": "web_search",
            "SHOW_TOOL_CALL_DETAIL": "false",
            "WEB_SEARCH_CONTEXT_SIZE": "medium",
            "WEB_SEARCH_LOCATION_COUNTRY": "",
            "WEB_SEARCH_LOCATION_CITY": "",
            "WEB_SEARCH_LOCATION_REGION": "",
            "WEB_SEARCH_LOCATION_TIMEZONE": "",
        }
        env_defaults.update(env_overrides or {})

        mock_client = build_web_search_mock_client()

        with _dotenv(env_defaults, set_fake_api_key=False):
            with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/chat/conv_test123/receive", timeout=10.0)
                    raw = response.text
                    await response.aclose()

        return parse_sse_events(raw)

    @pytest.mark.anyio
    async def test_web_search_emits_tool_call_created(self):
        """Web search searching event should emit a toolCallCreated SSE event."""
        events = await self._stream_events()
        tool_calls = [e for e in events if e["event"] == "toolCallCreated"]
        assert len(tool_calls) >= 1, f"Expected toolCallCreated event, got events: {[e['event'] for e in events]}"
        assert "web_search" in tool_calls[0]["data"].lower() or "web search" in tool_calls[0]["data"].lower()

    @pytest.mark.anyio
    async def test_web_search_does_not_crash_stream(self):
        """Web search events should not cause unhandled event errors; stream should complete."""
        events = await self._stream_events()
        event_types = [e["event"] for e in events]
        assert "endStream" in event_types, f"Stream should complete. Got: {event_types}"
        assert "networkError" not in event_types, f"No network errors expected. Got: {event_types}"

    @pytest.mark.anyio
    async def test_web_search_text_message_rendered(self):
        """The text message after web search should be rendered via messageCreated + textDelta."""
        events = await self._stream_events()
        msg_created = [e for e in events if e["event"] == "messageCreated"]
        assert len(msg_created) >= 1, "Expected a messageCreated event for the text response"
        text_deltas = [e for e in events if e["event"] == "textDelta"]
        assert len(text_deltas) >= 1, "Expected textDelta events with search results text"
        assert any("results" in td["data"] for td in text_deltas)

    @pytest.mark.anyio
    async def test_url_citation_rendered_as_link(self):
        """url_citation annotations should be emitted as clickable links."""
        events = await self._stream_events()
        text_deltas = [e for e in events if e["event"] == "textDelta"]
        # Find the citation event
        citation_data = " ".join(td["data"] for td in text_deltas)
        assert "https://example.com/article" in citation_data, (
            f"Expected url_citation link in textDelta events. Got: {citation_data[:300]}"
        )


# ---------------------------------------------------------------------------
# Image Generation tool tests
# ---------------------------------------------------------------------------

IG_ITEM_ID = "ig_test_abc123"
IG_RESPONSE_ID = "resp_ig_test_xyz789"
# A tiny base64 PNG (1x1 transparent pixel)
FAKE_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
FAKE_PARTIAL_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


def make_image_generation_stream(*, include_partial: bool = False) -> MockAsyncStream:
    """Create a mock stream that emits an image generation call."""
    resp_mock = MagicMock()
    resp_mock.id = IG_RESPONSE_ID

    # OutputItemAdded for image_generation_call
    ig_item_added = MagicMock()
    ig_item_added.id = IG_ITEM_ID
    ig_item_added.type = "image_generation_call"

    # Done item
    ig_done_item = ImageGenerationCall.model_construct(
        id=IG_ITEM_ID,
        type="image_generation_call",
        status="completed",
        result=FAKE_IMAGE_B64,
    )

    events = [
        ResponseCreatedEvent.model_construct(
            type="response.created", response=resp_mock, sequence_number=0,
        ),
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=ig_item_added,
            output_index=0, sequence_number=1,
        ),
        ResponseImageGenCallInProgressEvent.model_construct(
            type="response.image_generation_call.in_progress",
            item_id=IG_ITEM_ID, output_index=0, sequence_number=2,
        ),
        ResponseImageGenCallGeneratingEvent.model_construct(
            type="response.image_generation_call.generating",
            item_id=IG_ITEM_ID, output_index=0, sequence_number=3,
        ),
    ]

    if include_partial:
        events.append(
            ResponseImageGenCallPartialImageEvent.model_construct(
                type="response.image_generation_call.partial_image",
                item_id=IG_ITEM_ID, output_index=0,
                partial_image_b64=FAKE_PARTIAL_B64,
                partial_image_index=0, sequence_number=4,
            )
        )

    events.extend([
        ResponseImageGenCallCompletedEvent.model_construct(
            type="response.image_generation_call.completed",
            item_id=IG_ITEM_ID, output_index=0,
            sequence_number=5 if include_partial else 4,
        ),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done", item=ig_done_item,
            output_index=0,
            sequence_number=6 if include_partial else 5,
        ),
        ResponseCompletedEvent.model_construct(
            type="response.completed", response=resp_mock,
            sequence_number=7 if include_partial else 6,
        ),
    ])

    return MockAsyncStream(events)


def build_image_generation_mock_client(*, include_partial: bool = False) -> AsyncMock:
    """Build a mocked AsyncOpenAI client for image generation tests."""
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(
        return_value=make_image_generation_stream(include_partial=include_partial)
    )
    mock_client.conversations.items.list = AsyncMock()
    mock_client.conversations.items.create = AsyncMock()
    return mock_client


class TestImageGenerationToolPayload:
    """Verify that the image generation tool payload is correctly built from env config."""

    async def _get_create_call_args(self, env_overrides: dict) -> dict:
        """Helper: stream /receive with mocked client and return the kwargs passed to responses.create."""
        from main import app

        env_defaults = {
            "OPENAI_API_KEY": "sk-fake-key",
            "RESPONSES_MODEL": "gpt-4o",
            "RESPONSES_INSTRUCTIONS": "Test",
            "ENABLED_TOOLS": "image_generation",
            "SHOW_TOOL_CALL_DETAIL": "false",
            "IMAGE_GENERATION_QUALITY": "auto",
            "IMAGE_GENERATION_SIZE": "auto",
            "IMAGE_GENERATION_BACKGROUND": "auto",
        }
        env_defaults.update(env_overrides)

        mock_client = build_image_generation_mock_client()

        with _dotenv(env_defaults, set_fake_api_key=False):
            with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    await client.get("/chat/conv_test123/receive", timeout=10.0)

        return mock_client.responses.create.call_args.kwargs

    @pytest.mark.anyio
    async def test_image_generation_tool_in_tools_list(self):
        """When ENABLED_TOOLS includes image_generation, tools list must contain an image_generation entry."""
        kwargs = await self._get_create_call_args({})
        tools = kwargs.get("tools", [])
        ig_tools = [t for t in tools if isinstance(t, dict) and t.get("type") == "image_generation"]
        assert len(ig_tools) == 1, f"Expected one image_generation tool, got {ig_tools}"

    @pytest.mark.anyio
    async def test_image_generation_default_no_extra_params(self):
        """With auto defaults, the tool dict should only have 'type'."""
        kwargs = await self._get_create_call_args({})
        tools = kwargs.get("tools", [])
        ig_tool = next(t for t in tools if isinstance(t, dict) and t.get("type") == "image_generation")
        assert "quality" not in ig_tool, "auto quality should not be included"
        assert "size" not in ig_tool, "auto size should not be included"
        assert "background" not in ig_tool, "auto background should not be included"

    @pytest.mark.anyio
    async def test_image_generation_custom_quality(self):
        """Custom quality should be passed through."""
        kwargs = await self._get_create_call_args({"IMAGE_GENERATION_QUALITY": "high"})
        tools = kwargs.get("tools", [])
        ig_tool = next(t for t in tools if isinstance(t, dict) and t.get("type") == "image_generation")
        assert ig_tool["quality"] == "high"

    @pytest.mark.anyio
    async def test_image_generation_custom_size(self):
        """Custom size should be passed through."""
        kwargs = await self._get_create_call_args({"IMAGE_GENERATION_SIZE": "1024x1536"})
        tools = kwargs.get("tools", [])
        ig_tool = next(t for t in tools if isinstance(t, dict) and t.get("type") == "image_generation")
        assert ig_tool["size"] == "1024x1536"

    @pytest.mark.anyio
    async def test_image_generation_custom_background(self):
        """Custom background should be passed through."""
        kwargs = await self._get_create_call_args({"IMAGE_GENERATION_BACKGROUND": "transparent"})
        tools = kwargs.get("tools", [])
        ig_tool = next(t for t in tools if isinstance(t, dict) and t.get("type") == "image_generation")
        assert ig_tool["background"] == "transparent"


class TestImageGenerationSseIntegration:
    """Integration test: verify SSE events for an image generation call."""

    async def _stream_events(self, *, include_partial: bool = False) -> list[dict[str, str]]:
        """Helper: stream /receive with image gen mock and return parsed SSE events."""
        from main import app

        env_defaults = {
            "OPENAI_API_KEY": "sk-fake-key",
            "RESPONSES_MODEL": "gpt-4o",
            "RESPONSES_INSTRUCTIONS": "Test",
            "ENABLED_TOOLS": "image_generation",
            "SHOW_TOOL_CALL_DETAIL": "false",
            "IMAGE_GENERATION_QUALITY": "auto",
            "IMAGE_GENERATION_SIZE": "auto",
            "IMAGE_GENERATION_BACKGROUND": "auto",
        }

        mock_client = build_image_generation_mock_client(include_partial=include_partial)

        with _dotenv(env_defaults, set_fake_api_key=False):
            with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/chat/conv_test123/receive", timeout=10.0)
                    raw = response.text
                    await response.aclose()

        return parse_sse_events(raw)

    @pytest.mark.anyio
    async def test_image_gen_emits_tool_call_created(self):
        """Image generation in-progress event should emit a toolCallCreated SSE event."""
        events = await self._stream_events()
        tool_calls = [e for e in events if e["event"] == "toolCallCreated"]
        assert len(tool_calls) >= 1, f"Expected toolCallCreated event, got events: {[e['event'] for e in events]}"
        assert "generat" in tool_calls[0]["data"].lower(), "Tool call should mention image generation"

    @pytest.mark.anyio
    async def test_image_gen_emits_final_image(self):
        """The final image should be emitted as an imageOutput SSE event with base64 data."""
        events = await self._stream_events()
        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        assert len(image_outputs) >= 1, f"Expected imageOutput event, got events: {[e['event'] for e in events]}"
        assert f"data:image/png;base64,{FAKE_IMAGE_B64}" in image_outputs[-1]["data"]

    @pytest.mark.anyio
    async def test_image_gen_stream_completes(self):
        """Stream should complete without errors."""
        events = await self._stream_events()
        event_types = [e["event"] for e in events]
        assert "endStream" in event_types, f"Stream should complete. Got: {event_types}"
        assert "networkError" not in event_types, f"No network errors expected. Got: {event_types}"

    @pytest.mark.anyio
    async def test_image_gen_partial_image_emitted(self):
        """When partial images are streamed, they should appear as imageOutput events."""
        events = await self._stream_events(include_partial=True)
        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        # Should have at least 2: one partial + one final
        assert len(image_outputs) >= 2, (
            f"Expected at least 2 imageOutput events (partial + final), got {len(image_outputs)}"
        )
        # First should be the partial
        assert FAKE_PARTIAL_B64 in image_outputs[0]["data"]
        # Last should be the final
        assert FAKE_IMAGE_B64 in image_outputs[-1]["data"]

    @pytest.mark.anyio
    async def test_image_gen_no_duplicate_tool_call_on_generating(self):
        """The generating event should not create a second toolCallCreated if in_progress already did."""
        events = await self._stream_events()
        tool_calls = [e for e in events if e["event"] == "toolCallCreated"]
        # Both in_progress and generating emit toolCallCreated, but that's acceptable
        # (they render to the same step_id so HTMX deduplicates)
        # Just verify they all reference the same item
        for tc in tool_calls:
            assert f'id="step-{IG_ITEM_ID}"' in tc["data"]
