"""
Live integration tests for web search tool support.

These tests hit the real OpenAI API to verify:
1. The web_search_preview tool payload is accepted
2. The streaming event sequence is handled correctly
3. url_citation annotations are received with expected structure
4. The full SSE pipeline (chat router) works end-to-end

Requires a valid OPENAI_API_KEY in .env or environment.
Mark: all tests use @pytest.mark.live to allow selective runs.
"""

import os

import pytest
from openai import AsyncOpenAI

from conftest import REAL_API_KEY, parse_sse_events, _dotenv

_REAL_API_KEY = REAL_API_KEY
_has_real_key = bool(_REAL_API_KEY)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _has_real_key, reason="No real OPENAI_API_KEY available"),
]

# Use a cheap, fast model for integration tests
MODEL = "gpt-4.1-mini"

# Common tool config: low context size to minimize cost/latency
WS_TOOL = {"type": "web_search_preview", "search_context_size": "low"}

# Force web search invocation so tests are deterministic
FORCE_WS = {"type": "web_search_preview"}


@pytest.fixture
async def client():
    c = AsyncOpenAI(api_key=_REAL_API_KEY)
    yield c
    await c.close()


class TestWebSearchApiAcceptance:
    """Verify the OpenAI API accepts our web search tool payload shapes."""

    @pytest.mark.anyio
    async def test_minimal_web_search_payload_accepted(self, client: AsyncOpenAI):
        """API should accept a minimal web_search_preview tool."""
        stream = await client.responses.create(
            model=MODEL,
            tools=[{"type": "web_search_preview"}],
            tool_choice=FORCE_WS,
            input="What is Python?",
            stream=True,
        )
        event_types = []
        async with stream as events:
            async for event in events:
                event_types.append(type(event).__name__)

        assert "ResponseCompletedEvent" in event_types

    @pytest.mark.anyio
    async def test_web_search_with_context_size_accepted(self, client: AsyncOpenAI):
        """API should accept search_context_size parameter."""
        stream = await client.responses.create(
            model=MODEL,
            tools=[{"type": "web_search_preview", "search_context_size": "high"}],
            tool_choice=FORCE_WS,
            input="What is Python?",
            stream=True,
        )
        event_types = []
        async with stream as events:
            async for event in events:
                event_types.append(type(event).__name__)

        assert "ResponseCompletedEvent" in event_types

    @pytest.mark.anyio
    async def test_web_search_with_location_accepted(self, client: AsyncOpenAI):
        """API should accept the full user_location payload."""
        stream = await client.responses.create(
            model=MODEL,
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "low",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "city": "New York",
                    "region": "New York",
                    "timezone": "America/New_York",
                },
            }],
            tool_choice=FORCE_WS,
            input="What is Python?",
            stream=True,
        )
        event_types = []
        async with stream as events:
            async for event in events:
                event_types.append(type(event).__name__)

        assert "ResponseCompletedEvent" in event_types


class TestWebSearchEventSequence:
    """Verify the real event sequence from a web search call."""

    @pytest.mark.anyio
    async def test_web_search_produces_expected_events(self, client: AsyncOpenAI):
        """A web search query should produce the expected event types."""
        stream = await client.responses.create(
            model=MODEL,
            tools=[WS_TOOL],
            tool_choice=FORCE_WS,
            input="What new features were in Python 3.13? Cite sources.",
            stream=True,
        )

        event_types = []
        async with stream as events:
            async for event in events:
                event_types.append(type(event).__name__)

        # Web search events must appear
        assert "ResponseWebSearchCallSearchingEvent" in event_types, (
            f"Expected ResponseWebSearchCallSearchingEvent in {event_types}"
        )
        assert "ResponseWebSearchCallCompletedEvent" in event_types, (
            f"Expected ResponseWebSearchCallCompletedEvent in {event_types}"
        )

        # A text response must follow
        assert "ResponseTextDeltaEvent" in event_types, (
            f"Expected ResponseTextDeltaEvent in {event_types}"
        )

        # Stream must complete
        assert "ResponseCompletedEvent" in event_types

    @pytest.mark.anyio
    async def test_web_search_event_order(self, client: AsyncOpenAI):
        """Web search events should come before the text message events."""
        stream = await client.responses.create(
            model=MODEL,
            tools=[WS_TOOL],
            tool_choice=FORCE_WS,
            input="What is the latest stable release of Rust? Cite your sources.",
            stream=True,
        )

        event_types = []
        async with stream as events:
            async for event in events:
                event_types.append(type(event).__name__)

        # Find indices
        search_idx = event_types.index("ResponseWebSearchCallSearchingEvent")
        completed_idx = event_types.index("ResponseWebSearchCallCompletedEvent")
        text_idx = event_types.index("ResponseTextDeltaEvent")

        assert search_idx < completed_idx < text_idx, (
            f"Expected searching < completed < text, got "
            f"searching={search_idx}, completed={completed_idx}, text={text_idx}"
        )


class TestWebSearchCitations:
    """Verify url_citation annotations from a real web search."""

    @pytest.mark.anyio
    async def test_url_citations_present(self, client: AsyncOpenAI):
        """A 'cite your sources' query should produce url_citation annotations."""
        from openai.types.responses import ResponseOutputTextAnnotationAddedEvent

        stream = await client.responses.create(
            model=MODEL,
            tools=[WS_TOOL],
            tool_choice=FORCE_WS,
            input="What new features were in Python 3.13? Cite your sources.",
            stream=True,
        )

        annotations = []
        async with stream as events:
            async for event in events:
                if isinstance(event, ResponseOutputTextAnnotationAddedEvent):
                    annotations.append(event.annotation)

        url_citations = [a for a in annotations if a.get("type") == "url_citation"]
        assert len(url_citations) >= 1, (
            f"Expected at least one url_citation annotation, got {annotations}"
        )

    @pytest.mark.anyio
    async def test_url_citation_structure(self, client: AsyncOpenAI):
        """url_citation annotations should have the expected fields."""
        from openai.types.responses import ResponseOutputTextAnnotationAddedEvent

        stream = await client.responses.create(
            model=MODEL,
            tools=[WS_TOOL],
            tool_choice=FORCE_WS,
            input="What is the latest Python release? Cite your sources.",
            stream=True,
        )

        annotations = []
        async with stream as events:
            async for event in events:
                if isinstance(event, ResponseOutputTextAnnotationAddedEvent):
                    if event.annotation.get("type") == "url_citation":
                        annotations.append(event.annotation)

        assert len(annotations) >= 1, "Expected at least one url_citation"

        citation = annotations[0]
        assert "url" in citation, f"url_citation missing 'url': {citation}"
        assert "title" in citation, f"url_citation missing 'title': {citation}"
        assert "start_index" in citation, f"url_citation missing 'start_index': {citation}"
        assert "end_index" in citation, f"url_citation missing 'end_index': {citation}"
        assert citation["url"].startswith("http"), f"url should be a URL: {citation['url']}"
        assert isinstance(citation["title"], str) and len(citation["title"]) > 0


class TestWebSearchSsePipeline:
    """End-to-end test: verify the chat router SSE pipeline handles web search correctly."""

    @pytest.mark.anyio
    async def test_full_sse_pipeline_with_web_search(self):
        """Hit the actual /receive endpoint with web_search enabled and verify
        the SSE stream contains the expected event types without errors."""
        from httpx import ASGITransport, AsyncClient
        from main import app

        env = {
            "OPENAI_API_KEY": _REAL_API_KEY,
            "RESPONSES_MODEL": MODEL,
            "RESPONSES_INSTRUCTIONS": "Be brief. Always search the web before answering. Cite sources.",
            "ENABLED_TOOLS": "web_search",
            "WEB_SEARCH_CONTEXT_SIZE": "low",
            "SHOW_TOOL_CALL_DETAIL": "false",
            "WEB_SEARCH_LOCATION_COUNTRY": "",
            "WEB_SEARCH_LOCATION_CITY": "",
            "WEB_SEARCH_LOCATION_REGION": "",
            "WEB_SEARCH_LOCATION_TIMEZONE": "",
        }

        # We need a real conversation. Create one, send a message, then stream.
        real_client = AsyncOpenAI(api_key=_REAL_API_KEY)
        conversation = await real_client.conversations.create()
        conv_id = conversation.id
        await real_client.conversations.items.create(
            conversation_id=conv_id,
            items=[{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Search the web: what is the latest Python release? Cite sources."}],
            }],
        )

        # _dotenv writes .env and restores it; set_fake_api_key=False so
        # the real key reaches the Depends(lambda: AsyncOpenAI()) constructor.
        with _dotenv(env, set_fake_api_key=False):
            # Also set os.environ so the FastAPI Depends picks up the real key
            # (it runs before load_dotenv in the route body).
            os.environ["OPENAI_API_KEY"] = _REAL_API_KEY
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/chat/{conv_id}/receive",
                    timeout=30.0,
                )
                raw = response.text
                await response.aclose()

        events = parse_sse_events(raw)
        event_types = [e["event"] for e in events]

        # Must complete without errors
        assert "endStream" in event_types, (
            f"Expected endStream in SSE events. Got: {event_types}"
        )
        assert "networkError" not in event_types, (
            f"Unexpected networkError in SSE events. Got: {event_types}"
        )

        # Must have toolCallCreated for web search
        assert "toolCallCreated" in event_types, (
            f"Expected toolCallCreated in SSE events. Got: {event_types}"
        )

        # toolCallCreated should mention web_search
        tool_call_events = [e for e in events if e["event"] == "toolCallCreated"]
        assert any("web_search" in e["data"].lower() or "web search" in e["data"].lower()
                    for e in tool_call_events), (
            f"toolCallCreated should mention web_search. Got: {[e['data'][:100] for e in tool_call_events]}"
        )

        # Must have text content
        assert "textDelta" in event_types, (
            f"Expected textDelta in SSE events. Got: {event_types}"
        )

        # Should have url_citation links (we asked for citations)
        text_data = " ".join(e["data"] for e in events if e["event"] == "textDelta")
        assert "https://" in text_data or "http://" in text_data, (
            f"Expected url_citation links in text output. Got text: {text_data[:500]}"
        )
