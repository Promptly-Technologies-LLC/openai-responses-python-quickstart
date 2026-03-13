"""
Tests for file carousel feature:
1. ResponseOutputItemDoneEvent with ResponseCodeInterpreterToolCall triggers container file listing
   and emits a single fileOutput SSE event with all file cards
2. The file-card template renders correctly for image and non-image files
3. Inline code interpreter images have onclick for click-to-zoom
4. Only assistant-created files appear in the carousel (user files are skipped)
5. Container listing errors are handled gracefully
6. The carousel uses innerHTML OOB swap for full replacement (no duplicates across turns)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseContentPartDoneEvent,
)
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)

from conftest import parse_sse_events, _dotenv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CI_ITEM_ID = "ci_carousel_abc123"
MSG_ITEM_ID = "msg_carousel_456"
CI_RESPONSE_ID = "resp_carousel_xyz789"
CI_CONTAINER_ID = "cntr_carousel_container123"
FILE_ID_IMG = "cfile_carousel_img1"
FILE_ID_CSV = "cfile_carousel_csv1"
FILE_ID_USER = "cfile_carousel_user1"


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


def make_container_file(file_id: str, path: str, source: str = "assistant"):
    """Create a mock container file object."""
    f = MagicMock()
    f.id = file_id
    f.path = path
    f.source = source
    f.container_id = CI_CONTAINER_ID
    return f


def make_stream_with_code_interpreter(
    container_files: list | None = None,
    include_annotation: bool = True,
) -> tuple["MockAsyncStream", list]:
    """Create a mock stream with a code interpreter call.
    Returns (stream, container_files_list)."""
    resp_mock = MagicMock()
    resp_mock.id = CI_RESPONSE_ID

    msg_item = MagicMock()
    msg_item.id = MSG_ITEM_ID
    msg_item.type = "message"

    ci_item_added = MagicMock()
    ci_item_added.id = CI_ITEM_ID
    ci_item_added.type = "code_interpreter_call"

    ci_tool_call = ResponseCodeInterpreterToolCall.model_construct(
        id=CI_ITEM_ID,
        type="code_interpreter_call",
        code="import pandas as pd\ndf.to_csv('data.csv')",
        container_id=CI_CONTAINER_ID,
        outputs=None,
        status="completed",
    )

    events = [
        ResponseCreatedEvent.model_construct(
            type="response.created", response=resp_mock, sequence_number=0,
        ),
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=msg_item,
            output_index=0, sequence_number=1,
        ),
        ResponseTextDeltaEvent.model_construct(
            type="response.text.delta",
            delta="Here are your files:",
            item_id=MSG_ITEM_ID, output_index=0,
            content_index=0, sequence_number=2,
        ),
        ResponseTextDoneEvent.model_construct(
            type="response.output_text.done",
            text="Here are your files:",
            item_id=MSG_ITEM_ID, output_index=0,
            content_index=0, sequence_number=3,
        ),
        ResponseContentPartDoneEvent.model_construct(
            type="response.content_part.done",
            part=MagicMock(),
            item_id=MSG_ITEM_ID, output_index=0,
            content_index=0, sequence_number=4,
        ),
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=ci_item_added,
            output_index=1, sequence_number=5,
        ),
        ResponseCodeInterpreterCallInProgressEvent.model_construct(
            type="response.code_interpreter_call.in_progress",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=6,
        ),
        ResponseCodeInterpreterCallCodeDoneEvent.model_construct(
            type="response.code_interpreter_call.code.done",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=7,
        ),
        ResponseCodeInterpreterCallInterpretingEvent.model_construct(
            type="response.code_interpreter_call.interpreting",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=8,
        ),
    ]

    seq = 9

    # Optionally add an annotation for the image
    if include_annotation:
        events.append(
            ResponseOutputTextAnnotationAddedEvent.model_construct(
                type="response.output_text.annotation.added",
                annotation={
                    "type": "container_file_citation",
                    "container_id": CI_CONTAINER_ID,
                    "file_id": FILE_ID_IMG,
                    "filename": f"{FILE_ID_IMG}.png",
                    "start_index": 0,
                    "end_index": 0,
                },
                annotation_index=0,
                item_id=MSG_ITEM_ID,
                output_index=0,
                content_index=0,
                sequence_number=seq,
            )
        )
        seq += 1

    events.extend([
        ResponseCodeInterpreterCallCompletedEvent.model_construct(
            type="response.code_interpreter_call.completed",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=seq,
        ),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done", item=ci_tool_call,
            output_index=1, sequence_number=seq + 1,
        ),
        ResponseCompletedEvent.model_construct(
            type="response.completed", response=resp_mock, sequence_number=seq + 2,
        ),
    ])

    if container_files is None:
        container_files = [
            make_container_file(FILE_ID_IMG, "/mnt/data/chart.png"),
            make_container_file(FILE_ID_CSV, "/mnt/data/data.csv"),
        ]

    return MockAsyncStream(events), container_files


def build_mock_client(stream: MockAsyncStream, container_files: list) -> AsyncMock:
    """Build a mocked AsyncOpenAI client."""
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=stream)
    mock_client.conversations.items.list = AsyncMock()
    mock_client.conversations.items.create = AsyncMock()

    # Mock containers.files.list
    files_list_result = MagicMock()
    files_list_result.data = container_files
    mock_client.containers.files.list = AsyncMock(return_value=files_list_result)

    # Mock containers.files.retrieve
    async def mock_retrieve(file_id, container_id=None):
        file_mock = MagicMock()
        file_mock.path = f"/mnt/data/{file_id}.csv"
        return file_mock
    mock_client.containers.files.retrieve = AsyncMock(side_effect=mock_retrieve)

    return mock_client


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestFileCarousel:
    """Integration tests for the file carousel feature."""

    async def _stream_events(
        self, stream: MockAsyncStream, container_files: list,
        env_overrides: dict | None = None,
    ) -> list[dict[str, str]]:
        from main import app

        env_defaults = {
            "OPENAI_API_KEY": "sk-fake-key",
            "RESPONSES_MODEL": "gpt-4o",
            "RESPONSES_INSTRUCTIONS": "Test",
            "ENABLED_TOOLS": "code_interpreter",
            "SHOW_TOOL_CALL_DETAIL": "false",
        }
        env_defaults.update(env_overrides or {})

        mock_client = build_mock_client(stream, container_files)

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
    async def test_file_carousel_emits_single_file_output_event(self):
        """Code interpreter completion should emit exactly one fileOutput SSE event
        containing all container files."""
        stream, container_files = make_stream_with_code_interpreter()
        events = await self._stream_events(stream, container_files)

        file_outputs = [e for e in events if e["event"] == "fileOutput"]
        assert len(file_outputs) == 1, (
            f"Expected exactly 1 fileOutput event. Got {len(file_outputs)}: "
            f"{[e['event'] for e in events]}"
        )
        # Both files should be in the single event
        html = file_outputs[0]["data"]
        assert "chart.png" in html
        assert "data.csv" in html

    @pytest.mark.anyio
    async def test_file_card_image_has_thumbnail(self):
        """File card for an image file should contain an <img> tag."""
        stream, container_files = make_stream_with_code_interpreter()
        events = await self._stream_events(stream, container_files)

        file_outputs = [e for e in events if e["event"] == "fileOutput"]
        assert len(file_outputs) == 1
        html = file_outputs[0]["data"]
        assert "<img" in html, f"Expected an image thumbnail in carousel. Got: {html}"
        assert "chart.png" in html

    @pytest.mark.anyio
    async def test_file_card_non_image_has_icon(self):
        """File card for a non-image file should contain the document icon."""
        stream, container_files = make_stream_with_code_interpreter()
        events = await self._stream_events(stream, container_files)

        file_outputs = [e for e in events if e["event"] == "fileOutput"]
        assert len(file_outputs) == 1
        html = file_outputs[0]["data"]
        assert "fileCardIcon" in html, f"Expected a document icon in carousel. Got: {html}"
        assert "data.csv" in html

    @pytest.mark.anyio
    async def test_file_card_has_download_links(self):
        """File cards should contain download links."""
        stream, container_files = make_stream_with_code_interpreter()
        events = await self._stream_events(stream, container_files)

        file_outputs = [e for e in events if e["event"] == "fileOutput"]
        assert len(file_outputs) == 1
        html = file_outputs[0]["data"]
        assert "openai_content" in html, (
            f"File cards should contain download URLs. Got: {html}"
        )

    @pytest.mark.anyio
    async def test_file_output_uses_innerhtml_oob_swap(self):
        """fileOutput should use innerHTML OOB swap to fully replace carousel contents,
        preventing duplicate files across turns."""
        stream, container_files = make_stream_with_code_interpreter()
        events = await self._stream_events(stream, container_files)

        file_outputs = [e for e in events if e["event"] == "fileOutput"]
        assert len(file_outputs) == 1
        html = file_outputs[0]["data"]
        assert 'hx-swap-oob="innerHTML:#file-carousel"' in html, (
            f"fileOutput should use innerHTML OOB swap. Got: {html}"
        )

    @pytest.mark.anyio
    async def test_user_files_excluded_from_carousel(self):
        """Files with source != 'assistant' should not appear in the carousel."""
        container_files = [
            make_container_file(FILE_ID_IMG, "/mnt/data/chart.png", source="assistant"),
            make_container_file(FILE_ID_USER, "/mnt/data/input.txt", source="user"),
        ]
        stream, _ = make_stream_with_code_interpreter(container_files=container_files)
        events = await self._stream_events(stream, container_files)

        file_outputs = [e for e in events if e["event"] == "fileOutput"]
        assert len(file_outputs) == 1
        html = file_outputs[0]["data"]
        assert "chart.png" in html
        assert "input.txt" not in html, (
            f"User file should not appear in carousel. Got: {html}"
        )

    @pytest.mark.anyio
    async def test_no_container_files_no_file_output(self):
        """When container has no assistant files, no fileOutput events should be emitted."""
        container_files: list = []
        stream, _ = make_stream_with_code_interpreter(
            container_files=container_files, include_annotation=False
        )
        events = await self._stream_events(stream, container_files)

        file_outputs = [e for e in events if e["event"] == "fileOutput"]
        assert len(file_outputs) == 0, (
            f"Expected no fileOutput events. Got: {file_outputs}"
        )

    @pytest.mark.anyio
    async def test_container_listing_error_handled_gracefully(self):
        """If container file listing fails, stream should still complete normally."""
        stream, _ = make_stream_with_code_interpreter(include_annotation=False)
        from main import app

        env_defaults = {
            "OPENAI_API_KEY": "sk-fake-key",
            "RESPONSES_MODEL": "gpt-4o",
            "RESPONSES_INSTRUCTIONS": "Test",
            "ENABLED_TOOLS": "code_interpreter",
            "SHOW_TOOL_CALL_DETAIL": "false",
        }

        mock_client = build_mock_client(stream, [])
        mock_client.containers.files.list = AsyncMock(side_effect=Exception("API error"))

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

        events = parse_sse_events(raw)
        event_types = [e["event"] for e in events]
        assert "endStream" in event_types, f"Stream should complete. Got: {event_types}"
        assert "networkError" not in event_types, f"No network errors expected. Got: {event_types}"


class TestFileCarouselLayout:
    """Test that the file carousel appears below messages, not above them.

    The chatContainer uses flex-direction: column-reverse, which means items
    with lower CSS `order` values appear visually lower (closer to bottom).
    The visual stacking from bottom to top follows ascending order values.

    Required visual layout (top to bottom):
        messages (highest order) > file carousel > input form (lowest order)
    """

    def _parse_order(self, css: str, selector: str) -> int | None:
        """Extract the CSS order value for a given selector from raw CSS text."""
        import re
        # Find the block for this selector
        # Handle both .className { and .className\n{
        pattern = re.escape(selector) + r'\s*\{([^}]*)\}'
        match = re.search(pattern, css)
        if not match:
            return None
        block = match.group(1)
        order_match = re.search(r'order:\s*(\d+)', block)
        if not order_match:
            return None
        return int(order_match.group(1))

    def test_carousel_order_between_messages_and_form(self):
        """In column-reverse, carousel order must be > form order and < messages order
        so it appears visually between them (below messages, above form)."""
        from pathlib import Path
        css = Path("static/styles.css").read_text()

        form_order = self._parse_order(css, ".inputForm")
        carousel_order = self._parse_order(css, ".fileCarousel")
        messages_order = self._parse_order(css, ".messages")

        assert form_order is not None, "inputForm must have a CSS order property"
        assert carousel_order is not None, "fileCarousel must have a CSS order property"
        assert messages_order is not None, "messages must have a CSS order property"

        assert form_order < carousel_order < messages_order, (
            f"In column-reverse, order must be: form ({form_order}) < "
            f"carousel ({carousel_order}) < messages ({messages_order}) "
            f"for correct visual stacking"
        )



    """Test that inline code interpreter images have onclick for zoom."""

    async def _stream_events(self, stream: MockAsyncStream, container_files: list) -> list[dict[str, str]]:
        from main import app

        env_defaults = {
            "OPENAI_API_KEY": "sk-fake-key",
            "RESPONSES_MODEL": "gpt-4o",
            "RESPONSES_INSTRUCTIONS": "Test",
            "ENABLED_TOOLS": "code_interpreter",
            "SHOW_TOOL_CALL_DETAIL": "false",
        }

        mock_client = build_mock_client(stream, container_files)

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
    async def test_inline_image_has_onclick_zoom(self):
        """Inline code interpreter images should have onclick='openImagePreview(this.src)'."""
        stream, container_files = make_stream_with_code_interpreter(include_annotation=True)
        events = await self._stream_events(stream, container_files)

        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        assert len(image_outputs) >= 1, (
            f"Expected imageOutput events. Got: {[e['event'] for e in events]}"
        )
        html = image_outputs[0]["data"]
        assert "openImagePreview" in html, (
            f"Inline image should have openImagePreview onclick. Got: {html}"
        )
        assert 'cursor:pointer' in html, (
            f"Inline image should have pointer cursor. Got: {html}"
        )
