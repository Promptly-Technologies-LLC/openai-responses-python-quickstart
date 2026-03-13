"""
Tests for code interpreter image output rendering:
1. container_file_citation annotations for image files emit imageOutput SSE events
2. The imageOutput event contains an <img> tag with the correct download URL
3. Non-image container_file_citation annotations still use textReplacement
4. Annotation handler uses event.item_id (not current_item_id) for correct OOB targeting
5. Stream completes without errors when code interpreter produces images
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallCodeDeltaEvent,
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

CI_ITEM_ID = "ci_test_abc123"
MSG_ITEM_ID = "msg_test_456"
CI_RESPONSE_ID = "resp_ci_test_xyz789"
CI_CONTAINER_ID = "cntr_test_container123"
FILE_ID = "cfile_test_image123"
FILE_ID_2 = "cfile_test_image456"
CSV_FILE_ID = "cfile_test_csv789"


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


def make_code_interpreter_image_stream(
    annotation_count: int = 1,
    include_csv: bool = False,
) -> MockAsyncStream:
    """Create a mock stream that emits a code interpreter call with container_file_citation
    annotations for image output, matching the actual API behavior observed in live testing.

    The actual API event sequence is:
    1. Text message (with text deltas, done)
    2. Code interpreter call (in progress, code deltas, done, interpreting)
    3. container_file_citation annotation (fires AFTER CI, targets the text message via item_id)
    4. Code interpreter completed
    5. OutputItemDone for code interpreter (outputs: None)
    6. ResponseCompleted
    """
    resp_mock = MagicMock()
    resp_mock.id = CI_RESPONSE_ID

    # Text message output item
    msg_item = MagicMock()
    msg_item.id = MSG_ITEM_ID
    msg_item.type = "message"

    # Code interpreter call output item
    ci_item_added = MagicMock()
    ci_item_added.id = CI_ITEM_ID
    ci_item_added.type = "code_interpreter_call"

    # Code interpreter tool call (outputs=None in streaming, as observed in live API)
    ci_tool_call = ResponseCodeInterpreterToolCall.model_construct(
        id=CI_ITEM_ID,
        type="code_interpreter_call",
        code="import matplotlib.pyplot as plt\nplt.plot([1,2,3])\nplt.savefig('chart.png')",
        container_id=CI_CONTAINER_ID,
        outputs=None,
        status="completed",
    )

    events = [
        # 1. Response created
        ResponseCreatedEvent.model_construct(
            type="response.created", response=resp_mock, sequence_number=0,
        ),
        # 2. Text message
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=msg_item,
            output_index=0, sequence_number=1,
        ),
        ResponseTextDeltaEvent.model_construct(
            type="response.text.delta",
            delta="Here is the plot you requested:",
            item_id=MSG_ITEM_ID, output_index=0,
            content_index=0, sequence_number=2,
        ),
        ResponseTextDoneEvent.model_construct(
            type="response.output_text.done",
            text="Here is the plot you requested:",
            item_id=MSG_ITEM_ID, output_index=0,
            content_index=0, sequence_number=3,
        ),
        ResponseContentPartDoneEvent.model_construct(
            type="response.content_part.done",
            part=MagicMock(),
            item_id=MSG_ITEM_ID, output_index=0,
            content_index=0, sequence_number=4,
        ),
        # 3. Code interpreter call
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=ci_item_added,
            output_index=1, sequence_number=5,
        ),
        ResponseCodeInterpreterCallInProgressEvent.model_construct(
            type="response.code_interpreter_call.in_progress",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=6,
        ),
        ResponseCodeInterpreterCallCodeDeltaEvent.model_construct(
            type="response.code_interpreter_call.code.delta",
            delta="import matplotlib", item_id=CI_ITEM_ID,
            output_index=1, sequence_number=7,
        ),
        ResponseCodeInterpreterCallCodeDoneEvent.model_construct(
            type="response.code_interpreter_call.code.done",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=8,
        ),
        ResponseCodeInterpreterCallInterpretingEvent.model_construct(
            type="response.code_interpreter_call.interpreting",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=9,
        ),
    ]

    # 4. container_file_citation annotations (fire AFTER CI interpreting, targeting the text message)
    seq = 10
    file_ids = [FILE_ID, FILE_ID_2]
    for i in range(annotation_count):
        events.append(
            ResponseOutputTextAnnotationAddedEvent.model_construct(
                type="response.output_text.annotation.added",
                annotation={
                    "type": "container_file_citation",
                    "container_id": CI_CONTAINER_ID,
                    "file_id": file_ids[i],
                    "filename": f"{file_ids[i]}.png",
                    "start_index": 0,
                    "end_index": 0,
                },
                annotation_index=i,
                item_id=MSG_ITEM_ID,
                output_index=0,
                content_index=0,
                sequence_number=seq,
            )
        )
        seq += 1

    # Optional: CSV file annotation (non-image)
    if include_csv:
        events.append(
            ResponseOutputTextAnnotationAddedEvent.model_construct(
                type="response.output_text.annotation.added",
                annotation={
                    "type": "container_file_citation",
                    "container_id": CI_CONTAINER_ID,
                    "file_id": CSV_FILE_ID,
                    "filename": f"{CSV_FILE_ID}.csv",
                    "start_index": 0,
                    "end_index": 0,
                },
                annotation_index=annotation_count,
                item_id=MSG_ITEM_ID,
                output_index=0,
                content_index=0,
                sequence_number=seq,
            )
        )
        seq += 1

    # 5. Code interpreter completed + OutputItemDone (outputs: None)
    events.extend([
        ResponseCodeInterpreterCallCompletedEvent.model_construct(
            type="response.code_interpreter_call.completed",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=seq,
        ),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done", item=ci_tool_call,
            output_index=1, sequence_number=seq + 1,
        ),
        # 6. Response completed
        ResponseCompletedEvent.model_construct(
            type="response.completed", response=resp_mock, sequence_number=seq + 2,
        ),
    ])

    return MockAsyncStream(events)


def make_code_interpreter_no_annotation_stream() -> MockAsyncStream:
    """Create a mock stream with code interpreter that produces no annotations (no file output)."""
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
        code="print('hello world')",
        container_id=CI_CONTAINER_ID,
        outputs=None,
        status="completed",
    )

    return MockAsyncStream([
        ResponseCreatedEvent.model_construct(
            type="response.created", response=resp_mock, sequence_number=0,
        ),
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=msg_item,
            output_index=0, sequence_number=1,
        ),
        ResponseTextDeltaEvent.model_construct(
            type="response.text.delta",
            delta="The output was: hello world",
            item_id=MSG_ITEM_ID, output_index=0,
            content_index=0, sequence_number=2,
        ),
        ResponseCodeInterpreterCallInProgressEvent.model_construct(
            type="response.code_interpreter_call.in_progress",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=3,
        ),
        ResponseCodeInterpreterCallCompletedEvent.model_construct(
            type="response.code_interpreter_call.completed",
            item_id=CI_ITEM_ID, output_index=1, sequence_number=4,
        ),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done", item=ci_tool_call,
            output_index=1, sequence_number=5,
        ),
        ResponseCompletedEvent.model_construct(
            type="response.completed", response=resp_mock, sequence_number=6,
        ),
    ])


def build_code_interpreter_mock_client(stream: MockAsyncStream) -> AsyncMock:
    """Build a mocked AsyncOpenAI client for code interpreter tests."""
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=stream)
    mock_client.conversations.items.list = AsyncMock()
    mock_client.conversations.items.create = AsyncMock()

    # Mock containers.files.retrieve to return file metadata
    async def mock_retrieve(file_id, container_id=None):
        file_mock = MagicMock()
        file_mock.path = f"/mnt/data/{file_id}.png"
        return file_mock

    mock_client.containers.files.retrieve = AsyncMock(side_effect=mock_retrieve)

    return mock_client


# ---------------------------------------------------------------------------
# Integration tests: SSE stream for code interpreter image outputs
# ---------------------------------------------------------------------------

async def _stream_ci_events(
    stream: MockAsyncStream, env_overrides: dict | None = None
) -> list[dict[str, str]]:
    """Helper: set env, patch OpenAI client, stream /receive, return parsed SSE events."""
    from main import app

    env_defaults = {
        "OPENAI_API_KEY": "sk-fake-key",
        "RESPONSES_MODEL": "gpt-4o",
        "RESPONSES_INSTRUCTIONS": "Test",
        "ENABLED_TOOLS": "code_interpreter",
        "SHOW_TOOL_CALL_DETAIL": "false",
    }
    env_defaults.update(env_overrides or {})

    mock_client = build_code_interpreter_mock_client(stream)

    with _dotenv(env_defaults, set_fake_api_key=False):
        with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/chat/conv_test123/receive",
                    timeout=10.0,
                )
                raw = response.text
                # Explicitly close the response to prevent dangling connections
                # that cause "Event loop is closed" errors during teardown.
                await response.aclose()

    return parse_sse_events(raw)


class TestCodeInterpreterImageAnnotation:
    """Integration test: verify that container_file_citation annotations for image
    files emit imageOutput SSE events with the correct download URL."""

    @pytest.mark.anyio
    async def test_image_annotation_emits_image_output_sse_event(self):
        """When a container_file_citation annotation has a .png filename,
        an imageOutput SSE event must be emitted."""
        stream = make_code_interpreter_image_stream()
        events = await _stream_ci_events(stream)

        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        assert len(image_outputs) >= 1, (
            f"Expected at least one imageOutput event. Got events: {[e['event'] for e in events]}"
        )

    @pytest.mark.anyio
    async def test_image_output_contains_img_tag_with_download_url(self):
        """The imageOutput SSE event must contain an <img> tag pointing to the
        container file download endpoint."""
        stream = make_code_interpreter_image_stream()
        events = await _stream_ci_events(stream)

        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        assert len(image_outputs) >= 1

        html = image_outputs[0]["data"]
        assert "<img" in html, f"imageOutput must contain an <img> tag. Got: {html}"
        # Should contain the download URL path for the container file
        assert CI_CONTAINER_ID in html, f"imageOutput must reference the container. Got: {html}"
        assert FILE_ID in html, f"imageOutput must reference the file. Got: {html}"
        assert "openai_content" in html, f"imageOutput must point to download endpoint. Got: {html}"

    @pytest.mark.anyio
    async def test_multiple_image_annotations_emit_multiple_events(self):
        """When code interpreter produces multiple images, each container_file_citation
        annotation for an image should emit its own imageOutput event."""
        stream = make_code_interpreter_image_stream(annotation_count=2)
        events = await _stream_ci_events(stream)

        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        assert len(image_outputs) == 2, (
            f"Expected 2 imageOutput events for 2 images. Got {len(image_outputs)}: {image_outputs}"
        )

        # Verify each file ID is referenced
        all_data = " ".join(e["data"] for e in image_outputs)
        assert FILE_ID in all_data, f"First file ID not found. Data: {all_data}"
        assert FILE_ID_2 in all_data, f"Second file ID not found. Data: {all_data}"

    @pytest.mark.anyio
    async def test_no_annotation_does_not_emit_image_event(self):
        """When code interpreter produces no file annotations, no imageOutput event should be emitted."""
        stream = make_code_interpreter_no_annotation_stream()
        events = await _stream_ci_events(stream)

        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        assert len(image_outputs) == 0, (
            f"Expected no imageOutput events. Got: {image_outputs}"
        )

    @pytest.mark.anyio
    async def test_non_image_annotation_does_not_emit_image_event(self):
        """A container_file_citation for a .csv file should NOT emit imageOutput."""
        stream = make_code_interpreter_image_stream(annotation_count=0, include_csv=True)
        events = await _stream_ci_events(stream)

        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        assert len(image_outputs) == 0, (
            f"Expected no imageOutput events for CSV file. Got: {image_outputs}"
        )

        # Should emit a textReplacement instead
        text_replacements = [e for e in events if e["event"] == "textReplacement"]
        assert len(text_replacements) >= 1, (
            f"Expected textReplacement for CSV file. Got events: {[e['event'] for e in events]}"
        )

    @pytest.mark.anyio
    async def test_mixed_image_and_csv_annotations(self):
        """When both image and non-image annotations exist, only images get imageOutput."""
        stream = make_code_interpreter_image_stream(annotation_count=1, include_csv=True)
        events = await _stream_ci_events(stream)

        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        assert len(image_outputs) == 1, (
            f"Expected exactly 1 imageOutput (for the .png). Got {len(image_outputs)}"
        )

        text_replacements = [e for e in events if e["event"] == "textReplacement"]
        assert len(text_replacements) >= 1, (
            f"Expected at least 1 textReplacement (for the .csv). Got events: {[e['event'] for e in events]}"
        )

    @pytest.mark.anyio
    async def test_stream_completes_after_image_annotation(self):
        """The SSE stream should complete normally after emitting image outputs."""
        stream = make_code_interpreter_image_stream()
        events = await _stream_ci_events(stream)

        event_types = [e["event"] for e in events]
        assert "endStream" in event_types, f"Stream should complete. Got: {event_types}"
        assert "networkError" not in event_types, f"No network errors expected. Got: {event_types}"

    @pytest.mark.anyio
    async def test_text_message_still_rendered(self):
        """Text messages should still be rendered alongside image outputs."""
        stream = make_code_interpreter_image_stream()
        events = await _stream_ci_events(stream)

        msg_created = [e for e in events if e["event"] == "messageCreated"]
        assert len(msg_created) >= 1, "Expected a messageCreated event for the text response"

        text_deltas = [e for e in events if e["event"] == "textDelta"]
        assert len(text_deltas) >= 1, "Expected textDelta events"
        assert any("plot" in td["data"] for td in text_deltas)

    @pytest.mark.anyio
    async def test_annotation_uses_event_item_id_not_current_item_id(self):
        """The imageOutput event must target the correct text message element,
        not the code interpreter element. This verifies the handler uses
        event.item_id rather than the stale current_item_id."""
        stream = make_code_interpreter_image_stream()
        events = await _stream_ci_events(stream)

        image_outputs = [e for e in events if e["event"] == "imageOutput"]
        assert len(image_outputs) >= 1

        # The imageOutput should NOT reference the code interpreter item ID
        html = image_outputs[0]["data"]
        assert CI_ITEM_ID not in html, (
            f"imageOutput should not target the CI item. Got: {html}"
        )

    @pytest.mark.anyio
    async def test_tool_call_created_emitted_for_code_interpreter(self):
        """Code interpreter in-progress event should emit toolCallCreated."""
        stream = make_code_interpreter_image_stream()
        events = await _stream_ci_events(stream)

        tool_calls = [e for e in events if e["event"] == "toolCallCreated"]
        assert len(tool_calls) >= 1, (
            f"Expected toolCallCreated event. Got events: {[e['event'] for e in events]}"
        )
        assert "code_interpreter" in tool_calls[0]["data"]


class TestContainerFileAnnotationTargeting:
    """Verify that container_file_citation annotations for non-image files
    also use event.item_id for correct OOB targeting."""

    @pytest.mark.anyio
    async def test_csv_annotation_targets_text_message_not_ci(self):
        """textReplacement for a CSV container_file_citation should target the
        text message element (MSG_ITEM_ID), not the code interpreter element."""
        stream = make_code_interpreter_image_stream(annotation_count=0, include_csv=True)
        events = await _stream_ci_events(stream)

        text_replacements = [e for e in events if e["event"] == "textReplacement"]
        assert len(text_replacements) >= 1

        html = text_replacements[0]["data"]
        # Should target the text message, not the CI item
        assert MSG_ITEM_ID in html, (
            f"textReplacement should target the text message element. Got: {html}"
        )
        assert CI_ITEM_ID not in html, (
            f"textReplacement should NOT target the CI element. Got: {html}"
        )
