"""
Live integration tests for image upload (vision) support.

These tests hit the real OpenAI API to verify:
1. Uploading an image file with purpose="vision" is accepted
2. The Responses API accepts input_image with file_id
3. The model produces a text response describing the image
4. The full SSE pipeline works end-to-end with an image message

Requires a valid OPENAI_API_KEY in .env or environment.
Mark: all tests use @pytest.mark.live to allow selective runs.
"""

import io
import os
import struct
import zlib

import pytest
from openai import AsyncOpenAI

from conftest import REAL_API_KEY, parse_sse_events, _dotenv

_REAL_API_KEY = REAL_API_KEY
_has_real_key = bool(_REAL_API_KEY)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _has_real_key, reason="No real OPENAI_API_KEY available"),
]

MODEL = "gpt-4.1-mini"


def make_tiny_png(width: int = 2, height: int = 2, color: tuple = (255, 0, 0)) -> bytes:
    """Create a minimal valid PNG image in memory."""
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = chunk(b"IHDR", ihdr_data)

    # Raw image data: filter byte (0) + RGB pixels per row
    raw = b""
    for _ in range(height):
        raw += b"\x00"  # filter byte
        for _ in range(width):
            raw += bytes(color)

    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")

    return header + ihdr + idat + iend


@pytest.fixture
async def client():
    c = AsyncOpenAI(api_key=_REAL_API_KEY)
    yield c
    await c.close()


class TestVisionApiAcceptance:
    """Verify the OpenAI API accepts image uploads and vision input."""

    @pytest.mark.anyio
    async def test_upload_image_with_vision_purpose(self, client: AsyncOpenAI):
        """API should accept file upload with purpose='vision'."""
        png_bytes = make_tiny_png()
        result = await client.files.create(
            file=("test.png", png_bytes),
            purpose="vision",
        )
        assert result.id.startswith("file-")

        # Cleanup
        await client.files.delete(result.id)

    @pytest.mark.anyio
    async def test_vision_input_produces_text_response(self, client: AsyncOpenAI):
        """Sending an image via file_id should produce a text response."""
        png_bytes = make_tiny_png(color=(0, 0, 255))
        uploaded = await client.files.create(
            file=("blue.png", png_bytes),
            purpose="vision",
        )

        try:
            stream = await client.responses.create(
                model=MODEL,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image briefly."},
                        {"type": "input_image", "file_id": uploaded.id},
                    ],
                }],
                stream=True,
            )

            event_types = []
            text_parts = []
            async with stream as events:
                async for event in events:
                    event_types.append(type(event).__name__)
                    if type(event).__name__ == "ResponseTextDeltaEvent" and event.delta:
                        text_parts.append(event.delta)

            assert "ResponseCompletedEvent" in event_types
            assert "ResponseTextDeltaEvent" in event_types
            full_text = "".join(text_parts)
            assert len(full_text) > 0, "Expected non-empty text response"
        finally:
            await client.files.delete(uploaded.id)


class TestVisionSsePipeline:
    """End-to-end: verify the chat router SSE pipeline handles image messages."""

    @pytest.mark.anyio
    async def test_full_sse_pipeline_with_image(self):
        """Send a message with an image via the /send endpoint, then verify
        the /receive SSE stream produces a text response."""
        from httpx import ASGITransport, AsyncClient
        from main import app

        env = {
            "OPENAI_API_KEY": _REAL_API_KEY,
            "RESPONSES_MODEL": MODEL,
            "RESPONSES_INSTRUCTIONS": "Describe images briefly.",
            "ENABLED_TOOLS": "",
            "SHOW_TOOL_CALL_DETAIL": "false",
        }

        real_client = AsyncOpenAI(api_key=_REAL_API_KEY)
        conversation = await real_client.conversations.create()
        conv_id = conversation.id

        png_bytes = make_tiny_png(color=(0, 255, 0))

        with _dotenv(env, set_fake_api_key=False):
            os.environ["OPENAI_API_KEY"] = _REAL_API_KEY
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Send a message with an image
                send_response = await client.post(
                    f"/chat/{conv_id}/send",
                    data={"userInput": "What color is this image?"},
                    files={"images": ("green.png", io.BytesIO(png_bytes), "image/png")},
                    timeout=15.0,
                )
                assert send_response.status_code == 200
                assert "What color is this image?" in send_response.text
                # Should have image thumbnail
                assert "<img" in send_response.text

                # Now stream the response
                receive_response = await client.get(
                    f"/chat/{conv_id}/receive",
                    timeout=30.0,
                )
                raw = receive_response.text

        events = parse_sse_events(raw)
        event_types = [e["event"] for e in events]

        assert "endStream" in event_types, (
            f"Expected endStream in SSE events. Got: {event_types}"
        )
        assert "networkError" not in event_types, (
            f"Unexpected networkError in SSE events. Got: {event_types}"
        )
        assert "textDelta" in event_types, (
            f"Expected textDelta in SSE events. Got: {event_types}"
        )
