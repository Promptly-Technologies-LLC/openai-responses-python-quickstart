"""Unit tests for audio transcription endpoint."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


@pytest.mark.anyio
async def test_transcribe_returns_text():
    """POST /audio/transcribe with an audio file returns transcribed text."""
    mock_client = AsyncMock()
    transcription = MagicMock()
    transcription.text = "Hello, world!"
    mock_client.audio.transcriptions.create = AsyncMock(return_value=transcription)

    with patch("routers.audio.AsyncOpenAI", return_value=mock_client):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            fake_audio = io.BytesIO(b"\x00" * 100)
            response = await ac.post(
                "/audio/transcribe",
                files={"audio": ("recording.webm", fake_audio, "audio/webm")},
            )

    assert response.status_code == 200
    assert response.text == "Hello, world!"

    # Verify whisper-1 model was used
    call_kwargs = mock_client.audio.transcriptions.create.call_args
    assert call_kwargs.kwargs.get("model") == "whisper-1"


@pytest.mark.anyio
async def test_transcribe_without_file_returns_422():
    """POST /audio/transcribe without a file returns 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/audio/transcribe")

    assert response.status_code == 422


@pytest.mark.anyio
async def test_mic_button_present_in_index():
    """The index page should contain a mic button."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")

    # May redirect to setup if env vars missing, so check both
    if response.status_code == 200:
        assert "micButton" in response.text
    else:
        # If redirected to setup, verify the index template has the button
        # by reading the template file directly
        with open("templates/index.html") as f:
            template = f.read()
        assert "micButton" in template
