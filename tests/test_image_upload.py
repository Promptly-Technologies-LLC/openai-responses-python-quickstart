"""Unit tests for image upload in chat send endpoint."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


def make_file_object(file_id: str, filename: str):
    """Create a mock OpenAI FileObject."""
    file_obj = MagicMock()
    file_obj.id = file_id
    file_obj.filename = filename
    return file_obj


def make_conversation_item():
    """Create a mock conversation item response."""
    item = MagicMock()
    item.id = "item-123"
    return item


@pytest.mark.anyio
async def test_send_text_only_no_image():
    """Sending a message without an image should work as before (text only)."""
    mock_client = AsyncMock()
    mock_client.conversations.items.create = AsyncMock(return_value=make_conversation_item())

    with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/chat/conv-123/send",
                data={"userInput": "Hello, world!"},
            )

    assert response.status_code == 200
    html = response.text
    assert "Hello, world!" in html

    # Verify conversation item was created with text-only content
    call_kwargs = mock_client.conversations.items.create.call_args
    items = call_kwargs.kwargs.get("items") or call_kwargs[1].get("items")
    content = items[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_text"


@pytest.mark.anyio
async def test_send_with_image_upload():
    """Sending a message with an image should upload it and include file_id in content."""
    mock_client = AsyncMock()
    mock_client.conversations.items.create = AsyncMock(return_value=make_conversation_item())
    mock_client.files.create = AsyncMock(
        return_value=make_file_object("file-img-123", "test.png")
    )

    # Create a small PNG-like file
    fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/chat/conv-123/send",
                data={"userInput": "What is in this image?"},
                files={"image": ("test.png", fake_image, "image/png")},
            )

    assert response.status_code == 200
    html = response.text
    assert "What is in this image?" in html

    # Verify the image was uploaded to OpenAI with purpose="vision"
    mock_client.files.create.assert_called_once()
    upload_kwargs = mock_client.files.create.call_args
    assert upload_kwargs.kwargs.get("purpose") == "vision"

    # Verify conversation item includes both text and image content
    call_kwargs = mock_client.conversations.items.create.call_args
    items = call_kwargs.kwargs.get("items") or call_kwargs[1].get("items")
    content = items[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["file_id"] == "file-img-123"


@pytest.mark.anyio
async def test_send_with_image_shows_thumbnail_in_response():
    """The HTML response should include an image thumbnail when an image is attached."""
    mock_client = AsyncMock()
    mock_client.conversations.items.create = AsyncMock(return_value=make_conversation_item())
    mock_client.files.create = AsyncMock(
        return_value=make_file_object("file-img-456", "photo.jpg")
    )

    fake_image = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/chat/conv-123/send",
                data={"userInput": "Describe this photo"},
                files={"image": ("photo.jpg", fake_image, "image/jpeg")},
            )

    assert response.status_code == 200
    html = response.text
    # Should contain an img tag pointing to the file content endpoint
    assert "file-img-456" in html
    assert "<img" in html


@pytest.mark.anyio
async def test_send_with_image_field_omitted():
    """If no image field is sent at all, treat as text-only."""
    mock_client = AsyncMock()
    mock_client.conversations.items.create = AsyncMock(return_value=make_conversation_item())

    with patch("routers.chat.AsyncOpenAI", return_value=mock_client):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/chat/conv-123/send",
                data={"userInput": "Just text"},
            )

    assert response.status_code == 200

    # Should NOT have called files.create
    mock_client.files.create.assert_not_called()

    # Content should be text-only
    call_kwargs = mock_client.conversations.items.create.call_args
    items = call_kwargs.kwargs.get("items") or call_kwargs[1].get("items")
    content = items[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_text"
