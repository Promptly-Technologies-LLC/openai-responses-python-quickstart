"""Unit tests for multi-file upload ensuring all files appear in the response."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


def make_vector_store_file(file_id: str, status: str = "in_progress"):
    """Create a mock VectorStoreFile object."""
    vs_file = MagicMock()
    vs_file.id = file_id
    vs_file.status = status
    vs_file.last_error = None
    return vs_file


def make_file_object(file_id: str, filename: str):
    """Create a mock FileObject."""
    file_obj = MagicMock()
    file_obj.id = file_id
    file_obj.filename = filename
    return file_obj


@pytest.mark.anyio
async def test_upload_multiple_files_all_appear_in_response():
    """All uploaded files should appear in the response even if the list API
    hasn't caught up yet (eventual consistency)."""

    mock_client = AsyncMock()

    # files.create returns file objects with IDs
    file_ids = ["file-aaa", "file-bbb", "file-ccc"]
    filenames = ["doc1.txt", "doc2.txt", "doc3.txt"]

    mock_client.files.create = AsyncMock(
        side_effect=[make_file_object(fid, fn) for fid, fn in zip(file_ids, filenames)]
    )

    # vector_stores.files.create returns VectorStoreFile objects
    mock_client.vector_stores.files.create = AsyncMock(
        side_effect=[make_vector_store_file(fid) for fid in file_ids]
    )

    # Simulate eventual consistency: list API returns only the first file
    list_result = MagicMock()
    list_result.data = [make_vector_store_file("file-aaa", "completed")]
    mock_client.vector_stores.files.list = AsyncMock(return_value=list_result)

    # files.retrieve for the one file that shows up in list
    mock_client.files.retrieve = AsyncMock(
        return_value=make_file_object("file-aaa", "doc1.txt")
    )

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.store_file"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/files/",
                files=[
                    ("files", ("doc1.txt", io.BytesIO(b"content1"), "text/plain")),
                    ("files", ("doc2.txt", io.BytesIO(b"content2"), "text/plain")),
                    ("files", ("doc3.txt", io.BytesIO(b"content3"), "text/plain")),
                ],
                data={"purpose": "assistants"},
            )

    assert response.status_code == 200
    html = response.text

    # All three filenames must be present in the returned HTML
    assert "doc1.txt" in html
    assert "doc2.txt" in html
    assert "doc3.txt" in html

    # Should have exactly 3 file items
    assert html.count("fileItem") == 3


@pytest.mark.anyio
async def test_upload_multiple_files_no_duplicates_when_list_is_complete():
    """When the list API returns all files, there should be no duplicates."""

    mock_client = AsyncMock()

    file_ids = ["file-aaa", "file-bbb"]
    filenames = ["doc1.txt", "doc2.txt"]

    mock_client.files.create = AsyncMock(
        side_effect=[make_file_object(fid, fn) for fid, fn in zip(file_ids, filenames)]
    )

    mock_client.vector_stores.files.create = AsyncMock(
        side_effect=[make_vector_store_file(fid) for fid in file_ids]
    )

    # List API returns both files (no consistency lag)
    list_result = MagicMock()
    list_result.data = [
        make_vector_store_file("file-aaa", "completed"),
        make_vector_store_file("file-bbb", "completed"),
    ]
    mock_client.vector_stores.files.list = AsyncMock(return_value=list_result)

    mock_client.files.retrieve = AsyncMock(
        side_effect=[
            make_file_object("file-aaa", "doc1.txt"),
            make_file_object("file-bbb", "doc2.txt"),
        ]
    )

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.store_file"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/files/",
                files=[
                    ("files", ("doc1.txt", io.BytesIO(b"content1"), "text/plain")),
                    ("files", ("doc2.txt", io.BytesIO(b"content2"), "text/plain")),
                ],
                data={"purpose": "assistants"},
            )

    assert response.status_code == 200
    html = response.text

    # Should have exactly 2 file items (no duplicates)
    assert html.count("fileItem") == 2
    assert "doc1.txt" in html
    assert "doc2.txt" in html
