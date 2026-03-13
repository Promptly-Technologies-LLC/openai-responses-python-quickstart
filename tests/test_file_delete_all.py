"""Unit tests for the delete-all-files endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


def make_vector_store_file(file_id: str, status: str = "completed"):
    vs_file = MagicMock()
    vs_file.id = file_id
    vs_file.status = status
    vs_file.last_error = None
    return vs_file


def make_file_object(file_id: str, filename: str):
    file_obj = MagicMock()
    file_obj.id = file_id
    file_obj.filename = filename
    return file_obj


def make_deleted_response(deleted: bool = True):
    resp = MagicMock()
    resp.deleted = deleted
    return resp


@pytest.mark.anyio
async def test_delete_all_removes_all_files():
    """DELETE /files/ should delete all files and return an empty list."""
    mock_client = AsyncMock()

    file_ids = ["file-aaa", "file-bbb", "file-ccc"]
    filenames = ["doc1.txt", "doc2.txt", "doc3.txt"]

    # List returns 3 files (used to discover what to delete)
    list_result = MagicMock()
    list_result.data = [make_vector_store_file(fid) for fid in file_ids]
    list_result.has_more = False
    mock_client.vector_stores.files.list = AsyncMock(return_value=list_result)

    # files.retrieve returns filenames for local deletion
    mock_client.files.retrieve = AsyncMock(
        side_effect=[make_file_object(fid, fn) for fid, fn in zip(file_ids, filenames)]
    )

    # Deletions succeed
    mock_client.vector_stores.files.delete = AsyncMock(
        return_value=make_deleted_response(deleted=True)
    )
    mock_client.files.delete = AsyncMock(return_value=MagicMock())

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.delete_local_file") as mock_local_delete,
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.delete("/files/")

    assert response.status_code == 200
    html = response.text

    # Should show empty state
    assert "No files found" in html
    assert "fileItem" not in html

    # All 3 files should have been deleted from VS and OpenAI
    assert mock_client.vector_stores.files.delete.call_count == 3
    assert mock_client.files.delete.call_count == 3

    # All 3 local files should have been deleted
    assert mock_local_delete.call_count == 3


@pytest.mark.anyio
async def test_delete_all_with_no_files():
    """DELETE /files/ with no files should return empty state without errors."""
    mock_client = AsyncMock()

    list_result = MagicMock()
    list_result.data = []
    list_result.has_more = False
    mock_client.vector_stores.files.list = AsyncMock(return_value=list_result)

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.delete("/files/")

    assert response.status_code == 200
    html = response.text
    assert "No files found" in html
    assert "errorMessage" not in html


@pytest.mark.anyio
async def test_delete_all_continues_on_individual_failure():
    """If one file fails to delete, the others should still be deleted."""
    mock_client = AsyncMock()

    file_ids = ["file-aaa", "file-bbb", "file-ccc"]

    list_result = MagicMock()
    list_result.data = [make_vector_store_file(fid) for fid in file_ids]
    list_result.has_more = False
    mock_client.vector_stores.files.list = AsyncMock(return_value=list_result)

    # file-bbb retrieve fails
    async def retrieve_side_effect(file_id):
        if file_id == "file-bbb":
            raise Exception("Not found")
        return make_file_object(file_id, f"{file_id}.txt")

    mock_client.files.retrieve = AsyncMock(side_effect=retrieve_side_effect)

    # VS delete: file-bbb fails, others succeed
    async def vs_delete_side_effect(*, vector_store_id, file_id):
        if file_id == "file-bbb":
            raise Exception("Delete failed")
        return make_deleted_response(deleted=True)

    mock_client.vector_stores.files.delete = AsyncMock(side_effect=vs_delete_side_effect)
    mock_client.files.delete = AsyncMock(return_value=MagicMock())

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.delete_local_file"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.delete("/files/")

    assert response.status_code == 200
    html = response.text

    # Should report the error
    assert "errorMessage" in html

    # The other 2 files should have been deleted from OpenAI
    assert mock_client.files.delete.call_count == 2


@pytest.mark.anyio
async def test_delete_all_paginates_through_all_files():
    """DELETE /files/ should paginate through all pages to delete everything."""
    mock_client = AsyncMock()

    # Page 1: 2 files, has_more=True
    page1 = MagicMock()
    page1.data = [make_vector_store_file("file-1"), make_vector_store_file("file-2")]
    page1.has_more = True
    page1.last_id = "file-2"

    # Page 2: 1 file, has_more=False
    page2 = MagicMock()
    page2.data = [make_vector_store_file("file-3")]
    page2.has_more = False
    page2.last_id = "file-3"

    mock_client.vector_stores.files.list = AsyncMock(side_effect=[page1, page2])

    mock_client.files.retrieve = AsyncMock(
        side_effect=[
            make_file_object("file-1", "a.txt"),
            make_file_object("file-2", "b.txt"),
            make_file_object("file-3", "c.txt"),
        ]
    )
    mock_client.vector_stores.files.delete = AsyncMock(
        return_value=make_deleted_response(deleted=True)
    )
    mock_client.files.delete = AsyncMock(return_value=MagicMock())

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.delete_local_file"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.delete("/files/")

    assert response.status_code == 200

    # All 3 files across both pages should have been deleted
    assert mock_client.vector_stores.files.delete.call_count == 3
    assert mock_client.files.delete.call_count == 3
