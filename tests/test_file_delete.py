"""Regression tests for file deletion — issue #27.

After deleting a file, the file should be immediately removed from the list
even if the OpenAI API hasn't caught up yet (eventual consistency).
"""

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
async def test_deleted_file_not_in_response_when_api_still_returns_it():
    """Regression test for issue #27.

    After deletion, the deleted file should NOT appear in the returned HTML
    even if the vector store list API still returns it (eventual consistency).
    """
    mock_client = AsyncMock()

    deleted_file_id = "file-to-delete"
    other_file_id = "file-to-keep"

    # Pre-deletion retrieve (to get filename for local delete)
    mock_client.files.retrieve = AsyncMock(
        return_value=make_file_object(deleted_file_id, "delete-me.txt")
    )

    # Vector store and file deletion succeed
    mock_client.vector_stores.files.delete = AsyncMock(
        return_value=make_deleted_response(deleted=True)
    )
    mock_client.files.delete = AsyncMock(return_value=MagicMock())

    # Simulate eventual consistency: list still returns the deleted file
    list_result = MagicMock()
    list_result.data = [
        make_vector_store_file(deleted_file_id, "completed"),
        make_vector_store_file(other_file_id, "completed"),
    ]
    mock_client.vector_stores.files.list = AsyncMock(return_value=list_result)

    # files.retrieve for get_files_for_vector_store: deleted file raises, other succeeds
    async def retrieve_side_effect(file_id):
        if file_id == deleted_file_id:
            raise Exception("File not found")
        return make_file_object(other_file_id, "keep-me.txt")

    mock_client.files.retrieve = AsyncMock(side_effect=retrieve_side_effect)

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.delete_local_file"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.delete(f"/files/{deleted_file_id}")

    assert response.status_code == 200
    html = response.text

    # Deleted file must NOT appear at all
    assert "delete-me.txt" not in html
    assert deleted_file_id not in html

    # The other file should still be present
    assert "keep-me.txt" in html
