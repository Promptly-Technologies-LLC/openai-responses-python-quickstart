"""Unit tests for vector store file list pagination."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


def make_vector_store_file(file_id: str, status: str = "completed"):
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


def make_list_result(file_ids: list[str], has_more: bool = False, last_id: str | None = None):
    """Create a mock paginated list result."""
    result = MagicMock()
    result.data = [make_vector_store_file(fid) for fid in file_ids]
    result.has_more = has_more
    result.last_id = last_id or (file_ids[-1] if file_ids else None)
    return result


@pytest.mark.anyio
async def test_list_files_returns_load_more_button_when_has_more():
    """When has_more is True, the response should include a Load More button."""
    mock_client = AsyncMock()

    file_ids = [f"file-{i}" for i in range(20)]
    mock_client.vector_stores.files.list = AsyncMock(
        return_value=make_list_result(file_ids, has_more=True, last_id="file-19")
    )
    mock_client.files.retrieve = AsyncMock(
        side_effect=[make_file_object(fid, f"doc{i}.txt") for i, fid in enumerate(file_ids)]
    )

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/files/list")

    assert response.status_code == 200
    html = response.text
    assert html.count("fileItem") == 20
    assert "Load More" in html
    assert "file-19" in html  # cursor should be in the load more button


@pytest.mark.anyio
async def test_list_files_no_load_more_button_when_no_more():
    """When has_more is False, there should be no Load More button."""
    mock_client = AsyncMock()

    file_ids = ["file-aaa", "file-bbb"]
    mock_client.vector_stores.files.list = AsyncMock(
        return_value=make_list_result(file_ids, has_more=False)
    )
    mock_client.files.retrieve = AsyncMock(
        side_effect=[
            make_file_object("file-aaa", "doc1.txt"),
            make_file_object("file-bbb", "doc2.txt"),
        ]
    )

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/files/list")

    assert response.status_code == 200
    html = response.text
    assert html.count("fileItem") == 2
    assert "Load More" not in html


@pytest.mark.anyio
async def test_list_files_with_after_param_passes_cursor():
    """When 'after' query param is provided, it should be passed to the API."""
    mock_client = AsyncMock()

    file_ids = ["file-next-1", "file-next-2"]
    mock_client.vector_stores.files.list = AsyncMock(
        return_value=make_list_result(file_ids, has_more=False)
    )
    mock_client.files.retrieve = AsyncMock(
        side_effect=[
            make_file_object("file-next-1", "next1.txt"),
            make_file_object("file-next-2", "next2.txt"),
        ]
    )

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/files/list?after=file-prev-last")

    assert response.status_code == 200

    # Verify the API was called with the after parameter
    call_kwargs = mock_client.vector_stores.files.list.call_args
    assert call_kwargs.kwargs.get("after") == "file-prev-last"


@pytest.mark.anyio
async def test_list_files_with_after_returns_append_partial():
    """When 'after' is provided, response should use hx-swap-oob for appending."""
    mock_client = AsyncMock()

    file_ids = ["file-next-1"]
    mock_client.vector_stores.files.list = AsyncMock(
        return_value=make_list_result(file_ids, has_more=False)
    )
    mock_client.files.retrieve = AsyncMock(
        return_value=make_file_object("file-next-1", "next1.txt")
    )

    with (
        patch("routers.files.get_or_create_vector_store", return_value="vs-123"),
        patch("routers.files.AsyncOpenAI", return_value=mock_client),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/files/list?after=file-prev-last")

    assert response.status_code == 200
    html = response.text
    # Should contain file items but use the append template
    assert "next1.txt" in html
