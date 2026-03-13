"""
Live integration tests for vector store file list pagination.

These tests hit the real OpenAI API to verify:
1. Files persist across multiple uploads (all files appear in the list)
2. Pagination works correctly when the file count exceeds the page limit

Requires a valid OPENAI_API_KEY in .env or environment.
Mark: all tests use @pytest.mark.live to allow selective runs.
"""

import asyncio

import pytest
from openai import AsyncOpenAI

from conftest import REAL_API_KEY
from utils.files import get_files_for_vector_store

_REAL_API_KEY = REAL_API_KEY
_has_real_key = bool(_REAL_API_KEY)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _has_real_key, reason="No real OPENAI_API_KEY available"),
]


@pytest.fixture
async def client():
    c = AsyncOpenAI(api_key=_REAL_API_KEY)
    yield c
    await c.close()


@pytest.fixture
async def vector_store(client: AsyncOpenAI):
    """Create a temporary vector store and clean it up after the test."""
    vs = await client.vector_stores.create(name="pagination-test")
    yield vs
    # Cleanup: delete all files then the vector store
    files = await client.vector_stores.files.list(vector_store_id=vs.id)
    for f in files.data:
        try:
            await client.vector_stores.files.delete(
                vector_store_id=vs.id, file_id=f.id
            )
            await client.files.delete(file_id=f.id)
        except Exception:
            pass
    await client.vector_stores.delete(vector_store_id=vs.id)


async def upload_test_files(
    client: AsyncOpenAI, vector_store_id: str, count: int, prefix: str = "file"
) -> list[str]:
    """Upload `count` small text files and add them to the vector store.
    Returns the list of uploaded file IDs.
    """
    file_ids: list[str] = []
    for i in range(count):
        content = f"Test content for {prefix}-{i}".encode()
        openai_file = await client.files.create(
            file=(f"{prefix}-{i}.txt", content),
            purpose="assistants",
        )
        await client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=openai_file.id,
        )
        file_ids.append(openai_file.id)
    return file_ids


async def wait_for_files(
    client: AsyncOpenAI,
    vector_store_id: str,
    expected_count: int,
    timeout_seconds: int = 60,
) -> None:
    """Poll until the vector store file list returns at least expected_count files."""
    for _ in range(timeout_seconds):
        result = await client.vector_stores.files.list(
            vector_store_id=vector_store_id, limit=100
        )
        if len(result.data) >= expected_count:
            return
        await asyncio.sleep(1)
    raise TimeoutError(
        f"Expected {expected_count} files in vector store {vector_store_id} "
        f"but only found {len(result.data)} after {timeout_seconds}s"
    )


class TestFilePersistence:
    """Verify all uploaded files appear in the list across multiple uploads."""

    @pytest.mark.anyio
    async def test_files_persist_after_single_batch_upload(
        self, client: AsyncOpenAI, vector_store
    ):
        """Upload a batch of files and verify all appear in the list."""
        file_ids = await upload_test_files(client, vector_store.id, count=3, prefix="batch1")
        await wait_for_files(client, vector_store.id, expected_count=3)

        result = await get_files_for_vector_store(vector_store.id, client)
        listed_ids = {f["id"] for f in result["files"]}

        for fid in file_ids:
            assert fid in listed_ids, f"File {fid} missing from list after upload"

    @pytest.mark.anyio
    async def test_files_persist_after_multiple_batch_uploads(
        self, client: AsyncOpenAI, vector_store
    ):
        """Upload two batches and verify ALL files from both batches appear."""
        batch1_ids = await upload_test_files(client, vector_store.id, count=3, prefix="batchA")
        await wait_for_files(client, vector_store.id, expected_count=3)

        batch2_ids = await upload_test_files(client, vector_store.id, count=3, prefix="batchB")
        await wait_for_files(client, vector_store.id, expected_count=6)

        result = await get_files_for_vector_store(vector_store.id, client)
        listed_ids = {f["id"] for f in result["files"]}

        all_ids = batch1_ids + batch2_ids
        for fid in all_ids:
            assert fid in listed_ids, f"File {fid} missing from list after second batch upload"

        assert len(result["files"]) == 6


class TestPaginationBehavior:
    """Verify pagination works when file count exceeds page size."""

    @pytest.mark.anyio
    async def test_pagination_with_small_limit(
        self, client: AsyncOpenAI, vector_store
    ):
        """Upload 5 files with limit=2 and verify pagination returns all files."""
        file_ids = await upload_test_files(client, vector_store.id, count=5, prefix="page")
        await wait_for_files(client, vector_store.id, expected_count=5)

        # First page
        result = await get_files_for_vector_store(
            vector_store.id, client, limit=2
        )
        assert len(result["files"]) == 2
        assert result["has_more"] is True
        assert result["last_id"] is not None

        # Collect all files across pages
        all_files = list(result["files"])
        cursor = result["last_id"]

        while result["has_more"]:
            result = await get_files_for_vector_store(
                vector_store.id, client, after=cursor, limit=2
            )
            all_files.extend(result["files"])
            cursor = result["last_id"]

        # All 5 files should be present
        listed_ids = {f["id"] for f in all_files}
        for fid in file_ids:
            assert fid in listed_ids, f"File {fid} missing when paginating"

        assert len(all_files) == 5

    @pytest.mark.anyio
    async def test_last_page_has_no_more(
        self, client: AsyncOpenAI, vector_store
    ):
        """The final page should have has_more=False."""
        await upload_test_files(client, vector_store.id, count=3, prefix="last")
        await wait_for_files(client, vector_store.id, expected_count=3)

        # Request with limit large enough to get all files
        result = await get_files_for_vector_store(
            vector_store.id, client, limit=100
        )
        assert result["has_more"] is False
        assert len(result["files"]) == 3
