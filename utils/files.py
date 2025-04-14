import logging
from typing import List, Dict, Any
from fastapi import Depends
from openai import AsyncOpenAI
from openai.types.file_object import FileObject
from openai.types.vector_stores.vector_store_file import VectorStoreFile
logger = logging.getLogger("uvicorn.error")

# Helper function to get or create a vector store
async def get_or_create_vector_store(assistantId: str, client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())) -> str:
    assistant = await client.beta.assistants.retrieve(assistantId)
    if assistant.tool_resources and assistant.tool_resources.file_search and assistant.tool_resources.file_search.vector_store_ids:
        return assistant.tool_resources.file_search.vector_store_ids[0]
    vector_store = await client.vector_stores.create(name="sample-assistant-vector-store") # TODO: Make this dynamic
    await client.beta.assistants.update(
        assistantId,
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store.id],
            },
        }
    )
    return vector_store.id


async def get_files_for_vector_store(vector_store_id: str, client: AsyncOpenAI) -> List[Dict[str, str | None]]:
    """Helper function to fetch and format file list."""
    try:
        vector_store_files = await client.vector_stores.files.list(
            vector_store_id=vector_store_id,
            order="desc",
        )
        files_data: List[Dict[str, str | None]] = []

        vs_file: VectorStoreFile
        for vs_file in vector_store_files.data:
            try:
                # Retrieve the base file object for filename
                file_object: FileObject = await client.files.retrieve(vs_file.id)
                # The vs_file object itself contains the status
                files_data.append({
                    "id": vs_file.id,
                    "filename": file_object.filename or "unknown_filename",
                    "status": vs_file.status,
                    "last_error": vs_file.last_error.message if vs_file.last_error else None, # Include error message if failed
                    "status_details": getattr(vs_file, 'status_details', None) # Include if available
                })
            except Exception as file_retrieve_error:
                 # If retrieving the base FileObject fails, still list the VS file entry
                 logger.error(f"Failed to retrieve file object {vs_file.id}: {file_retrieve_error}")
                 files_data.append({
                    "id": vs_file.id,
                    "filename": f"File ID: {vs_file.id} (retrieval failed)",
                    "status": vs_file.status,
                    "last_error": vs_file.last_error.message if vs_file.last_error else None,
                    "status_details": getattr(vs_file, 'status_details', None)
                })

        return files_data
    except Exception as e:
        logger.error(f"Error listing files for vector store {vector_store_id}: {e}")
        # Return empty list or re-raise depending on desired behavior
        return []