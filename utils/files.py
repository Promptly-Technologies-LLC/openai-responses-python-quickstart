import logging
import os
from typing import List, Dict
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from openai.types.file_object import FileObject
from openai.types.vector_stores.vector_store_file import VectorStoreFile

logger = logging.getLogger("uvicorn.error")

# Helper function to get or create a vector store (env-based, not assistant-bound)
async def get_or_create_vector_store(client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())) -> str:
    from utils.create_assistant import update_env_file
    vs_id = os.getenv("VECTOR_STORE_ID")
    if vs_id:
        return vs_id
    vector_store = await client.vector_stores.create(name="quickstart-vector-store")
    update_env_file("VECTOR_STORE_ID", vector_store.id, logger)
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


def store_file(file_name: str, file_content: bytes):
    """Store a file in the uploads directory."""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True) # Create directory if it doesn't exist
    file_path = os.path.join(upload_dir, file_name)
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"File stored successfully: {file_path}")
    except IOError as e:
        logger.error(f"Error writing file {file_path}: {e}")
        # Re-raise a more specific exception or handle as needed
        # For now, letting the caller's try/except handle it might be okay
        raise


def retrieve_file(file_name: str):
    """Retrieve a file from the uploads directory."""
    upload_dir = "uploads"
    file_path = os.path.join(upload_dir, file_name)

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    # Basic security check: prevent path traversal
    if not os.path.abspath(file_path).startswith(os.path.abspath(upload_dir)):
        logger.error(f"Attempted path traversal: {file_path}")
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        return FileResponse(path=file_path, filename=file_name, media_type='application/octet-stream')
    except Exception as e:
        logger.error(f"Error serving file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Error serving file")


def delete_local_file(file_name: str):
    """Delete a file from the local uploads directory."""
    upload_dir = "uploads"
    file_path = os.path.join(upload_dir, file_name)

    # Basic security check
    if not os.path.abspath(file_path).startswith(os.path.abspath(upload_dir)):
        logger.error(f"Attempted path traversal during delete: {file_path}")
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Successfully deleted local file: {file_path}")
        else:
            logger.warning(f"Attempted to delete local file, but it was not found: {file_path}")
    except OSError as e:
        logger.error(f"Error deleting local file {file_path}: {e}")