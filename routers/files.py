import os
import logging
from typing import Literal
from dotenv import load_dotenv
from fastapi import (
    APIRouter, Request, UploadFile, File, HTTPException, Depends, Path, Form
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from utils.files import get_or_create_vector_store, get_files_for_vector_store
from utils.streaming import stream_file_content

logger = logging.getLogger("uvicorn.error")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Get assistant ID from environment variables
load_dotenv(override=True)
assistant_id_env = os.getenv("ASSISTANT_ID")
if not assistant_id_env:
    raise ValueError("ASSISTANT_ID environment variable not set")
default_assistant_id: str = assistant_id_env

router = APIRouter(
    prefix="/assistants/{assistant_id}/files",
    tags=["assistants_files"]
)


@router.get("/list", response_class=HTMLResponse)
async def list_files(
    request: Request,
    assistant_id: str = Path(..., description="The ID of the Assistant"),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    """Lists files and returns an HTML partial."""
    try:
        vector_store_id = await get_or_create_vector_store(assistant_id, client)
        files = await get_files_for_vector_store(vector_store_id, client)
        return templates.TemplateResponse(
            "components/file-list.html", 
            {"request": request, "files": files}
        )
    except Exception as e:
        logger.error(f"Error generating file list HTML for assistant {assistant_id}: {e}")
        # Return an error message within the HTML structure
        return HTMLResponse(content=f'<div id="file-list-container"><p class="error-message">Error loading files: {e}</p></div>')


# Modified upload_file
@router.post("/", response_class=HTMLResponse)
async def upload_file(
    request: Request,
    assistant_id: str = Path(..., description="The ID of the Assistant"), 
    file: UploadFile = File(...), 
    purpose: Literal["assistants", "vision"] = Form(default="assistants"),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    """Uploads a file, adds it to the vector store, and returns the updated file list HTML."""
    try:
        vector_store_id = await get_or_create_vector_store(assistant_id, client)
    except Exception as e:
        logger.error(f"Error getting or creating vector store for assistant {assistant_id}: {e}")
        return templates.TemplateResponse(
            "components/file-list.html", 
            {"request": request, "error_message": f"Error getting or creating vector store for assistant"}
        )

    error_message = None
    try:
        # Upload the file to OpenAI
        openai_file = await client.files.create(
            file=(file.filename, file.file),
            purpose=purpose
        )
        
        # Add the uploaded file to the vector store
        await client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=openai_file.id
        )
    except Exception as e:
        logger.error(f"Error uploading file for assistant {assistant_id}: {e}")
        error_message = f"Error uploading file for assistant"

    # Fetch the updated list of files and render the partial
    files = []
    try:
        if vector_store_id:
            files = await get_files_for_vector_store(vector_store_id, client)
    except Exception as e:
        logger.error(f"Error fetching files for assistant {assistant_id}: {e}")
        error_message = f"Error fetching files for assistant"

    # Return the response, conditionally including error message
    return templates.TemplateResponse(
        "components/file-list.html", 
        {
            "request": request, 
            "files": files,
            **({"error_message": error_message} if error_message else {})
        }
    )


# Modified delete_file
@router.delete("/{file_id}", response_class=HTMLResponse) # Changed path to include file_id
async def delete_file(
    request: Request, # Add request for template rendering
    assistant_id: str = Path(..., description="The ID of the Assistant"), 
    file_id: str = Path(..., description="The ID of the file to delete"),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    """Deletes a file from the vector store and OpenAI account, then returns the updated file list HTML."""
    error_message = None
    files = []
    vector_store_id = None
    
    try:
        vector_store_id = await get_or_create_vector_store(assistant_id, client)
        
        # 1. Delete the file association from the vector store
        try:
            deleted_vs_file = await client.vector_stores.files.delete(
                vector_store_id=vector_store_id, 
                file_id=file_id
            )

            # 2. If vector store deletion was successful, attempt to delete the base file object
            if deleted_vs_file.deleted:
                 try:
                     await client.files.delete(file_id=file_id)
                 except Exception as file_delete_error:
                     # Log the warning but don't set error_message, as VS deletion succeeded
                     logger.warning(f"File {file_id} removed from vector store {vector_store_id}, but failed to delete base file object: {file_delete_error}")
            else:
                 # Log the warning and potentially set an error if VS deletion failed
                 logger.warning(f"Failed to remove file {file_id} association from vector store {vector_store_id}")
                 # Decide if this constitutes a full error for the user
                 error_message = f"Failed to remove file from vector store." 

        except Exception as delete_error:
            logger.error(f"Error during file deletion process for file {file_id}, assistant {assistant_id}: {delete_error}")
            error_message = f"Error deleting file: {delete_error}"

    except Exception as vs_error:
        logger.error(f"Error getting or creating vector store for assistant {assistant_id}: {vs_error}")
        error_message = f"Error accessing vector store: {vs_error}"

    # Always try to fetch the current list of files, even if deletion had issues
    try:
        if vector_store_id: # Only fetch if we got the ID
            files = await get_files_for_vector_store(vector_store_id, client)
        elif not error_message: # If we couldn't get VS ID and have no other error, set one
             error_message = "Could not retrieve vector store information."
             
    except Exception as fetch_error:
        logger.error(f"Error fetching file list after delete attempt for assistant {assistant_id}: {fetch_error}")
        # If an error message wasn't already set, set one now. Otherwise, keep the original error.
        if not error_message:
            error_message = f"Error fetching file list: {fetch_error}"

    # Return the response, conditionally including error message
    return templates.TemplateResponse(
        "components/file-list.html",
        {
            "request": request,
            "files": files,
            **({"error_message": error_message} if error_message else {})
        }
    )


# --- Streaming file content routes ---

@router.get("/{file_id}")
async def download_assistant_file(
    file_id: str = Path(..., description="The ID of the file to retrieve"),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> StreamingResponse:
    try:
        file = await client.files.retrieve(file_id)
        file_content = await client.files.content(file_id)
        
        if not hasattr(file_content, 'content'):
            raise HTTPException(status_code=500, detail="File content not available")
            
        return StreamingResponse(
            stream_file_content(file_content.content),
            headers={"Content-Disposition": f'attachment; filename=\"{file.filename or "download"}\"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{file_id}/content")
async def get_assistant_image_content(
    file_id: str,
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> StreamingResponse:
    """
    Streams file content from OpenAI API.
    This route is used to serve images and other files generated by the code interpreter.
    """
    try:
        # Get the file content from OpenAI
        file_content = await client.files.content(file_id)
        file_bytes = file_content.read()  # Remove await since read() returns bytes directly

        # Return the file content as a streaming response
        # Note: In a production environment, you might want to add caching
        return StreamingResponse(
            content=iter([file_bytes]),
            media_type="image/png"  # You might want to make this dynamic based on file type
        )
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
        raise HTTPException(status_code=500, detail=str(e))
