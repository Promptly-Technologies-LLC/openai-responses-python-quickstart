import logging
from typing import Literal
from fastapi import (
    APIRouter, Request, UploadFile, File, HTTPException, Depends, Path, Form
)
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from utils.files import get_or_create_vector_store, get_files_for_vector_store, store_file, retrieve_file, delete_local_file
from utils.streaming import stream_file_content

logger = logging.getLogger("uvicorn.error")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

router = APIRouter(
    prefix="/files",
    tags=["files"]
)


@router.get("/list", response_class=HTMLResponse)
async def list_files(
    request: Request,
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    """Lists files and returns an HTML partial."""
    try:
        vector_store_id = await get_or_create_vector_store(client)
        files = await get_files_for_vector_store(vector_store_id, client)
        return templates.TemplateResponse(
            "components/file-list.html", 
            {"request": request, "files": files}
        )
    except Exception as e:
        logger.error(f"Error generating file list HTML: {e}")
        # Return an error message within the HTML structure
        return HTMLResponse(content=f'<div id="file-list-container"><p class="error-message">Error loading files: {e}</p></div>')


# Modified upload_file
@router.post("/", response_class=HTMLResponse)
async def upload_file(
    request: Request,
    files: list[UploadFile] = File(...),
    purpose: Literal["assistants", "vision"] = Form(default="assistants"),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    """Uploads one or more files, adds them to the vector store, and returns the updated file list HTML."""
    try:
        vector_store_id = await get_or_create_vector_store(client)
    except Exception as e:
        logger.error(f"Error getting or creating vector store: {e}")
        return templates.TemplateResponse(
            "components/file-list.html",
            {"request": request, "error_message": "Error getting or creating vector store"}
        )

    error_messages: list[str] = []

    for file in files:
        file_content = None
        try:
            # 1. Read the file content first
            file_content = await file.read()
            if not file.filename:
                raise ValueError("File has no filename")
            if not file_content:
                raise ValueError("File content is empty")

            # 2. Upload the file content to OpenAI
            openai_file = await client.files.create(
                file=(file.filename, file_content),
                purpose=purpose
            )

            # 3. Add the uploaded file to the vector store
            await client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=openai_file.id
            )
            logger.info(f"File {file.filename} uploaded to OpenAI and added to vector store.")

            # 4. Store the file locally using the same content
            try:
                store_file(file.filename, file_content)
            except Exception as e:
                logger.error(f"Error storing file {file.filename} locally: {e}")
                error_messages.append(f"Error storing {file.filename} locally")

        except ValueError as ve:
            logger.error(f"File validation error for {file.filename}: {ve}")
            error_messages.append(f"{file.filename}: {ve}")
        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {e}")
            error_messages.append(f"Error uploading {file.filename}")

    # Fetch the updated list of files and render the partial
    file_list = []
    try:
        if vector_store_id:
            file_list = await get_files_for_vector_store(vector_store_id, client)
    except Exception as e:
        logger.error(f"Error fetching files: {e}")
        error_messages.append("Error fetching files for assistant")

    # Combine error messages if any
    error_message = "; ".join(error_messages) if error_messages else None

    # Return the response, conditionally including error message
    return templates.TemplateResponse(
        "components/file-list.html",
        {
            "request": request,
            "files": file_list,
            **({"error_message": error_message} if error_message else {})
        }
    )


# Modified delete_file
@router.delete("/{file_id}", response_class=HTMLResponse)
async def delete_file(
    request: Request,
    file_id: str = Path(..., description="The ID of the file to delete"),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    """Deletes a file from the vector store and OpenAI account, then returns the updated file list HTML."""
    error_message = None
    files = []
    vector_store_id = None
    
    try:
        vector_store_id = await get_or_create_vector_store(client)
        
        # Retrieve filename before attempting deletions
        file_to_delete_name = None
        try:
            retrieved_file = await client.files.retrieve(file_id)
            if retrieved_file and retrieved_file.filename:
                file_to_delete_name = retrieved_file.filename
                logger.info(f"Retrieved filename '{retrieved_file.filename}' for deletion.")
            else:
                logger.warning(f"Could not retrieve filename for file_id {file_id}")
        except Exception as retrieve_err:
            logger.error(f"Error retrieving file object {file_id} for filename: {retrieve_err}")

        # Attempt to delete stored file if filename was found
        if file_to_delete_name:
            try:
                delete_local_file(file_to_delete_name)
            except Exception as local_delete_err:
                # Log error but continue with OpenAI/VS deletion
                logger.error(f"Error deleting local file {file_to_delete_name}: {local_delete_err}")

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
                 error_message = "Failed to remove file from vector store." 

        except Exception as delete_error:
            logger.error(f"Error during file deletion process for file {file_id}: {delete_error}")
            error_message = f"Error deleting file: {delete_error}"

    except Exception as vs_error:
        logger.error(f"Error getting or creating vector store: {vs_error}")
        error_message = f"Error accessing vector store: {vs_error}"

    # Always try to fetch the current list of files, even if deletion had issues
    try:
        if vector_store_id:
            files = await get_files_for_vector_store(vector_store_id, client)
        elif not error_message:
             error_message = "Could not retrieve vector store information."
             
    except Exception as fetch_error:
        logger.error(f"Error fetching file list after delete attempt: {fetch_error}")
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


@router.get("/{file_name}")
async def download_stored_file(
    file_name: str = Path(..., description="The name of the file to retrieve")
) -> FileResponse:
    """This endpoint retrieves files uploaded TO openai as file search inputs
    and stored locally in the uploads directory (since OpenAI doesn't serve
    them for download)."""
    return retrieve_file(file_name)


@router.get("/{container_id}/{file_id}/openai_content")
async def download_container_file(
    container_id: str = Path(..., description="The ID of the container the file is stored in"),
    file_id: str = Path(..., description="The ID of the file stored in OpenAI"),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> StreamingResponse:
    """This endpoint retrieves files created by the code interpreter"""
    try:
        file = await client.containers.files.retrieve(file_id, container_id=container_id)
        # base_url workaround because container file download is not supported in the Python client yet
        client.base_url = f"https://api.openai.com/v1/containers/{container_id}"
        file_content = await client.files.content(file_id)
        client.base_url = "https://api.openai.com/v1"
        
        if not hasattr(file_content, 'content'):
            raise HTTPException(status_code=500, detail="File content not available")
            
        # Use stream_file_content helper
        return StreamingResponse(
            stream_file_content(file_content.content), # Assuming stream_file_content handles bytes
            headers={"Content-Disposition": f'attachment; filename="{file.path.split("/")[-1] or file_id}"'}
        )
    except Exception as e:
        logger.error(f"Error downloading file {file_id} from OpenAI: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file from OpenAI: {str(e)}")


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
