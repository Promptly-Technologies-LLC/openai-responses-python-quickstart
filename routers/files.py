import os
import logging
from dotenv import load_dotenv
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

logger = logging.getLogger("uvicorn.error")

# Get assistant ID from environment variables
load_dotenv()
assistant_id: str = os.getenv("ASSISTANT_ID")

router: APIRouter = APIRouter(
    prefix="/assistants/{assistant_id}/files",
    tags=["assistants_files"]
)

# Pydantic model for DELETE request body
class DeleteRequest(BaseModel):
    fileId: str

# Pydantic model for request parameters
class FileParams(BaseModel):
    file_id: str

# Helper function to get or create a vector store
async def get_or_create_vector_store(assistantId: str, client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())) -> str:
    assistant = await client.beta.assistants.retrieve(assistantId)
    if assistant.tool_resources and assistant.tool_resources.file_search and assistant.tool_resources.file_search.vector_store_ids:
        return assistant.tool_resources.file_search.vector_store_ids[0]
    
    vector_store = await client.beta.vectorStores.create(name="sample-assistant-vector-store")
    await client.beta.assistants.update(assistantId, {
        "tool_resources": {
            "file_search": {
                "vector_store_ids": [vector_store.id],
            },
        },
    })
    return vector_store.id


@router.get("/files/{file_id}")
async def get_file(file_id: str):
    try:
        # Retrieve file metadata and content concurrently
        client = AsyncOpenAI()
        file, file_content = await client.files.retrieve(file_id), await client.files.content(file_id)
        
        # Return the file content as a streaming response
        return StreamingResponse(
            file_content.body,
            headers={"Content-Disposition": f'attachment; filename="{file.filename}"'}
        )
    except Exception as e:
        # Handle exceptions and return an HTTP error response
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        client = AsyncOpenAI()

        # Handles multiple files through the form
        # Process file and upload to OpenAI
        vector_store_id = await get_or_create_vector_store(assistant_id)

        openai_file = await client.files.create(
            file=file.file,
            purpose="assistants"
        )
        await client.beta.vectorStores.files.create(vector_store_id, {
            "file_id": openai_file.id
        })
        return {"message": "File uploaded successfully"}
    except Exception as e:
        # Handle exceptions and return an HTTP error response
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_files(client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())):
    # List files in the vector store
    vector_store_id = await get_or_create_vector_store(assistant_id, client)
    file_list = await client.beta.vectorStores.files.list(vector_store_id)
    
    files_array = []
    for file in file_list.data:
        file_details = await client.files.retrieve(file.id)
        vector_file_details = await client.beta.vectorStores.files.retrieve(vector_store_id, file.id)
        files_array.append({
            "file_id": file.id,
            "filename": file_details.filename,
            "status": vector_file_details.status,
        })
    
    return files_array

@router.delete("/delete")
async def delete_file(request: Request, client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())):
    # Delete file from vector store
    body = await request.json()
    delete_request = DeleteRequest(**body)
    vector_store_id = await get_or_create_vector_store(assistant_id, client)
    await client.beta.vectorStores.files.delete(vector_store_id, delete_request.fileId)
    return {"message": "File deleted successfully"}
