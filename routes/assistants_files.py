from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI()

# Initialize OpenAI client
openai = AsyncOpenAI(api_key="your-api-key")

# Pydantic model for DELETE request body
class DeleteRequest(BaseModel):
    fileId: str

# Helper function to get or create a vector store
async def get_or_create_vector_store() -> str:
    assistant = await openai.beta.assistants.retrieve(assistantId)
    if assistant.tool_resources and assistant.tool_resources.file_search and assistant.tool_resources.file_search.vector_store_ids:
        return assistant.tool_resources.file_search.vector_store_ids[0]
    
    vector_store = await openai.beta.vectorStores.create(name="sample-assistant-vector-store")
    await openai.beta.assistants.update(assistantId, {
        "tool_resources": {
            "file_search": {
                "vector_store_ids": [vector_store.id],
            },
        },
    })
    return vector_store.id

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Process file and upload to OpenAI
    vector_store_id = await get_or_create_vector_store()
    openai_file = await openai.files.create(
        file=file.file,
        purpose="assistants"
    )
    await openai.beta.vectorStores.files.create(vector_store_id, {
        "file_id": openai_file.id
    })
    return {"message": "File uploaded successfully"}

@app.get("/files")
async def list_files():
    # List files in the vector store
    vector_store_id = await get_or_create_vector_store()
    file_list = await openai.beta.vectorStores.files.list(vector_store_id)
    
    files_array = []
    for file in file_list.data:
        file_details = await openai.files.retrieve(file.id)
        vector_file_details = await openai.beta.vectorStores.files.retrieve(vector_store_id, file.id)
        files_array.append({
            "file_id": file.id,
            "filename": file_details.filename,
            "status": vector_file_details.status,
        })
    
    return files_array

@app.delete("/delete")
async def delete_file(request: Request):
    # Delete file from vector store
    body = await request.json()
    delete_request = DeleteRequest(**body)
    vector_store_id = await get_or_create_vector_store()
    await openai.beta.vectorStores.files.delete(vector_store_id, delete_request.fileId)
    return {"message": "File deleted successfully"}
