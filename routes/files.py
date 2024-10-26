from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI()

# Pydantic model for request parameters
class FileParams(BaseModel):
    file_id: str

# Initialize the OpenAI client
openai_client = AsyncOpenAI()

@app.get("/files/{file_id}")
async def get_file(file_id: str):
    """
    Endpoint to download a file by file ID.
    """
    try:
        # Retrieve file metadata and content concurrently
        file, file_content = await openai_client.files.retrieve(file_id), await openai_client.files.content(file_id)
        
        # Return the file content as a streaming response
        return StreamingResponse(
            file_content.body,
            headers={"Content-Disposition": f'attachment; filename="{file.filename}"'}
        )
    except Exception as e:
        # Handle exceptions and return an HTTP error response
        raise HTTPException(status_code=500, detail=str(e))

