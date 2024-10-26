from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI()

# Pydantic model for the response
class ThreadResponse(BaseModel):
    threadId: str

# Initialize the OpenAI client
openai_client = AsyncOpenAI()

@app.post("/threads", response_model=ThreadResponse)
async def create_thread():
    """
    Create a new thread using OpenAI's API and return the thread ID.
    """
    try:
        # Create a new thread
        thread = await openai_client.beta.threads.create()
        return ThreadResponse(threadId=thread.id)
    except Exception as e:
        # Handle exceptions and return an HTTP error response
        raise HTTPException(status_code=500, detail=str(e))
