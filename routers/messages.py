import os
import logging
from dotenv import load_dotenv
from fastapi import APIRouter, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openai.resources.beta.threads.runs.runs import AsyncAssistantStreamManager
import json
from utils.threads import create_thread

logger: logging.Logger = logging.getLogger("uvicorn.error")

# Initialize the router
router: APIRouter = APIRouter()

# Send a new message to a thread
@router.post("/send_message")
async def post_message(
    userInput: str = Form(...),
    thread_id: str | None = Form(None),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> dict:
    # Create a new assistant chat thread if no thread ID is provided
    if not thread_id or thread_id == "None" or thread_id == "null":
        thread_id: str = await create_thread()

    # Create a new message in the thread
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=userInput
    )
    
    return {"thread_id": thread_id}

@router.get("/stream_response")
async def stream_response(
    thread_id: str | None = None,
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> StreamingResponse:
    if not thread_id:
        raise HTTPException(status_code=400, message="thread_id is required")
   
    # Create a generator to stream the response from the assistant
    load_dotenv()
    async def event_generator():
        stream: AsyncAssistantStreamManager = client.beta.threads.runs.stream(
            assistant_id=os.getenv("ASSISTANT_ID"),
            thread_id=thread_id
        )
        async with stream as stream_manager:
            async for text in stream_manager.text_deltas:
                yield f"data: {json.dumps({'text': text, 'thread_id': thread_id})}\n\n"
            
            # Send a done event when the stream is complete
            yield f"data: {json.dumps({'complete': True})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
