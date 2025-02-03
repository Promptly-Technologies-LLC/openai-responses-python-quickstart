import os
import logging
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openai.resources.beta.threads.runs.runs import AsyncAssistantStreamManager
import json

logger: logging.Logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

# Initialize the router
router: APIRouter = APIRouter(
    prefix="/assistants/{assistant_id}/messages",
    tags=["assistants_messages"]
)

# Load Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Send a new message to a thread
@router.post("/send")
async def post_message(
    userInput: str = Form(...),
    thread_id: str = Form(),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> dict:
    # Create a new message in the thread
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=userInput
    )

    return templates.TemplateResponse("components/chat-turn.html")

@router.get("/receive")
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
                yield f"data: {text}"
            
            # Send a done event when the stream is complete
            yield f"event: EndMessage"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
