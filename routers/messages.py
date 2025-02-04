import logging
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Form, Depends, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openai.resources.beta.threads.runs.runs import AsyncAssistantStreamManager

logger: logging.Logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

# Initialize the router
router: APIRouter = APIRouter(
    prefix="/assistants/{assistant_id}/messages/{thread_id}",
    tags=["assistants_messages"]
)

# Load Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Send a new message to a thread
@router.post("/send")
async def post_message(
    request: Request,
    assistant_id: str,
    thread_id: str,
    userInput: str = Form(...),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> dict:
    # Create a new message in the thread
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=userInput

    )

    return templates.TemplateResponse(
        "components/chat-turn.html",
        {
            "request": request,
            "user_input": userInput,
            "assistant_id": assistant_id,
            "thread_id": thread_id
        }
    )

@router.get("/receive")
async def stream_response(
    assistant_id: str,
    thread_id: str,
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> StreamingResponse:   
    # Create a generator to stream the response from the assistant
    async def event_generator():
        stream: AsyncAssistantStreamManager = client.beta.threads.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id
        )

        async with stream as stream_manager:
            async for text in stream_manager.text_deltas:
                logger.info(text)
                yield f"data: {text}\n\n"
            
            logger.info("Sending end message")

            # Send a done event when the stream is complete
            yield "event: EndMessage\ndata: DONE\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
