import logging
import time
from typing import Any
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Form, Depends, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from openai import AsyncOpenAI
from openai.resources.beta.threads.runs.runs import AsyncAssistantStreamManager
from openai.types.beta.assistant_stream_event import ThreadMessageCreated, ThreadMessageDelta, ThreadRunCompleted
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, Form, HTTPException
from pydantic import BaseModel


logger: logging.Logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)


router: APIRouter = APIRouter(
    prefix="/assistants/{assistant_id}/messages/{thread_id}",
    tags=["assistants_messages"]
)

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Utility function for submitting tool outputs to the assistant
class ToolCallOutputs(BaseModel):
    tool_outputs: Any
    runId: str

async def post_tool_outputs(client: AsyncOpenAI, data: dict, thread_id: str):

    try:
        # Parse the JSON body into the ToolCallOutputs model
        tool_call_outputs = ToolCallOutputs(**data)

        # Submit tool outputs stream
        stream = await client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id,
            tool_call_outputs.runId,
            {"tool_outputs": tool_call_outputs.tool_outputs}
        )

        # Return the stream as a response
        return stream.to_readable_stream()
    except Exception as e:
        logger.error(f"Error submitting tool outputs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Route to submit a new user message to a thread and mount a component that
# will start an assistant run stream
@router.post("/send")
async def send_message(
    request: Request,
    assistant_id: str,
    thread_id: str,
    userInput: str = Form(...),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    # Create a new message in the thread
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=userInput
    )

    # Render the component templates with the context
    user_message_html = templates.get_template("components/user-message.html").render(user_input=userInput)
    assistant_run_html = templates.get_template("components/assistant-run.html").render(
        assistant_id=assistant_id,
        thread_id=thread_id
    )

    return HTMLResponse(
        content=(
            user_message_html +
            assistant_run_html
        )
    )


# Route to stream the response from the assistant via server-sent events
@router.get("/receive")
async def stream_response(
    assistant_id: str,
    thread_id: str,
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> StreamingResponse:   
    
    # Create a generator to stream the response from the assistant
    async def event_generator():
        step_counter: int = 0
        stream_manager: AsyncAssistantStreamManager = client.beta.threads.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id
        )

        async with stream_manager as event_handler:
            async for event in event_handler:
                logger.info(f"{event}")
                
                if isinstance(event, ThreadMessageCreated):
                    step_counter += 1

                    yield (
                        f"event: messageCreated\n"
                        f"data: {templates.get_template("components/assistant-step.html").render(
                            step_type=f"assistantMessage",
                            stream_name=f"textDelta{step_counter}"
                        ).replace("\n", "")}\n\n"
                    )
                    time.sleep(0.25) # Give the client time to render the message

                if isinstance(event, ThreadMessageDelta):
                    logger.info(f"Sending delta with name textDelta{step_counter}")
                    yield (
                        f"event: textDelta{step_counter}\n"
                        f"data: {event.data.delta.content[0].text.value}\n\n"
                    )

                if isinstance(event, ThreadRunCompleted):
                    yield "event: endStream\ndata: DONE\n\n"

            # Send a done event when the stream is complete
            yield "event: endStream\ndata: DONE\n\n"
    

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
