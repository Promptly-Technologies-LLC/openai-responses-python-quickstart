import logging
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Form, Depends, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, AssistantEventHandler
from openai.resources.beta.threads.runs.runs import AsyncAssistantStreamManager
from openai.types.beta.threads.runs import RunStep, RunStepDelta
from typing_extensions import override
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, AssistantEventHandler
from fastapi import APIRouter, Depends, Form, HTTPException
from pydantic import BaseModel
from typing import Any

logger: logging.Logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)



router: APIRouter = APIRouter(
    prefix="/assistants/{assistant_id}/messages/{thread_id}",
    tags=["assistants_messages"]
)

# Load Jinja2 templates
templates = Jinja2Templates(directory="templates")

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


class CustomEventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()

    @override
    def on_tool_call_created(self, tool_call):
        yield f"<span class='tool-call'>Calling {tool_call.type} tool</span>\n"

    @override
    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                yield f"<span class='code'>{delta.code_interpreter.input}</span>\n"
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        yield f"<span class='console'>{output.logs}</span>\n"
        if delta.type == "function":
            yield
        if delta.type == "file_search":
            yield


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
        stream_manager: AsyncAssistantStreamManager = client.beta.threads.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id
        )

        event_handler = CustomEventHandler()

        async with stream_manager as event_handler:
            async for text in event_handler.text_deltas:
                yield f"data: {text.replace('\n', '<br>')}\n\n"

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
