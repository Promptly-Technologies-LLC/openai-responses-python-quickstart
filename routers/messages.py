import logging
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Form, Depends, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, AssistantEventHandler
from openai.resources.beta.threads.runs.runs import AsyncAssistantStreamManager
from typing_extensions import override
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, AssistantEventHandler
from fastapi import APIRouter, Depends, Form
from typing_extensions import override

logger: logging.Logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)


router: APIRouter = APIRouter(
    prefix="/assistants/{assistant_id}/messages/{thread_id}",
    tags=["assistants_messages"]
)

# Load Jinja2 templates
templates = Jinja2Templates(directory="templates")

class CustomEventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.message_content = ""

    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
        self.message_content += delta.value

    @override
    def on_text_done(self, text):
        print(f"\nassistant > done", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

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
