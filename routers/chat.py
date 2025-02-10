import logging
import time
from typing import Any
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Form, Depends, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from openai import AsyncOpenAI
from openai.resources.beta.threads.runs.runs import AsyncAssistantStreamManager
from openai.types.beta.assistant_stream_event import (
    ThreadMessageCreated, ThreadMessageDelta, ThreadRunCompleted,
    ThreadRunRequiresAction, ThreadRunStepCreated, ThreadRunStepDelta
)
from openai.types.beta.threads.run import RequiredAction
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, Form, HTTPException
from pydantic import BaseModel
import json




# Import our get_weather method
from utils.weather import get_weather
from utils.sse import sse_format

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
    """
    data is expected to be something like

    {
      "tool_outputs": {"location": "City", "temperature": 70, "conditions": "Sunny"},
      "runId": "some-run-id",
    }
    """
    try:
        outputs_list = [data["tool_outputs"]]

        stream_manager = client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=thread_id,
            run_id=data["runId"],
            tool_outputs=outputs_list,
        )

        return stream_manager

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

    async def event_generator():
        step_counter: int = 0
        required_action: RequiredAction | None = None
        stream_manager: AsyncAssistantStreamManager = client.beta.threads.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id
        )

        async with stream_manager as event_handler:
            async for event in event_handler:
                logger.info(f"{event}")
                
                if isinstance(event, ThreadMessageCreated):
                    step_counter += 1

                    yield sse_format(
                        "messageCreated",
                        templates.get_template("components/assistant-step.html").render(
                            step_type="assistantMessage",
                            stream_name=f"textDelta{step_counter}"
                        )
                    )
                    time.sleep(0.25) # Give the client time to render the message

                if isinstance(event, ThreadMessageDelta):
                    logger.info(f"Sending delta with name textDelta{step_counter}")
                    yield sse_format(
                        f"textDelta{step_counter}",
                        event.data.delta.content[0].text.value
                    )


                if isinstance(event, ThreadRunStepCreated) and event.data.type == "tool_calls":
                    yield sse_format(
                        f"toolCallCreated",
                        templates.get_template('components/assistant-step.html').render(
                            step_type='toolCall', stream_name=f'toolDelta{step_counter}'
                        )
                    )

                if isinstance(event, ThreadRunStepDelta) and event.data.type == "tool_calls":
                    if event.data.delta.step_details.tool_calls[0].function.name:
                        yield sse_format(
                            f"toolDelta{step_counter}",
                            event.data.delta.step_details.tool_calls[0].function.name + "\n"
                        )
                    elif event.data.delta.step_details.tool_calls[0].function.arguments:
                        yield sse_format(
                            f"toolDelta{step_counter}",
                            event.data.delta.step_details.tool_calls[0].function.arguments
                        )

                if isinstance(event, ThreadRunRequiresAction):
                    required_action = event.data.required_action
                    if required_action and required_action.submit_tool_outputs:
                        # Exit the for loop and context manager
                        break

                if isinstance(event, ThreadRunCompleted):
                    yield sse_format("endStream", "DONE")
    
        if required_action and required_action.submit_tool_outputs:
            # Get the weather
            for tool_call in required_action.submit_tool_outputs.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                    location = args.get("location", "Unknown")
                except Exception as err:
                    logger.error(f"Failed to parse function arguments: {err}")
                    location = "Unknown"

            weather_output = get_weather(location)
            logger.info(f"Weather output: {weather_output}")

            data_for_tool = {
                "tool_outputs": weather_output,
                "runId": event.data.id,
            }
            stream_manager: AsyncAssistantStreamManager = await post_tool_outputs(client, data_for_tool, thread_id)
        
            # We here need to run the whole stream management loop again

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
