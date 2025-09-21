import logging
import json
import asyncio
from datetime import datetime
from types import ModuleType
from typing import AsyncGenerator, Dict, Any, Callable, cast
from pydantic import ValidationError
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Form, Depends, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from openai.types.responses import (
    ResponseCreatedEvent, ResponseOutputItemAddedEvent,
    ResponseFunctionCallArgumentsDeltaEvent, ResponseFunctionCallArgumentsDoneEvent,
    ResponseCompletedEvent, ResponseTextDeltaEvent, ResponseRefusalDeltaEvent,
    ResponseFileSearchCallSearchingEvent, ResponseCodeInterpreterCallInProgressEvent,
    ResponseOutputTextAnnotationAddedEvent, ResponseContentPartAddedEvent,
    ResponseFileSearchCallInProgressEvent, ResponseFileSearchCallCompletedEvent,
    ResponseOutputItemDoneEvent, ResponseInProgressEvent, ResponseTextDoneEvent,
    ResponseContentPartDoneEvent, ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent, ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCodeInterpreterCallCompletedEvent, ResponseMcpListToolsInProgressEvent,
    ResponseMcpListToolsFailedEvent, ResponseMcpListToolsCompletedEvent,
    ResponseMcpCallArgumentsDoneEvent, ResponseMcpCallCompletedEvent,
    ResponseMcpCallInProgressEvent, ResponseMcpCallArgumentsDeltaEvent
)
from openai._types import NOT_GIVEN
from openai import AsyncOpenAI
from utils.function_calling import Context, FunctionResult
from utils.sse import sse_format
from urllib.parse import quote as url_quote
from routers.files import router as files_router


logger: logging.Logger = logging.getLogger("uvicorn.error")


router: APIRouter = APIRouter(
    prefix="/chat/{conversation_id}",
    tags=["chat"]
)

# Jinja2 templates
templates = Jinja2Templates(directory="templates")


def wrap_for_oob_swap(step_id: str, text_value: str) -> str:
    return f'<span hx-swap-oob="beforeend:#step-{step_id}">{text_value}</span>'


@router.post("/send")
async def send_message(
    request: Request,
    conversation_id: str,
    userInput: str = Form(...),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    # Create a new conversation item for the user's message
    await client.conversations.items.create(
        conversation_id=conversation_id,
        items=[{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": f"System: Today's date is {datetime.today().strftime('%Y-%m-%d')}\n{userInput}"
            }]
        }]
    )

    user_message_html = templates.get_template("components/user-message.html").render(
        request=request, user_input=userInput
    )
    assistant_run_html = templates.get_template("components/assistant-run.html").render(
        request=request, conversation_id=conversation_id
    )

    return HTMLResponse(content=(user_message_html + assistant_run_html))


# Route to stream the response from the assistant via server-sent events
@router.get("/receive")
async def stream_response(
    conversation_id: str,
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        # Load config from env
        import os
        import importlib
        from dotenv import load_dotenv
        from utils.config import ToolConfig
        from utils.function_calling import ToolRegistry
        load_dotenv(override=True)
        model = os.getenv("RESPONSES_MODEL", "gpt-5-mini")
        instructions = os.getenv("RESPONSES_INSTRUCTIONS")
        enabled_tools = [t.strip() for t in os.getenv("ENABLED_TOOLS", "").split(",") if t.strip()]
        show_tool_call_detail = os.getenv("SHOW_TOOL_CALL_DETAIL", "false").lower() in ["true", "1"]
        FUNCTION_REGISTRY = ToolRegistry()
        TEMPLATE_REGISTRY: dict[str, str | tuple[str, Callable]] = {}

        # Build tools
        tools: list[Dict[str, Any]] = []
        if "file_search" in enabled_tools:
            vector_store_id = os.getenv("VECTOR_STORE_ID")
            if vector_store_id and vector_store_id.replace("_", "").replace("-", "").isalnum():
                tools.append({"type": "file_search", "vector_store_ids": [vector_store_id]})
        if "code_interpreter" in enabled_tools:
            # Per Responses schema: container requires a type and container_id, not id
            tools.append({
                "type": "code_interpreter",
                "container": {"type": "auto"}
            })
        if "function" in enabled_tools or "mcp" in enabled_tools:
            try:
                with open("tool.config.json", "r") as f:
                    TOOL_CONFIG = ToolConfig.model_validate_json(f.read())
            except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Error loading tool.config.json: {e}")
                TOOL_CONFIG = ToolConfig(mcp_servers=[], custom_functions=[])
            if "function" in enabled_tools:
                for function_config in TOOL_CONFIG.custom_functions:
                    function_module: ModuleType = importlib.import_module(function_config.import_path)
                    if not hasattr(function_module, function_config.name):
                        raise ValueError(f"Invalid tool.config.json: Function {function_config.name} not found in {function_config.import_path}")
                    function_fn: Callable[..., Any] = cast(Callable[..., Any], getattr(function_module, function_config.name))
                    FUNCTION_REGISTRY.add_function(function_fn, name=function_config.name)
                    if function_config.template_path:
                        TEMPLATE_REGISTRY.update({
                            function_config.name: function_config.template_path,
                        })
                tool_defs = FUNCTION_REGISTRY.get_tool_def_list()
                tools.extend(tool_defs)
            if "mcp" in enabled_tools:
                tools.extend(TOOL_CONFIG.mcp_servers)

        stream = await client.responses.create(
            input="",
            conversation=conversation_id,
            model=model,
            tools=tools or NOT_GIVEN,
            instructions=instructions,
            parallel_tool_calls=False,
            stream=True
        )

        async def iterate_stream(s, response_id: str = "") -> AsyncGenerator[str, None]:
            nonlocal model, conversation_id, tools, instructions, FUNCTION_REGISTRY, show_tool_call_detail
            current_item_id: str = ""

            try:
                async with s as events:
                    async for event in events:
                        match event:
                            case ResponseCreatedEvent():
                                response_id = event.response.id

                            case ResponseInProgressEvent() | \
                                ResponseFileSearchCallInProgressEvent() | \
                                ResponseFileSearchCallCompletedEvent() | \
                                ResponseTextDoneEvent() | \
                                ResponseContentPartDoneEvent() | \
                                ResponseCodeInterpreterCallCodeDoneEvent() | \
                                ResponseCodeInterpreterCallInterpretingEvent() | \
                                ResponseCodeInterpreterCallCompletedEvent() | \
                                ResponseMcpListToolsInProgressEvent() | \
                                ResponseMcpListToolsCompletedEvent() | \
                                ResponseMcpCallArgumentsDoneEvent() | \
                                ResponseMcpCallCompletedEvent() | \
                                ResponseMcpCallInProgressEvent() | \
                                ResponseFunctionCallArgumentsDoneEvent():
                                # Don't need to handle "in progress" or intermediate "done" events
                                # (though long-running code interpreter interpreting might warrant handling)
                                continue

                            case ResponseMcpListToolsFailedEvent():
                                # TODO: handle this (currently triggers Network/stream error exception handler)
                                continue

                            case ResponseFileSearchCallSearchingEvent() | ResponseCodeInterpreterCallInProgressEvent():
                                tool = event.type.split(".")[1].split("_call")[0]
                                current_item_id = event.item_id
                                yield sse_format(
                                        "toolCallCreated",
                                        templates.get_template('components/assistant-step.html').render(
                                            step_type='toolCall',
                                            step_id=event.item_id,
                                            content=f"Calling {tool} tool..." + ("\n" if isinstance(event, ResponseCodeInterpreterCallInProgressEvent) else "")
                                        )
                                    )

                            case ResponseOutputItemAddedEvent():
                                # Skip reasoning steps by default (later make this configurable and/or mount a thinking indicator)
                                if event.item.id and event.item.type in ["message", "output_text"]:
                                    current_item_id = event.item.id
                                    yield sse_format(
                                        "messageCreated",
                                        templates.get_template("components/assistant-step.html").render(
                                            step_type="assistantMessage",
                                            step_id=event.item.id
                                        )
                                    )
                                if event.item.type in [
                                    "function_call", "mcp_call"
                                ]:
                                    current_item_id = event.item.id
                                    tool_name = event.item.name
                                    yield sse_format(
                                        "toolCallCreated",
                                        templates.get_template('components/assistant-step.html').render(
                                            step_type='toolCall',
                                            step_id=event.item.id,
                                            content=f"Calling {tool_name} tool..." + ("\n" if show_tool_call_detail else "")
                                        )
                                    )

                            case ResponseContentPartAddedEvent():
                                # This event indicates the start of annotations; skip creating a new assistantMessage
                                continue

                            case ResponseTextDeltaEvent() | ResponseRefusalDeltaEvent():
                                if event.delta and current_item_id:
                                    yield sse_format("textDelta", wrap_for_oob_swap(current_item_id, event.delta))

                            case ResponseOutputTextAnnotationAddedEvent():
                                logger.info(f"ResponseOutputTextAnnotationAddedEvent: {event}")
                                if event.annotation and current_item_id:
                                    if event.annotation["type"] == "file_citation":
                                        filename = event.annotation["filename"]
                                        # Emit a literal HTML anchor to avoid markdown parsing edge cases
                                        encoded_filename = url_quote(filename, safe="")
                                        file_url_path = files_router.url_path_for("download_stored_file", file_name=encoded_filename)
                                        citation = f"(<a href=\"{file_url_path}\">†</a>)"
                                        yield sse_format("textDelta", wrap_for_oob_swap(current_item_id, citation))
                                    elif event.annotation["type"] == "container_file_citation":
                                        container_id = event.annotation["container_id"]
                                        file_id = event.annotation["file_id"]
                                        file = await client.containers.files.retrieve(file_id, container_id=container_id)
                                        container_file_path = file.path
                                        file_url_path = files_router.url_path_for("download_container_file", container_id=container_id, file_id=file_id)
                                        replacement_payload = f"sandbox:{container_file_path}|{file_url_path}"
                                        yield sse_format("textReplacement", wrap_for_oob_swap(current_item_id, replacement_payload))
                                    else:
                                        logger.error(f"Unhandled annotation type: {event.annotation['type']}")

                            case ResponseCodeInterpreterCallCodeDeltaEvent():
                                if event.delta and current_item_id:
                                    yield sse_format("toolDelta", wrap_for_oob_swap(current_item_id, event.delta))

                            case ResponseFunctionCallArgumentsDeltaEvent() | ResponseMcpCallArgumentsDeltaEvent():
                                if show_tool_call_detail:
                                    current_item_id = event.item_id
                                    delta = event.delta
                                    yield sse_format("toolDelta", wrap_for_oob_swap(current_item_id, str(delta)))

                            case ResponseOutputItemDoneEvent():
                                if event.item.type == "function_call":
                                    current_item_id = event.item.id
                                    function_name = event.item.name
                                    arguments_json = json.loads(event.item.arguments)

                                    # Dispatch via registry
                                    result: FunctionResult[Any] = await FUNCTION_REGISTRY.call(function_name, arguments_json, context=Context())

                                    # Render output (custom widget for weather, generic otherwise)
                                    try:
                                        if function_name in TEMPLATE_REGISTRY:
                                            tpl = TEMPLATE_REGISTRY[function_name]
                                            if isinstance(tpl, tuple):
                                                tpl_name, context_builder = tpl
                                                html = templates.get_template(tpl_name).render(**context_builder(result))
                                            else:
                                                html = templates.get_template(tpl).render(tool=result)
                                            yield sse_format("toolOutput", html)
                                        else:
                                            payload = result.model_dump(exclude_none=True)
                                            yield sse_format("toolOutput", f"<pre>{json.dumps(payload, indent=2)}</pre>")
                                    except Exception as e:
                                        logger.error(f"Error rendering tool output for '{function_name}': {e}")
                                        # Fallback to raw JSON
                                        yield sse_format("toolOutput", f"<pre>{json.dumps(result.model_dump(exclude_none=True))}</pre>")

                                    # Submit outputs and continue streaming
                                    items = await client.conversations.items.list(
                                        conversation_id=conversation_id
                                    )
                                    function_call_item = next((item for item in items.data if item.id == current_item_id), None)
                                    if function_call_item:
                                        call_id = function_call_item.call_id
                                        await client.conversations.items.create(
                                            conversation_id=conversation_id,
                                            items=[{
                                                "type": "function_call_output",
                                                "call_id": call_id,
                                                "output": json.dumps(result.model_dump(exclude_none=True))
                                            }]
                                        )
                                        next_stream = await client.responses.create(
                                            input="",
                                            conversation=conversation_id,
                                            model=model,
                                            tools=tools or NOT_GIVEN,
                                            instructions=instructions,
                                            parallel_tool_calls=False,
                                            stream=True
                                        )
                                        async for out in iterate_stream(next_stream, response_id):
                                            yield out

                            case ResponseCompletedEvent():
                                yield sse_format("runCompleted", "<span hx-swap-oob=\"outerHTML:.dots\"></span>")
                                yield sse_format("endStream", "DONE")

                            case _:
                                logger.error(f"Unhandled event: {event}")
            except asyncio.CancelledError:
                # Important: let cancellation/cleanup signals propagate
                raise
            except Exception as e:
                logger.error(f"Network/stream error: {e}")
                # Ensure loader clears
                yield sse_format("runCompleted", "<span hx-swap-oob=\"outerHTML:.dots\"></span>")
                # Send a minimal payload so the client handler (which expects data) doesn't warn
                yield sse_format("networkError", "<span></span>")
                # Close the SSE source on the client
                yield sse_format("endStream", "ERROR")
                return

        async for sse in iterate_stream(stream):
            yield sse

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
