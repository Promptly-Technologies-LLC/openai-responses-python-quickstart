import logging
import json
import asyncio
from html import escape
from datetime import datetime
from types import ModuleType
from typing import AsyncGenerator, Dict, Any, Callable, cast
from pydantic import ValidationError
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Form, Depends, Request, UploadFile, File
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
    ResponseMcpCallInProgressEvent, ResponseMcpCallArgumentsDeltaEvent,
    ResponseWebSearchCallInProgressEvent, ResponseWebSearchCallSearchingEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseImageGenCallInProgressEvent, ResponseImageGenCallGeneratingEvent,
    ResponseImageGenCallCompletedEvent, ResponseImageGenCallPartialImageEvent,
)
from openai.types.responses.response_output_item import McpApprovalRequest, ImageGenerationCall
from openai.types.responses import ResponseFunctionToolCall, ResponseComputerToolCall
from openai.types.responses.response_code_interpreter_tool_call import ResponseCodeInterpreterToolCall
from openai._types import NOT_GIVEN
from openai import AsyncOpenAI
from utils.function_calling import Context, FunctionResult
from utils.computer_use import build_computer_tool, describe_actions, execute_computer_actions
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
    images: list[UploadFile] = File([]),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    # Build multimodal content array
    content: list[dict[str, str]] = [{
        "type": "input_text",
        "text": f"System: Today's date is {datetime.today().strftime('%Y-%m-%d')}\n{userInput}"
    }]

    # If images were uploaded, send each to OpenAI and add to content
    image_file_ids: list[str] = []
    for image in images:
        if image and image.filename and image.size:
            image_bytes = await image.read()
            if image_bytes:
                openai_file = await client.files.create(
                    file=(image.filename, image_bytes),
                    purpose="vision"
                )
                image_file_ids.append(openai_file.id)
                content.append({
                    "type": "input_image",
                    "file_id": openai_file.id,
                })

    # Create a new conversation item for the user's message
    await client.conversations.items.create(
        conversation_id=conversation_id,
        items=[{
            "type": "message",
            "role": "user",
            "content": content
        }]
    )

    user_message_html = templates.get_template("components/user-message.html").render(
        request=request, user_input=userInput, image_file_ids=image_file_ids
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
        if "web_search" in enabled_tools:
            ws_tool: Dict[str, Any] = {"type": "web_search_preview"}
            ctx_size = os.getenv("WEB_SEARCH_CONTEXT_SIZE", "medium").strip()
            if ctx_size in {"low", "medium", "high"}:
                ws_tool["search_context_size"] = ctx_size
            # Build user_location only if at least one field is set
            country = os.getenv("WEB_SEARCH_LOCATION_COUNTRY", "").strip()
            city = os.getenv("WEB_SEARCH_LOCATION_CITY", "").strip()
            region = os.getenv("WEB_SEARCH_LOCATION_REGION", "").strip()
            timezone = os.getenv("WEB_SEARCH_LOCATION_TIMEZONE", "").strip()
            if any([country, city, region, timezone]):
                loc: Dict[str, str] = {"type": "approximate"}
                if country:
                    loc["country"] = country
                if city:
                    loc["city"] = city
                if region:
                    loc["region"] = region
                if timezone:
                    loc["timezone"] = timezone
                ws_tool["user_location"] = loc
            tools.append(ws_tool)
        if "computer_use" in enabled_tools:
            tools.append(build_computer_tool())
        if "image_generation" in enabled_tools:
            ig_tool: Dict[str, Any] = {"type": "image_generation"}
            ig_quality = os.getenv("IMAGE_GENERATION_QUALITY", "auto").strip()
            if ig_quality in {"low", "medium", "high"}:
                ig_tool["quality"] = ig_quality
            ig_size = os.getenv("IMAGE_GENERATION_SIZE", "auto").strip()
            if ig_size in {"1024x1024", "1536x1024", "1024x1536"}:
                ig_tool["size"] = ig_size
            ig_background = os.getenv("IMAGE_GENERATION_BACKGROUND", "auto").strip()
            if ig_background in {"opaque", "transparent"}:
                ig_tool["background"] = ig_background
            tools.append(ig_tool)

        try:
            stream = await client.responses.create(  # type: ignore[call-overload]
                input="",
                conversation=conversation_id,
                model=model,
                tools=tools or NOT_GIVEN,
                instructions=instructions,
                parallel_tool_calls=False,
                stream=True
            )
        except Exception as e:
            logger.error(f"Unhandled error: {e}")
            yield sse_format("runCompleted", "<span hx-swap-oob=\"outerHTML:.dots\"></span>")
            yield sse_format("networkError", f"<span>{escape(str(e))}</span>")
            yield sse_format("endStream", "ERROR")
            return

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
                                ResponseFunctionCallArgumentsDoneEvent() | \
                                ResponseWebSearchCallInProgressEvent() | \
                                ResponseWebSearchCallCompletedEvent() | \
                                ResponseImageGenCallCompletedEvent():
                                # Don't need to handle "in progress" or intermediate "done" events
                                # (though long-running code interpreter interpreting might warrant handling)
                                continue

                            case ResponseMcpListToolsFailedEvent():
                                # TODO: handle this (currently triggers Network/stream error exception handler)
                                continue

                            case ResponseWebSearchCallSearchingEvent():
                                current_item_id = event.item_id
                                yield sse_format(
                                    "toolCallCreated",
                                    templates.get_template('components/assistant-step.html').render(
                                        step_type='toolCall',
                                        step_id=event.item_id,
                                        content="Calling web_search tool..."
                                    )
                                )

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
                                if event.item.type == "function_call":
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
                                if event.item.type == "mcp_call":
                                    current_item_id = event.item.id
                                    server_label = event.item.server_label
                                    tool_name = event.item.name
                                    # Skip if TOOL_CONFIG.mcp_servers item with same server_label has require_approval set to
                                    # "always", as approval form provides adequate notification to the user of the tool call
                                    if any(
                                        server["server_label"] == server_label and server["require_approval"] == "always"
                                        for server in TOOL_CONFIG.mcp_servers
                                    ):
                                        continue
                                    yield sse_format(
                                        "toolCallCreated",
                                        templates.get_template('components/assistant-step.html').render(
                                            step_type='toolCall',
                                            step_id=event.item.id,
                                            content=f"Calling {tool_name} tool..." + ("\n" if show_tool_call_detail else "")
                                        )
                                    )
                                if event.item.type == "computer_call":
                                    current_item_id = event.item.id
                                    yield sse_format(
                                        "toolCallCreated",
                                        templates.get_template('components/assistant-step.html').render(
                                            step_type='toolCall',
                                            step_id=event.item.id,
                                            content="Calling computer_use tool...\n"
                                        )
                                    )
                                # Handle MCP approval requests by rendering an approval UI card
                                if isinstance(event.item, McpApprovalRequest):
                                    current_item_id = event.item.id
                                    # Pretty print arguments JSON if possible
                                    pretty_args: str
                                    try:
                                        pretty_args = json.dumps(json.loads(event.item.arguments), indent=2)
                                    except Exception:
                                        pretty_args = event.item.arguments

                                    # Ensure approval request exists in conversation state with same id
                                    # (Note: this is a workaround to resolve a Responses API bug by triggering a
                                    # commit of conversation state on their server; else approval throws error)
                                    try:
                                        await client.conversations.items.create(
                                            conversation_id=conversation_id,
                                            items=[{
                                                "type": "mcp_approval_request",
                                                "id": event.item.id,
                                                "arguments": event.item.arguments,
                                                "name": event.item.name,
                                                "server_label": event.item.server_label,
                                            }]
                                        )
                                    except Exception as e:
                                        logger.debug(f"Failed to add MCP approval request to conversation: {e}")

                                    approval_card_html = templates.get_template("components/mcp-approval-request.html").render(
                                        conversation_id=conversation_id,
                                        approval_request=event.item,
                                        arguments_pretty=pretty_args
                                    )
                                    yield sse_format(
                                        "mcpApprovalRequest",
                                        templates.get_template('components/assistant-step.html').render(
                                            step_type='approvalRequest',
                                            step_id=event.item.id,
                                            content_html=approval_card_html
                                        )
                                    )

                            case ResponseImageGenCallInProgressEvent() | ResponseImageGenCallGeneratingEvent():
                                current_item_id = event.item_id
                                yield sse_format(
                                    "toolCallCreated",
                                    templates.get_template('components/assistant-step.html').render(
                                        step_type='toolCall',
                                        step_id=event.item_id,
                                        content="Generating image..."
                                    )
                                )

                            case ResponseImageGenCallPartialImageEvent():
                                # Display partial image as it streams in
                                img_html = (
                                    f'<div class="imageOutput">'
                                    f'<img src="data:image/png;base64,{event.partial_image_b64}" '
                                    f'alt="Partial image (generating...)" '
                                    f'onclick="openImagePreview(this.src)" style="cursor:pointer" />'
                                    f'</div>'
                                )
                                yield sse_format("imageOutput", img_html)

                            case ResponseContentPartAddedEvent():
                                # This event indicates the start of annotations; skip creating a new assistantMessage
                                continue

                            case ResponseTextDeltaEvent() | ResponseRefusalDeltaEvent():
                                if event.delta and current_item_id:
                                    yield sse_format("textDelta", wrap_for_oob_swap(current_item_id, event.delta))

                            case ResponseOutputTextAnnotationAddedEvent():
                                # Use event.item_id (not current_item_id) because annotations
                                # may fire after other output items (e.g. code interpreter)
                                # have changed current_item_id to a different element.
                                annotation_target_id = event.item_id or current_item_id
                                if event.annotation and annotation_target_id:
                                    if event.annotation["type"] == "file_citation":
                                        filename = event.annotation["filename"]
                                        # Emit a literal HTML anchor to avoid markdown parsing edge cases
                                        encoded_filename = url_quote(filename, safe="")
                                        file_url_path = files_router.url_path_for("download_stored_file", file_name=encoded_filename)
                                        citation = f"(<a href=\"{file_url_path}\">†</a>)"
                                        yield sse_format("textDelta", wrap_for_oob_swap(annotation_target_id, citation))
                                    elif event.annotation["type"] == "container_file_citation":
                                        container_id = event.annotation["container_id"]
                                        file_id = event.annotation["file_id"]
                                        filename = event.annotation.get("filename", "")
                                        file_url_path = files_router.url_path_for("download_container_file", container_id=container_id, file_id=file_id)
                                        # Check if the file is an image by extension
                                        image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
                                        if filename.lower().endswith(image_extensions):
                                            img_html = (
                                                f'<div class="imageOutput">'
                                                f'<img src="{file_url_path}" alt="Code interpreter output" onclick="openImagePreview(this.src)" style="cursor:pointer" />'
                                                f'</div>'
                                            )
                                            yield sse_format("imageOutput", img_html)
                                        else:
                                            file = await client.containers.files.retrieve(file_id, container_id=container_id)
                                            container_file_path = file.path
                                            replacement_payload = f"sandbox:{container_file_path}|{file_url_path}"
                                            yield sse_format("textReplacement", wrap_for_oob_swap(annotation_target_id, replacement_payload))
                                    elif event.annotation["type"] == "url_citation":
                                        url = event.annotation["url"]
                                        title = event.annotation.get("title", url)
                                        citation = f'(<a href="{escape(url)}" target="_blank" rel="noopener noreferrer">{escape(title)}</a>)'
                                        yield sse_format("textDelta", wrap_for_oob_swap(annotation_target_id, citation))
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
                                if isinstance(event.item, ResponseCodeInterpreterToolCall):
                                    container_id = event.item.container_id
                                    if container_id:
                                        try:
                                            container_files = await client.containers.files.list(container_id=container_id)
                                            cards: list[str] = []
                                            for f in container_files.data:
                                                if f.source != "assistant":
                                                    continue
                                                filename = f.path.split("/")[-1]
                                                file_url = files_router.url_path_for(
                                                    "download_container_file",
                                                    container_id=container_id,
                                                    file_id=f.id,
                                                )
                                                is_image = filename.lower().endswith(
                                                    (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
                                                )
                                                cards.append(templates.get_template("components/file-card.html").render(
                                                    file_url=file_url, filename=filename, is_image=is_image,
                                                ))
                                            if cards:
                                                carousel_html = f'<div hx-swap-oob="innerHTML:#file-carousel">{"".join(cards)}</div>'
                                                yield sse_format("fileOutput", carousel_html)
                                        except Exception as e:
                                            logger.error(f"Error listing container files: {e}")

                                elif isinstance(event.item, ResponseFunctionToolCall):
                                    current_item_id = event.item.id
                                    function_name = event.item.name
                                    arguments_json = json.loads(event.item.arguments)

                                    # Emit complete arguments into the collapsible details
                                    yield sse_format(
                                        "toolDelta",
                                        wrap_for_oob_swap(
                                            current_item_id,
                                            (
                                                '<pre class="toolCallArgs" '
                                                'data-tool-delta="replace">'
                                                f'{escape(json.dumps(arguments_json, indent=2))}'
                                                "</pre>"
                                            ),
                                        ),
                                    )

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
                                        next_stream = await client.responses.create(  # type: ignore[call-overload]
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

                                elif isinstance(event.item, ImageGenerationCall):
                                    current_item_id = event.item.id
                                    if event.item.result:
                                        img_html = (
                                            f'<div class="imageOutput">'
                                            f'<img src="data:image/png;base64,{event.item.result}" '
                                            f'alt="Generated image" '
                                            f'onclick="openImagePreview(this.src)" style="cursor:pointer" />'
                                            f'</div>'
                                        )
                                        yield sse_format("imageOutput", img_html)

                                elif isinstance(event.item, ResponseComputerToolCall):
                                    current_item_id = event.item.id
                                    call_id = event.item.call_id
                                    actions = event.item.actions or []
                                    action_desc = describe_actions(actions)
                                    pending_checks = event.item.pending_safety_checks or []

                                    # Show action details in the tool call collapsible
                                    yield sse_format(
                                        "toolDelta",
                                        wrap_for_oob_swap(
                                            current_item_id,
                                            f'<pre class="toolCallArgs" data-tool-delta="replace">{escape(action_desc)}</pre>'
                                        ),
                                    )

                                    # Execute the actions and capture a screenshot
                                    screenshot_base64 = await execute_computer_actions(actions, conversation_id)

                                    # Display the screenshot inline
                                    img_html = (
                                        f'<div class="imageOutput">'
                                        f'<img src="data:image/png;base64,{screenshot_base64}" '
                                        f'alt="Computer use screenshot" '
                                        f'onclick="openImagePreview(this.src)" style="cursor:pointer" />'
                                        f'</div>'
                                    )
                                    yield sse_format("imageOutput", img_html)

                                    # Submit computer_call_output and continue streaming
                                    acknowledged = [
                                        {"id": c.id, "code": c.code, "message": c.message}
                                        for c in pending_checks
                                    ]
                                    await client.conversations.items.create(
                                        conversation_id=conversation_id,
                                        items=[{
                                            "type": "computer_call_output",
                                            "call_id": call_id,
                                            "output": {
                                                "type": "computer_screenshot",
                                                "image_url": f"data:image/png;base64,{screenshot_base64}",
                                            },
                                            **({"acknowledged_safety_checks": acknowledged} if acknowledged else {}),
                                        }]
                                    )
                                    next_stream = await client.responses.create(  # type: ignore[call-overload]
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


@router.post("/approve")
async def approve_mcp_tool(
    request: Request,
    conversation_id: str,
    approval_request_id: str = Form(...),
    approve: bool = Form(...),
    reason: str | None = Form(None),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> HTMLResponse:
    # Create an approval response conversation item
    try:
        await client.conversations.items.create(
            conversation_id=conversation_id,
            items=[{
                "type": "mcp_approval_response",
                "approval_request_id": approval_request_id,
                "approve": approve,
                **({"reason": reason} if reason else {})
            }]
        )
    except Exception as e:
        logger.error(f"Failed to submit MCP approval response: {e}")
        # Render a minimal error card and allow retry
        error_html = f"<div class=\"assistantMessage\">Error submitting approval: {str(e)}</div>"
        return HTMLResponse(content=error_html)

    # After approval decision, start a new stream run by returning a new assistant-run include
    assistant_run_html = templates.get_template("components/assistant-run.html").render(
        request=request, conversation_id=conversation_id
    )
    # Also append a small acknowledgement step
    ack_html = templates.get_template("components/assistant-step.html").render(
        step_type="systemNote",
        step_id=f"ack-{approval_request_id}",
        content=("Approved MCP tool request" if approve else "Rejected MCP tool request") + (f": {reason}" if reason else "")
    )
    return HTMLResponse(content=(ack_html + assistant_run_html))
