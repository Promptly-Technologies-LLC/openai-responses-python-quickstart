import logging
import json
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional, cast
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Form, Depends, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from openai import AsyncOpenAI
from utils.custom_functions import get_weather, post_tool_outputs, get_function_tool_def
from utils.sse import sse_format


logger: logging.Logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)


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
            "content": [{"type": "input_text", "text": f"System: Today's date is {datetime.today().strftime('%Y-%m-%d')}\n{userInput}"}]
        }]
    )

    user_message_html = templates.get_template("components/user-message.html").render(user_input=userInput)
    assistant_run_html = templates.get_template("components/assistant-run.html").render(
        conversation_id=conversation_id
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
        from dotenv import load_dotenv
        load_dotenv(override=True)
        model = os.getenv("RESPONSES_MODEL", "gpt-4o")
        instructions = os.getenv("RESPONSES_INSTRUCTIONS")
        enabled_tools = [t.strip() for t in os.getenv("ENABLED_TOOLS", "").split(",") if t.strip()]

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
        if "function" in enabled_tools:
            tools.append(get_function_tool_def())

        stream = await client.responses.create(
            input="",
            conversation=conversation_id,
            model=model,
            tools=tools or None,
            instructions=instructions,
            parallel_tool_calls=False,
            stream=True
        )

        response_id: str = ""
        current_item_id: str = ""
        # Accumulate function call args per tool_call_id
        fn_args_buffer: Dict[str, str] = {}

        async def iterate_stream(s) -> AsyncGenerator[str, None]:
            nonlocal response_id, current_item_id, fn_args_buffer
            async with s as events:
                async for event in events:
                    etype = getattr(event, "type", None)
                    if not etype:
                        continue

                    if etype == "response.created":
                        rid_opt: Optional[str] = getattr(event, "id", None)
                        resp_obj = getattr(event, "response", None)
                        resp_id_opt: Optional[str] = getattr(resp_obj, "id", None) if resp_obj else None
                        response_id = cast(str, (rid_opt or resp_id_opt or ""))

                    elif etype == "response.output_item.added":
                        item = getattr(event, "item", None)
                        item_id = getattr(item, "id", "") if item else ""
                        if item_id:
                            current_item_id = item_id
                            yield sse_format(
                                "messageCreated",
                                templates.get_template("components/assistant-step.html").render(
                                    step_type="assistantMessage",
                                    step_id=item_id
                                )
                            )

                    elif etype == "response.output_text.delta":
                        # Handle inline citations and file links in text deltas
                        delta = getattr(event, "delta", None)
                        text_value = str(delta) if delta is not None else None
                        if text_value and current_item_id:
                            # TODO: update to match new pattern, https://platform.openai.com/docs/guides/tools-code-interpreter#work-with-files
                            # Replace patterns like 【...†filename】 with app download links
                            try:
                                import re
                                match = re.search(r"【.*?†(.*?)】", text_value)
                                if match:
                                    file_name_in_citation = match.group(1)
                                    file_url = f"/files/{file_name_in_citation}"
                                    text_value = text_value.replace(match.group(0), f"[†]({file_url})")
                            except Exception:
                                pass
                            yield sse_format("textDelta", wrap_for_oob_swap(current_item_id, text_value))

                    elif etype == "response.function_call_arguments.delta":
                        tool_call_id = getattr(event, "tool_call_id", "")
                        delta = getattr(event, "delta", "")
                        if tool_call_id:
                            # Emit a toolCallCreated once per tool_call_id
                            if tool_call_id not in fn_args_buffer:
                                yield sse_format(
                                    "toolCallCreated",
                                    templates.get_template('components/assistant-step.html').render(
                                        step_type='toolCall',
                                        step_id=current_item_id or tool_call_id
                                    )
                                )
                                fn_args_buffer[tool_call_id] = ""
                            fn_args_buffer[tool_call_id] = fn_args_buffer.get(tool_call_id, "") + str(delta)
                            if current_item_id:
                                yield sse_format("toolDelta", wrap_for_oob_swap(current_item_id, str(delta)))

                    elif etype == "response.function_call_arguments.done":
                        tool_call_id = getattr(event, "tool_call_id", "")
                        args_json = fn_args_buffer.get(tool_call_id, "{}")
                        # Execute function
                        try:
                            args = json.loads(args_json or "{}")
                            location = args.get("location", "Unknown")
                            dates_raw = args.get("dates", [datetime.today().strftime("%Y-%m-%d")])
                            weather_output = get_weather(location, dates_raw)
                            # Render widget
                            weather_widget_html = templates.get_template(
                                "components/weather-widget.html"
                            ).render(reports=weather_output)
                            yield sse_format("toolOutput", weather_widget_html)
                            # Submit outputs and continue streaming
                            next_stream = await post_tool_outputs(
                                client,
                                response_id=response_id,
                                tool_call_id=tool_call_id,
                                output=json.dumps(weather_output)
                            )
                            async for out in iterate_stream(next_stream):
                                yield out
                        except Exception as err:
                            yield sse_format("toolOutput", f"Function error: {err}")

                    elif etype == "response.completed":
                        yield sse_format("endStream", "DONE")

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
