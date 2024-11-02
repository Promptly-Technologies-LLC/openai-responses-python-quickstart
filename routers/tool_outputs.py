import logging
from typing import Any
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI

logger = logging.getLogger("uvicorn.error")

router = APIRouter()

openai = AsyncOpenAI()

class ToolCallOutputs(BaseModel):
    tool_outputs: Any
    runId: str

@router.post("/send_tool_outputs")
async def post_tool_outputs(thread_id: str, request: Request):
    try:
        # Parse the JSON body into the ToolCallOutputs model
        data = await request.json()
        tool_call_outputs = ToolCallOutputs(**data)

        # Submit tool outputs stream
        stream = await openai.beta.threads.runs.submit_tool_outputs_stream(
            thread_id,
            tool_call_outputs.runId,
            {"tool_outputs": tool_call_outputs.tool_outputs}
        )

        # Return the stream as a response
        return stream.to_readable_stream()
    except Exception as e:
        # Handle exceptions and return an HTTP error response
        raise HTTPException(status_code=500, detail=str(e))

