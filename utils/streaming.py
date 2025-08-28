from typing import AsyncIterable
from dataclasses import dataclass


@dataclass
class ResponseStreamState:
    """State carried through a Responses streaming session."""
    response_id: str = ""
    current_item_id: str = ""
    awaiting_tool_output: bool = False


async def stream_file_content(content: bytes) -> AsyncIterable[bytes]:
    yield content