from typing import Optional
import pydantic
from openai.types.responses.tool_param import Mcp

class CustomFunction(pydantic.BaseModel):
    name: str
    import_path: str
    template_path: Optional[str]

class ToolConfig(pydantic.BaseModel):
    mcp_servers: list[Mcp]
    custom_functions: list[CustomFunction]
