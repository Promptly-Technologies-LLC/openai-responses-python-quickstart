import os
import logging
from typing import List, Literal, Union
from openai import AsyncOpenAI
from openai.types.beta.code_interpreter_tool_param import CodeInterpreterToolParam
from openai.types.beta.file_search_tool_param import FileSearchToolParam
from openai.types.beta.file_search_tool_param import FileSearch
from openai.types.beta.function_tool_param import FunctionToolParam
from utils.custom_functions import get_function_tool_param


ToolTypes = Literal["code_interpreter", "file_search", "function"]
ToolParam = Union[CodeInterpreterToolParam, FileSearchToolParam, FunctionToolParam]


def get_tool_params(tool_types: List[ToolTypes]) -> List[ToolParam]:
    """
    Construct the list of tool parameters based on the selected tool types.
    """
    tool_params: List[ToolParam] = []
    if "code_interpreter" in tool_types:
        tool_params.append(CodeInterpreterToolParam(type="code_interpreter"))
    if "file_search" in tool_types:
        tool_params.append(
            FileSearchToolParam(
                type="file_search",
                file_search=FileSearch(max_num_results=5)
            )
        )
    if "function" in tool_types:
        tool_params.append(get_function_tool_param())
    return tool_params


def update_env_file(var_name: str, var_value: str, logger: logging.Logger):
    """
    Update the .env file with a new environment variable.

    If the .env file already contains the specified variable, it will be updated.
    The new value will be appended to the .env file if it doesn't exist.
    If the .env file does not exist, it will be created.

    Args:
        var_name: The name of the environment variable to update
        var_value: The value to set for the environment variable
        logger: Logger instance for output
    """
    lines = []
    # Read existing contents if file exists
    if os.path.exists('.env'):
        with open('.env', 'r') as env_file:
            lines = env_file.readlines()

        # Remove any existing line with this variable
        lines = [line for line in lines if not line.startswith(f"{var_name}=")]
    else:
        # Log when we're creating a new .env file
        logger.info("Creating new .env file")

    # Write back all lines including the new variable
    with open('.env', 'w') as env_file:
        env_file.writelines(lines)
        env_file.write(f"{var_name}={var_value}\n")
    
    logger.debug(f"Environment variable {var_name} written to .env: {var_value}")


async def create_or_update_assistant(
        client: AsyncOpenAI,
        assistant_id: str | None,
        tool_types: List[ToolTypes],
        model: str,
        logger: logging.Logger
) -> str | None:
    """
    Create or update the assistant based on the presence of an assistant_id.
    """
    assistant = None  # Initialize assistant
    try:
        tool_params = get_tool_params(tool_types)
        if assistant_id:
            # Update the existing assistant
            assistant = await client.beta.assistants.update(
                assistant_id,
                instructions="You are a helpful assistant.",
                name="Quickstart Assistant",
                model=model,
                tools=tool_params
            )
            logger.info(f"Updated assistant with ID: {assistant_id}")
        else:
            # Create a new assistant
            assistant = await client.beta.assistants.create(
                instructions="You are a helpful assistant.",
                name="Quickstart Assistant",
                model=model,
                tools=tool_params
            )
            logger.info(f"Created new assistant: {assistant.id}")

            # Update the .env file with the new assistant ID
            update_env_file("ASSISTANT_ID", assistant.id, logger)

    except Exception as e:
        action = "update" if assistant_id else "create"
        logger.error(f"Failed to {action} assistant: {e}")

    return assistant.id if assistant else None  # Conditionally return ID
