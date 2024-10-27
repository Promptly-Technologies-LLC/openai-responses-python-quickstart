import os
import sys
import logging
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.beta.assistant_create_params import AssistantCreateParams
from openai.types.beta.assistant_tool_param import CodeInterpreterToolParam, FileSearchToolParam, FunctionToolParam
from openai.types.beta.assistant import Assistant
from openai.types import FunctionDefinition

# Configure logger to stream to console
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger: logging.Logger = logging.getLogger(__name__)


request: AssistantCreateParams = AssistantCreateParams(
    instructions="You are a helpful assistant.",
    name="Quickstart Assistant",
    model="gpt-4o",
    tools=[
        CodeInterpreterToolParam(type="code_interpreter"),
        FileSearchToolParam(type="file_search"),
        FunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Determine weather in my location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
                strict=True,
            )
        ),
    ],
)


def update_env_file(assistant_id, logger):
    """
    Update the .env file with a new assistant ID.

    If the .env file already contains an ASSISTANT_ID, it will be removed.
    The new assistant ID will be appended to the .env file. If the .env file
    does not exist, it will be created.
    """
    if os.path.exists('.env'):
        with open('.env', 'r') as env_file:
            lines = env_file.readlines()

        # Remove any existing ASSISTANT_ID line
        lines = [line for line in lines if not line.startswith("ASSISTANT_ID=")]

        # Write back the modified lines
        with open('.env', 'w') as env_file:
            env_file.writelines(lines)

    # Write the new assistant ID to the .env file
    with open('.env', 'a') as env_file:
        env_file.write(f"ASSISTANT_ID={assistant_id}\n")
    logger.info(f"Assistant ID written to .env: {assistant_id}")


async def create_or_update_assistant(
        client: AsyncOpenAI,
        assistant_id: str,
        request: AssistantCreateParams,
        logger: logging.Logger
):
    """
    Create or update the assistant based on the presence of an assistant_id.
    """
    try:
        if assistant_id:
            # Update the existing assistant
            assistant: Assistant = await client.beta.assistants.update(
                assistant_id,
                **request
            )
            logger.info(f"Updated assistant with ID: {assistant_id}")
        else:
            # Create a new assistant
            assistant: Assistant = await client.beta.assistants.create(**request)
            logger.info(f"Created new assistant: {assistant}")

            # Update the .env file with the new assistant ID
            update_env_file(assistant.id, logger)

    except Exception as e:
        action = "update" if assistant_id else "create"
        logger.error(f"Failed to {action} assistant: {e}")


# Run the assistant creation in an asyncio event loop
if __name__ == "__main__":
    load_dotenv()
    assistant_id = os.getenv("ASSISTANT_ID")

    # Initialize the OpenAI client
    openai: AsyncOpenAI = AsyncOpenAI()

    # Run the main function in an asyncio event loop
    new_assistant_id: Assistant = asyncio.run(
        create_or_update_assistant(openai, assistant_id, request, logger)
    )
    logger.info(f"Assistant created with ID: {new_assistant_id}")
