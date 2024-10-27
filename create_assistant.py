import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.beta.assistant_create_params import AssistantCreateParams
from openai.types.beta.assistant_tool_param import CodeInterpreterToolParam, FileSearchToolParam, FunctionToolParam
from openai.types.beta.assistant import Assistant
from openai.types import FunctionDefinition

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


async def main():
    # Create a new assistant using the OpenAI client
    assistant: Assistant = await openai.beta.assistants.create(**request)
    return assistant


# Run the assistant creation in an asyncio event loop
if __name__ == "__main__":
    import logging
    import sys

    # Initialize the OpenAI client
    load_dotenv()
    openai: AsyncOpenAI = AsyncOpenAI()

    # Configure logger to stream to console
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger: logging.Logger = logging.getLogger(__name__)

    # Run the main function in an asyncio event loop
    assistant: Assistant = asyncio.run(main())
    logger.info(f"Assistant created with ID: {assistant.id}")
