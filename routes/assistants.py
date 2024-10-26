from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

app = FastAPI()

# Define the request model using Pydantic
class AssistantRequest(BaseModel):
    instructions: str = "You are a helpful assistant."
    name: str = "Quickstart Assistant"
    model: str = "gpt-4o"
    tools: list = Field(default_factory=lambda: [
        {"type": "code_interpreter"},
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Determine weather in my location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["c", "f"],
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {"type": "file_search"},
    ])

# Define the response model using Pydantic
class AssistantResponse(BaseModel):
    assistantId: str

# Initialize the OpenAI client
openai = AsyncOpenAI()

@app.post("/assistants", response_model=AssistantResponse)
async def create_assistant():
    try:
        # Create a new assistant using the OpenAI client
        assistant = await openai.beta.assistants.create(
            instructions="You are a helpful assistant.",
            name="Quickstart Assistant",
            model="gpt-4o",
            tools=[
                {"type": "code_interpreter"},
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Determine weather in my location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["c", "f"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                },
                {"type": "file_search"},
            ],
        )
        return AssistantResponse(assistantId=assistant.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
