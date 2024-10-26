# app/api/assistants/threads/[threadId]/messages/route.py
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from openai import AsyncOpenAI

# Initialize the router
router = APIRouter()

# Initialize the OpenAI client
openai = AsyncOpenAI()

# Pydantic model for request body
class MessageRequest(BaseModel):
    content: str

# Send a new message to a thread
@router.post("/assistants/threads/{thread_id}/messages")
async def post_message(request: Request, thread_id: str):
    # Parse the request body
    body = await request.json()
    message_request = MessageRequest(**body)

    # Create a new message in the thread
    await openai.beta.threads.messages.create(thread_id, {
        "role": "user",
        "content": message_request.content,
    })

    # Stream the response from the assistant
    stream = await openai.beta.threads.runs.stream(thread_id, {
        "assistant_id": "your_assistant_id",  # Replace with your actual assistant ID
    })

    # Return the response as a stream
    return Response(content=stream.to_readable_stream())

