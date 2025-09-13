import logging
from openai import AsyncOpenAI

logger = logging.getLogger("uvicorn.error")


async def create_conversation() -> str:
    """Create a new conversation and return its id."""
    try:
        client: AsyncOpenAI = AsyncOpenAI()
        conv = await client.conversations.create()
        return conv.id
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return ""
