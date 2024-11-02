from fastapi import APIRouter, Depends
from openai import AsyncOpenAI
from utils.create_assistant import create_or_update_assistant, request
import logging
import os
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger("uvicorn.error")

# Load environment variables
load_dotenv()

router = APIRouter(
    prefix="/assistants",
    tags=["assistants"]
)


@router.post("/create-update")
async def create_update_assistant(
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
):
    """
    Create a new assistant or update an existing one.
    Returns the assistant ID and status of the operation.
    """
    assistant_id = os.getenv("ASSISTANT_ID")

    assistant_id: str = await create_or_update_assistant(
        client=client,
        assistant_id=assistant_id,
        request=request,
        logger=logger
    )

    return {
        "message": f"Assistant {assistant_id} successfully created/updated"
    }
