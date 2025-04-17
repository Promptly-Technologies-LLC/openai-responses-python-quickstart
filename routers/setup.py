import logging
import os
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from urllib.parse import quote

from utils.create_assistant import create_or_update_assistant, ToolTypes
from utils.create_assistant import update_env_file

# Configure logger
logger = logging.getLogger("uvicorn.error")

# Load environment variables
load_dotenv()

router = APIRouter(prefix="/setup", tags=["Setup"])
templates = Jinja2Templates(directory="templates")

@router.put("/api-key")
async def set_openai_api_key(api_key: str = Form(...)) -> RedirectResponse:
    """
    Set the OpenAI API key in the application's environment variables.
    
    Args:
        api_key: OpenAI API key received from form submission
    
    Returns:
        RedirectResponse: Redirects to home page on success
    
    Raises:
        HTTPException: If there's an error updating the environment file
    """
    try:
        update_env_file("OPENAI_API_KEY", api_key, logger)
        return RedirectResponse(url="/", headers={"HX-Redirect": "/"}, status_code=303)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update API key: {str(e)}"
        )


# Add new setup route
@router.get("/")
async def read_setup(
    request: Request,
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI()),
    status: Optional[str] = None,
    message_text: Optional[str] = None
) -> Response:
    # Variable initializations
    current_tools: List[str] = []
    current_model: Optional[str] = None
    # Populate with all models extracted from user-provided HTML, sorted
    available_models: List[str] = sorted([
        "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k", 
        "gpt-4", "gpt-4-0125-preview", "gpt-4-0613", "gpt-4-1106-preview", 
        "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview", 
        "gpt-4.1", "gpt-4.1-2025-04-14", "gpt-4.1-mini", "gpt-4.1-mini-2025-04-14", 
        "gpt-4.1-nano", "gpt-4.1-nano-2025-04-14", 
        "gpt-4.5-preview", "gpt-4.5-preview-2025-02-27", 
        "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20", 
        "gpt-4o-mini", "gpt-4o-mini-2024-07-18", 
        "o1", "o1-2024-12-17", 
        "o3-mini", "o3-mini-2025-01-31"
    ])
    setup_message: str = ""

    # Check if env variables are set
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    assistant_id = os.getenv("ASSISTANT_ID")
    logger.info(f"Assistant ID: {assistant_id}")
    
    if not openai_api_key:
        setup_message = "OpenAI API key is missing."
    else:
        if assistant_id:
            try:
                assistant = await client.beta.assistants.retrieve(assistant_id)
                current_tools = [tool.type for tool in assistant.tools]
                current_model = assistant.model  # Get the model from the assistant
            except Exception as e:
                logger.error(f"Failed to retrieve assistant {assistant_id}: {e}")
                # If we can't retrieve the assistant, proceed as if it doesn't exist
                assistant_id = None
                setup_message = "Error retrieving existing assistant. Please create a new one."
    
    return templates.TemplateResponse(
        "setup.html",
        {
            "request": request,
            "setup_message": setup_message,
            "status": status, # Pass status from query params
            "status_message": message_text, # Pass message from query params
            "assistant_id": assistant_id,
            "current_tools": current_tools,
            "current_model": current_model,
            "available_models": available_models # Pass available models to template
        }
    )


@router.post("/assistant")
async def create_update_assistant(
    tool_types: List[ToolTypes] = Form(...),
    model: str = Form(...),
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> RedirectResponse:
    """
    Create a new assistant or update an existing one.
    Returns the assistant ID and status of the operation.
    """
    current_assistant_id = os.getenv("ASSISTANT_ID")
    action = "updated" if current_assistant_id else "created"
    new_assistant_id = await create_or_update_assistant(
        client=client,
        assistant_id=current_assistant_id,
        tool_types=tool_types,
        model=model,
        logger=logger
    )
    
    if not new_assistant_id:
        status = "error"
        message_text = f"Failed to {action} assistant."
    else:
        status = "success"
        message_text = f"Assistant {action} successfully."
        
    # URL encode the message text
    encoded_message = quote(message_text)
    redirect_url = f"/setup/?status={status}&message_text={encoded_message}"
    return RedirectResponse(url=redirect_url, status_code=303)
