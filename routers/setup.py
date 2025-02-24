import logging
import os
from typing import Optional
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI

from utils.create_assistant import create_or_update_assistant, request as assistant_create_request
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
async def read_setup(request: Request, message: Optional[str] = "") -> Response:
    # Check if assistant ID is missing
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    assistant_id = os.getenv("ASSISTANT_ID")
    
    if not openai_api_key:
        message = "OpenAI API key is missing."
    elif not assistant_id:
        message = "Assistant ID is missing."
    else:
        message = "All set up!"
    
    return templates.TemplateResponse(
        "setup.html",
        {"request": request, "message": message}
    )


@router.post("/assistant")
async def create_update_assistant(
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())
) -> RedirectResponse:
    """
    Create a new assistant or update an existing one.
    Returns the assistant ID and status of the operation.
    """
    current_assistant_id = os.getenv("ASSISTANT_ID")
    new_assistant_id = await create_or_update_assistant(
        client=client,
        assistant_id=current_assistant_id,
        request=assistant_create_request,
        logger=logger
    )
    
    if not new_assistant_id:
        raise HTTPException(status_code=400, detail="Failed to create or update assistant")
    
    return RedirectResponse(url="/", status_code=303)
