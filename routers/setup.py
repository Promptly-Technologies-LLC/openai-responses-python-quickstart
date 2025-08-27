import logging
import os
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from urllib.parse import quote

from utils.create_assistant import update_env_file

# Configure logger
logger = logging.getLogger("uvicorn.error")

# Load environment variables
load_dotenv()

router = APIRouter(prefix="/setup", tags=["Setup"])
templates = Jinja2Templates(directory="templates")

@router.post("/api-key")
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
        safe_key = api_key.strip().replace("\r", "").replace("\n", "")
        update_env_file("OPENAI_API_KEY", safe_key, logger)
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
    current_instructions: Optional[str] = None
    # Populate with all models extracted from user-provided HTML, sorted
    available_models: List[str] = sorted([ 
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", 
        "gpt-4o", "gpt-4o-mini", "o1", "o1-mini",
        "o3", "o3-pro", "o3-mini", "o4-mini",
        "o3-deep-research", "o4-mini-deep-research",
        "gpt-5-chat", "gpt-5-mini", "gpt-5-nano",
        "gpt-oss-120b", "gpt-oss-20b"
    ])
    setup_message: str = ""

    # Check if env variables are set
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Load existing app config from env
    current_model = os.getenv("RESPONSES_MODEL")
    current_instructions = os.getenv("RESPONSES_INSTRUCTIONS")
    enabled_tools_csv = os.getenv("ENABLED_TOOLS", "")
    current_tools = [t.strip() for t in enabled_tools_csv.split(",") if t.strip()]

    if not openai_api_key:
        setup_message = "OpenAI API key is missing."
    
    return templates.TemplateResponse(
        "setup.html",
        {
            "request": request,
            "setup_message": setup_message,
            "status": status, # Pass status from query params
            "status_message": message_text, # Pass message from query params
            "current_tools": current_tools,
            "current_model": current_model,
            "current_instructions": current_instructions,
            "available_models": available_models # Pass available models to template
        }
    )


@router.post("/config")
async def save_app_config(
    tool_types: List[str] = Form(default=[]),
    model: str = Form(...),
    instructions: str = Form(...)
) -> RedirectResponse:
    try:
        update_env_file("RESPONSES_MODEL", model, logger)
        update_env_file("RESPONSES_INSTRUCTIONS", instructions, logger)
        enabled_tools_csv = ",".join(tool_types)
        update_env_file("ENABLED_TOOLS", enabled_tools_csv, logger)
        status = "success"
        message_text = "Configuration saved."
    except Exception as e:
        status = "error"
        message_text = f"Failed to save configuration: {e}"

    encoded_message = quote(message_text)
    redirect_url = f"/setup/?status={status}&message_text={encoded_message}"
    return RedirectResponse(url=redirect_url, status_code=303)
