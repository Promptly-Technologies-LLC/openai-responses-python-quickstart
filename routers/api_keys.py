import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import RedirectResponse
from utils.create_assistant import update_env_file

# Configure logger
logger: logging.Logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api-keys", tags=["API Keys"])

@router.post("/set")
async def set_openai_api_key(api_key: str = Form()):
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
        return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update API key: {str(e)}"
        )
