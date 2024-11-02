import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.create_assistant import update_env_file

# Configure logger
logger: logging.Logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api-keys", tags=["API Keys"])

class APIKeyRequest(BaseModel):
    api_key: str

@router.post("")
async def set_openai_api_key(request: APIKeyRequest):
    """
    Set the OpenAI API key in the application's environment variables.
    
    Args:
        request: APIKeyRequest containing the API key
    
    Returns:
        dict: Success message
    
    Raises:
        HTTPException: If there's an error updating the environment file
    """
    try:
        update_env_file("OPENAI_API_KEY", request.api_key, logger)
        return {"message": "API key updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update API key: {str(e)}"
        )
