import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, Response, HTMLResponse
from routers import chat, files, setup
from utils.conversations import create_conversation
from fastapi.exceptions import HTTPException, RequestValidationError


logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional startup logic
    yield
    # Optional shutdown logic

app = FastAPI(lifespan=lifespan)

# Mount routers
app.include_router(chat.router)
app.include_router(files.router)
app.include_router(setup.router)

# Mount static files (e.g., CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "static")), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> Response:
    logger.error(f"Unhandled error: {exc}")
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error_message": str(exc)},
        status_code=500
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the detailed validation errors
    logger.error(f"Validation error: {exc.errors()}") 
    error_details = "; ".join([f"{err['loc'][-1]}: {err['msg']}" for err in exc.errors()])

    # Check if it's an htmx request
    if request.headers.get("hx-request") == "true":
        # Return an HTML fragment suitable for htmx swapping
        error_html = f'<div id="file-list-container"><p class="errorMessage">Validation Error: {error_details}</p></div>' # Assuming target is file-list-container
        return HTMLResponse(content=error_html, status_code=200)
    else:
        # Return the full error page for standard requests
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Invalid input: {error_details}"},
            status_code=422,
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
    logger.error(f"HTTP error: {exc.detail}")
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error_message": exc.detail},
        status_code=exc.status_code
    )


# TODO: Implement some kind of thread id storage or management logic to allow
# user to load an old thread, delete an old thread, etc. instead of start new
@app.get("/")
async def read_home(
    request: Request,
    conversation_id: Optional[str] = None,
    messages: List[Dict[str, Any]] = []
) -> Response:
    logger.info("Home page requested")
    
    # Check if environment variables are missing
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    responses_model = os.getenv("RESPONSES_MODEL")
    if not openai_api_key or not responses_model:
        return RedirectResponse(url=app.url_path_for("read_setup"))

    # Create a new conversation if none provided
    if not conversation_id or conversation_id == "None" or conversation_id == "null":
        conversation_id = await create_conversation()
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "messages": messages,
            "conversation_id": conversation_id
        }
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
