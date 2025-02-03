import os
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from routers import files, messages, tools, api_keys, assistants


logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional startup logic
    yield
    # Optional shutdown logic

app = FastAPI(lifespan=lifespan)

# Mount routers
app.include_router(messages.router)
app.include_router(files.router)
app.include_router(tools.router)
app.include_router(api_keys.router)
app.include_router(assistants.router)

# Mount static files (e.g., CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "static")), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_home(request: Request):
    logger.info("Home page requested")
    
    # Check if environment variables are missing
    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("ASSISTANT_ID"):
        return RedirectResponse(url="/setup")
    
    categories = {
        "Basic chat": "basic-chat",
        "File search": "file-search",
        "Function calling": "function-calling",
        "All": "all",
    }
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": categories
        }
    )

@app.get("/basic-chat")
async def read_basic_chat(request: Request, messages: list = [], thread_id: str = None):
    # Get assistant ID from environment variables
    load_dotenv()
    assistant_id = os.getenv("ASSISTANT_ID")
    
    return templates.TemplateResponse(
        "examples/basic-chat.html",
        {
            "request": request,
            "assistant_id": assistant_id,  # Add assistant_id to template context
            "messages": messages,
            "thread_id": thread_id
        }
    )

@app.get("/file-search")
async def read_file_search(request: Request, messages: list = [], thread_id: str = None):
    # Get assistant ID from environment variables
    load_dotenv()
    assistant_id = os.getenv("ASSISTANT_ID")
    
    return templates.TemplateResponse(
        "examples/file-search.html",
        {
            "request": request,
            "messages": messages,
            "thread_id": thread_id,
            "assistant_id": assistant_id,  # Add assistant_id to template context
        }
    )

@app.get("/function-calling")
async def read_function_calling(request: Request, messages: list = [], thread_id: str = None):
    # Get assistant ID from environment variables
    load_dotenv()
    assistant_id = os.getenv("ASSISTANT_ID")
    
    # Define the condition class map
    conditionClassMap = {
        "Cloudy": "weatherBGCloudy",
        "Sunny": "weatherBGSunny",
        "Rainy": "weatherBGRainy",
        "Snowy": "weatherBGSnowy",
        "Windy": "weatherBGWindy",
    }
    
    return templates.TemplateResponse(
        "examples/function-calling.html", 
        {
            "conditionClassMap": conditionClassMap,
            "location": "---",
            "temperature": "---",
            "conditions": "Sunny",
            "isEmpty": True,
            "thread_id": thread_id,
            "messages": messages,
            "assistant_id": assistant_id,  # Add assistant_id to template context
        }
    )


@app.get("/all")
async def read_all(request: Request, messages: list = [], thread_id: str = None):
    # Get assistant ID from environment variables
    load_dotenv()
    assistant_id = os.getenv("ASSISTANT_ID")
    
    return templates.TemplateResponse(
        "examples/all.html",
        {
            "request": request,
            "assistant_id": assistant_id,  # Add assistant_id to template context
            "thread_id": thread_id,
            "messages": messages
        }
    )

# Add new setup route
@app.get("/setup")
async def read_setup(request: Request, message: str = None):
    # Check if assistant ID is missing
    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY"):
        message="OpenAI API key is missing."
    elif not os.getenv("ASSISTANT_ID"):
        message="Assistant ID is missing."
    else:
        message="All set up!"
    
    return templates.TemplateResponse(
        "setup.html",
        {"request": request, "message": message}
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
