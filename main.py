import os
import logging
import dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routers import messages, assistants_files, tool_outputs


logger = logging.getLogger("uvicorn.error")

dotenv.load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional startup logic
    yield
    # Optional shutdown logic

app = FastAPI(lifespan=lifespan)

# Mount routers
app.include_router(messages.router)
app.include_router(assistants_files.router)
app.include_router(tool_outputs.router)


# Mount static files (e.g., CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "static")), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_home(request: Request):
    logger.info("Home page requested")
    
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
    return templates.TemplateResponse(
        "examples/basic-chat.html",
        {
            "request": request,
            "messages": messages,
            "thread_id": thread_id
        }
    )

@app.get("/file-search")
async def read_file_search(request: Request, messages: list = [], thread_id: str = None):
    return templates.TemplateResponse(
        "examples/file-search.html",
        {
            "request": request,
            "messages": messages,
            "thread_id": thread_id,
        }
    )

@app.get("/function-calling")
async def read_function_calling(request: Request, messages: list = [], thread_id: str = None):
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
            "messages": messages
        }
    )

@app.get("/all")
async def read_all(request: Request, messages: list = [], thread_id: str = None):
    return templates.TemplateResponse(
        "examples/all.html",
        {
            "request": request,
            "thread_id": thread_id,
            "messages": messages
        }
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
