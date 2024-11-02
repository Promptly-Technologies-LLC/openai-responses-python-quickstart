# OpenAI Assistants API Quickstart with Python, Jinja2, and FastAPI

A quick-start template using the OpenAI [Assistants API](https://platform.openai.com/docs/assistants/overview) with [Python](https://www.python.org/), [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/), and [FastAPI](https://fastapi.tiangolo.com/).

**Note:** This repository is still under construction; only basic chat is currently functional.

## Quickstart Setup

### 1. Clone repo

```shell
git clone https://github.com/Promptly-Technologies-LLC/openai-assistants-python-quickstart.git
cd openai-assistants-python-quickstart
```

### 2. Install dependencies

```shell
uv venv
uv pip install -r pyproject.toml
```

### 3. Run the FastAPI server

```shell
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Navigate to [http://localhost:8000](http://localhost:8000).

### 5. Set your OpenAI API key and create an assistant in the GUI

If your OPENAI_API_KEY or ASSISTANT_ID are not set, you will be redirected to the setup page where you can set them. (The values will be saved in a .env file in the root of the project.) Once set, you will be redirected to the home page and can begin a chat session.
