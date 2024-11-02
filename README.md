# OpenAI Assistants API Quickstart with Python, Jinja2, and FastAPI

A quick-start template using the OpenAI [Assistants API](https://platform.openai.com/docs/assistants/overview) with [Python](https://www.python.org/), [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/), and [FastAPI](https://fastapi.tiangolo.com/).

**Note:** This repository is under construction and not yet functional.

## Quickstart Setup

### 1. Clone repo

```shell
git clone https://github.com/Promptly-Technologies-LLC/openai-assistants-python-quickstart.git
cd openai-assistants-python-quickstart
```

### 2. Set your [OpenAI API key](https://platform.openai.com/api-keys)

```shell
cp .env.example .env
```

### 3. Install dependencies

```shell
uv venv
uv pip install -r pyproject.toml
```

### 4. Create an assistant

```shell
uv run create_assistant.py
```

### 5. Run the FastAPI server

```shell
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Navigate to [http://localhost:8000](http://localhost:8000).
- `api/assistants/threads/[threadId]/actions` - `POST`: inform assistant of the result of a function it decided to call
- `api/assistants/files` - `GET`/`POST`/`DELETE`: fetch, upload, and delete assistant files for file search

## Feedback

Let us know if you have any thoughts, questions, or feedback in [this form](https://docs.google.com/forms/d/e/1FAIpQLScn_RSBryMXCZjCyWV4_ebctksVvQYWkrq90iN21l1HLv3kPg/viewform?usp=sf_link)!
