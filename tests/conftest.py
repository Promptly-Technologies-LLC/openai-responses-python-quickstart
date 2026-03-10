"""Pytest configuration and fixtures for Playwright tests."""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest
from playwright.sync_api import Page, Browser

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_BACKUP = PROJECT_ROOT / ".env.test_backup"


@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for the test server."""
    return "http://127.0.0.1:8000"


@pytest.fixture(scope="session")
def app_server(base_url: str) -> Generator[subprocess.Popen, None, None]:
    """Start the FastAPI server for the test session."""
    # Start uvicorn server
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    max_retries = 30
    for _ in range(max_retries):
        try:
            import urllib.request
            urllib.request.urlopen(f"{base_url}/setup/", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError("Server failed to start")

    yield proc

    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def env_backup() -> Generator[None, None, None]:
    """Back up and restore .env file around each test."""
    # Backup existing .env
    if ENV_FILE.exists():
        shutil.copy(ENV_FILE, ENV_BACKUP)

    yield

    # Restore original .env
    if ENV_BACKUP.exists():
        shutil.copy(ENV_BACKUP, ENV_FILE)
        ENV_BACKUP.unlink()
    elif ENV_FILE.exists():
        # If there was no original .env, remove the test one
        ENV_FILE.unlink()


def write_env_file(**kwargs: str) -> None:
    """Write a .env file with the given key-value pairs."""
    lines = [f'{key}="{value}"' for key, value in kwargs.items()]
    ENV_FILE.write_text("\n".join(lines) + "\n")


@pytest.fixture
def env_no_api_key(env_backup: None) -> None:
    """Set up .env with no API key (explicitly empty to override any existing value)."""
    write_env_file(
        OPENAI_API_KEY="",  # Explicitly empty to override any existing value
        RESPONSES_MODEL="gpt-4o",
        RESPONSES_INSTRUCTIONS="You are a helpful assistant.",
        ENABLED_TOOLS="",
    )


@pytest.fixture
def env_api_key_no_tools(env_backup: None) -> None:
    """Set up .env with API key but no tools enabled."""
    write_env_file(
        OPENAI_API_KEY="sk-test-fake-key-for-testing",
        RESPONSES_MODEL="gpt-4o",
        RESPONSES_INSTRUCTIONS="You are a helpful assistant.",
        ENABLED_TOOLS="",
    )


@pytest.fixture
def env_file_search_only(env_backup: None) -> None:
    """Set up .env with API key and file_search tool only."""
    write_env_file(
        OPENAI_API_KEY="sk-test-fake-key-for-testing",
        RESPONSES_MODEL="gpt-4o",
        RESPONSES_INSTRUCTIONS="You are a helpful assistant.",
        ENABLED_TOOLS="file_search",
    )


@pytest.fixture
def env_function_only(env_backup: None) -> None:
    """Set up .env with API key and function tool only."""
    write_env_file(
        OPENAI_API_KEY="sk-test-fake-key-for-testing",
        RESPONSES_MODEL="gpt-4o",
        RESPONSES_INSTRUCTIONS="You are a helpful assistant.",
        ENABLED_TOOLS="function",
    )


@pytest.fixture
def env_mcp_only(env_backup: None) -> None:
    """Set up .env with API key and mcp tool only."""
    write_env_file(
        OPENAI_API_KEY="sk-test-fake-key-for-testing",
        RESPONSES_MODEL="gpt-4o",
        RESPONSES_INSTRUCTIONS="You are a helpful assistant.",
        ENABLED_TOOLS="mcp",
    )


@pytest.fixture
def env_file_search_and_function(env_backup: None) -> None:
    """Set up .env with API key and file_search + function tools."""
    write_env_file(
        OPENAI_API_KEY="sk-test-fake-key-for-testing",
        RESPONSES_MODEL="gpt-4o",
        RESPONSES_INSTRUCTIONS="You are a helpful assistant.",
        ENABLED_TOOLS="file_search,function",
    )


@pytest.fixture
def env_file_search_and_mcp(env_backup: None) -> None:
    """Set up .env with API key and file_search + mcp tools."""
    write_env_file(
        OPENAI_API_KEY="sk-test-fake-key-for-testing",
        RESPONSES_MODEL="gpt-4o",
        RESPONSES_INSTRUCTIONS="You are a helpful assistant.",
        ENABLED_TOOLS="file_search,mcp",
    )


@pytest.fixture
def env_function_and_mcp(env_backup: None) -> None:
    """Set up .env with API key and function + mcp tools."""
    write_env_file(
        OPENAI_API_KEY="sk-test-fake-key-for-testing",
        RESPONSES_MODEL="gpt-4o",
        RESPONSES_INSTRUCTIONS="You are a helpful assistant.",
        ENABLED_TOOLS="function,mcp",
    )


@pytest.fixture
def env_all_tools(env_backup: None) -> None:
    """Set up .env with API key and all three conditional tools."""
    write_env_file(
        OPENAI_API_KEY="sk-test-fake-key-for-testing",
        RESPONSES_MODEL="gpt-4o",
        RESPONSES_INSTRUCTIONS="You are a helpful assistant.",
        ENABLED_TOOLS="file_search,function,mcp",
    )
