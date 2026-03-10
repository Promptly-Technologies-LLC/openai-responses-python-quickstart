import os
import sys
import time
import socket
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Playwright infrastructure — live server + environment helpers
# ---------------------------------------------------------------------------

# A fake API key that satisfies AsyncOpenAI()'s constructor without hitting
# the real OpenAI API.  Tests that need the page to behave as if there is NO
# key write OPENAI_API_KEY= (empty) to .env; load_dotenv(override=True) in
# the route body then sets the in-process env to "".  The Depends lambda
# runs *before* load_dotenv, so it always sees the fake key and doesn't raise.
_FAKE_API_KEY = "sk-fake-playwright-test-key"
_BASE_ENV: dict[str, str] = {
    "OPENAI_API_KEY": _FAKE_API_KEY,
    "RESPONSES_MODEL": "gpt-4o",
    "ENABLED_TOOLS": "",
}


@pytest.fixture(autouse=True)
def _isolate_asyncio_running_loop(request: pytest.FixtureRequest) -> Iterator[None]:
    """
    Save and restore asyncio._running_loop around anyio tests.

    Playwright's sync API calls asyncio._set_running_loop(self._loop) after
    each sync operation to mark its paused greenlet loop as "running" from the
    main thread's perspective.  anyio's asyncio.Runner raises "Cannot run the
    event loop while another loop is running" if that marker is non-None when
    it starts.

    We therefore clear the marker before each anyio test and restore it
    afterwards so that Playwright's session teardown (browser.close) can still
    reach its paused loop.
    """
    if not request.node.get_closest_marker("anyio"):
        yield
        return

    import asyncio.events as _aio_events

    saved = _aio_events._get_running_loop()
    _aio_events._set_running_loop(None)
    try:
        yield
    finally:
        # Clear any loop reference anyio left behind, then restore Playwright's.
        _aio_events._set_running_loop(None)
        if saved is not None:
            _aio_events._set_running_loop(saved)


@pytest.fixture(scope="session")
def app_server() -> Generator[int, None, None]:
    """Start the FastAPI app in a background thread on a free port."""
    import uvicorn
    from main import app  # imported here to avoid polluting the global scope

    # Seed the process environment so AsyncOpenAI() can always be instantiated
    # (it validates the key at construction time, before load_dotenv runs).
    os.environ.setdefault("OPENAI_API_KEY", _FAKE_API_KEY)
    os.environ.setdefault("RESPONSES_MODEL", "gpt-4o")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait until the server is actually accepting connections (up to 5 s).
    for _ in range(50):
        time.sleep(0.1)
        if server.started:
            break

    yield port

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def base_url(app_server: int) -> str:  # type: ignore[override]
    """Return the base URL of the running test server."""
    return f"http://127.0.0.1:{app_server}"


@contextmanager
def _dotenv(overrides: dict[str, str]) -> Iterator[None]:
    """
    Write a temporary .env for the duration of a single test, then restore.

    Also keeps OPENAI_API_KEY set to the fake value in os.environ so that the
    FastAPI Depends(lambda: AsyncOpenAI()) in route signatures never raises
    before load_dotenv(override=True) has a chance to run inside the route body.
    """
    env_path = PROJECT_ROOT / ".env"

    # Persist the original .env (may not exist).
    try:
        original_text = env_path.read_text()
    except FileNotFoundError:
        original_text = None

    # Track the os.environ state for every key we will touch.
    all_keys = set(overrides) | {"OPENAI_API_KEY"}
    original_osenv = {k: os.environ.get(k) for k in all_keys}

    # Always keep a fake key in the process env for the Depends constructor.
    os.environ["OPENAI_API_KEY"] = _FAKE_API_KEY

    # Write the test-specific .env; the route body's load_dotenv will read it.
    env_path.write_text(
        "\n".join(f"{k}={v}" for k, v in overrides.items()) + "\n"
    )

    try:
        yield
    finally:
        # Restore .env.
        if original_text is not None:
            env_path.write_text(original_text)
        else:
            env_path.unlink(missing_ok=True)

        # Restore os.environ.
        for k, orig in original_osenv.items():
            if orig is not None:
                os.environ[k] = orig
            elif k in os.environ:
                del os.environ[k]


# ---------------------------------------------------------------------------
# Environment fixtures used by test_setup_page_rendering.py
# ---------------------------------------------------------------------------

@pytest.fixture
def env_no_api_key(app_server: int) -> Iterator[None]:
    """Setup page should show the API-key form (OPENAI_API_KEY empty)."""
    with _dotenv({"RESPONSES_MODEL": "gpt-4o", "OPENAI_API_KEY": ""}):
        yield


@pytest.fixture
def env_api_key_no_tools(app_server: int) -> Iterator[None]:
    with _dotenv(_BASE_ENV):
        yield


@pytest.fixture
def env_file_search_only(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "file_search"}):
        yield


@pytest.fixture
def env_function_only(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "function"}):
        yield


@pytest.fixture
def env_mcp_only(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "mcp"}):
        yield


@pytest.fixture
def env_file_search_and_function(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "file_search,function"}):
        yield


@pytest.fixture
def env_file_search_and_mcp(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "file_search,mcp"}):
        yield


@pytest.fixture
def env_function_and_mcp(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "function,mcp"}):
        yield


@pytest.fixture
def env_all_tools(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "file_search,function,mcp"}):
        yield
