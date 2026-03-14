import os
import re
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

# ---------------------------------------------------------------------------
# Capture the real API key from .env BEFORE any fixtures modify it.
# This is used by live integration tests that need to hit the real API.
# ---------------------------------------------------------------------------
def _read_real_api_key() -> str:
    """Read the real API key from os.environ or .env file.

    Called once at conftest import time — before any test fixtures modify
    the environment.  Falls back to .env on disk if os.environ is empty.
    """
    # 1. Check os.environ first (may be set by the shell or a prior load_dotenv)
    key = os.environ.get("OPENAI_API_KEY", "")
    if key and not key.startswith("sk-fake"):
        return key
    # 2. Fall back to reading .env directly
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        m = re.match(r"^OPENAI_API_KEY=(.+)$", line)
        if m:
            val = m.group(1).strip()
            if val and not val.startswith("sk-fake"):
                return val
    return ""

REAL_API_KEY: str = _read_real_api_key()


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


# ---------------------------------------------------------------------------
# Suppress "Event loop is closed" RuntimeError from anyio test runner cleanup.
#
# On Python 3.13 + httpx + anyio, streaming response cleanup can race with
# event loop shutdown.  anyio's TestRunner captures these as async exceptions
# and re-raises them after the test passes.  Since they are harmless cleanup
# artifacts (not test failures), we filter them out.
# ---------------------------------------------------------------------------
try:
    import anyio._backends._asyncio as _anyio_asyncio_backend

    _original_raise_async = _anyio_asyncio_backend.TestRunner._raise_async_exceptions

    def _filtered_raise_async(self):  # type: ignore[no-untyped-def]
        self._exceptions = [
            e for e in self._exceptions
            if not (isinstance(e, RuntimeError) and "Event loop is closed" in str(e))
        ]
        _original_raise_async(self)

    _anyio_asyncio_backend.TestRunner._raise_async_exceptions = _filtered_raise_async  # type: ignore[assignment]
except (ImportError, AttributeError):
    pass  # Different anyio version; skip the patch


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


@pytest.fixture(scope="session", autouse=True)
def _backup_dotenv() -> Iterator[None]:
    """Back up .env at session start and restore it at session end.

    This ensures a crashed or interrupted test run does not permanently
    corrupt the user's .env file.
    """
    env_path = PROJECT_ROOT / ".env"
    try:
        original = env_path.read_text()
    except FileNotFoundError:
        original = None

    yield

    if original is not None:
        env_path.write_text(original)
    else:
        env_path.unlink(missing_ok=True)


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
def _dotenv(overrides: dict[str, str], *, set_fake_api_key: bool = True) -> Iterator[None]:
    """
    Write a temporary .env for the duration of a single test, then restore.

    Saves and restores both the .env file and os.environ for every key in
    *overrides*.  When *set_fake_api_key* is True (the default, used by
    Playwright tests), OPENAI_API_KEY is also force-set in os.environ so
    the FastAPI ``Depends(lambda: AsyncOpenAI())`` never raises before
    ``load_dotenv(override=True)`` has a chance to run inside the route body.
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
    if set_fake_api_key:
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


def parse_sse_events(raw: str) -> list[dict[str, str]]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events: list[dict[str, str]] = []
    current_event: str | None = None
    current_data_lines: list[str] = []

    for line in raw.split("\n"):
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            current_data_lines.append(line[len("data: "):])
        elif line == "" and current_event is not None:
            events.append({
                "event": current_event,
                "data": "\n".join(current_data_lines),
            })
            current_event = None
            current_data_lines = []

    return events


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
def env_web_search_only(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "web_search"}):
        yield


@pytest.fixture
def env_web_search_with_config(app_server: int) -> Iterator[None]:
    with _dotenv({
        **_BASE_ENV,
        "ENABLED_TOOLS": "web_search",
        "WEB_SEARCH_CONTEXT_SIZE": "high",
        "WEB_SEARCH_LOCATION_COUNTRY": "US",
        "WEB_SEARCH_LOCATION_CITY": "New York",
        "WEB_SEARCH_LOCATION_REGION": "New York",
        "WEB_SEARCH_LOCATION_TIMEZONE": "America/New_York",
    }):
        yield


@pytest.fixture
def env_computer_use_only(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "computer_use"}):
        yield


@pytest.fixture
def env_image_generation_only(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "image_generation"}):
        yield


@pytest.fixture
def env_image_generation_with_config(app_server: int) -> Iterator[None]:
    with _dotenv({
        **_BASE_ENV,
        "ENABLED_TOOLS": "image_generation",
        "IMAGE_GENERATION_QUALITY": "high",
        "IMAGE_GENERATION_SIZE": "1024x1536",
        "IMAGE_GENERATION_BACKGROUND": "transparent",
    }):
        yield


@pytest.fixture
def env_all_tools(app_server: int) -> Iterator[None]:
    with _dotenv({**_BASE_ENV, "ENABLED_TOOLS": "file_search,function,mcp,web_search,computer_use,image_generation"}):
        yield
