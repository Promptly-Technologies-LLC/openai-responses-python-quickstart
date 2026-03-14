import asyncio
import base64
import logging
from typing import Protocol, Union, runtime_checkable

from playwright.async_api import async_playwright, Browser, Page, Playwright

from openai.types.responses.computer_action import (
    Click,
    DoubleClick,
    Drag,
    Keypress,
    Move,
    Screenshot,
    Scroll,
    Type,
    Wait,
)

logger = logging.getLogger("uvicorn.error")

Action = Union[
    Click, DoubleClick, Drag, Keypress,
    Move, Screenshot, Scroll, Type, Wait,
]


@runtime_checkable
class ComputerSession(Protocol):
    """Protocol for computer use sessions.

    Implement this to provide a custom backend (e.g., VNC, xdotool, pyautogui)
    instead of the default Playwright-based BrowserSession.

    Methods:
        screenshot: Capture the current screen as a base64-encoded PNG string.
        execute: Perform an action and return a base64-encoded PNG screenshot.
        close: Release resources held by this session.
    """

    async def screenshot(self) -> str: ...
    async def execute(self, action: Action) -> str: ...
    async def close(self) -> None: ...


@runtime_checkable
class ComputerSessionManager(Protocol):
    """Protocol for managing computer use sessions by conversation ID.

    Implement this to provide custom session lifecycle management.

    Methods:
        get_or_create: Return an existing session or create a new one.
        close: Close and remove a specific session.
        close_all: Close all sessions (called during app shutdown).
    """

    def get_or_create(
        self, conversation_id: str, width: int = 1024, height: int = 768,
    ) -> ComputerSession: ...
    async def close(self, conversation_id: str) -> None: ...
    async def close_all(self) -> None: ...

_LANDING_PAGE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Browser</title></head>
<body style="margin:0;display:flex;justify-content:center;align-items:center;
             height:100vh;background:#f0f0f0;font-family:sans-serif;">
  <div style="text-align:center;">
    <h1 style="font-size:28px;margin-bottom:24px;">Browser &mdash; Enter a URL to get started</h1>
    <form onsubmit="window.location=document.getElementById('url').value;return false;"
          style="display:flex;gap:8px;justify-content:center;">
      <input id="url" type="text"
             placeholder="Enter a URL, e.g. https://example.com"
             style="width:500px;padding:12px 16px;font-size:18px;border:2px solid #ccc;
                    border-radius:6px;"
             autofocus />
      <button type="submit"
              style="padding:12px 24px;font-size:18px;background:#2563eb;color:white;
                     border:none;border-radius:6px;cursor:pointer;">
        Go
      </button>
    </form>
  </div>
</body>
</html>
"""

# Map OpenAI key names to Playwright key names
_KEY_MAP: dict[str, str] = {
    "ctrl": "Control",
    "shift": "Shift",
    "alt": "Alt",
    "meta": "Meta",
    "enter": "Enter",
    "return": "Enter",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "escape": "Escape",
    "esc": "Escape",
    "space": " ",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
}


def _map_key(key: str) -> str:
    """Map an OpenAI key name to a Playwright key name."""
    return _KEY_MAP.get(key.lower(), key)


def _map_button(button: str) -> str:
    """Map an OpenAI button name to a Playwright button name."""
    if button == "wheel":
        return "middle"
    if button in {"left", "right", "middle"}:
        return button
    return "left"


class BrowserSession:
    """Manages a headless Playwright browser page for computer use actions."""

    def __init__(self, width: int = 1024, height: int = 768):
        self.width = width
        self.height = height
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def _ensure_page(self) -> Page:
        """Lazily launch the browser and return the active page."""
        if self._page and not self._page.is_closed():
            return self._page

        if not self._playwright:
            self._playwright = await async_playwright().start()
        if not self._browser or not self._browser.is_connected():
            self._browser = await self._playwright.chromium.launch(headless=True)

        context = await self._browser.new_context(
            viewport={"width": self.width, "height": self.height},
            device_scale_factor=1,
        )
        self._page = await context.new_page()
        await self._page.set_content(_LANDING_PAGE_HTML, wait_until="load")
        return self._page

    async def screenshot(self) -> str:
        """Capture a full-page screenshot and return as base64-encoded PNG."""
        page = await self._ensure_page()
        png_bytes = await page.screenshot(type="png")
        return base64.b64encode(png_bytes).decode("ascii")

    async def execute(self, action: Action) -> str:
        """Execute an action and return a base64-encoded PNG screenshot."""
        page = await self._ensure_page()

        match action.type:
            case "click":
                await page.mouse.click(
                    action.x, action.y,
                    button=_map_button(action.button),
                )

            case "double_click":
                await page.mouse.dblclick(action.x, action.y)

            case "drag":
                if action.path and len(action.path) >= 2:
                    start = action.path[0]
                    await page.mouse.move(start.x, start.y)
                    await page.mouse.down()
                    for point in action.path[1:]:
                        await page.mouse.move(point.x, point.y)
                    await page.mouse.up()

            case "keypress":
                combo = "+".join(_map_key(k) for k in action.keys)
                await page.keyboard.press(combo)

            case "move":
                await page.mouse.move(action.x, action.y)

            case "screenshot":
                pass  # Just take the screenshot below

            case "scroll":
                await page.mouse.move(action.x, action.y)
                await page.mouse.wheel(action.scroll_x, action.scroll_y)

            case "type":
                await page.keyboard.type(action.text)

            case "wait":
                await asyncio.sleep(2)

        return await self.screenshot()

    async def close(self) -> None:
        """Close the browser and clean up resources."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                logger.debug("Browser already disconnected during close")
            self._browser = None
            self._page = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                logger.debug("Playwright already stopped during close")
            self._playwright = None


class BrowserSessionManager:
    """Manages browser sessions keyed by conversation ID."""

    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSession] = {}

    def get_or_create(
        self,
        conversation_id: str,
        width: int = 1024,
        height: int = 768,
    ) -> BrowserSession:
        """Get an existing session or create a new one for a conversation."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = BrowserSession(width=width, height=height)
        return self._sessions[conversation_id]

    async def close(self, conversation_id: str) -> None:
        """Close and remove a specific session."""
        session = self._sessions.pop(conversation_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        """Close all sessions."""
        for session in self._sessions.values():
            await session.close()
        self._sessions.clear()


# Module-level singleton — replace with your own ComputerSessionManager implementation
# to use a different backend (e.g., VNC, xdotool, pyautogui).
session_manager: ComputerSessionManager = BrowserSessionManager()


def build_computer_tool(**kwargs: object) -> dict[str, str]:
    """Build the API tool dict for the computer use tool.

    The 'computer' tool type accepts no configuration parameters;
    display dimensions and environment are local-only settings used
    by BrowserSession but not sent to the API.
    """
    return {"type": "computer"}


def describe_action(action: Action) -> str:
    """Return a human-readable description of a single computer use action."""
    match action.type:
        case "click":
            return f"click({action.x}, {action.y}, button={action.button})"
        case "double_click":
            return f"double_click({action.x}, {action.y})"
        case "drag":
            coords = " -> ".join(f"({p.x}, {p.y})" for p in action.path)
            return f"drag({coords})"
        case "keypress":
            return f"keypress({', '.join(action.keys)})"
        case "move":
            return f"move({action.x}, {action.y})"
        case "screenshot":
            return "screenshot()"
        case "scroll":
            return f"scroll({action.x}, {action.y}, dx={action.scroll_x}, dy={action.scroll_y})"
        case "type":
            return f"type({action.text!r})"
        case "wait":
            return "wait()"
        case _:
            return f"unknown_action({action.type})"


def describe_actions(actions: list[Action]) -> str:
    """Return a human-readable description of a list of computer use actions."""
    return "\n".join(describe_action(a) for a in actions)


async def execute_computer_actions(actions: list[Action], conversation_id: str) -> str:
    """Execute a list of computer use actions and return a base64-encoded PNG screenshot.

    Uses a persistent browser session for the given conversation ID.
    The session is created on first use with the configured display dimensions.
    Actions are executed in order; a screenshot is taken after the last action.
    """
    import os
    width = int(os.getenv("COMPUTER_USE_DISPLAY_WIDTH", "1024"))
    height = int(os.getenv("COMPUTER_USE_DISPLAY_HEIGHT", "768"))

    session = session_manager.get_or_create(conversation_id, width=width, height=height)
    result = ""
    for action in actions:
        result = await session.execute(action)
    # If no actions were provided, just take a screenshot
    if not result:
        result = await session.screenshot()
    return result
