"""
Tests for computer use tool utilities.

Tests cover:
1. describe_action(): human-readable descriptions for all 9 action types
2. Key and button mapping helpers
3. BrowserSession: Playwright-based action execution and screenshots
4. BrowserSessionManager: session lifecycle management
5. execute_computer_action(): integration with session manager
6. build_computer_tool(): correct tool dict for the API
"""

import base64
import io

import pytest
from PIL import Image

from openai.types.responses.computer_action import (
    Click,
    DoubleClick,
    Drag,
    DragPath,
    Keypress,
    Move,
    Screenshot,
    Scroll,
    Type,
    Wait,
)

from utils.computer_use import (
    BrowserSession,
    BrowserSessionManager,
    build_computer_tool,
    describe_action,
    execute_computer_actions,
    _map_key,
    _map_button,
)


class TestDescribeAction:
    """Tests for describe_action() with all 9 action types."""

    def test_click(self):
        action = Click(type="click", x=100, y=200, button="left")
        assert describe_action(action) == "click(100, 200, button=left)"

    def test_click_right_button(self):
        action = Click(type="click", x=50, y=75, button="right")
        assert describe_action(action) == "click(50, 75, button=right)"

    def test_double_click(self):
        action = DoubleClick(type="double_click", x=300, y=400)
        assert describe_action(action) == "double_click(300, 400)"

    def test_drag(self):
        action = Drag(
            type="drag",
            path=[
                DragPath(x=10, y=20),
                DragPath(x=30, y=40),
            ],
        )
        assert describe_action(action) == "drag((10, 20) -> (30, 40))"

    def test_keypress(self):
        action = Keypress(type="keypress", keys=["ctrl", "c"])
        assert describe_action(action) == "keypress(ctrl, c)"

    def test_move(self):
        action = Move(type="move", x=500, y=600)
        assert describe_action(action) == "move(500, 600)"

    def test_screenshot(self):
        action = Screenshot(type="screenshot")
        assert describe_action(action) == "screenshot()"

    def test_scroll(self):
        action = Scroll(type="scroll", x=100, y=200, scroll_x=0, scroll_y=-3)
        assert describe_action(action) == "scroll(100, 200, dx=0, dy=-3)"

    def test_type(self):
        action = Type(type="type", text="hello world")
        assert describe_action(action) == "type('hello world')"

    def test_wait(self):
        action = Wait(type="wait")
        assert describe_action(action) == "wait()"


class TestKeyMapping:
    """Tests for _map_key() OpenAI -> Playwright key name mapping."""

    def test_modifier_keys(self):
        assert _map_key("ctrl") == "Control"
        assert _map_key("shift") == "Shift"
        assert _map_key("alt") == "Alt"
        assert _map_key("meta") == "Meta"

    def test_case_insensitive(self):
        assert _map_key("CTRL") == "Control"
        assert _map_key("Shift") == "Shift"

    def test_arrow_keys(self):
        assert _map_key("up") == "ArrowUp"
        assert _map_key("down") == "ArrowDown"
        assert _map_key("left") == "ArrowLeft"
        assert _map_key("right") == "ArrowRight"

    def test_special_keys(self):
        assert _map_key("enter") == "Enter"
        assert _map_key("return") == "Enter"
        assert _map_key("tab") == "Tab"
        assert _map_key("backspace") == "Backspace"
        assert _map_key("delete") == "Delete"
        assert _map_key("escape") == "Escape"
        assert _map_key("esc") == "Escape"
        assert _map_key("space") == " "

    def test_function_keys(self):
        assert _map_key("f1") == "F1"
        assert _map_key("f12") == "F12"

    def test_passthrough(self):
        assert _map_key("a") == "a"
        assert _map_key("1") == "1"


class TestButtonMapping:
    """Tests for _map_button() OpenAI -> Playwright button mapping."""

    def test_standard_buttons(self):
        assert _map_button("left") == "left"
        assert _map_button("right") == "right"
        assert _map_button("middle") == "middle"

    def test_wheel_maps_to_middle(self):
        assert _map_button("wheel") == "middle"

    def test_unknown_defaults_to_left(self):
        assert _map_button("back") == "left"
        assert _map_button("forward") == "left"


class TestBrowserSession:
    """Tests for BrowserSession with real Playwright."""

    @pytest.fixture
    async def session(self):
        s = BrowserSession(width=800, height=600)
        yield s
        await s.close()

    @pytest.mark.anyio
    async def test_screenshot_returns_valid_png(self, session: BrowserSession):
        result = await session.screenshot()
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"\x89PNG"

    @pytest.mark.anyio
    async def test_screenshot_has_correct_dimensions(self, session: BrowserSession):
        result = await session.screenshot()
        img = Image.open(io.BytesIO(base64.b64decode(result)))
        assert img.size == (800, 600)

    @pytest.mark.anyio
    async def test_execute_click(self, session: BrowserSession):
        action = Click(type="click", x=100, y=100, button="left")
        result = await session.execute(action)
        assert len(result) > 0
        assert base64.b64decode(result)[:4] == b"\x89PNG"

    @pytest.mark.anyio
    async def test_execute_type(self, session: BrowserSession):
        action = Type(type="type", text="hello")
        result = await session.execute(action)
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_execute_screenshot(self, session: BrowserSession):
        action = Screenshot(type="screenshot")
        result = await session.execute(action)
        assert base64.b64decode(result)[:4] == b"\x89PNG"

    @pytest.mark.anyio
    async def test_execute_scroll(self, session: BrowserSession):
        action = Scroll(type="scroll", x=400, y=300, scroll_x=0, scroll_y=100)
        result = await session.execute(action)
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_execute_move(self, session: BrowserSession):
        action = Move(type="move", x=200, y=150)
        result = await session.execute(action)
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_execute_keypress(self, session: BrowserSession):
        action = Keypress(type="keypress", keys=["tab"])
        result = await session.execute(action)
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_execute_double_click(self, session: BrowserSession):
        action = DoubleClick(type="double_click", x=100, y=100)
        result = await session.execute(action)
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_execute_drag(self, session: BrowserSession):
        action = Drag(
            type="drag",
            path=[DragPath(x=10, y=10), DragPath(x=200, y=200)],
        )
        result = await session.execute(action)
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_landing_page_has_url_input(self, session: BrowserSession):
        page = await session._ensure_page()
        url_input = await page.query_selector("#url")
        assert url_input is not None

    @pytest.mark.anyio
    async def test_landing_page_has_go_button(self, session: BrowserSession):
        page = await session._ensure_page()
        button = await page.query_selector("button[type=submit]")
        assert button is not None
        text = await button.text_content()
        assert text is not None
        assert "Go" in text

    @pytest.mark.anyio
    async def test_landing_page_navigation(self, session: BrowserSession):
        page = await session._ensure_page()
        # Intercept the navigation request and serve a fake page
        await page.route("https://example.com/", lambda route: route.fulfill(
            status=200,
            content_type="text/html",
            body="<h1>Hello</h1>",
        ))
        await page.fill("#url", "https://example.com/")
        await page.click("button[type=submit]")
        await page.wait_for_load_state("load")
        content = await page.content()
        assert "Hello" in content

    @pytest.mark.anyio
    async def test_landing_page_not_blank(self, session: BrowserSession):
        result = await session.screenshot()
        img = Image.open(io.BytesIO(base64.b64decode(result)))
        colors = img.getcolors(maxcolors=10)
        # A blank white page would have exactly 1 color; landing page has more
        assert colors is None or len(colors) > 1

    @pytest.mark.anyio
    async def test_close_is_idempotent(self, session: BrowserSession):
        await session.close()
        await session.close()  # Should not raise

    @pytest.mark.anyio
    async def test_relaunch_after_close(self, session: BrowserSession):
        await session.screenshot()  # Ensure browser is launched
        await session.close()
        # Should relaunch on next use
        result = await session.screenshot()
        assert base64.b64decode(result)[:4] == b"\x89PNG"


class TestBrowserSessionManager:
    """Tests for BrowserSessionManager lifecycle."""

    @pytest.fixture
    def manager(self):
        return BrowserSessionManager()

    def test_get_or_create_returns_same_session(self, manager: BrowserSessionManager):
        s1 = manager.get_or_create("conv-1")
        s2 = manager.get_or_create("conv-1")
        assert s1 is s2

    def test_different_conversations_get_different_sessions(self, manager: BrowserSessionManager):
        s1 = manager.get_or_create("conv-1")
        s2 = manager.get_or_create("conv-2")
        assert s1 is not s2

    @pytest.mark.anyio
    async def test_close_removes_session(self, manager: BrowserSessionManager):
        manager.get_or_create("conv-1")
        await manager.close("conv-1")
        # Next call should create a new session
        s2 = manager.get_or_create("conv-1")
        assert s2 is not None

    @pytest.mark.anyio
    async def test_close_all(self, manager: BrowserSessionManager):
        manager.get_or_create("conv-1")
        manager.get_or_create("conv-2")
        await manager.close_all()
        # Internal state should be cleared
        assert len(manager._sessions) == 0

    @pytest.mark.anyio
    async def test_close_nonexistent_is_safe(self, manager: BrowserSessionManager):
        await manager.close("nonexistent")  # Should not raise


class TestBuildComputerTool:
    """Tests for build_computer_tool() API tool dict construction."""

    def test_returns_computer_type(self):
        """Tool dict should use type 'computer' (not 'computer_use_preview')."""
        tool = build_computer_tool()
        assert tool["type"] == "computer"

    def test_no_display_width_in_tool(self):
        """display_width should NOT be sent to the API (not a valid param for 'computer' type)."""
        tool = build_computer_tool(display_width=1280, display_height=720, environment="browser")
        assert "display_width" not in tool

    def test_no_display_height_in_tool(self):
        """display_height should NOT be sent to the API."""
        tool = build_computer_tool(display_width=1280, display_height=720, environment="browser")
        assert "display_height" not in tool

    def test_no_environment_in_tool(self):
        """environment should NOT be sent to the API."""
        tool = build_computer_tool(display_width=1280, display_height=720, environment="mac")
        assert "environment" not in tool

    def test_only_type_key(self):
        """Tool dict should contain only the 'type' key."""
        tool = build_computer_tool(display_width=1024, display_height=768, environment="browser")
        assert tool == {"type": "computer"}


class TestExecuteComputerAction:
    """Integration tests for execute_computer_action()."""

    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch):
        monkeypatch.setenv("COMPUTER_USE_DISPLAY_WIDTH", "640")
        monkeypatch.setenv("COMPUTER_USE_DISPLAY_HEIGHT", "480")

    @pytest.fixture(autouse=True)
    async def _cleanup(self):
        yield
        from utils.computer_use import session_manager
        await session_manager.close_all()

    @pytest.mark.anyio
    async def test_returns_valid_png(self):
        action = Screenshot(type="screenshot")
        result = await execute_computer_actions([action],"test-conv")
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"\x89PNG"

    @pytest.mark.anyio
    async def test_reuses_session_for_same_conversation(self):
        from utils.computer_use import session_manager

        action = Screenshot(type="screenshot")
        await execute_computer_actions([action],"test-conv")
        await execute_computer_actions([action],"test-conv")
        assert "test-conv" in session_manager._sessions

    @pytest.mark.anyio
    async def test_different_conversations_different_sessions(self):
        from utils.computer_use import session_manager

        action = Screenshot(type="screenshot")
        await execute_computer_actions([action],"conv-a")
        await execute_computer_actions([action],"conv-b")
        assert "conv-a" in session_manager._sessions
        assert "conv-b" in session_manager._sessions
        assert session_manager._sessions["conv-a"] is not session_manager._sessions["conv-b"]
