"""
Regression tests for setup page conditional rendering.

The setup page renders different sections based on:
1. Whether OPENAI_API_KEY is set (if not, shows API key form only)
2. Which tools are enabled in ENABLED_TOOLS env var:
   - file_search: Shows file upload section
   - function: Shows custom functions section
   - mcp: Shows MCP servers section
   - function OR mcp: Shows "Save tool config" button
"""

from playwright.sync_api import Page, expect


class TestApiKeyMissing:
    """Tests for when OPENAI_API_KEY is not set."""

    def test_shows_api_key_form_when_key_missing(
        self, page: Page, base_url: str, app_server, env_no_api_key
    ):
        """When API key is missing, only the API key input form should be visible."""
        page.goto(f"{base_url}/setup/")

        # API key form should be visible
        api_key_input = page.locator('input[name="api_key"]')
        expect(api_key_input).to_be_visible()

        save_api_key_button = page.locator('button:has-text("Save API Key")')
        expect(save_api_key_button).to_be_visible()

        # Setup message should mention API key
        setup_message = page.locator(".setupMessage")
        expect(setup_message).to_contain_text("API key")

    def test_hides_config_form_when_key_missing(
        self, page: Page, base_url: str, app_server, env_no_api_key
    ):
        """When API key is missing, the main config form should NOT be visible."""
        page.goto(f"{base_url}/setup/")

        # Config form elements should NOT be visible
        model_select = page.locator("#model-select")
        expect(model_select).not_to_be_visible()

        instructions_input = page.locator("#instructions-input")
        expect(instructions_input).not_to_be_visible()

        save_config_button = page.locator('button:has-text("Save Configuration")')
        expect(save_config_button).not_to_be_visible()


class TestApiKeyPresent:
    """Tests for when OPENAI_API_KEY is set."""

    def test_shows_config_form_when_key_present(
        self, page: Page, base_url: str, app_server, env_api_key_no_tools
    ):
        """When API key is present, the main config form should be visible."""
        page.goto(f"{base_url}/setup/")

        # Config form elements should be visible
        model_select = page.locator("#model-select")
        expect(model_select).to_be_visible()

        instructions_input = page.locator("#instructions-input")
        expect(instructions_input).to_be_visible()

        save_config_button = page.locator('button:has-text("Save Configuration")')
        expect(save_config_button).to_be_visible()

    def test_hides_api_key_form_when_key_present(
        self, page: Page, base_url: str, app_server, env_api_key_no_tools
    ):
        """When API key is present, the API key input form should NOT be visible."""
        page.goto(f"{base_url}/setup/")

        api_key_input = page.locator('input[name="api_key"]')
        expect(api_key_input).not_to_be_visible()

        save_api_key_button = page.locator('button:has-text("Save API Key")')
        expect(save_api_key_button).not_to_be_visible()

    def test_tool_checkboxes_visible(
        self, page: Page, base_url: str, app_server, env_api_key_no_tools
    ):
        """Tool selection checkboxes should be visible in the config form."""
        page.goto(f"{base_url}/setup/")

        # All tool checkboxes should be visible
        code_interpreter_cb = page.locator('input[value="code_interpreter"]')
        expect(code_interpreter_cb).to_be_visible()

        file_search_cb = page.locator('input[value="file_search"]')
        expect(file_search_cb).to_be_visible()

        function_cb = page.locator('input[value="function"]')
        expect(function_cb).to_be_visible()

        mcp_cb = page.locator('input[value="mcp"]')
        expect(mcp_cb).to_be_visible()


class TestNoToolsEnabled:
    """Tests for when no conditional tools are enabled."""

    def test_no_tool_sections_visible(
        self, page: Page, base_url: str, app_server, env_api_key_no_tools
    ):
        """When no tools are enabled, no tool-specific sections should appear."""
        page.goto(f"{base_url}/setup/")

        # File search section should NOT be visible
        upload_form = page.locator("#upload-form")
        expect(upload_form).not_to_be_visible()

        # Custom functions section should NOT be visible
        registry_rows = page.locator("#registry-rows")
        expect(registry_rows).not_to_be_visible()

        # MCP section should NOT be visible
        mcp_rows = page.locator("#mcp-rows")
        expect(mcp_rows).not_to_be_visible()

        # Save tool config button should NOT be visible
        save_tool_config = page.locator('button:has-text("Save tool config")')
        expect(save_tool_config).not_to_be_visible()


class TestFileSearchTool:
    """Tests for file_search tool conditional rendering."""

    def test_shows_file_upload_section(
        self, page: Page, base_url: str, app_server, env_file_search_only
    ):
        """When file_search is enabled, file upload section should be visible."""
        page.goto(f"{base_url}/setup/")

        # File upload section should be visible
        upload_files_heading = page.locator('h3:has-text("Upload Files")')
        expect(upload_files_heading).to_be_visible()

        upload_form = page.locator("#upload-form")
        expect(upload_form).to_be_visible()

        file_input = page.locator('input[type="file"]')
        expect(file_input).to_be_visible()

        upload_button = page.locator('button:has-text("Upload File")')
        expect(upload_button).to_be_visible()

    def test_file_search_checkbox_checked(
        self, page: Page, base_url: str, app_server, env_file_search_only
    ):
        """The file_search checkbox should be checked."""
        page.goto(f"{base_url}/setup/")

        file_search_cb = page.locator('input[value="file_search"]')
        expect(file_search_cb).to_be_checked()

    def test_no_function_or_mcp_sections(
        self, page: Page, base_url: str, app_server, env_file_search_only
    ):
        """Function and MCP sections should NOT be visible with file_search only."""
        page.goto(f"{base_url}/setup/")

        registry_rows = page.locator("#registry-rows")
        expect(registry_rows).not_to_be_visible()

        mcp_rows = page.locator("#mcp-rows")
        expect(mcp_rows).not_to_be_visible()

        # Save tool config button should NOT be visible (only appears with function or mcp)
        save_tool_config = page.locator('button:has-text("Save tool config")')
        expect(save_tool_config).not_to_be_visible()


class TestFunctionTool:
    """Tests for function tool conditional rendering."""

    def test_shows_custom_functions_section(
        self, page: Page, base_url: str, app_server, env_function_only
    ):
        """When function is enabled, custom functions section should be visible."""
        page.goto(f"{base_url}/setup/")

        # Custom functions section should be visible
        functions_heading = page.locator('h3:has-text("Register Custom Functions")')
        expect(functions_heading).to_be_visible()

        registry_rows = page.locator("#registry-rows")
        expect(registry_rows).to_be_visible()

        # Should have at least one registry row
        registry_row = page.locator(".registry-row").first
        expect(registry_row).to_be_visible()

        # Add function button should be visible
        add_function_button = page.locator('button:has-text("Add Function")')
        expect(add_function_button).to_be_visible()

    def test_function_checkbox_checked(
        self, page: Page, base_url: str, app_server, env_function_only
    ):
        """The function checkbox should be checked."""
        page.goto(f"{base_url}/setup/")

        function_cb = page.locator('input[value="function"]')
        expect(function_cb).to_be_checked()

    def test_shows_save_tool_config_button(
        self, page: Page, base_url: str, app_server, env_function_only
    ):
        """Save tool config button should be visible when function is enabled."""
        page.goto(f"{base_url}/setup/")

        save_tool_config = page.locator('button:has-text("Save tool config")')
        expect(save_tool_config).to_be_visible()

    def test_no_file_search_or_mcp_sections(
        self, page: Page, base_url: str, app_server, env_function_only
    ):
        """File search and MCP sections should NOT be visible with function only."""
        page.goto(f"{base_url}/setup/")

        upload_form = page.locator("#upload-form")
        expect(upload_form).not_to_be_visible()

        mcp_rows = page.locator("#mcp-rows")
        expect(mcp_rows).not_to_be_visible()


class TestMcpTool:
    """Tests for mcp tool conditional rendering."""

    def test_shows_mcp_servers_section(
        self, page: Page, base_url: str, app_server, env_mcp_only
    ):
        """When mcp is enabled, MCP servers section should be visible."""
        page.goto(f"{base_url}/setup/")

        # MCP servers section should be visible
        mcp_heading = page.locator('h3:has-text("Register MCP Servers")')
        expect(mcp_heading).to_be_visible()

        mcp_rows = page.locator("#mcp-rows")
        expect(mcp_rows).to_be_visible()

        # Should have at least one MCP row
        mcp_row = page.locator(".mcp-row").first
        expect(mcp_row).to_be_visible()

        # Add MCP Server button should be visible
        add_mcp_button = page.locator('button:has-text("Add MCP Server")')
        expect(add_mcp_button).to_be_visible()

    def test_mcp_checkbox_checked(
        self, page: Page, base_url: str, app_server, env_mcp_only
    ):
        """The mcp checkbox should be checked."""
        page.goto(f"{base_url}/setup/")

        mcp_cb = page.locator('input[value="mcp"]')
        expect(mcp_cb).to_be_checked()

    def test_shows_save_tool_config_button(
        self, page: Page, base_url: str, app_server, env_mcp_only
    ):
        """Save tool config button should be visible when mcp is enabled."""
        page.goto(f"{base_url}/setup/")

        save_tool_config = page.locator('button:has-text("Save tool config")')
        expect(save_tool_config).to_be_visible()

    def test_no_file_search_or_function_sections(
        self, page: Page, base_url: str, app_server, env_mcp_only
    ):
        """File search and function sections should NOT be visible with mcp only."""
        page.goto(f"{base_url}/setup/")

        upload_form = page.locator("#upload-form")
        expect(upload_form).not_to_be_visible()

        registry_rows = page.locator("#registry-rows")
        expect(registry_rows).not_to_be_visible()


class TestFileSearchAndFunction:
    """Tests for file_search + function combination."""

    def test_shows_both_sections(
        self, page: Page, base_url: str, app_server, env_file_search_and_function
    ):
        """Both file upload and custom functions sections should be visible."""
        page.goto(f"{base_url}/setup/")

        # File upload section
        upload_form = page.locator("#upload-form")
        expect(upload_form).to_be_visible()

        # Custom functions section
        registry_rows = page.locator("#registry-rows")
        expect(registry_rows).to_be_visible()

        # Save tool config button
        save_tool_config = page.locator('button:has-text("Save tool config")')
        expect(save_tool_config).to_be_visible()

    def test_mcp_section_not_visible(
        self, page: Page, base_url: str, app_server, env_file_search_and_function
    ):
        """MCP section should NOT be visible."""
        page.goto(f"{base_url}/setup/")

        mcp_rows = page.locator("#mcp-rows")
        expect(mcp_rows).not_to_be_visible()


class TestFileSearchAndMcp:
    """Tests for file_search + mcp combination."""

    def test_shows_both_sections(
        self, page: Page, base_url: str, app_server, env_file_search_and_mcp
    ):
        """Both file upload and MCP servers sections should be visible."""
        page.goto(f"{base_url}/setup/")

        # File upload section
        upload_form = page.locator("#upload-form")
        expect(upload_form).to_be_visible()

        # MCP servers section
        mcp_rows = page.locator("#mcp-rows")
        expect(mcp_rows).to_be_visible()

        # Save tool config button
        save_tool_config = page.locator('button:has-text("Save tool config")')
        expect(save_tool_config).to_be_visible()

    def test_function_section_not_visible(
        self, page: Page, base_url: str, app_server, env_file_search_and_mcp
    ):
        """Function section should NOT be visible."""
        page.goto(f"{base_url}/setup/")

        registry_rows = page.locator("#registry-rows")
        expect(registry_rows).not_to_be_visible()


class TestFunctionAndMcp:
    """Tests for function + mcp combination."""

    def test_shows_both_sections(
        self, page: Page, base_url: str, app_server, env_function_and_mcp
    ):
        """Both custom functions and MCP servers sections should be visible."""
        page.goto(f"{base_url}/setup/")

        # Custom functions section
        registry_rows = page.locator("#registry-rows")
        expect(registry_rows).to_be_visible()

        # MCP servers section
        mcp_rows = page.locator("#mcp-rows")
        expect(mcp_rows).to_be_visible()

        # Save tool config button
        save_tool_config = page.locator('button:has-text("Save tool config")')
        expect(save_tool_config).to_be_visible()

    def test_file_search_section_not_visible(
        self, page: Page, base_url: str, app_server, env_function_and_mcp
    ):
        """File search section should NOT be visible."""
        page.goto(f"{base_url}/setup/")

        upload_form = page.locator("#upload-form")
        expect(upload_form).not_to_be_visible()


class TestAllToolsEnabled:
    """Tests for when all three conditional tools are enabled."""

    def test_shows_all_sections(
        self, page: Page, base_url: str, app_server, env_all_tools
    ):
        """All tool sections should be visible."""
        page.goto(f"{base_url}/setup/")

        # File upload section
        upload_form = page.locator("#upload-form")
        expect(upload_form).to_be_visible()

        # Custom functions section
        registry_rows = page.locator("#registry-rows")
        expect(registry_rows).to_be_visible()

        # MCP servers section
        mcp_rows = page.locator("#mcp-rows")
        expect(mcp_rows).to_be_visible()

        # Save tool config button
        save_tool_config = page.locator('button:has-text("Save tool config")')
        expect(save_tool_config).to_be_visible()

    def test_all_tool_checkboxes_checked(
        self, page: Page, base_url: str, app_server, env_all_tools
    ):
        """All three tool checkboxes should be checked."""
        page.goto(f"{base_url}/setup/")

        file_search_cb = page.locator('input[value="file_search"]')
        expect(file_search_cb).to_be_checked()

        function_cb = page.locator('input[value="function"]')
        expect(function_cb).to_be_checked()

        mcp_cb = page.locator('input[value="mcp"]')
        expect(mcp_cb).to_be_checked()
