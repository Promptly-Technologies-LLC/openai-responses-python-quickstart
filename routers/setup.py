import logging
import os
import json
from typing import Optional, List
from itertools import zip_longest
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from urllib.parse import quote

from utils.config import update_env_file, generate_registry_file, read_registry_entries, read_mcp_servers, CustomFunction

# Configure logger
logger = logging.getLogger("uvicorn.error")

# Load environment variables
load_dotenv()

router = APIRouter(prefix="/setup", tags=["Setup"])
templates = Jinja2Templates(directory="templates")

@router.post("/api-key")
async def set_openai_api_key(api_key: str = Form(...)) -> RedirectResponse:
    """
    Set the OpenAI API key in the application's environment variables.
    
    Args:
        api_key: OpenAI API key received from form submission
    
    Returns:
        RedirectResponse: Redirects to home page on success
    
    Raises:
        HTTPException: If there's an error updating the environment file
    """
    try:
        safe_key = api_key.strip().replace("\r", "").replace("\n", "")
        update_env_file("OPENAI_API_KEY", safe_key)
        return RedirectResponse(url="/", headers={"HX-Redirect": "/"}, status_code=303)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update API key: {str(e)}"
        )


# Add new setup route
@router.get("/")
async def read_setup(
    request: Request,
    client: AsyncOpenAI = Depends(lambda: AsyncOpenAI()),
    status: Optional[str] = None,
    message_text: Optional[str] = None
) -> Response:
    # Variable initializations
    current_tools: List[str] = []
    current_model: Optional[str] = None
    current_instructions: Optional[str] = None
    # Populate with all models extracted from user-provided HTML, sorted
    available_models: List[str] = sorted([ 
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", 
        "gpt-4o", "gpt-4o-mini", "o1", "o1-mini",
        "o3", "o3-pro", "o3-mini", "o4-mini",
        "o3-deep-research", "o4-mini-deep-research",
        "gpt-5", "gpt-5.4", "gpt-5-mini", "gpt-5-nano",
        "gpt-oss-120b", "gpt-oss-20b"
    ])
    setup_message: str = ""

    # Check if env variables are set
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Load existing app config from env
    current_model = os.getenv("RESPONSES_MODEL")
    current_instructions = os.getenv("RESPONSES_INSTRUCTIONS")
    enabled_tools_csv = os.getenv("ENABLED_TOOLS", "")
    current_tools = [t.strip() for t in enabled_tools_csv.split(",") if t.strip()]
    # SHOW_TOOL_CALL_DETAIL flag
    show_detail_env = os.getenv("SHOW_TOOL_CALL_DETAIL", "false")
    current_show_tool_call_detail = show_detail_env.lower() in {"1", "true", "yes", "on"}

    # Web search config
    web_search_context_size = os.getenv("WEB_SEARCH_CONTEXT_SIZE", "medium")
    web_search_country = os.getenv("WEB_SEARCH_LOCATION_COUNTRY", "")
    web_search_city = os.getenv("WEB_SEARCH_LOCATION_CITY", "")
    web_search_region = os.getenv("WEB_SEARCH_LOCATION_REGION", "")
    web_search_timezone = os.getenv("WEB_SEARCH_LOCATION_TIMEZONE", "")

    # Image generation config
    image_generation_quality = os.getenv("IMAGE_GENERATION_QUALITY", "auto")
    image_generation_size = os.getenv("IMAGE_GENERATION_SIZE", "auto")
    image_generation_background = os.getenv("IMAGE_GENERATION_BACKGROUND", "auto")

    if not openai_api_key:
        setup_message = "OpenAI API key is missing."
    
    # Prepare MCP servers with JSON-encoded headers for safe templating
    existing_mcp_servers_raw = read_mcp_servers()
    existing_mcp_servers: list[dict] = []
    for srv in existing_mcp_servers_raw:
        # srv is a TypedDict; cast to plain dict and compute headers_json
        entry = {
            "server_label": srv.get("server_label", ""),
            **({"server_url": srv["server_url"]} if "server_url" in srv else {}),
            **({"connector_id": srv["connector_id"]} if "connector_id" in srv else {}),
            **({"authorization": srv["authorization"]} if "authorization" in srv else {}),
        }
        # Normalize require_approval to a simple string always/never for the UI
        try:
            if "require_approval" in srv:
                ra = srv.get("require_approval")
                if isinstance(ra, str):
                    entry["require_approval"] = "always" if ra.lower() == "always" else "never"
                elif isinstance(ra, dict):
                    entry["require_approval"] = "always" if "always" in ra else "never"
            else:
                entry["require_approval"] = "never"
        except Exception:
            entry["require_approval"] = "never"
        if "headers" in srv and isinstance(srv["headers"], dict):
            try:
                entry["headers_json"] = json.dumps(srv["headers"]) or ""
            except Exception:
                entry["headers_json"] = ""
        else:
            entry["headers_json"] = ""
        existing_mcp_servers.append(entry)

    return templates.TemplateResponse(
        request, "setup.html",
        {
            "setup_message": setup_message,
            "status": status,
            "status_message": message_text,
            "current_tools": current_tools,
            "current_model": current_model,
            "current_instructions": current_instructions,
            "current_show_tool_call_detail": current_show_tool_call_detail,
            "available_models": available_models,
            "existing_registry_entries": read_registry_entries(),
            "existing_mcp_servers": existing_mcp_servers,
            "web_search_context_size": web_search_context_size,
            "web_search_country": web_search_country,
            "web_search_city": web_search_city,
            "web_search_region": web_search_region,
            "web_search_timezone": web_search_timezone,
            "image_generation_quality": image_generation_quality,
            "image_generation_size": image_generation_size,
            "image_generation_background": image_generation_background,
        }
    )


# HTMX endpoints for registry row add/delete
@router.get("/registry-row")
async def new_registry_row(request: Request) -> Response:
    """Return a single registry row fragment.

    Computes the next index from incoming lists (sent via hx-include) or
    from an explicit `index` query parameter.
    """
    qp = request.query_params
    try:
        # Determine next index robustly from any of the lists or explicit param
        provided_index = qp.get("index")
        if provided_index is not None:
            index = int(provided_index)
        else:
            fn_len = len(qp.getlist("reg_function_names"))
            imp_len = len(qp.getlist("reg_import_paths"))
            tpl_len = len(qp.getlist("reg_template_paths"))
            index = max(fn_len, imp_len, tpl_len)
    except Exception:
        index = 0

    return templates.TemplateResponse(
        request, "components/registry-row.html",
        {"index": index},
    )


@router.delete("/registry-row")
async def delete_registry_row() -> Response:
    """Return empty HTML so the client can remove the row using hx-swap=outerHTML."""
    return Response(content="", media_type="text/html", status_code=200)


@router.get("/mcp-row")
async def new_mcp_row(request: Request) -> Response:
    """Return a single MCP row fragment.

    Computes the next index from incoming lists (sent via hx-include) or
    from an explicit `index` query parameter.
    """
    qp = request.query_params
    try:
        provided_index = qp.get("index")
        if provided_index is not None:
            index = int(provided_index)
        else:
            lbl_len = len(qp.getlist("mcp_labels"))
            url_len = len(qp.getlist("mcp_server_urls"))
            conn_len = len(qp.getlist("mcp_connector_ids"))
            auth_len = len(qp.getlist("mcp_authorizations"))
            hdr_len = len(qp.getlist("mcp_headers_jsons"))
            req_len = len(qp.getlist("mcp_require_approvals"))
            index = max(lbl_len, url_len, conn_len, auth_len, hdr_len, req_len)
    except Exception:
        index = 0

    return templates.TemplateResponse(
        request, "components/mcp-row.html",
        {"index": index},
    )


@router.delete("/mcp-row")
async def delete_mcp_row() -> Response:
    """Return empty HTML so the client can remove the row using hx-swap=outerHTML."""
    return Response(content="", media_type="text/html", status_code=200)


@router.post("/config")
async def save_app_config(
    action: Optional[str] = Form(default=None),
    tool_types: List[str] = Form(default=[]),
    model: Optional[str] = Form(default=None),
    instructions: Optional[str] = Form(default=None),
    show_tool_call_detail: Optional[str] = Form(default=None),
    reg_function_names: List[str] = Form(default=[]),
    reg_import_paths: List[str] = Form(default=[]),
    reg_template_paths: List[str] = Form(default=[]),
    mcp_labels: List[str] = Form(default=[]),
    mcp_server_urls: List[str] = Form(default=[]),
    mcp_connector_ids: List[str] = Form(default=[]),
    mcp_authorizations: List[str] = Form(default=[]),
    mcp_headers_jsons: List[str] = Form(default=[]),
    mcp_require_approvals: List[str] = Form(default=[]),
    web_search_context_size: Optional[str] = Form(default=None),
    web_search_country: Optional[str] = Form(default=None),
    web_search_city: Optional[str] = Form(default=None),
    web_search_region: Optional[str] = Form(default=None),
    web_search_timezone: Optional[str] = Form(default=None),
    image_generation_quality: Optional[str] = Form(default=None),
    image_generation_size: Optional[str] = Form(default=None),
    image_generation_background: Optional[str] = Form(default=None),
) -> RedirectResponse:
    status = "success"
    message_text = ""

    try:
        if action == "regenerate_registry":
            # Preserve existing entries if no meaningful inputs were submitted
            # Custom functions
            has_function_input = any(
                ((fn or "").strip() or (imp or "").strip() or (tpl or "").strip())
                for fn, imp, tpl in zip_longest(reg_function_names, reg_import_paths, reg_template_paths, fillvalue="")
            )
            if has_function_input:
                entries: list[CustomFunction] = []
                for fn, imp, tpl in zip(reg_function_names, reg_import_paths, reg_template_paths):
                    if fn.strip() and imp.strip():
                        entries.append(
                            CustomFunction(name=fn.strip(), import_path=imp.strip(), template_path=(tpl or "").strip())
                        )
            else:
                entries = read_registry_entries()

            # MCP servers
            mcp_servers: list[dict] = []
            # Ensure lists have the same length by padding with empty strings
            max_len = max(
                len(mcp_labels),
                len(mcp_server_urls),
                len(mcp_connector_ids),
                len(mcp_authorizations),
                len(mcp_headers_jsons),
                len(mcp_require_approvals),
            )

            def get_or_empty(items: List[str], i: int) -> str:
                try:
                    return (items[i] or "").strip()
                except IndexError:
                    return ""

            has_mcp_input = any(
                get_or_empty(mcp_labels, i)
                or get_or_empty(mcp_server_urls, i)
                or get_or_empty(mcp_connector_ids, i)
                or get_or_empty(mcp_authorizations, i)
                or get_or_empty(mcp_headers_jsons, i)
                for i in range(max_len)
            )

            if has_mcp_input:
                for i in range(max_len):
                    label = get_or_empty(mcp_labels, i)
                    server_url = get_or_empty(mcp_server_urls, i)
                    connector_id = get_or_empty(mcp_connector_ids, i)
                    authorization = get_or_empty(mcp_authorizations, i)
                    headers_json = get_or_empty(mcp_headers_jsons, i)
                    req_approval = get_or_empty(mcp_require_approvals, i).lower() or "never"
                    if req_approval not in {"always", "never"}:
                        req_approval = "never"
                    if not label:
                        continue
                    # must have either server_url or connector_id
                    if not server_url and not connector_id:
                        continue
                    entry: dict = {
                        "type": "mcp",
                        "server_label": label,
                        # Submitted from frontend (currently hidden static value)
                        "require_approval": req_approval,
                    }
                    if server_url:
                        entry["server_url"] = server_url
                    if connector_id:
                        entry["connector_id"] = connector_id
                    if authorization:
                        entry["authorization"] = authorization
                    if headers_json:
                        try:
                            headers_obj = json.loads(headers_json)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid headers JSON for MCP server '{label}': {e}")
                        if not isinstance(headers_obj, dict):
                            raise ValueError(f"Headers for MCP server '{label}' must be a JSON object")
                        # Optionally coerce all values to strings
                        coerced: dict[str, str] = {str(k): str(v) for k, v in headers_obj.items()}
                        entry["headers"] = coerced
                    mcp_servers.append(entry)
            else:
                mcp_servers = read_mcp_servers()

            generate_registry_file(entries, mcp_servers=mcp_servers)
            status = "success"
            message_text = "Tool config saved."
        elif action == "save_web_search_config":
            # Save web search configuration
            ctx_size = (web_search_context_size or "medium").strip()
            if ctx_size not in {"low", "medium", "high"}:
                ctx_size = "medium"
            update_env_file("WEB_SEARCH_CONTEXT_SIZE", ctx_size)
            update_env_file("WEB_SEARCH_LOCATION_COUNTRY", (web_search_country or "").strip())
            update_env_file("WEB_SEARCH_LOCATION_CITY", (web_search_city or "").strip())
            update_env_file("WEB_SEARCH_LOCATION_REGION", (web_search_region or "").strip())
            update_env_file("WEB_SEARCH_LOCATION_TIMEZONE", (web_search_timezone or "").strip())
            status = "success"
            message_text = "Web search configuration saved."
        elif action == "save_image_generation_config":
            quality = (image_generation_quality or "auto").strip()
            if quality not in {"auto", "low", "medium", "high"}:
                quality = "auto"
            size = (image_generation_size or "auto").strip()
            if size not in {"auto", "1024x1024", "1536x1024", "1024x1536"}:
                size = "auto"
            background = (image_generation_background or "auto").strip()
            if background not in {"auto", "opaque", "transparent"}:
                background = "auto"
            update_env_file("IMAGE_GENERATION_QUALITY", quality)
            update_env_file("IMAGE_GENERATION_SIZE", size)
            update_env_file("IMAGE_GENERATION_BACKGROUND", background)
            status = "success"
            message_text = "Image generation configuration saved."
        else:
            # Standard app config save
            if model is None or instructions is None:
                raise ValueError("Missing model or instructions")
            update_env_file("RESPONSES_MODEL", model)
            update_env_file("RESPONSES_INSTRUCTIONS", instructions)
            enabled_tools_csv = ",".join(tool_types)
            update_env_file("ENABLED_TOOLS", enabled_tools_csv)
            # Persist SHOW_TOOL_CALL_DETAIL flag
            show_detail_value = "true" if (show_tool_call_detail or "").lower() in {"1", "true", "yes", "on"} else "false"
            update_env_file("SHOW_TOOL_CALL_DETAIL", show_detail_value)
            status = "success"
            message_text = "Configuration saved."
    except Exception as e:
        status = "error"
        message_text = f"Operation failed: {e}"

    encoded_message = quote(message_text)
    redirect_url = f"/setup/?status={status}&message_text={encoded_message}"
    return RedirectResponse(url=redirect_url, status_code=303)
