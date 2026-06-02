import asyncio
import logging
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Request, Response
from app.config.env_loader import get_env_debug_snapshot, load_backend_env

load_backend_env()

from app.route import health, properties, booking, faq, chat, mobile, admin
from app.route import stripe_webhook
from app.route import test as test_router
from datetime import datetime, timezone
from fastapi.middleware.cors import CORSMiddleware
from app.config.model_config_loader import get_model_config_snapshot
from app.services.redis_store import get_session_snapshot
from app.services.observability.langfuse_observer import (
    LANGFUSE_ENABLED,
    LANGFUSE_BASE_URL,
    LANGFUSE_ENVIRONMENT,
    LANGFUSE_RELEASE,
    LANGFUSE_PROMPTS_ENABLED,
    LANGFUSE_SAMPLE_RATE,
    LANGFUSE_REDACT_INPUTS,
    _is_configured
)

app = FastAPI(title="AI Concierge & Calling Agent")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def log_resolved_model_config():
    """
    Log secret-safe environment and model configuration snapshots at startup.
    
    This is intended to be run on application startup and records the output of
    get_model_config_snapshot() at INFO level for debugging and observability.
    """
    logger.info("[Config] Dotenv sources: %s", get_env_debug_snapshot())
    logger.info("[Config] Resolved model config: %s", get_model_config_snapshot())

@app.get("/")
async def root():
    """
    Return basic API metadata and a mapping of public endpoint names to their paths.
    
    Returns:
        info (dict): Dictionary with keys:
            - "message": API identification string.
            - "status": Current service status.
            - "endpoints": Mapping of endpoint names to their URL paths.
    """
    return {
        "message": "AI Concierge & Calling Agent API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "properties": "/properties",
            "booking": "/booking", 
            "faq": "/faq",
            "chat": "/chat",
            "mobile": "/mobile",
            "docs": "/docs",
            "static": "/static"
        }
    }

from app.services.dynamic_config import (
    LEGACY_RULES, get_intent_catalog, get_routing_policies, get_guardrails, get_vocabulary
)

@app.get("/debug/config", tags=["debug"])
async def debug_config():
    """
    Return a diagnostic snapshot of dynamically loaded configuration, policies, lexicons, and resolved model mappings.
    
    The returned payload aggregates legacy flags, intent/routing/guardrail/vocabulary dumps, resolved model configuration, and optional metadata (booking schema, tool registry, agent/response policies, and admin config version/reload timestamp) when those components are available.
    
    Returns:
        payload (dict): A mapping containing:
            - legacy_mode: boolean flag for legacy rules.
            - intent_catalog: serialized intent catalog.
            - routing_policies: serialized routing policies.
            - guardrails: serialized guardrails.
            - vocabulary: serialized vocabulary.
            - resolved_models: resolved model configuration snapshot.
            - booking_schema_version (optional): version string of the booking schema.
            - booking_schema_tools (optional): list of booking schema tool names.
            - tool_registry_version (optional): version string of the tool registry.
            - tool_registry_tools (optional): list of tool registry tool names.
            - agent_policy_version (optional): version string of the agent policy.
            - agent_policy_tools (optional): list of agent policy tool names.
            - response_policies_version (optional): version string of the response policies.
            - response_policies_tools (optional): list of response policy tool names.
            - config_version (optional): admin config version metadata.
            - last_reload_at (optional): admin config last reload timestamp.
    """
    payload = {
        "legacy_mode": LEGACY_RULES,
        "intent_catalog": get_intent_catalog().model_dump(),
        "routing_policies": get_routing_policies().model_dump(),
        "guardrails": get_guardrails().model_dump(),
        "vocabulary": get_vocabulary().model_dump(),
        "dotenv": get_env_debug_snapshot(),
        "resolved_models": get_model_config_snapshot(),
    }
    try:
        from app.config.booking_schema_loader import booking_schema as _bs
        payload["booking_schema_version"] = _bs.version
        payload["booking_schema_tools"] = list(_bs.tools.keys())
    except Exception:
        pass

    try:
        from app.config.tool_registry_loader import registry as _reg
        payload["tool_registry_version"] = _reg.version
        payload["tool_registry_tools"] = list(_reg.tools.keys())
    except Exception:
        pass

    try:
        from app.config.agent_policy_loader import policy as _pol
        payload["agent_policy_version"] = _pol.version
        payload["agent_policy_tools"] = list(_pol.tools.keys())
    except Exception:
        pass

    try:
        from app.config.response_policies_loader import policies as _rp
        payload["response_policies_version"] = _rp.version
        payload["response_policies_tools"] = list(_rp.tools.keys())
    except Exception:
        pass

    try:
        from app.route.admin import _CONFIG_VERSION, _LAST_RELOAD_AT
        payload["config_version"] = _CONFIG_VERSION
        payload["last_reload_at"] = _LAST_RELOAD_AT
    except Exception:
        pass

    try:
        from app.config.service_coverage_loader import get_service_coverage_snapshot
        payload["service_coverage"] = get_service_coverage_snapshot()
    except Exception:
        pass

    return payload


@app.get("/debug/session/{session_id}", tags=["debug"])
async def debug_session(session_id: str) -> dict:
    snapshot = await get_session_snapshot(session_id)
    state = snapshot.get("state", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(state, dict):
        state = {}

    soft_state = state.get("soft_state")
    if not isinstance(soft_state, dict):
        soft_state = state

    allowed_keys = {
        "active_flow",
        "active_property_options_generated_at",
        "active_property_options_map",
        "active_property_options_shown_count",
        "active_property_options_total_found",
        "all_search_results",
        "booking_property_id",
        "booking_required_fields",
        "booking_selected_property",
        "booking_stage",
        "current_page",
        "last_filters",
        "last_presented_view",
        "last_rejected_property_id",
        "last_search",
        "last_selected_property_at",
        "last_selected_property_id",
        "option_map",
        "page_size",
        "visible_results",
    }
    safe_soft_state = {
        key: soft_state[key]
        for key in sorted(allowed_keys)
        if key in soft_state
    }

    return {
        "session_id": session_id,
        "state_keys": sorted(state.keys()),
        "soft_state": safe_soft_state,
    }


@app.get("/debug/model-config", tags=["debug"])
async def debug_model_config():
    """
    Return a snapshot of resolved model configuration identifiers and metadata without exposing credentials.
    
    The snapshot contains resolved model names/identifiers and related configuration metadata useful for debugging model routing and selection. Secret values or credentials are not included.
    
    Returns:
        dict: Mapping of resolved model configuration details (identifiers and metadata).
    """
    return get_model_config_snapshot()


@app.get("/debug/adk-model-config", tags=["debug"])
async def debug_adk_model_config():
    """
    Return selected ADK agent model identifiers and related metadata without exposing credentials.
    
    Returns:
        payload (dict): Mapping with keys:
            - snapshot: Resolved model configuration snapshot from get_model_config_snapshot().
            - adk_dispatcher_model: Dispatcher model identifier from the ADK agents module, or `None` if unavailable.
            - adk_voice_model: Voice model identifier from the ADK agents module, or `None` if unavailable.
            - dispatcher_llm: String representation of the dispatcher LLM object, or an empty string if unavailable.
            - voice_llm: String representation of the voice LLM object, or an empty string if unavailable.
            - adk_agents_path: Filesystem path to the ADK agents module (`__file__`), or `None` if unavailable.
    """
    from app.agents import adk_agents as a
    return {
        "snapshot": get_model_config_snapshot(),
        "adk_dispatcher_model": getattr(a, "DISPATCHER_MODEL", None),
        "adk_voice_model": getattr(a, "VOICE_MODEL", None),
        "dispatcher_llm": str(getattr(a, "dispatcher_llm", "")),
        "voice_llm": str(getattr(a, "voice_llm", "")),
        "adk_agents_path": getattr(a, "__file__", None),
    }

@app.post("/echo")
async def post_echo(payload: dict):
    """
    Echoes the given payload along with a method label and UTC timestamp.
    
    Parameters:
        payload (dict): The data to include in the echoed response.
    
    Returns:
        dict: A mapping with keys:
            - `method`: the HTTP method label ("POST").
            - `received_at`: ISO 8601 UTC timestamp when the payload was received.
            - `data`: the original `payload` value.
    """
    return {
        "method": "POST",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload
    }

@app.put("/echo")
async def put_echo(payload: dict):
    return {
        "method": "PUT",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload
    }

@app.patch("/echo")
async def patch_echo(payload: dict):
    return {
        "method": "PATCH",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload
    }

@app.delete("/resource/{item_id}")
async def delete_resource(item_id: str):
    return {
        "message": "Deleted",
        "status": "deleted",
        "deleted": True,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "item_id": item_id
    }

@app.head("/ping")
async def head_ping():
    return Response(headers={"X-App": "AI-Concierge", "X-Ping": "pong"})

@app.options("/echo")
async def options_echo(request: Request):
    return Response(
        status_code=204,
        headers={
            "Allow": "OPTIONS, GET, POST, PUT, PATCH, DELETE, HEAD"
        }
    )

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(stripe_webhook.router, prefix="/api/v1", tags=["webhooks"])


@app.get("/debug/langfuse", tags=["debug"])
async def debug_langfuse():
    """
    Return Langfuse observability configuration status without exposing secrets.
    """
    from urllib.parse import urlparse
    
    base_url_host = "unknown"
    if LANGFUSE_BASE_URL:
        try:
            parsed = urlparse(LANGFUSE_BASE_URL)
            base_url_host = parsed.hostname or parsed.path or "unknown"
        except Exception:
            base_url_host = "unknown"

    return {
        "enabled": LANGFUSE_ENABLED,
        "configured": _is_configured(),
        "base_url_host": base_url_host,
        "environment": LANGFUSE_ENVIRONMENT,
        "release": LANGFUSE_RELEASE,
        "prompts_enabled": LANGFUSE_PROMPTS_ENABLED,
        "redaction_enabled": LANGFUSE_REDACT_INPUTS,
        "sample_rate": LANGFUSE_SAMPLE_RATE,
    }

app.include_router(admin.router, tags=["admin"])
