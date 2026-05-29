import asyncio
import logging
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import app.services.config
from fastapi import FastAPI, Request, Response
from app.route import health, properties, booking, faq, chat, mobile, admin
from app.route import stripe_webhook
from app.route import test as test_router
from datetime import datetime, timezone
from fastapi.middleware.cors import CORSMiddleware
from app.config.model_config_loader import get_model_config_snapshot

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
    Log the resolved model configuration snapshot to the module logger.
    
    This is intended to be run on application startup and records the output of
    get_model_config_snapshot() at INFO level for debugging and observability.
    """
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

    return payload


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
app.include_router(admin.router, tags=["admin"])

