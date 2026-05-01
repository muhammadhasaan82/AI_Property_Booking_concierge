import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Request, Response
from app.route import health, properties, booking, faq, chat, mobile, admin
from app.route import stripe_webhook
from app.route import test as test_router
from datetime import datetime, timezone
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Concierge & Calling Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
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
    """Inspect the internal state of all dynamically loaded rules and lexicons."""
    payload = {
        "legacy_mode": LEGACY_RULES,
        "intent_catalog": get_intent_catalog().model_dump(),
        "routing_policies": get_routing_policies().model_dump(),
        "guardrails": get_guardrails().model_dump(),
        "vocabulary": get_vocabulary().model_dump()
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

@app.post("/echo")
async def post_echo(payload: dict):
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

