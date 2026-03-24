# main.py
import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Request, Response

from app.route import health, properties, booking, faq, chat, mobile
from app.route import stripe_webhook
from app.route import test as test_router
from datetime import datetime, timezone
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Concierge & Calling Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your frontend origin(s) in production
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
    return {
        "legacy_mode": LEGACY_RULES,
        "intent_catalog": get_intent_catalog().model_dump(),
        "routing_policies": get_routing_policies().model_dump(),
        "guardrails": get_guardrails().model_dump(),
        "vocabulary": get_vocabulary().model_dump()
    }

# POST example: echoes JSON payload
@app.post("/echo")
async def post_echo(payload: dict):
    return {
        "method": "POST",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload
    }

# PUT example: full update semantics
@app.put("/echo")
async def put_echo(payload: dict):
    return {
        "method": "PUT",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload
    }

# PATCH example: partial update semantics
@app.patch("/echo")
async def patch_echo(payload: dict):
    return {
        "method": "PATCH",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload
    }

# DELETE example: delete a resource by id
@app.delete("/resource/{item_id}")
async def delete_resource(item_id: str):
    return {
        "message": "Deleted",
        "status": "deleted",
        "deleted": True,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "item_id": item_id
    }

# HEAD example: lightweight liveness header-only
@app.head("/ping")
async def head_ping():
    return Response(headers={"X-App": "AI-Concierge", "X-Ping": "pong"})

# OPTIONS example: advertise allowed methods for /echo
@app.options("/echo")
async def options_echo(request: Request):
    return Response(
        status_code=204,
        headers={
            "Allow": "OPTIONS, GET, POST, PUT, PATCH, DELETE, HEAD"
        }
    )

app.include_router(health.router, prefix="/api/v1", tags=["health"])
# app.include_router(properties.router, prefix="/api/v1", tags=["properties"])
# app.include_router(booking.router, prefix="/api/v1", tags=["booking"])
# app.include_router(faq.router, prefix="/api/v1", tags=["faq"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(stripe_webhook.router, prefix="/api/v1", tags=["webhooks"])
# app.include_router(test_router.router, prefix="/api/v1", tags=["test"])

# Mobile API endpoints
# app.include_router(mobile.router, prefix="/api/v1", tags=["mobile"])


