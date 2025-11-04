# main.py
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from route import health, properties, booking, faq, chat, mobile
from route import test as test_router
from datetime import datetime, timezone
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
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


    @app.get("/chatbot", include_in_schema=False)
    async def serve_chatbot():
        """Serve the integrated React-based chatbot UI."""
        return FileResponse("public/ai-estate-chatbot.html")

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
# app.include_router(test_router.router, prefix="/api/v1", tags=["test"])

# Mobile API endpoints
# app.include_router(mobile.router, prefix="/api/v1", tags=["mobile"])

# Serve your existing static UI (voice page)
app.mount("/static", StaticFiles(directory="public", html=True), name="public")

# -------------------- Realtime session minting (for /static index.html) --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_ORG = os.getenv("OPENAI_ORG", "").strip()

@app.post("/session")
async def create_session():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    url = "https://api.openai.com/v1/realtime/client_secrets"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "realtime=v1",
    }
    if OPENAI_ORG:
        headers["OpenAI-Organization"] = OPENAI_ORG
    payload = {
        "session": {
            "type": "realtime",
            # Use a current realtime model; adjust if your account has a different name available
            "model": os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17"),
            "instructions": (
                "You are a helpful multilingual voice assistant. "
                "You can communicate in English, Arabic, Korean, and Urdu. "
                "Respond in the same language the user speaks, or in English if unsure. "
                "Be brief and clear in your responses."
            ),
        }
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as e:
        print(f"[/session] httpx error: {e}", flush=True)
        return JSONResponse(status_code=502, content={"error": f"Upstream connect error: {e}"})

    if r.status_code != 200:
        print(f"[/session] OpenAI returned {r.status_code}: {r.text[:400]}", flush=True)
        return JSONResponse(status_code=r.status_code, content={"error": r.text})

    try:
        data = r.json()
    except Exception as e:
        print(f"[/session] Bad JSON parse: {e}, raw={r.text[:400]}", flush=True)
        return JSONResponse(status_code=500, content={"error": f"Bad JSON from OpenAI: {e}", "raw": r.text})

    cs = data.get("client_secret") or {}
    value = cs.get("value") if isinstance(cs, dict) else None
    expires_at = cs.get("expires_at") if isinstance(cs, dict) else None
    if not value:
        # Surface the full upstream for debugging in dev
        return JSONResponse(status_code=502, content={
            "error": "Missing client_secret.value from OpenAI",
            "upstream": data,
        })
    return {"client_secret": {"value": value, "expires_at": expires_at}}


class TranscriptPayload(BaseModel):
    line: str


@app.post("/transcript")
async def receive_transcript(payload: TranscriptPayload):
    line = (payload.line or "").strip()
    if not line:
        return {"ok": True, "received": 0}
    try:
        print(line, flush=True)
    except Exception:
        pass
    return {"ok": True, "received": 1}