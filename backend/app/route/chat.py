from typing import Any, Dict, Optional
from uuid import uuid4
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.adk_runner import run_adk_turn

router = APIRouter()

class ChatIn(BaseModel):
    user_id: Optional[str] = None
    message: str
    session_id: Optional[str] = None

@router.post("/chat/message")
async def chat_message(body: ChatIn):
    user_id = body.user_id or "api_user"
    session_id = body.session_id or str(uuid4())

    chunks = []
    async for chunk in run_adk_turn(
        user_id=user_id,
        session_id=session_id,
        message=body.message,
    ):
        chunks.append(chunk)

    return {"reply": "".join(chunks)}
