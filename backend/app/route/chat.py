"""FastAPI chat routes backed by the shared ADK runner service."""
from __future__ import annotations

from typing import Optional
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.adk_runner import run_adk_turn

router = APIRouter()


class ChatMessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatMessageResponse(BaseModel):
    reply: str
    user_id: str
    session_id: str


@router.post("/chat/message", response_model=ChatMessageResponse)
async def chat_message(request: ChatMessageRequest) -> ChatMessageResponse:
    user_id = request.user_id or "api_user"
    session_id = request.session_id or str(uuid4())

    chunks: list[str] = []
    async for chunk in run_adk_turn(
        user_id=user_id,
        session_id=session_id,
        message=request.message,
    ):
        chunks.append(chunk)

    return ChatMessageResponse(
        reply="".join(chunks),
        user_id=user_id,
        session_id=session_id,
    )
