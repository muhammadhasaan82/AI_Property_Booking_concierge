from typing import Any, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from services.graph import run_chat_graph

router = APIRouter()

class ChatIn(BaseModel):
    user_id: Optional[str] = None
    message: str
    # Optional hints/context you pass in from the client
    filters: Optional[Dict[str, Any]] = None
    booking_args: Optional[Dict[str, Any]] = None
    status_args: Optional[Dict[str, Any]] = None
    payment_args: Optional[Dict[str, Any]] = None

@router.post("/chat/message")
async def chat_message(body: ChatIn):
    result = await run_chat_graph(
        message=body.message,
        filters=body.filters,
        booking_args=body.booking_args,
        status_args=body.status_args,
        payment_args=body.payment_args,
    )
    return result