
from fastapi import APIRouter
from fastapi import Response

router = APIRouter()

@router.get("/health")
def health():
    return {"ok": True}

@router.options("/chat/message")
async def chat_message_options():
    return Response(status_code=204, headers={"Allow": "POST, OPTIONS"})
