from datetime import datetime, timezone
from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.put("/test/echo")
async def put_echo(payload: dict):
    return {
        "method": "PUT",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }


@router.patch("/test/echo")
async def patch_echo(payload: dict):
    return {
        "method": "PATCH",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }


@router.delete("/test/resource/{item_id}")
async def delete_resource(item_id: str):
    return {
        "message": "Deleted",
        "status": "deleted",
        "deleted": True,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "item_id": item_id,
    }


@router.head("/test/ping")
async def head_ping():
    return Response(headers={"X-App": "AI-Concierge", "X-Ping": "pong"})


@router.options("/test/echo")
async def options_echo(request: Request):
    return Response(
        status_code=204,
        headers={
            "Allow": "OPTIONS, GET, POST, PUT, PATCH, DELETE, HEAD",
        },
    )




