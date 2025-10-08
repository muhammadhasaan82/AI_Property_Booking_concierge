from __future__ import annotations
import os
from typing import Optional, Dict, Any

try:
    from supabase import create_client, Client  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    create_client = None  # type: ignore
    Client = None  # type: ignore


def _client() -> Optional["Client"]:
    url = os.getenv("SUPABASE_URL"); key = os.getenv("SUPABASE_KEY")
    if not (url and key) or not create_client:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def log_chat(user_message: str, bot_response: str) -> bool:
    c = _client()
    if not c:
        return False
    try:
        c.table("chat_history").insert({
            "user_message": user_message,
            "bot_response": bot_response,
        }).execute()
        return True
    except Exception:
        return False


def insert_booking_details(row: Dict[str, Any]) -> bool:
    c = _client()
    if not c:
        return False
    try:
        c.table("booking_details").insert(row).execute()
        return True
    except Exception:
        return False


