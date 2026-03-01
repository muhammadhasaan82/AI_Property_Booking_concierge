from __future__ import annotations
import os
from typing import Optional, Dict, Any

try:
    from supabase import create_client, Client  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    create_client = None  # type: ignore
    Client = None  # type: ignore


def _client() -> Optional["Client"]:
    url = os.getenv("SUPABASE_URL")
    # Prefer service-role key for server-side writes; fall back to project key/anon.
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
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


def insert_successful_booking(row: Dict[str, Any]) -> bool:
    """
    Persist only successful bookings for later status checks in chat.
    Expected table: public.successful_bookings
    """
    c = _client()
    if not c:
        return False
    try:
        payload = dict(row or {})
        if payload.get("booking_id") is not None:
            payload["booking_id"] = str(payload["booking_id"])
        c.table("successful_bookings").upsert(payload).execute()
        return True
    except Exception:
        return False


def get_successful_booking_status(booking_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch booking summary/status from successful_bookings table.
    Returns None if not found or any error.
    """
    c = _client()
    if not c or not booking_id:
        return None
    try:
        r = (
            c.table("successful_bookings")
            .select("booking_id,status,check_in,check_out,user_email,user_name,property_title,city,payment_url")
            .eq("booking_id", str(booking_id))
            .limit(1)
            .execute()
        )
        rows = getattr(r, "data", None) or []
        if rows:
            return rows[0]
    except Exception:
        return None
    return None


def log_feedback(user_message: str, bot_response: str, rating: str, comment: Optional[str] = None) -> bool:
    """Log user feedback (thumbs up/down) to Supabase.

    Args:
        user_message: The user's original message
        bot_response: The bot's reply
        rating: 'positive' or 'negative'
        comment: Optional user comment
    """
    c = _client()
    if not c:
        return False
    try:
        c.table("feedback").insert({
            "user_message": user_message,
            "bot_response": bot_response,
            "rating": rating,
            "comment": comment,
        }).execute()
        return True
    except Exception:
        return False


