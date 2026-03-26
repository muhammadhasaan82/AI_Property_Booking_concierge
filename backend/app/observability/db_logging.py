from __future__ import annotations

from typing import Optional, Dict, Any

from ..services import db_client


async def log_chat(user_message: str, bot_response: str) -> bool:
    try:
        await db_client.execute(
            """
            insert into public.chat_history (user_message, bot_response)
            values (%s, %s);
            """,
            (user_message, bot_response),
        )
        return True
    except Exception:
        return False


async def insert_booking_details(row: Dict[str, Any]) -> bool:
    if not row:
        return False
    keys = list(row.keys())
    cols = ", ".join(keys)
    placeholders = ", ".join(["%s"] * len(keys))
    values = [row[k] for k in keys]
    try:
        await db_client.execute(
            f"insert into public.booking_details ({cols}) values ({placeholders});",
            values,
        )
        return True
    except Exception:
        return False


async def insert_successful_booking(row: Dict[str, Any]) -> bool:
    if not row:
        return False
    payload = dict(row)
    if payload.get("booking_id") is not None:
        payload["booking_id"] = str(payload["booking_id"])

    keys = list(payload.keys())
    cols = ", ".join(keys)
    placeholders = ", ".join(["%s"] * len(keys))
    updates = ", ".join(f"{k}=excluded.{k}" for k in keys if k != "booking_id")
    values = [payload[k] for k in keys]
    try:
        await db_client.execute(
            f"""
            insert into public.successful_bookings ({cols})
            values ({placeholders})
            on conflict (booking_id)
            do update set {updates};
            """,
            values,
        )
        return True
    except Exception:
        return False


async def get_successful_booking_status(booking_id: str) -> Optional[Dict[str, Any]]:
    if not booking_id:
        return None
    try:
        return await db_client.fetch_one(
            """
            select booking_id,status,check_in,check_out,user_email,user_name,property_title,city,payment_url
            from public.successful_bookings
            where booking_id = %s
            limit 1;
            """,
            (str(booking_id),),
        )
    except Exception:
        return None


async def log_feedback(user_message: str, bot_response: str, rating: str, comment: Optional[str] = None) -> bool:
    try:
        await db_client.execute(
            """
            insert into public.feedback (user_message, bot_response, rating, comment)
            values (%s, %s, %s, %s);
            """,
            (user_message, bot_response, rating, comment),
        )
        return True
    except Exception:
        return False
