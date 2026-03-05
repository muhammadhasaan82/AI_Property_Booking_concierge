from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from .config import MOCK_MODE, PAYMENT_BASE_URL
from . import db_client

logger = logging.getLogger(__name__)

_USE_MOCK = MOCK_MODE
_USERS: Dict[str, Dict[str, Any]] = {}
_BOOKINGS: Dict[str, Dict[str, Any]] = {}


def _mock_user_key(email: str) -> str:
    return email.strip().lower()


async def get_or_create_user(name: str, email: str, phone: Optional[str] = None) -> str:
    if not email:
        raise ValueError("email required")

    if _USE_MOCK:
        key = _mock_user_key(email)
        if key not in _USERS:
            _USERS[key] = {
                "id": f"user_{uuid.uuid4().hex[:8]}",
                "name": name,
                "email": email,
                "phone": phone,
            }
        else:
            _USERS[key]["name"] = name or _USERS[key].get("name")
            _USERS[key]["phone"] = phone or _USERS[key].get("phone")
        return str(_USERS[key]["id"])

    try:
        row = await db_client.fetch_one(
            """
            insert into public.users (name, email, phone)
            values (%s, %s, %s)
            on conflict (email)
            do update set
                name = excluded.name,
                phone = coalesce(excluded.phone, public.users.phone)
            returning id;
            """,
            (name, email, phone),
        )
        if row and row.get("id"):
            return str(row["id"])
    except Exception as exc:
        logger.warning("[BOOKING] get_or_create_user failed, using transient mock: %s", exc)

    key = _mock_user_key(email)
    _USERS[key] = {
        "id": f"user_{uuid.uuid4().hex[:8]}",
        "name": name,
        "email": email,
        "phone": phone,
        "note": "transient-mock",
    }
    return str(_USERS[key]["id"])


async def create_booking(payload: Dict[str, Any]) -> Dict[str, Any]:
    required = ["user_id", "property_id", "check_in", "check_out"]
    missing = [k for k in required if not payload.get(k)]
    if missing:
        return {"ok": False, "error": f"missing: {', '.join(missing)}"}

    booking_id_for_url = uuid.uuid4().hex[:8].upper()
    payment_url = f"{PAYMENT_BASE_URL}/{booking_id_for_url}"
    guests = int(payload.get("guests", 1))

    if _USE_MOCK:
        rec = {
            "id": booking_id_for_url,
            **payload,
            "guests": guests,
            "status": "pending",
            "payment_url": payment_url,
        }
        _BOOKINGS[booking_id_for_url] = rec
        return {"ok": True, "booking_id": booking_id_for_url, "status": "pending", "payment_url": payment_url}

    try:
        row = await db_client.fetch_one(
            """
            insert into public.bookings
                (user_id, property_id, check_in, check_out, guests, phone, status, payment_url)
            values
                (%s, %s, %s::date, %s::date, %s, %s, %s, %s)
            returning id, status;
            """,
            (
                payload.get("user_id"),
                payload.get("property_id"),
                payload.get("check_in"),
                payload.get("check_out"),
                guests,
                payload.get("phone"),
                "pending",
                payment_url,
            ),
        )
        if row and row.get("id"):
            return {
                "ok": True,
                "booking_id": str(row["id"]),
                "status": row.get("status", "pending"),
                "payment_url": payment_url,
            }
    except Exception as exc:
        logger.warning("[BOOKING] create_booking failed, using transient mock: %s", exc)

    rec = {
        "id": booking_id_for_url,
        **payload,
        "guests": guests,
        "status": "pending",
        "payment_url": payment_url,
        "note": "transient-mock",
    }
    _BOOKINGS[booking_id_for_url] = rec
    return {
        "ok": True,
        "booking_id": booking_id_for_url,
        "status": "pending",
        "payment_url": payment_url,
        "note": "transient-mock",
    }


async def get_booking_status(booking_id: str) -> Dict[str, Any]:
    if not booking_id:
        return {"ok": False, "error": "booking_id required"}

    if booking_id in _BOOKINGS:
        rec = _BOOKINGS[booking_id]
        return {
            "ok": True,
            "status": rec.get("status"),
            "check_in": rec.get("check_in"),
            "check_out": rec.get("check_out"),
        }

    try:
        row = await db_client.fetch_one(
            """
            select status, check_in, check_out
            from public.bookings
            where id = %s
            limit 1;
            """,
            (booking_id,),
        )
        if row:
            return {
                "ok": True,
                "status": row.get("status", "unknown"),
                "check_in": row.get("check_in"),
                "check_out": row.get("check_out"),
            }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    return {"ok": False, "error": "not found"}


async def update_booking_status(booking_id: str, current_status: str, new_status: str) -> Dict[str, Any]:
    if not booking_id or not new_status:
        return {"ok": False, "error": "booking_id and new_status required"}

    if booking_id in _BOOKINGS:
        _BOOKINGS[booking_id]["status"] = new_status
        return {"ok": True}

    try:
        rowcount = await db_client.execute(
            """
            update public.bookings
            set status = %s
            where id = %s and (%s = '' or status = %s);
            """,
            (new_status, booking_id, current_status or "", current_status or ""),
        )
        if rowcount > 0:
            return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    return {"ok": False, "error": "not found"}
