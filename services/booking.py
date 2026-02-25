# services/booking.py
# Booking/storage helpers — fully async with Supabase REST via httpx.
from __future__ import annotations
import os
import uuid
from typing import Dict, Any, Optional

import httpx

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
_SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

# In-memory mock DB for local/dev usage
_USERS: Dict[str, Dict[str, Any]] = {}
_BOOKINGS: Dict[str, Dict[str, Any]] = {}


def _mock_user_key(email: str) -> str:
    return email.strip().lower()


def _supabase_headers() -> Dict[str, str]:
    return {
        "apikey": _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _rest_url(table: str) -> str:
    """Build Supabase PostgREST URL for a table."""
    base = _SUPABASE_URL.rstrip("/")
    return f"{base}/rest/v1/{table}"


async def get_or_create_user(name: str, email: str, phone: Optional[str] = None) -> str:
    """
    Returns a user_id. Uses mock store if Supabase env not set.
    Fully async — safe to call from FastAPI/LangGraph without blocking.
    """
    if not email:
        raise ValueError("email required")

    # Mock mode (no Supabase env)
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        key = _mock_user_key(email)
        if key not in _USERS:
            _USERS[key] = {
                "id": f"user_{uuid.uuid4().hex[:8]}",
                "name": name,
                "email": email,
                "phone": phone,
            }
        else:
            _USERS[key].update({
                "name": name or _USERS[key]["name"],
                "phone": phone or _USERS[key].get("phone"),
            })
        return _USERS[key]["id"]

    # Real Supabase path — async via httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Upsert user
            data = {"name": name, "email": email, "phone": phone}
            r = await client.post(
                _rest_url("users"),
                headers={
                    **_supabase_headers(),
                    "Prefer": "return=representation,resolution=merge-duplicates",
                },
                json=data,
            )
            if r.status_code in (200, 201):
                rows = r.json()
                user = rows[0] if isinstance(rows, list) and rows else {}
                if user.get("id"):
                    return str(user["id"])

            # Fallback: select existing
            r2 = await client.get(
                f"{_rest_url('users')}?email=eq.{email}&select=id",
                headers=_supabase_headers(),
            )
            if r2.status_code == 200:
                rows2 = r2.json()
                if rows2 and rows2[0].get("id"):
                    return str(rows2[0]["id"])

            raise RuntimeError(f"Supabase user upsert failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[BOOKING] Supabase user creation failed: {e}, falling back to mock")
        key = _mock_user_key(email)
        _USERS[key] = {
            "id": f"user_{uuid.uuid4().hex[:8]}",
            "name": name,
            "email": email,
            "phone": phone,
            "note": f"mock:{e}",
        }
        return _USERS[key]["id"]


async def create_booking(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects: user_id, property_id, check_in, check_out, guests, phone
    Returns: {"ok": True, "booking_id": "...", "status": "pending", "payment_url": "..."}
    """
    required = ["user_id", "property_id", "check_in", "check_out"]
    missing = [k for k in required if not payload.get(k)]
    if missing:
        return {"ok": False, "error": f"missing: {', '.join(missing)}"}

    # Mock path
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        booking_id = uuid.uuid4().hex[:8].upper()
        rec = {
            "id": booking_id,
            "user_id": payload["user_id"],
            "property_id": payload["property_id"],
            "check_in": payload["check_in"],
            "check_out": payload["check_out"],
            "guests": int(payload.get("guests", 1)),
            "phone": payload.get("phone"),
            "status": "pending",
            "payment_url": f"https://example.com/pay/{booking_id}",
        }
        _BOOKINGS[booking_id] = rec
        return {
            "ok": True,
            "booking_id": booking_id,
            "status": rec["status"],
            "payment_url": rec["payment_url"],
        }

    # Real Supabase (async)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                _rest_url("bookings"),
                headers=_supabase_headers(),
                json=payload,
            )
            if r.status_code in (200, 201):
                rows = r.json()
                row = rows[0] if isinstance(rows, list) and rows else {}
                if row.get("id"):
                    payment_url = f"https://example.com/pay/{row['id']}"
                    return {
                        "ok": True,
                        "booking_id": row["id"],
                        "status": row.get("status", "pending"),
                        "payment_url": payment_url,
                    }
            raise RuntimeError(f"insert booking failed: {r.status_code}")
    except Exception as e:
        print(f"[BOOKING] Supabase booking creation failed: {e}")
        return {"ok": False, "error": str(e)}


async def get_booking_status(booking_id: str) -> Dict[str, Any]:
    if not booking_id:
        return {"ok": False, "error": "booking_id required"}

    # Mock
    if booking_id in _BOOKINGS:
        rec = _BOOKINGS[booking_id]
        return {
            "ok": True,
            "status": rec.get("status"),
            "check_in": rec.get("check_in"),
            "check_out": rec.get("check_out"),
        }

    # Real Supabase (async)
    if _SUPABASE_URL and _SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{_rest_url('bookings')}?id=eq.{booking_id}&select=status,check_in,check_out",
                    headers={**_supabase_headers(), "Accept": "application/vnd.pgrst.object+json"},
                )
                if r.status_code == 200:
                    data = r.json()
                    return {
                        "ok": True,
                        "status": data.get("status", "unknown"),
                        "check_in": data.get("check_in"),
                        "check_out": data.get("check_out"),
                    }
                return {"ok": False, "error": "not found"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "not found"}


async def update_booking_status(booking_id: str, current_status: str, new_status: str) -> Dict[str, Any]:
    if not booking_id or not new_status:
        return {"ok": False, "error": "booking_id and new_status required"}

    # Mock
    if booking_id in _BOOKINGS:
        _BOOKINGS[booking_id]["status"] = new_status
        return {"ok": True}

    # Real Supabase (async)
    if _SUPABASE_URL and _SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.patch(
                    f"{_rest_url('bookings')}?id=eq.{booking_id}",
                    headers=_supabase_headers(),
                    json={"status": new_status},
                )
                if r.status_code in (200, 204):
                    return {"ok": True}
                return {"ok": False, "error": "not found"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "not found"}
