# services/booking.py
# Booking/storage helpers — fully async via Rust DB Gateway → Supabase PostgreSQL.
# Falls back to Supabase REST directly if the Rust gateway is unavailable.
from __future__ import annotations
import logging
import os
import uuid
from typing import Dict, Any, Optional

import httpx

from .config import MOCK_MODE, PAYMENT_BASE_URL

logger = logging.getLogger(__name__)

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
_SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

# Rust DB Gateway (port 3002 by default)
_DB_GATEWAY_URL = os.getenv("DB_GATEWAY_URL", "http://localhost:3002").rstrip("/")

# Only use mock if explicitly requested
_USE_MOCK = MOCK_MODE

if MOCK_MODE:
    logger.warning("[BOOKING] MOCK_MODE=true: all booking operations use in-memory store")

# In-memory mock DB (only used when MOCK_MODE=true)
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


async def _try_rust_gateway(path: str, method: str = "POST", body: dict | None = None) -> dict | None:
    """Attempt to call the Rust DB Gateway; returns None on failure (silently falls back)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = f"{_DB_GATEWAY_URL}{path}"
            if method == "POST":
                r = await client.post(url, json=body or {})
            elif method == "GET":
                r = await client.get(url)
            elif method == "PATCH":
                r = await client.patch(url, json=body or {})
            else:
                return None
            if r.status_code in (200, 201):
                return r.json()
    except Exception:
        pass
    return None


async def get_or_create_user(name: str, email: str, phone: Optional[str] = None) -> str:
    """
    Returns a user_id. Routes through Rust DB Gateway → Supabase REST → mock.
    """
    if not email:
        raise ValueError("email required")

    # Mock mode
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
            _USERS[key].update({
                "name": name or _USERS[key]["name"],
                "phone": phone or _USERS[key].get("phone"),
            })
        return _USERS[key]["id"]

    # ── Primary: Rust DB Gateway ──────────────────────────────────────────────
    result = await _try_rust_gateway("/users/upsert", "POST", {
        "name": name, "email": email, "phone": phone
    })
    if result and result.get("ok") and result.get("user_id"):
        return str(result["user_id"])

    # ── Fallback: Supabase REST directly ──────────────────────────────────────
    if _SUPABASE_URL and _SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
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

                r2 = await client.get(
                    f"{_rest_url('users')}?email=eq.{email}&select=id",
                    headers=_supabase_headers(),
                )
                if r2.status_code == 200:
                    rows2 = r2.json()
                    if rows2 and rows2[0].get("id"):
                        return str(rows2[0]["id"])

        except Exception as e:
            logger.warning("[BOOKING] Supabase REST user upsert failed: %s", e)

    # ── Emergency mock fallback ───────────────────────────────────────────────
    logger.warning("[BOOKING] All DB paths failed for user %s — using transient mock", email)
    key = _mock_user_key(email)
    _USERS[key] = {
        "id": f"user_{uuid.uuid4().hex[:8]}",
        "name": name,
        "email": email,
        "phone": phone,
        "note": "transient-mock",
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

    booking_id_for_url = uuid.uuid4().hex[:8].upper()
    payment_url = f"{PAYMENT_BASE_URL}/{booking_id_for_url}"

    # Mock path
    if _USE_MOCK:
        rec = {
            "id": booking_id_for_url,
            **payload,
            "guests": int(payload.get("guests", 1)),
            "status": "pending",
            "payment_url": payment_url,
        }
        _BOOKINGS[booking_id_for_url] = rec
        return {"ok": True, "booking_id": booking_id_for_url, "status": "pending", "payment_url": payment_url}

    # ── Primary: Rust DB Gateway ──────────────────────────────────────────────
    result = await _try_rust_gateway("/bookings/create", "POST", {
        **payload,
        "guests": int(payload.get("guests", 1)),
        "payment_url": payment_url,
    })
    if result and result.get("ok") and result.get("booking_id"):
        return {
            "ok": True,
            "booking_id": result["booking_id"],
            "status": result.get("status", "pending"),
            "payment_url": payment_url,
        }

    # ── Fallback: Supabase REST directly ──────────────────────────────────────
    if _SUPABASE_URL and _SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    _rest_url("bookings"),
                    headers=_supabase_headers(),
                    json={**payload, "guests": int(payload.get("guests", 1)), "payment_url": payment_url},
                )
                if r.status_code in (200, 201):
                    rows = r.json()
                    row = rows[0] if isinstance(rows, list) and rows else {}
                    if row.get("id"):
                        return {
                            "ok": True,
                            "booking_id": row["id"],
                            "status": row.get("status", "pending"),
                            "payment_url": payment_url,
                        }
        except Exception as e:
            logger.warning("[BOOKING] Supabase REST booking creation failed: %s", e)

    return {"ok": False, "error": "All database paths unavailable"}


async def get_booking_status(booking_id: str) -> Dict[str, Any]:
    if not booking_id:
        return {"ok": False, "error": "booking_id required"}

    # Mock path
    if booking_id in _BOOKINGS:
        rec = _BOOKINGS[booking_id]
        return {"ok": True, "status": rec.get("status"), "check_in": rec.get("check_in"), "check_out": rec.get("check_out")}

    # ── Primary: Rust DB Gateway ──────────────────────────────────────────────
    result = await _try_rust_gateway(f"/bookings/{booking_id}/status", "GET")
    if result and result.get("ok"):
        return result

    # ── Fallback: Supabase REST directly ──────────────────────────────────────
    if _SUPABASE_URL and _SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{_rest_url('bookings')}?id=eq.{booking_id}&select=status,check_in,check_out",
                    headers={**_supabase_headers(), "Accept": "application/vnd.pgrst.object+json"},
                )
                if r.status_code == 200:
                    data = r.json()
                    return {"ok": True, "status": data.get("status", "unknown"),
                            "check_in": data.get("check_in"), "check_out": data.get("check_out")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "not found"}


async def update_booking_status(booking_id: str, current_status: str, new_status: str) -> Dict[str, Any]:
    if not booking_id or not new_status:
        return {"ok": False, "error": "booking_id and new_status required"}

    # Mock path
    if booking_id in _BOOKINGS:
        _BOOKINGS[booking_id]["status"] = new_status
        return {"ok": True}

    # ── Primary: Rust DB Gateway ──────────────────────────────────────────────
    result = await _try_rust_gateway(f"/bookings/{booking_id}/status", "PATCH", {"status": new_status})
    if result and result.get("ok"):
        return result

    # ── Fallback: Supabase REST directly ──────────────────────────────────────
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
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "not found"}
