# services/booking.py
# Booking/storage helpers with automatic mock fallback if Supabase env is missing.
from __future__ import annotations
import os
import uuid
from typing import Dict, Any

_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# In-memory mock DB for local/dev usage
_USERS: Dict[str, Dict[str, Any]] = {}
_BOOKINGS: Dict[str, Dict[str, Any]] = {}

def _mock_user_key(email: str) -> str:
    return email.strip().lower()

def get_or_create_user(name: str, email: str, phone: str | None = None) -> str:
    """
    Returns a user_id. Uses mock store if Supabase env not set.
    """
    if not email:
        raise ValueError("email required")
    # Mock mode (no Supabase env)
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        key = _mock_user_key(email)
        if key not in _USERS:
            _USERS[key] = {"id": f"user_{uuid.uuid4().hex[:8]}", "name": name, "email": email, "phone": phone}
        else:
            # Update basic fields
            _USERS[key].update({"name": name or _USERS[key]["name"], "phone": phone or _USERS[key].get("phone")})
        return _USERS[key]["id"]

    # Real Supabase path (optional): implement if you want real persistence
    # Import lazily to avoid dependency if not configured
    try:
        from supabase import create_client, Client  # type: ignore
        client: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        # Upsert user
        data = {"name": name, "email": email, "phone": phone}
        res = client.table("users").upsert(data, on_conflict="email").execute()
        if not res.data:
            # Try select if upsert returns nothing
            res = client.table("users").select("*").eq("email", email).execute()
        user = (res.data or [{}])[0]
        if not user.get("id"):
            raise RuntimeError("Supabase user upsert failed")
        return str(user["id"])
    except Exception as e:
        # Fall back to mock if Supabase errors out
        print(f"[BOOKING] Supabase user creation failed: {e}, falling back to mock")
        key = _mock_user_key(email)
        _USERS[key] = {"id": f"user_{uuid.uuid4().hex[:8]}", "name": name, "email": email, "phone": phone, "note": f"mock:{e}"}
        return _USERS[key]["id"]

def create_booking(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects:
      user_id, property_id, check_in, check_out, guests, phone
    Returns:
      {"ok": True, "booking_id": "...", "status": "confirmed", "payment_url": "..."}
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
            "status": "confirmed",
            "payment_url": f"https://example.com/pay/{booking_id}",
        }
        _BOOKINGS[booking_id] = rec
        return {"ok": True, "booking_id": booking_id, "status": rec["status"], "payment_url": rec["payment_url"]}

    # Real Supabase (optional)
    try:
        from supabase import create_client, Client  # type: ignore
        client: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        res = client.table("bookings").insert(payload).execute()
        row = (res.data or [{}])[0]
        if not row.get("id"):
            raise RuntimeError("insert booking failed")
        # Create a fake payment link or use your payments table/webhook
        payment_url = f"https://example.com/pay/{row['id']}"
        return {"ok": True, "booking_id": row["id"], "status": row.get("status", "confirmed"), "payment_url": payment_url}
    except Exception as e:
        print(f"[BOOKING] Supabase booking creation failed: {e}")
        return {"ok": False, "error": str(e)}

def get_booking_status(booking_id: str) -> Dict[str, Any]:
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
    # Real Supabase (optional)
    if _SUPABASE_URL and _SUPABASE_KEY:
        try:
            from supabase import create_client, Client  # type: ignore
            client: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
            res = client.table("bookings").select("status, check_in, check_out").eq("id", booking_id).single().execute()
            if not res.data:
                return {"ok": False, "error": "not found"}
            return {
                "ok": True,
                "status": res.data.get("status", "unknown"),
                "check_in": res.data.get("check_in"),
                "check_out": res.data.get("check_out"),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "not found"}

def update_booking_status(booking_id: str, current_status: str, new_status: str) -> Dict[str, Any]:
    if not booking_id or not new_status:
        return {"ok": False, "error": "booking_id and new_status required"}
    # Mock
    if booking_id in _BOOKINGS:
        _BOOKINGS[booking_id]["status"] = new_status
        return {"ok": True}
    # Real Supabase (optional)
    if _SUPABASE_URL and _SUPABASE_KEY:
        try:
            from supabase import create_client, Client  # type: ignore
            client: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
            res = client.table("bookings").update({"status": new_status}).eq("id", booking_id).execute()
            if not res.data:
                return {"ok": False, "error": "not found"}
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "not found"}
