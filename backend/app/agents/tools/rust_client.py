# services/rust_client.py
# Async HTTP client for the Rust autonomous gateway.
# Provides transparent fallback when the Rust service is unavailable.

from __future__ import annotations
import os
import hashlib
import json
from typing import Any, Dict, Optional

import httpx

RUST_GATEWAY_URL = os.getenv("RUST_GATEWAY_URL", "http://localhost:3001")
RUST_TIMEOUT = float(os.getenv("RUST_TIMEOUT", "5.0"))

_rust_available: Optional[bool] = None  # Cached health status


async def _check_health() -> bool:
    """Quick health check against the Rust gateway."""
    global _rust_available
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{RUST_GATEWAY_URL}/health")
            _rust_available = r.status_code == 200
    except Exception:
        _rust_available = False
    return _rust_available


async def execute_tool(data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Call the Rust autonomous gateway's POST /execute endpoint.
    Uses TOON encoding for internal Python ↔ Rust communication.
    Returns the Rust response, or {"fallback": True} if the gateway is unavailable.
    """
    try:
        payload = {"data": data}
        if context:
            payload["context"] = context

        from app.services import toon
        toon_payload = toon.toon_encode(payload)
        headers = {
            "Content-Type": "application/toon",
            "Accept": "application/toon",
        }

        async with httpx.AsyncClient(timeout=RUST_TIMEOUT) as client:
            r = await client.post(
                f"{RUST_GATEWAY_URL}/execute",
                content=toon_payload,
                headers=headers,
            )

        if r.status_code == 200:
            return toon.toon_decode(r.text)
        else:
            print(f"[RUST] Gateway returned {r.status_code}: {r.text[:200]}")
            return {"fallback": True, "error": f"HTTP {r.status_code}"}
    except httpx.ConnectError:
        print("[RUST] Gateway unavailable (connection refused), using Python fallback")
        return {"fallback": True, "error": "connection_refused"}
    except httpx.TimeoutException:
        print("[RUST] Gateway timed out, using Python fallback")
        return {"fallback": True, "error": "timeout"}
    except Exception as e:
        print(f"[RUST] Unexpected error: {type(e).__name__} - {repr(e)}, using Python fallback")
        return {"fallback": True, "error": repr(e)}


async def search_properties(
    location: Optional[str] = None,
    budget: Optional[float] = None,
    beds: Optional[int] = None,
    amenities: Optional[list] = None,
    property_type: Optional[str] = None,
    properties: Optional[list] = None,
) -> Dict[str, Any]:
    """Call the Rust property search tool directly."""
    data: Dict[str, Any] = {}
    if location: data["location"] = location
    if budget is not None: data["budget"] = budget
    if beds is not None: data["beds"] = beds
    if amenities: data["amenities"] = amenities
    if property_type: data["property_type"] = property_type
    if properties: data["properties"] = properties
    return await execute_tool(data)


async def validate_booking(
    property_id: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
    email: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the Rust booking validator tool directly."""
    data: Dict[str, Any] = {
        "property_id": property_id,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
    }
    if email: data["email"] = email
    return await execute_tool(data)


async def compute_pricing(
    price_per_night: float,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    nights: Optional[int] = None,
    guests: int = 1,
    tax_rate: float = 0.10,
) -> Dict[str, Any]:
    """Call the Rust pricing tool directly."""
    data: Dict[str, Any] = {
        "price_per_night": price_per_night,
        "guests": guests,
        "tax_rate": tax_rate,
    }
    if check_in: data["check_in"] = check_in
    if check_out: data["check_out"] = check_out
    if nights is not None: data["nights"] = nights
    return await execute_tool(data)


async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Call the Rust sentiment analysis tool directly."""
    return await execute_tool({"text": text, "analyze_sentiment": True})


async def check_fraud(
    email: Optional[str] = None,
    phone: Optional[str] = None,
    amount: Optional[float] = None,
    guests: Optional[int] = None,
) -> Dict[str, Any]:
    """Call the Rust fraud check tool directly."""
    data: Dict[str, Any] = {"check_fraud": True}
    if email: data["email"] = email
    if phone: data["phone"] = phone
    if amount is not None: data["amount"] = amount
    if guests is not None: data["guests"] = guests
    return await execute_tool(data)
