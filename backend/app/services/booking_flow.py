from __future__ import annotations
"""
Thin compatibility facade for the booking workflow package.

Existing imports from ``app.services.booking_flow`` continue to work unchanged.
"""

from app.services.booking.amendment import (
    detect_booking_amendment_intent,
    handle_booking_amendment_turn,
)
from app.services.booking.cancellation import handle_booking_cancellation_turn
from app.services.booking.constants import BOOKING_ID_PATTERN
from app.services.booking.creation import (
    confirm_booking_review,
    handle_active_booking_turn,
    handle_review_modification_request,
    list_available_cities_payload,
    resume_booking_flow,
    start_booking_for_selected_property,
)
from app.services.booking.date_parser import _extract_dates_by_association, _find_all_dates
from app.services.booking.extraction import _extract_booking_id
from app.services.booking.formatting import _receipt_reply
from app.services.booking.state import (
    has_active_booking_session,
    has_post_confirmation_amendment_context,
)
from app.services.booking.status import handle_booking_status_check

__all__ = [
    "BOOKING_ID_PATTERN",
    "_extract_booking_id",
    "_extract_dates_by_association",
    "_find_all_dates",
    "_receipt_reply",
    "confirm_booking_review",
    "detect_booking_amendment_intent",
    "handle_active_booking_turn",
    "handle_booking_amendment_turn",
    "handle_booking_cancellation_turn",
    "handle_booking_status_check",
    "handle_review_modification_request",
    "has_active_booking_session",
    "has_post_confirmation_amendment_context",
    "list_available_cities_payload",
    "resume_booking_flow",
    "start_booking_for_selected_property",
]


# Backward-compatible private helpers used by existing tests/imports.
from app.services.booking.extraction import (  # noqa: F401
    _extract_amendment_name_value,
    _extract_amendment_updates,
    _sanitize_message_for_amendment_extraction,
)
from app.services.booking.receipt import (  # noqa: F401
    receipt_updates_to_successful_booking_columns,
    successful_booking_row_to_receipt,
)

from app.agents.tools.search import get_all_available_cities  # noqa: F401

# Compatibility wrapper: tests monkeypatch booking_flow.get_all_available_cities.
async def list_available_cities_payload():
    result = await get_all_available_cities()

    if isinstance(result, dict):
        cities = (
            result.get("cities")
            or result.get("available_cities")
            or result.get("data")
            or result.get("results")
        )
        if not cities and result.get("deterministic_reply"):
            return result
    else:
        cities = result

    cities = list(cities or [])
    cities = sorted(str(city) for city in cities)
    reply = (
        "Our service is currently available in: "
        + ", ".join(cities)
        + "."
        if cities
        else "I could not find any available cities right now."
    )

    return {
        "status": "ok",
        "deterministic_reply": reply,
        "answer": reply,
        "source": "available_cities",
        "cities": cities,
    }

# Compatibility wrapper: legacy tests call this synchronously and monkeypatch
# booking_flow.get_all_available_cities.
def list_available_cities_payload():
    import asyncio
    import inspect

    from app.agents.status_codes import Status

    result = get_all_available_cities()

    if inspect.isawaitable(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(result)
        else:
            raise RuntimeError(
                "Synchronous list_available_cities_payload cannot await "
                "get_all_available_cities inside a running event loop."
            )

    if isinstance(result, dict):
        cities = (
            result.get("cities")
            or result.get("available_cities")
            or result.get("data")
            or result.get("results")
        )
        if not cities and result.get("deterministic_reply"):
            result.setdefault("status", Status.CITIES_FOUND)
            return result
    else:
        cities = result

    cities = sorted(str(city) for city in list(cities or []))
    reply = (
        "Our service is currently available in: "
        + ", ".join(cities)
        + "."
        if cities
        else "I could not find any available cities right now."
    )

    return {
        "status": Status.CITIES_FOUND,
        "deterministic_reply": reply,
        "answer": reply,
        "source": "available_cities",
        "cities": cities,
    }
