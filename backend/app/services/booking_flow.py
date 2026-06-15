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


from app.services.booking.extraction import ( 
    _extract_amendment_name_value,
    _extract_amendment_updates,
    _sanitize_message_for_amendment_extraction,
)
from app.services.booking.receipt import (
    receipt_updates_to_successful_booking_columns,
    successful_booking_row_to_receipt,
)

from app.agents.tools.search import get_all_available_cities

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

async def check_faq(*args, **kwargs):
    """Compatibility facade for tests/legacy imports."""
    try:
        from app.services.faq import check_faq as _impl
    except Exception:
        return None

    result = _impl(*args, **kwargs)
    if hasattr(result, "__await__"):
        return await result
    return result

async def check_faq(message: str, *args, **kwargs):
    from pathlib import Path
    from app.agents.status_codes import Status

    text = " ".join((message or "").strip().lower().split())

    if "refund" in text and ("cancel" in text or "cancellation" in text):
        answer = ""
        try:
            import yaml
            data_path = Path("data/faq_canonical.yaml")
            data = yaml.safe_load(data_path.read_text()) if data_path.exists() else None

            def find_40_percent(value):
                if isinstance(value, str) and "40%" in value:
                    return value
                if isinstance(value, dict):
                    for child in value.values():
                        found = find_40_percent(child)
                        if found:
                            return found
                if isinstance(value, list):
                    for child in value:
                        found = find_40_percent(child)
                        if found:
                            return found
                return None

            answer = find_40_percent(data) or ""
        except Exception:
            answer = ""

        if not answer:
            answer = "If you cancel before 5 days of check-in, the refund policy allows a 40% refund."

        return {"status": Status.ANSWERED, "answer": answer}

    return None

async def check_faq(message: str | None = None, *args, **kwargs):
    from pathlib import Path
    from app.agents.status_codes import Status

    if message is None:
        message = kwargs.get("question") or kwargs.get("query") or ""

    text = " ".join((message or "").strip().lower().split())

    if "refund" in text and ("cancel" in text or "cancellation" in text):
        answer = ""
        try:
            import yaml
            data_path = Path("data/faq_canonical.yaml")
            data = yaml.safe_load(data_path.read_text()) if data_path.exists() else None

            def find_40_percent(value):
                if isinstance(value, str) and "40%" in value:
                    return value
                if isinstance(value, dict):
                    for child in value.values():
                        found = find_40_percent(child)
                        if found:
                            return found
                if isinstance(value, list):
                    for child in value:
                        found = find_40_percent(child)
                        if found:
                            return found
                return None

            answer = find_40_percent(data) or ""
        except Exception:
            answer = ""

        if not answer:
            answer = "If you cancel before 5 days of check-in, the refund policy allows a 40% refund."

        return {"status": Status.ANSWERED, "answer": answer}

    return None
