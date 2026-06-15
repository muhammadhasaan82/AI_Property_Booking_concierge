from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.config.agent_config_loader import cfg
from app.agents.status_codes import Status
from app.services.booking.amendment import detect_booking_amendment_intent
from app.services.booking.cancellation import _detect_booking_cancellation_intent
from app.services.booking.constants import _BOOKING_STATUS_INTENT_PATTERNS, BOOKING_ID_PATTERN
from app.services.booking.extraction import _extract_booking_id
from app.services.booking.formatting import _render_booking_status_reply
from app.services.booking.receipt import (
    _latest_session_receipt,
    _lookup_booking_in_session,
    _store_current_receipt,
    successful_booking_row_to_receipt,
)
from app.services.booking.state import _normalize

def _detect_booking_status_intent(message: str) -> bool:
    """
    Deterministically detect whether the user message expresses a booking-status intent.

    Matches the normalized message against a configured set of regex patterns
    (e.g. "my booking ID is", "check my booking status", "registration ID").
    Also returns True when a booking ID token (BK-YYYYMMDD-XXXXXXXX) is present
    in the message even without an explicit intent phrase.

    Parameters:
        message (str): The raw user message.

    Returns:
        bool: True if the message is recognised as a booking-status request.
    """
    if not message:
        return False
    normalized = _normalize(message)
    if not normalized:
        return False
    for pattern in _BOOKING_STATUS_INTENT_PATTERNS:
        if pattern.search(normalized):
            return True
    if BOOKING_ID_PATTERN.search(message):
        return True
    return False

async def handle_booking_status_check(
    message: str,
    soft_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Deterministic booking-status handler — checks session soft_state before falling back to DB/ADK.

    Behaviour:
      1. Detects booking-status intent in the message.
      2. Extracts a booking ID (BK-YYYYMMDD-XXXXXXXX) if present.
      3. If ID is present and found in session → returns deterministic status.
      4. If ID is present but NOT found in session → attempts DB lookup
         (`public.bookings`, then `public.successful_bookings`);
         if both fail → returns not-found message.
      5. If no ID but session has a booking_receipt → returns latest booking status.
      6. If no ID and no receipt → asks user for booking ID.
      7. If booking-status intent is NOT detected → returns None (let pipeline continue).

    Parameters:
        message (str): The user's raw message.
        soft_state (Dict[str, Any]): Current session soft_state (mutable dict from Redis snapshot).

    Returns:
        Optional[Dict[str, Any]]: A payload with `deterministic_reply` when handled,
        or None when booking-status intent is not detected (pass-through).
    """
    if not _detect_booking_status_intent(message):
        return None

    if soft_state.get("booking_cancellation_stage") or soft_state.get("booking_cancellation_pending"):
        return None
    if soft_state.get("pending_booking_amendment"):
        return None
    amendment_stages = {
        "awaiting_amendment_choice",
        "awaiting_amendment_values",
        "awaiting_amendment_confirmation",
    }
    if str(soft_state.get("booking_stage") or "") in amendment_stages:
        return None
    if _detect_booking_cancellation_intent(message):
        return None
    if detect_booking_amendment_intent(message, soft_state):
        return None

    booking_id = _extract_booking_id(message)

    if booking_id:
        receipt = _lookup_booking_in_session(booking_id, soft_state)
        if receipt:
            _store_current_receipt(soft_state, receipt)
            return {
                "status": Status.FOUND,
                "receipt": receipt,
                "deterministic_reply": _render_booking_status_reply(receipt),
            }

        try:
            from app.services.booking.persistence import get_booking_status
            from app.observability.db_logging import get_successful_booking_status

            db_result = await get_booking_status(booking_id)
            if db_result.get("ok"):
                merged = {
                    "booking_id": booking_id,
                    "status": db_result.get("status") or cfg.booking_confirmed_status,
                    "check_in": db_result.get("check_in") or "",
                    "check_out": db_result.get("check_out") or "",
                }
                _store_current_receipt(soft_state, merged)
                return {
                    "status": Status.FOUND,
                    "receipt": merged,
                    "deterministic_reply": _render_booking_status_reply(merged),
                }

            db_row = await get_successful_booking_status(booking_id)
            if db_row:
                merged = successful_booking_row_to_receipt(db_row)
                merged.setdefault("booking_id", booking_id)
                _store_current_receipt(soft_state, merged)
                return {
                    "status": Status.FOUND,
                    "receipt": merged,
                    "deterministic_reply": _render_booking_status_reply(merged),
                }
        except Exception as exc:
            logger.debug("[booking_status] DB lookup failed for %s: %s", booking_id, exc)

        not_found_template = str(
            getattr(cfg, "booking_status_not_found_template", "") or ""
        ).strip()
        if not not_found_template:
            not_found_template = "It looks like your booking wasn't found in our system."
        return {
            "status": Status.BOOKING_NOT_FOUND,
            "deterministic_reply": not_found_template,
        }

    latest = _latest_session_receipt(soft_state)
    if latest:
        _store_current_receipt(soft_state, latest)
        return {
            "status": Status.FOUND,
            "receipt": latest,
            "deterministic_reply": _render_booking_status_reply(latest),
        }

    ask_template = str(
        getattr(cfg, "booking_status_ask_for_id_template", "") or ""
    ).strip()
    if not ask_template:
        ask_template = (
            "I'd be happy to check your booking status. "
            "Could you share your registration ID? "
            "It looks like BK-YYYYMMDD-XXXXXXXX."
        )
    return {
        "status": Status.GATHERING_INFO,
        "deterministic_reply": ask_template,
    }

