from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import re

from app.config.agent_config_loader import cfg
from app.services.booking.constants import (
    BOOKING_ID_PATTERN,
    _CANCELLATION_PATTERNS,
    _NEGATED_CANCELLATION_PATTERNS,
    _NO_TOKENS,
    _YES_TOKENS,
)
from app.services.booking.extraction import _extract_booking_id
from app.services.booking.formatting import _receipt_reply
from app.services.booking.receipt import successful_booking_row_to_receipt
from app.services.booking.state import _normalize

def _detect_booking_cancellation_intent(message: str) -> bool:
    if not message:
        return False
    normalized = _normalize(message)
    if not normalized:
        return False
    for pattern in _CANCELLATION_PATTERNS:
        if pattern.search(normalized):
            return True

    if re.search(r"\b(delete|cancel|remove)\b", normalized, re.I) and BOOKING_ID_PATTERN.search(message):
        return True

    return False

def _is_negated_cancellation_request(message: str) -> bool:
    if not message:
        return False
    normalized = _normalize(message)
    if not normalized:
        return False

    for pattern in _NEGATED_CANCELLATION_PATTERNS:
        if pattern.search(normalized):
            return True

    negated_request_patterns = (
        r"\b(?:do\s+not|don'?t|dont)\s+want\s+to\s+(?:cancel|delete|remove)\b",
        r"\b(?:do\s+not|don'?t|dont)\s+(?:cancel|delete|remove)\s+(?:this\s+|my\s+|the\s+)?booking\b",
        r"\bkeep\s+(?:this\s+|my\s+|the\s+)?booking\b",
    )
    return any(re.search(pattern, normalized, re.I) for pattern in negated_request_patterns)

def _clear_cancellation_soft_state(soft_state: Dict[str, Any]) -> None:
    for key in (
        "booking_cancellation_pending",
        "booking_cancellation_stage",
        "booking_cancellation_id",
        "booking_cancellation_receipt",
    ):
        soft_state.pop(key, None)

def _is_yes(message: str) -> bool:
    if not message:
        return False
    normalized = _normalize(message)
    tokens = set(normalized.split())
    if any(w in tokens for w in _YES_TOKENS):
        return True
    if re.search(r"\b(delete|cancel|remove)\b", normalized, re.I):
        return True
    return False

def _is_no(message: str) -> bool:
    if not message:
        return False
    normalized = _normalize(message)
    tokens = set(normalized.split())
    if any(w in tokens for w in _NO_TOKENS):
        return True
    for pattern in _NEGATED_CANCELLATION_PATTERNS:
        if pattern.search(normalized):
            return True
    if re.search(r"\b(don't|dont|keep)\b", normalized, re.I):
        return True
    return False

async def handle_booking_cancellation_turn(
    message: str,
    soft_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Deterministic booking cancellation/deletion handler.
    """
    is_cancel_intent = _detect_booking_cancellation_intent(message)
    stage = soft_state.get("booking_cancellation_stage")

    if not is_cancel_intent and not stage:
        return None

    from app.services.booking.persistence import get_booking_status, update_booking_status
    from app.observability.db_logging import get_successful_booking_status, update_successful_booking

    if stage == "awaiting_confirmation":
        if _is_no(message):
            soft_state.pop("booking_cancellation_pending", None)
            soft_state.pop("booking_cancellation_stage", None)
            soft_state.pop("booking_cancellation_id", None)
            soft_state.pop("booking_cancellation_receipt", None)
            return {
                "status": "cancellation_rejected",
                "deterministic_reply": "Okay, I have kept your booking unchanged."
            }
        elif _is_yes(message):
            booking_id = soft_state.get("booking_cancellation_id")
            if booking_id:
                sb_ok = await update_successful_booking(booking_id, {"status": "cancelled"})
                booking_result = await update_booking_status(booking_id, "", "cancelled")
                bookings_ok = isinstance(booking_result, dict) and bool(booking_result.get("ok"))

                if not sb_ok and not bookings_ok:
                    return {
                        "status": "error",
                        "deterministic_reply": (
                            "I couldn't cancel your booking right now. "
                            "Please try confirming the cancellation again."
                        ),
                    }

                if soft_state.get("booking_receipt", {}).get("booking_id") == booking_id:
                    soft_state["booking_receipt"]["status"] = "cancelled"
                if soft_state.get("booking_registration_id") == booking_id:
                    soft_state["booking_status"] = "cancelled"

                soft_state.pop("booking_cancellation_pending", None)
                soft_state.pop("booking_cancellation_stage", None)
                soft_state.pop("booking_cancellation_id", None)
                soft_state.pop("booking_cancellation_receipt", None)

                return {
                    "status": "cancelled",
                    "deterministic_reply": f"Your booking {booking_id} has been successfully cancelled.",
                }
            else:
                soft_state.pop("booking_cancellation_pending", None)
                soft_state.pop("booking_cancellation_stage", None)
                soft_state.pop("booking_cancellation_id", None)
                soft_state.pop("booking_cancellation_receipt", None)
                return {
                    "status": "error",
                    "deterministic_reply": "Something went wrong. Please try again."
                }
        else:
            receipt = soft_state.get("booking_cancellation_receipt")
            receipt_rendered = _receipt_reply(receipt) if receipt else ""
            return {
                "status": "awaiting_cancellation_confirmation",
                "deterministic_reply": "I didn't quite get that. Please confirm if you want to cancel the booking. Say 'yes' to cancel or 'no' to keep it."
            }

    booking_id = _extract_booking_id(message)
    if not booking_id:
        booking_id = soft_state.get("booking_cancellation_id") or soft_state.get("booking_registration_id") or soft_state.get("booking_receipt", {}).get("booking_id")

    if not booking_id:
        soft_state["booking_cancellation_pending"] = True
        soft_state["booking_cancellation_stage"] = "awaiting_id"
        return {
            "status": "gathering_cancellation_id",
            "deterministic_reply": "I'd be happy to help you cancel your booking. Could you please provide your booking registration ID? It looks like BK-YYYYMMDD-XXXXXXXX."
        }

    db_row = await get_successful_booking_status(booking_id)
    if db_row:
        receipt = successful_booking_row_to_receipt(db_row)
        receipt.setdefault("booking_id", booking_id)
    else:
        db_result = await get_booking_status(booking_id)
        if db_result.get("ok"):
            receipt = {
                "booking_id": booking_id,
                "status": db_result.get("status") or cfg.booking_confirmed_status,
                "check_in": db_result.get("check_in") or "",
                "check_out": db_result.get("check_out") or "",
            }
        else:
            receipt = None

    if not receipt:
        soft_state["booking_cancellation_pending"] = True
        soft_state["booking_cancellation_stage"] = "awaiting_id"
        return {
            "status": "booking_not_found",
            "deterministic_reply": f"It looks like booking {booking_id} wasn't found in our system. Please double-check your registration ID and provide it again."
        }

  
    receipt_rendered = _receipt_reply(receipt)
    soft_state["booking_cancellation_pending"] = True
    soft_state["booking_cancellation_stage"] = "awaiting_confirmation"
    soft_state["booking_cancellation_id"] = booking_id
    soft_state["booking_cancellation_receipt"] = receipt
    
    return {
        "status": "awaiting_cancellation_confirmation",
        "receipt": receipt,
        "deterministic_reply": f"Here are your booking details:\n{receipt_rendered}\n\nAre you sure you want to cancel this booking? Please say 'yes' to confirm or 'no' to keep it."
    }

def _is_cancellation_confirmation(message: str) -> bool:
    text = " ".join((message or "").strip().lower().split())
    if text in {"yes", "y", "confirm", "yes cancel", "yes delete"}:
        return True
    return text.startswith("yes") and any(word in text for word in ("cancel", "delete", "booking"))


def _is_cancellation_rejection(message: str) -> bool:
    text = " ".join((message or "").strip().lower().split())
    return text in {
        "no",
        "n",
        "nope",
        "do not cancel",
        "don't cancel",
        "dont cancel",
        "do not delete",
        "dont delete",
        "don't delete",
        "keep it",
        "nevermind",
        "never mind",
    }

_handle_booking_cancellation_turn_original_v1 = handle_booking_cancellation_turn

async def handle_booking_cancellation_turn(message: str, soft_state: dict):
    if not isinstance(soft_state, dict):
        return await _handle_booking_cancellation_turn_original_v1(message, soft_state)

    stage = str(soft_state.get("booking_cancellation_stage") or "").strip().lower()
    pending = bool(soft_state.get("booking_cancellation_pending")) or stage == "awaiting_confirmation"
    text = " ".join((message or "").strip().lower().split())

    if stage != "awaiting_confirmation" and _is_negated_cancellation_request(message):
        _clear_cancellation_soft_state(soft_state)
        return {
            "status": "cancellation_not_requested",
            "deterministic_reply": "Okay, I will not cancel your booking. Your booking remains unchanged.",
        }

    confirmations = {"yes", "y", "confirm", "yes cancel", "yes delete"}
    rejections = {
        "no",
        "n",
        "nope",
        "do not cancel",
        "don't cancel",
        "dont cancel",
        "do not delete",
        "don't delete",
        "dont delete",
        "keep it",
        "nevermind",
        "never mind",
    }

    is_confirm = text in confirmations or (text.startswith("yes") and any(w in text for w in ("cancel", "delete", "booking")))
    is_reject = text in rejections

    if pending and stage == "awaiting_confirmation":
        booking_id = soft_state.get("booking_cancellation_id")
        if not booking_id:
            receipt = soft_state.get("booking_cancellation_receipt")
            if isinstance(receipt, dict):
                booking_id = receipt.get("booking_id")

        if is_reject:
            for key in (
                "booking_cancellation_pending",
                "booking_cancellation_stage",
                "booking_cancellation_id",
                "booking_cancellation_receipt",
            ):
                soft_state.pop(key, None)

            return {
                "status": "cancelled_rejected",
                "deterministic_reply": "I've kept your booking unchanged.",
            }

        if is_confirm:
            from app.observability import db_logging
            import app.services.booking as booking_pkg

            sb_ok = await db_logging.update_successful_booking(str(booking_id), {"status": "cancelled"})
            booking_ok = await booking_pkg.update_booking_status(str(booking_id), "", "cancelled")

            booking_ok_bool = (
                bool(booking_ok.get("ok"))
                if isinstance(booking_ok, dict)
                else bool(booking_ok)
            )

            if sb_ok or booking_ok_bool:
                for key in (
                    "booking_cancellation_pending",
                    "booking_cancellation_stage",
                    "booking_cancellation_id",
                    "booking_cancellation_receipt",
                ):
                    soft_state.pop(key, None)

                return {
                    "status": "booking_cancelled",
                    "deterministic_reply": f"Your booking {booking_id} has been successfully cancelled.",
                }

            soft_state["booking_cancellation_pending"] = True
            soft_state["booking_cancellation_stage"] = "awaiting_confirmation"
            soft_state["booking_cancellation_id"] = booking_id

            return {
                "status": "error",
                "deterministic_reply": "I couldn't cancel the booking yet. Please try confirming the cancellation again.",
            }

    return await _handle_booking_cancellation_turn_original_v1(message, soft_state)
