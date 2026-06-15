from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.agents.state.booking_state import clear_awaiting_field, set_awaiting_field
from app.agents.status_codes import Status
from app.agents.tools.search import return_to_previous_results
from app.config.agent_config_loader import cfg
from app.config.booking_schema_loader import (
    get_amendable_fields,
    get_amendment_context_markers,
    get_amendment_display_name,
    get_amendment_field_groups,
    get_amendment_intent_verbs,
    get_field_display_name,
    validate_field,
)
from app.agents.tools.helpers import _coerce_float, _coerce_int
from app.services.booking.constants import BOOKING_ID_PATTERN
from app.services.booking.extraction import (
    _extract_amendment_updates,
    _extract_booking_id,
    _extract_email,
    _extract_phone,
    _extract_updates_from_message,
    _is_valid_candidate_for_field,
    _sanitize_message_for_amendment_extraction,
)
from app.services.booking.formatting import (
    _amendment_choice_reply,
    _amendment_values_reply,
    _friendly_prompt_for_field,
    _receipt_reply,
    _receipt_review_from_candidate,
)
from app.services.booking.receipt import (
    _latest_session_receipt,
    _lookup_booking_in_session,
    _store_current_receipt,
    receipt_updates_to_successful_booking_columns,
    successful_booking_row_to_receipt,
)
from app.services.booking.state import (
    _field_alias_map,
    _normalize,
    _selected_property,
    has_post_confirmation_amendment_context,
)
from app.services.booking.validation import (
    _missing_amendment_fields,
    _validate_amendment_candidate,
)
import re

def _has_booking_amendment_context(message: str, soft_state: Dict[str, Any]) -> bool:
    if _extract_booking_id(message):
        return True
    if not isinstance(soft_state, dict):
        return False
    if isinstance(soft_state.get("booking_receipt"), dict):
        return True
    if soft_state.get("booking_registration_id"):
        return True
    return bool(soft_state.get("pending_booking") or soft_state.get("booking_review"))

def _has_booking_reference_marker(message: str) -> bool:
    if _extract_booking_id(message):
        return True
    normalized = _normalize(message)
    markers = get_amendment_context_markers()
    if not markers:
        markers = ["this booking", "my booking", "reservation"]
    return any(re.search(rf"\b{re.escape(_normalize(marker))}\b", normalized) for marker in markers)

def detect_booking_amendment_intent(message: str, soft_state: Dict[str, Any]) -> bool:
    if not _has_booking_amendment_context(message, soft_state):
        return False
    normalized = _normalize(message)
    if not normalized:
        return False
    fields = _amendment_fields_from_message(message)
    if fields and _has_booking_reference_marker(message):
        return True
    verbs = get_amendment_intent_verbs()
    if not verbs:
        verbs = ["change", "update", "edit", "modify", "fix", "correct"]
    return any(re.search(rf"\b{re.escape(verb)}\b", normalized) for verb in verbs)

def _amendable_field_alias_map() -> Dict[str, List[str]]:
    base_aliases = _field_alias_map()
    allowed = set(get_amendable_fields())
    return {
        field: aliases
        for field, aliases in base_aliases.items()
        if field in allowed
    }

def _amendment_fields_from_message(message: str) -> List[str]:
    normalized = _normalize(message)
    if not normalized:
        return []
    fields: List[str] = []
    allowed = set(get_amendable_fields())
    for group in get_amendment_field_groups().values():
        aliases = group.get("aliases", [])
        if any(re.search(rf"\b{re.escape(_normalize(alias))}\b", normalized) for alias in aliases):
            for field in group.get("fields", []):
                if field in allowed:
                    fields.append(field)
    for field, aliases in _amendable_field_alias_map().items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(_normalize(alias))}\b", normalized):
                fields.append(field)
                break
    ordered_unique_fields = list(dict.fromkeys(fields))
    field_order = {field: index for index, field in enumerate(get_amendable_fields())}
    return sorted(
        ordered_unique_fields,
        key=lambda field: field_order.get(field, len(field_order)),
    )

async def _confirm_amendment(soft_state: Dict[str, Any]) -> Dict[str, Any]:
    pending = soft_state.get("pending_booking_amendment")
    if not isinstance(pending, dict):
        return {
            "status": Status.GATHERING_INFO,
            "deterministic_reply": _amendment_choice_reply(),
        }
    receipt = dict(pending.get("candidate_receipt") or {})
    booking_id = str(receipt.get("booking_id") or "").strip()
    if not booking_id:
        return {
            "status": Status.GATHERING_INFO,
            "deterministic_reply": "Please share the booking ID for this amendment.",
        }

    changed_fields = list(pending.get("fields") or [])
    changed_updates = {field: receipt.get(field) for field in changed_fields if field in receipt}
    if any(field in changed_fields for field in ("check_in", "check_out")):
        changed_updates["nights"] = receipt.get("nights")
        changed_updates["total_amount"] = receipt.get("total_amount")
    if any(field in changed_fields for field in ("guests", "property_id")):
        changed_updates["total_amount"] = receipt.get("total_amount")
    if "property_id" in changed_fields:
        changed_updates["property_title"] = receipt.get("property_title")
        changed_updates["price_per_night"] = receipt.get("price_per_night")

    persisted = False
    try:
        from app.observability.db_logging import update_successful_booking

        persisted = await update_successful_booking(
            booking_id,
            receipt_updates_to_successful_booking_columns(changed_updates),
        )
    except Exception as exc:
        logger.warning("[booking_flow] Could not persist booking amendment: %s", exc)

    if not persisted:
        soft_state["booking_stage"] = "awaiting_amendment_confirmation"
        soft_state["active_flow"] = "booking"
        return {
            "status": Status.ERROR,
            "receipt": receipt,
            "deterministic_reply": (
                "I couldn't save the updated booking details right now. "
                "Your amendment is still pending; please try confirming again."
            ),
        }

    _store_current_receipt(soft_state, receipt)
    soft_state.pop("pending_booking_amendment", None)
    clear_awaiting_field(soft_state)
    return {
        "status": Status.BOOKING_CONFIRMED,
        "receipt": receipt,
        "deterministic_reply": _receipt_reply(receipt),
    }

async def handle_booking_amendment_turn(
    message: str,
    soft_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    stage = str(soft_state.get("booking_stage") or "").strip()

    if not has_post_confirmation_amendment_context(message, soft_state):
        return None

    normalized = _normalize(message)

    if stage == "awaiting_amendment_confirmation":
        if re.search(r"\b(yes|confirm|correct|looks good|proceed|ok|okay)\b", normalized):
            return await _confirm_amendment(soft_state)
        if re.search(r"\b(no|change|update|edit|modify|fix|correct)\b", normalized):
            pending = soft_state.get("pending_booking_amendment")
            if isinstance(pending, dict):
                fields = _amendment_fields_from_message(message) or list(pending.get("fields") or [])
                soft_state["pending_booking_amendment"] = {
                    "base_receipt": dict(pending.get("candidate_receipt") or pending.get("base_receipt") or {}),
                    "fields": fields,
                }
                soft_state["booking_stage"] = "awaiting_amendment_values"
                set_awaiting_field(soft_state, fields)
                return {
                    "status": Status.GATHERING_INFO,
                    "missing_fields": fields,
                    "deterministic_reply": _amendment_values_reply(fields),
                }
        return {
            "status": Status.REVIEW_PENDING,
            "deterministic_reply": "Please confirm if these updated booking details are correct, or tell me what to change.",
        }

    if stage == "awaiting_amendment_values":
        pending = soft_state.get("pending_booking_amendment")
        if not isinstance(pending, dict):
            soft_state["booking_stage"] = "awaiting_amendment_choice"
            return {
                "status": Status.GATHERING_INFO,
                "deterministic_reply": _amendment_choice_reply(),
            }
        base_receipt = pending.get("base_receipt")
        if not isinstance(base_receipt, dict):
            base_receipt = await _current_amendment_receipt(message, soft_state) or {}
        receipt = dict(base_receipt)
        fields = list(pending.get("fields") or [])
    else:
        if stage != "awaiting_amendment_choice" and not detect_booking_amendment_intent(message, soft_state):
            return None
        receipt = await _current_amendment_receipt(message, soft_state)
        if not receipt:
            return None
        fields = _amendment_fields_from_message(message)
        if not fields:
            soft_state["booking_stage"] = "awaiting_amendment_choice"
            soft_state["active_flow"] = "booking"
            clear_awaiting_field(soft_state)
            soft_state["pending_booking_amendment"] = {"base_receipt": dict(receipt), "fields": []}
            return {
                "status": Status.GATHERING_INFO,
                "deterministic_reply": _amendment_choice_reply(),
            }
        if "property_id" in fields:
            soft_state["booking_stage"] = "awaiting_property_reselection"
            soft_state["pending_booking_amendment"] = {"base_receipt": dict(receipt), "fields": fields}
            return return_to_previous_results(soft_state)

    updates, extraction_errors = _extract_amendment_updates(message, fields)
    missing_fields = _missing_amendment_fields(fields, updates, extraction_errors)
    if missing_fields or extraction_errors:
        ask_fields = list(dict.fromkeys(list(extraction_errors.keys()) + missing_fields))
        soft_state["active_flow"] = "booking"
        soft_state["booking_stage"] = "awaiting_amendment_values"
        soft_state["pending_booking_amendment"] = {
            "base_receipt": dict(receipt),
            "fields": fields,
        }
        set_awaiting_field(soft_state, ask_fields)
        reply = next(iter(extraction_errors.values()), None) or _amendment_values_reply(ask_fields)
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": ask_fields,
            "deterministic_reply": reply,
        }

    candidate, validation_errors = _validate_amendment_candidate(
        receipt,
        updates,
        extraction_errors,
        soft_state,
    )
    if validation_errors:
        ask_fields = list(validation_errors.keys())
        soft_state["active_flow"] = "booking"
        soft_state["booking_stage"] = "awaiting_amendment_values"
        soft_state["pending_booking_amendment"] = {
            "base_receipt": dict(receipt),
            "fields": fields,
        }
        set_awaiting_field(soft_state, ask_fields)
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": ask_fields,
            "deterministic_reply": next(iter(validation_errors.values())),
        }

    soft_state["active_flow"] = "booking"
    soft_state["booking_stage"] = "awaiting_amendment_confirmation"
    soft_state["pending_booking_amendment"] = {
        "base_receipt": dict(receipt),
        "candidate_receipt": dict(candidate),
        "fields": fields,
    }
    soft_state["last_presented_view"] = "booking_receipt"
    clear_awaiting_field(soft_state)
    return {
        "status": Status.REVIEW_PENDING,
        "receipt": candidate,
        "deterministic_reply": _receipt_review_from_candidate(candidate),
    }

async def _current_amendment_receipt(message: str, soft_state: dict) -> dict | None:
    """Resolve the current receipt for an amendment flow.

    Priority:
    1. existing soft_state["booking_receipt"]
    2. pending amendment candidate receipt
    3. booking id in message/state via successful_bookings lookup
    """
    import re

    if not isinstance(soft_state, dict):
        return None

    pending = soft_state.get("pending_booking_amendment")
    if isinstance(pending, dict):
        candidate = pending.get("candidate_receipt")
        if isinstance(candidate, dict) and candidate:
            return dict(candidate)

    receipt = soft_state.get("booking_receipt")
    if isinstance(receipt, dict) and receipt:
        return dict(receipt)

    booking_id = (
        soft_state.get("booking_registration_id")
        or soft_state.get("booking_id")
    )

    if not booking_id:
        match = re.search(r"\bBK-[A-Z0-9-]+\b", message or "", re.I)
        if match:
            booking_id = match.group(0).upper()

    if not booking_id:
        return None

    from app.observability import db_logging

    row = await db_logging.get_successful_booking_status(str(booking_id))
    if not isinstance(row, dict) or not row:
        return None

    resolved = {
        "booking_id": row.get("booking_id") or str(booking_id),
        "property_title": row.get("property_title") or row.get("property_name") or "",
        "guest_name": row.get("guest_name") or row.get("user_name") or row.get("full_name") or "",
        "guest_email": row.get("guest_email") or row.get("user_email") or row.get("email") or "",
        "guest_phone": row.get("guest_phone") or row.get("user_phone") or row.get("phone") or "",
        "check_in": row.get("check_in") or row.get("checkin") or "",
        "check_out": row.get("check_out") or row.get("checkout") or "",
        "guests": row.get("guests") or row.get("guest_count") or "",
        "nights": row.get("nights") or "",
        "price_per_night": row.get("price_per_night") or 0.0,
        "total_amount": row.get("total_amount") or row.get("total") or 0.0,
        "status": row.get("status") or "confirmed",
    }

    soft_state["booking_receipt"] = dict(resolved)
    soft_state["booking_registration_id"] = resolved["booking_id"]
    soft_state["booking_status"] = resolved["status"]
    soft_state["booking_stage"] = soft_state.get("booking_stage") or "confirmed"
    soft_state["active_flow"] = "booking"

    return resolved

# Compatibility wrapper: ensure amended check-in/check-out recompute nights and total.
from datetime import datetime as _amendment_datetime

_validate_amendment_candidate_original_recompute_v1 = _validate_amendment_candidate

def _validate_amendment_candidate(receipt, updates, extraction_errors, soft_state):
    candidate, errors = _validate_amendment_candidate_original_recompute_v1(
        receipt,
        updates,
        extraction_errors,
        soft_state,
    )

    if not any(field in errors for field in ("check_in", "check_out")):
        check_in = candidate.get("check_in")
        check_out = candidate.get("check_out")
        if check_in and check_out:
            try:
                check_in_dt = _amendment_datetime.strptime(str(check_in), cfg.date_format)
                check_out_dt = _amendment_datetime.strptime(str(check_out), cfg.date_format)
                if check_out_dt > check_in_dt:
                    nights = max((check_out_dt - check_in_dt).days, 1)
                    candidate["nights"] = nights
                    price = _coerce_float(candidate.get("price_per_night")) or 0.0
                    candidate["total_amount"] = round(nights * price, 2)
            except Exception:
                pass

    return candidate, errors
