from __future__ import annotations
from app.agents.status_codes import Status
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.config.agent_config_loader import cfg
from app.agents.state.booking_state import (
    clear_awaiting_field,
    compute_missing_booking_fields,
    get_booking_state,
    set_awaiting_field,
)
from app.config.booking_schema_loader import (
    get_amendable_fields,
    get_ask_order,
    next_field_to_ask,
    validate_field,
)
from app.agents.tools.helpers import _coerce_float, _coerce_int
from app.services.booking.formatting import (
    _checkout_validation_message,
    _friendly_prompt_for_field,
    _occupancy_validation_message,
    _review_reply,
    _review_summary_from_state,
)
from app.services.booking.state import (
    _replace_booking_state,
    _seed_property_fields,
    _selected_property,
    _sync_review_state,
)

def _next_missing_field(state: Dict[str, Any]) -> Optional[str]:
    missing = compute_missing_booking_fields(state)
    filtered = [field for field in missing if field in set(cfg.booking_details_request_fields)]
    if filtered:
        return next_field_to_ask(filtered)
    return next_field_to_ask(missing)

def _validate_and_commit_state(
    soft_state: Dict[str, Any],
    updates: Dict[str, Any],
    mentioned_fields: List[str],
    extraction_errors: Dict[str, str],
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    current_state = dict(get_booking_state(soft_state))
    _seed_property_fields(soft_state, current_state)

    candidate_state = dict(current_state)
    errors: Dict[str, str] = dict(extraction_errors)
    for field, value in updates.items():
        temp_state = dict(candidate_state)
        temp_state[field] = value
        ok, err = validate_field(field, value, current_state=temp_state)
        if ok:
            candidate_state[field] = value
        else:
            errors[field] = err or _friendly_prompt_for_field(field)

    guests_value = _coerce_int(candidate_state.get("guests"))
    selected_property = _selected_property(soft_state)
    occupancy_max = _coerce_int(selected_property.get("occupancy_max")) if isinstance(selected_property, dict) else None
    if occupancy_max is not None and guests_value is not None and guests_value > occupancy_max:
        candidate_state.pop("guests", None)
        errors["guests"] = _occupancy_validation_message(occupancy_max)

    if "check_in" in errors:
        candidate_state.pop("check_in", None)
        candidate_state.pop("check_out", None)
        errors.pop("check_out", None)
    else:
        check_in = candidate_state.get("check_in") or updates.get("check_in")
        check_out = candidate_state.get("check_out") or updates.get("check_out")
        if check_in and check_out:
            try:
                check_in_dt = datetime.strptime(str(check_in), cfg.date_format)
                check_out_dt = datetime.strptime(str(check_out), cfg.date_format)
                if check_out_dt <= check_in_dt:
                    candidate_state.pop("check_out", None)
                    errors["check_out"] = (
                        _checkout_validation_message(str(check_in))
                        or "Check-out cannot be earlier than your check-in date."
                    )
            except Exception:
                pass

    committed = _replace_booking_state(soft_state, candidate_state)
    missing_field = _next_missing_field(committed)
    error_field = None
    if errors:
        if "check_in" in errors:
            error_field = "check_in"
        elif "check_out" in errors:
            error_field = "check_out"
        else:
            for field in get_ask_order():
                if field in errors:
                    error_field = field
                    break
            if error_field is None:
                error_field = next(iter(errors))

    if error_field:
        set_awaiting_field(soft_state, [error_field])
        return committed, error_field, errors[error_field]

    if missing_field:
        set_awaiting_field(soft_state, [missing_field])
        return committed, missing_field, _friendly_prompt_for_field(missing_field)

    clear_awaiting_field(soft_state)
    return committed, None, None

def _review_payload_from_state(soft_state: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    summary = _review_summary_from_state(soft_state, state)
    soft_state["active_flow"] = "booking"
    soft_state["booking_stage"] = "awaiting_confirmation"
    soft_state["last_presented_view"] = "booking_review"
    _sync_review_state(soft_state, summary)
    return {
        "status": Status.REVIEW_PENDING,
        "summary": summary,
        "deterministic_reply": _review_reply(summary),
    }

def _missing_amendment_fields(fields: List[str], updates: Dict[str, Any], errors: Dict[str, str]) -> List[str]:
    missing = []
    for field in fields:
        if field == "property_id":
            continue
        if field not in updates and field not in errors:
            missing.append(field)
    return missing

def _validate_amendment_candidate(
    receipt: Dict[str, Any],
    updates: Dict[str, Any],
    extraction_errors: Dict[str, str],
    soft_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    candidate = dict(receipt)
    errors: Dict[str, str] = dict(extraction_errors)
    allowed = set(get_amendable_fields())

    for field, value in updates.items():
        if field not in allowed or field == "property_id":
            continue
        temp = dict(candidate)
        temp[field] = value
        ok, err = validate_field(field, value, current_state=temp)
        if ok:
            candidate[field] = value
        else:
            errors[field] = err or _friendly_prompt_for_field(field)

    guests_value = _coerce_int(candidate.get("guests"))
    selected_property = _selected_property(soft_state)
    occupancy_max = _coerce_int(selected_property.get("occupancy_max")) if isinstance(selected_property, dict) else None
    if occupancy_max is not None and guests_value is not None and guests_value > occupancy_max:
        candidate["guests"] = receipt.get("guests")
        errors["guests"] = _occupancy_validation_message(occupancy_max)

    check_in = candidate.get("check_in")
    check_out = candidate.get("check_out")
    if check_in and check_out:
        try:
            check_in_dt = datetime.strptime(str(check_in), cfg.date_format)
            check_out_dt = datetime.strptime(str(check_out), cfg.date_format)
            if check_out_dt <= check_in_dt:
                invalid_field = "check_out" if "check_out" in updates else "check_in"
                candidate[invalid_field] = receipt.get(invalid_field)
                errors[invalid_field] = (
                    _checkout_validation_message(str(candidate.get("check_in") or check_in))
                    or "Check-out cannot be earlier than your check-in date."
                )
        except Exception:
            pass

    if "check_in" in candidate and "check_out" in candidate and not any(
        field in errors for field in ("check_in", "check_out")
    ):
        try:
            check_in_dt = datetime.strptime(str(candidate["check_in"]), cfg.date_format)
            check_out_dt = datetime.strptime(str(candidate["check_out"]), cfg.date_format)
            nights = max((check_out_dt - check_in_dt).days, 1)
            candidate["nights"] = nights
            price = _coerce_float(candidate.get("price_per_night")) or 0.0
            candidate["total_amount"] = round(nights * price, 2)
        except Exception:
            pass

    candidate.setdefault("booking_id", receipt.get("booking_id"))
    candidate.setdefault("status", receipt.get("status") or cfg.booking_confirmed_status)
    return candidate, errors

# Compatibility override: tests and UX expect this exact checkout-order wording.
def _checkout_validation_message(check_in=None) -> str:
    return "Check-out cannot be earlier than your check-in date."
