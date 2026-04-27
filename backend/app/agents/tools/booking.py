"""
app/agents/tools/booking.py
----------------------------
Tools: request_booking_details, review_booking_details, process_v2_booking
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from ..status_codes import Source, Status
from .helpers import (
    _compute_nights_and_total,
    _diff_booking_summary,
    _finalize_payload,
    _get_soft_state,
    _missing_critical_data,
    _validate_booking_fields,
)
from ..state.booking_state import (
    clear_awaiting_field,
    clear_booking_state,
    compute_missing_booking_fields,
    extract_booking_updates,
    hydrate_booking_state_from_pending,
    set_awaiting_field,
    update_booking_state,
)
from app.config.agent_config_loader import cfg

logger = logging.getLogger(__name__)


def _resolve_property_id_from_state(
    property_id: Optional[str],
    soft_state: Optional[Dict[str, Any]],
) -> Optional[str]:
    if property_id:
        return property_id
    if not isinstance(soft_state, dict):
        return None
    pending = soft_state.get("pending_booking")
    if isinstance(pending, dict):
        pending_id = pending.get("property_id")
        if pending_id:
            return str(pending_id)
    booking_state = soft_state.get("booking_state")
    if isinstance(booking_state, dict):
        state_id = booking_state.get("property_id")
        if state_id:
            return str(state_id)
    last_selected = soft_state.get("last_selected_property_id")
    if last_selected:
        return str(last_selected)
    return None

async def request_booking_details(
    missing_info: Optional[str] = None,
    missing_fields: Optional[List[str]] = None,
    property_id: Optional[str] = None,
    property_title: Optional[str] = None,
    guest_name: Optional[str] = None,
    guest_email: Optional[str] = None,
    guest_phone: Optional[str] = None,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    guests: Optional[int] = None,
    price_per_night: Optional[float] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Use this tool when you need to gather missing booking information from the user.

    CRITICAL: Call this tool whenever the user wants to book a property but has NOT
    yet provided ALL of the following: full name, email, phone, check-in date,
    check-out date, and number of guests.

    Args:
        missing_info: A comma-separated list of what is still needed.
        missing_fields: Optional explicit list of missing fields.
        property_id: Optional property identifier to persist.
        property_title: Optional property title to persist.
        guest_name: Optional guest name to persist.
        guest_email: Optional guest email to persist.
        guest_phone: Optional guest phone to persist.
        check_in: Optional check-in date to persist.
        check_out: Optional check-out date to persist.
        guests: Optional guest count to persist.
        price_per_night: Optional nightly price to persist.
        action_intent: Optional context flag.
        context_flag: Optional secondary context flag.
        tool_context: ADK tool context for soft-state persistence.
    """
    resolved_fields = []
    if missing_fields:
        resolved_fields = [str(f).strip() for f in missing_fields if str(f).strip()]
    elif missing_info:
        resolved_fields = [f.strip() for f in missing_info.split(",") if f.strip()]
    soft_state = _get_soft_state(tool_context)
    property_id = _resolve_property_id_from_state(property_id, soft_state)
    booking_state = {}
    updates: Dict[str, Any] = {}
    if isinstance(soft_state, dict):
        updates = extract_booking_updates(
            property_id=property_id,
            property_title=property_title,
            guest_name=guest_name,
            guest_email=guest_email,
            guest_phone=guest_phone,
            check_in=check_in,
            check_out=check_out,
            guests=guests,
            price_per_night=price_per_night,
        )
        booking_state = update_booking_state(soft_state, updates)
        booking_state = hydrate_booking_state_from_pending(soft_state)

    computed_missing = compute_missing_booking_fields(booking_state) if booking_state else []
    has_updates = bool(updates)

    is_amendment = isinstance(soft_state, dict) and soft_state.get("pending_booking") is not None

    if booking_state:
        if has_updates:
            resolved_fields = computed_missing
        elif resolved_fields:
            allowed = set(cfg.booking_required_fields + cfg.booking_required_numeric_fields)
            resolved_fields = [f for f in resolved_fields if f in allowed]
            if not resolved_fields:
                resolved_fields = computed_missing
        else:
            resolved_fields = computed_missing
    if is_amendment and has_updates:
        previous_summary = soft_state.get("pending_booking", {})
        update_context = _diff_booking_summary(previous_summary, booking_state)
        for key, value in updates.items():
            if value is not None:
                soft_state["pending_booking"][key] = value
        return _finalize_payload(
            {
                "status": Status.AMENDMENT_ACKNOWLEDGED,
                "update_context": update_context,
                "updated_fields": list(updates.keys()),
                "current_state": soft_state.get("pending_booking", {}),
                "remaining_missing": resolved_fields if resolved_fields else [],
            },
            action_intent, context_flag,
        )
    if booking_state and has_updates and not resolved_fields:
        return await review_booking_details(
            property_id=booking_state.get("property_id"),
            property_title=booking_state.get("property_title"),
            guest_name=booking_state.get("guest_name"),
            guest_email=booking_state.get("guest_email"),
            guest_phone=booking_state.get("guest_phone"),
            check_in=booking_state.get("check_in"),
            check_out=booking_state.get("check_out"),
            guests=booking_state.get("guests"),
            price_per_night=booking_state.get("price_per_night"),
            action_intent=action_intent,
            context_flag=context_flag,
            tool_context=tool_context,
        )

    if not resolved_fields:
        return _missing_critical_data(
            ["missing_info"],
            "Booking details are needed but no missing-field list was provided.",
            action_intent, context_flag,
        )

    if isinstance(soft_state, dict):
        set_awaiting_field(soft_state, resolved_fields)

    return _finalize_payload(
        {"status": Status.GATHERING_INFO, "missing_fields": resolved_fields},
        action_intent, context_flag,
    )

async def review_booking_details(
    property_id: Optional[str] = None,
    property_title: Optional[str] = None,
    guest_name: Optional[str] = None,
    guest_email: Optional[str] = None,
    guest_phone: Optional[str] = None,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    guests: Optional[int] = None,
    price_per_night: Optional[float] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Present a full booking summary for the user to review BEFORE final confirmation.

    Call this tool ONCE when ALL booking details have been collected but the user
    has NOT yet explicitly authorized final booking.
    This gives the user a chance to review and correct any mistakes.
    If the user wants to change a detail, silently update your context and call
    this tool again with the corrected values — do NOT call process_v2_booking yet.

    Args:
        property_id: The unique ID of the property.
        property_title: The display title of the property.
        guest_name: The guest's full name.
        guest_email: The guest's email address.
        guest_phone: The guest's phone number.
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.
        guests: Number of guests.
        price_per_night: The nightly price of the property.
    """
    soft_state = _get_soft_state(tool_context)
    property_id = _resolve_property_id_from_state(property_id, soft_state)
    missing, guests_value, price_value = _validate_booking_fields(
        property_id, property_title, guest_name, guest_email,
        guest_phone, check_in, check_out, guests, price_per_night,
    )
    if missing:
        return _missing_critical_data(
            missing, "Booking review needs a complete set of details.", action_intent, context_flag,
        )

    nights, total_price = _compute_nights_and_total(check_in, check_out, price_value)

    summary = {
        "property": property_title,
        "property_id": property_id,
        "guest_name": guest_name,
        "guest_email": guest_email,
        "guest_phone": guest_phone,
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "guests": guests_value,
        "price_per_night": price_value,
        "total": total_price,
    }

    update_context = None
    if isinstance(soft_state, dict):
        previous_summary = soft_state.get("pending_booking")
        update_context = _diff_booking_summary(previous_summary, summary)
        soft_state["pending_booking"] = summary
        soft_state["pending_booking_updated_at"] = time.time()
        update_booking_state(
            soft_state,
            extract_booking_updates(
                property_id=property_id,
                property_title=property_title,
                guest_name=guest_name,
                guest_email=guest_email,
                guest_phone=guest_phone,
                check_in=check_in,
                check_out=check_out,
                guests=guests_value,
                price_per_night=price_value,
            ),
        )
        clear_awaiting_field(soft_state)

    payload: Dict[str, Any] = {"status": Status.REVIEW_PENDING, "summary": summary}
    if update_context and update_context.get("was_update"):
        payload["update_context"] = update_context
    return _finalize_payload(payload, action_intent, context_flag)

async def process_v2_booking(
    property_id: Optional[str] = None,
    property_title: Optional[str] = None,
    guest_name: Optional[str] = None,
    guest_email: Optional[str] = None,
    guest_phone: Optional[str] = None,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    guests: Optional[int] = None,
    price_per_night: Optional[float] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Finalise and commit the booking ONLY after the user has explicitly confirmed.

    CRITICAL SEQUENCE:
    1. Use `request_booking_details` if any detail is missing.
    2. Use `review_booking_details` once all details are collected — let the user confirm.
    3. Call THIS tool ONLY after the user explicitly authorizes final booking.
    Never call this tool if the user has not seen and approved the review summary.
    All dates must be in YYYY-MM-DD format.

    Args:
        property_id: The unique ID of the property to book.
        property_title: The display title of the property.
        guest_name: The guest's full name.
        guest_email: The guest's email address.
        guest_phone: The guest's phone number.
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.
        guests: Number of guests.
        price_per_night: The nightly price of the property.
    """
    soft_state = _get_soft_state(tool_context)
    property_id = _resolve_property_id_from_state(property_id, soft_state)
    missing, guests_value, price_value = _validate_booking_fields(
        property_id, property_title, guest_name, guest_email,
        guest_phone, check_in, check_out, guests, price_per_night,
    )
    if missing:
        return _missing_critical_data(
            missing, "Booking confirmation needs a complete set of details.",
            action_intent, context_flag,
        )

    nights, total_price = _compute_nights_and_total(check_in, check_out, price_value)
    booking_id = str(uuid.uuid4())

    try:
        from ...observability.db_logging import insert_successful_booking
        await insert_successful_booking({
            "booking_id": booking_id,
            "user_name": guest_name,
            "user_email": guest_email,
            "user_phone": guest_phone,
            "property_title": property_title,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests_value,
            "nights": nights,
            "total_amount": total_price,
            "status": cfg.booking_confirmed_status,
            "source": cfg.booking_source_tag,
        })
    except Exception as e:
        logger.warning("[V2 Booking] Could not persist booking to DB: %s", e)

    payload: Dict[str, Any] = {
        "status": Status.BOOKING_CONFIRMED,
        "receipt": {
            "booking_id": booking_id,
            "property_title": property_title,
            "guest_name": guest_name,
            "guest_email": guest_email,
            "guest_phone": guest_phone,
            "check_in": check_in,
            "check_out": check_out,
            "nights": nights,
            "guests": guests_value,
            "price_per_night": price_value,
            "total_amount": total_price,
        },
    }

    if isinstance(soft_state, dict):
        soft_state.pop("pending_booking", None)
        soft_state.pop("pending_booking_updated_at", None)
        clear_booking_state(soft_state)

    return _finalize_payload(payload, action_intent, context_flag)
