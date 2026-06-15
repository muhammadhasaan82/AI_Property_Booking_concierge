from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4
import re

from app.agents.state.booking_state import clear_awaiting_field, set_awaiting_field
from app.agents.status_codes import Status
from app.agents.tools.search import get_all_available_cities, return_to_previous_results
from app.agents.tools.support import check_faq
from app.config.agent_config_loader import cfg
from app.services.observability.langfuse_observer import get_observer, summarize_booking_state
from app.services.faq_interruption import clear_faq_interruption, sync_alias_keys
from app.services.booking.amendment import (
    handle_booking_amendment_turn,
    has_post_confirmation_amendment_context,
)
from app.services.booking.constants import _ACTIVE_BOOKING_STAGES, _POST_CONFIRMATION_AMENDMENT_STAGES
from app.services.booking.extraction import _extract_updates_from_message
from app.services.booking.faq import _enrich_booking_faq_payload, _is_booking_faq
from app.services.booking.formatting import (
    _details_request_reply,
    _friendly_prompt_for_field,
    _receipt_reply,
    _review_reply,
    _review_summary_from_state,
)
from app.services.booking.state import (
    _ensure_property_seeded,
    _modification_field_from_message,
    _normalize,
    _property_title_from_soft_state,
    _resolve_selected_property_from_soft_state,
    _sync_review_state,
)
from app.services.booking.validation import (
    _review_payload_from_state,
    _validate_and_commit_state,
)

def start_booking_for_selected_property(soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selected_id_raw = soft_state.get("last_selected_property_id")
    if selected_id_raw is None or not str(selected_id_raw).strip():
        return None

    selected_id = str(selected_id_raw).strip()
    selected_property = _resolve_selected_property_from_soft_state(soft_state, selected_id)
    soft_state["booking_property_id"] = selected_id
    if selected_property:
        soft_state["booking_selected_property"] = dict(selected_property)
        soft_state["selected_property"] = dict(selected_property)
    soft_state["booking_required_fields"] = list(cfg.booking_details_request_fields)
    clear_faq_interruption(soft_state)
    sync_alias_keys(soft_state)

    state = _ensure_property_seeded(soft_state)
    property_title = str(state.get("property_title") or _property_title_from_soft_state(soft_state))
    validated_state, next_field, reply = _validate_and_commit_state(
        soft_state,
        updates={},
        mentioned_fields=[],
        extraction_errors={},
    )
    if next_field is None:
        clear_awaiting_field(soft_state)
        return _review_payload_from_state(soft_state, validated_state)

    observer = get_observer()
    trace = observer.trace(name="booking_flow")
    trace.update(metadata={
        "previous_booking_stage": soft_state.get("booking_stage"),
        "next_booking_stage": "collecting_details",
        "missing_fields": list(cfg.booking_details_request_fields),
    })
    trace.end()
    soft_state["active_flow"] = "booking"
    soft_state["booking_stage"] = "collecting_details"
    soft_state["last_presented_view"] = "booking_details_request"
    set_awaiting_field(soft_state, [next_field])
    collected_detail_fields = [
        field for field in cfg.booking_details_request_fields
        if validated_state.get(field) not in (None, "")
    ]
    return {
        "status": "booking_details_required",
        "property_id": selected_id,
        "property": soft_state.get("booking_selected_property"),
        "required_fields": list(cfg.booking_details_request_fields),
        "deterministic_reply": (
            reply
            if collected_detail_fields
            else _details_request_reply(property_title, list(cfg.booking_details_request_fields))
        ),
    }

def resume_booking_flow(soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    observer = get_observer()
    previous_stage = str(soft_state.get("booking_stage") or "").strip()
    
    with observer.trace(name="booking_flow", metadata={
        "previous_booking_stage": previous_stage,
        "soft_state_summary": summarize_booking_state(soft_state)
    }) as trace:
        stage = str(soft_state.get("booking_stage") or "").strip()
    if stage not in _ACTIVE_BOOKING_STAGES:
        return None
    soft_state["active_flow"] = "booking"
    clear_faq_interruption(soft_state)
    sync_alias_keys(soft_state)
    state = _ensure_property_seeded(soft_state)
    if stage == "awaiting_confirmation":
        summary = soft_state.get("booking_review") or soft_state.get("pending_booking")
        if isinstance(summary, dict):
            return {
                "status": Status.REVIEW_PENDING,
                "summary": summary,
                "deterministic_reply": _review_reply(summary),
            }
    property_title = str(state.get("property_title") or _property_title_from_soft_state(soft_state))
    next_field = soft_state.get("awaiting_field")
    if isinstance(next_field, str) and next_field.strip():
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": [next_field],
            "deterministic_reply": _friendly_prompt_for_field(next_field),
        }
    missing_fields = [
        field
        for field in cfg.booking_details_request_fields
        if state.get(field) in (None, "")
    ]
    if missing_fields:
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": missing_fields,
            "deterministic_reply": _details_request_reply(property_title, missing_fields),
        }
    summary = _review_summary_from_state(soft_state, state)
    soft_state["booking_stage"] = "awaiting_confirmation"
    _sync_review_state(soft_state, summary)
    return {
        "status": Status.REVIEW_PENDING,
        "summary": summary,
        "deterministic_reply": _review_reply(summary),
    }

def handle_review_modification_request(message: str, soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    field = _modification_field_from_message(message)
    if field == "property_id":
        soft_state["booking_stage"] = "awaiting_property_reselection"
        soft_state["last_presented_view"] = "property_list"
        return return_to_previous_results(soft_state)

    if field:
        soft_state["booking_stage"] = "modifying_details"
        set_awaiting_field(soft_state, [field])
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": [field],
            "deterministic_reply": _friendly_prompt_for_field(field),
        }

    soft_state["booking_stage"] = "awaiting_modification_choice"
    clear_awaiting_field(soft_state)
    return {
        "status": Status.GATHERING_INFO,
        "deterministic_reply": str(getattr(cfg, "booking_modification_prompt_template", "") or "").strip(),
    }

async def confirm_booking_review(soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary = soft_state.get("booking_review") or soft_state.get("pending_booking")
    if not isinstance(summary, dict):
        state = _ensure_property_seeded(soft_state)
        summary = _review_summary_from_state(soft_state, state)

    registration_id = (
        f"{cfg.booking_registration_id_prefix}-{datetime.utcnow():%Y%m%d}-{uuid4().hex[:8].upper()}"
    )
    receipt = {
        "booking_id": registration_id,
        "property_title": summary.get("property_title") or summary.get("property"),
        "guest_name": summary.get("guest_name"),
        "guest_email": summary.get("guest_email"),
        "guest_phone": summary.get("guest_phone"),
        "check_in": summary.get("check_in"),
        "check_out": summary.get("check_out"),
        "nights": summary.get("nights"),
        "guests": summary.get("guests"),
        "price_per_night": summary.get("price_per_night"),
        "total_amount": summary.get("total"),
        "status": cfg.booking_confirmed_status,
    }

    try:
        from app.observability.db_logging import insert_successful_booking

        await insert_successful_booking(
            {
                "booking_id": registration_id,
                "user_name": summary.get("guest_name"),
                "user_email": summary.get("guest_email"),
                "user_phone": summary.get("guest_phone"),
                "property_title": summary.get("property_title") or summary.get("property"),
                "check_in": summary.get("check_in"),
                "check_out": summary.get("check_out"),
                "guests": summary.get("guests"),
                "nights": summary.get("nights"),
                "price_per_night": summary.get("price_per_night"),
                "total_amount": summary.get("total"),
                "status": cfg.booking_confirmed_status,
                "source": cfg.booking_source_tag,
            }
        )
    except Exception as exc:
        logger.warning("[booking_flow] Could not persist confirmed booking: %s", exc)

    observer = get_observer()
    trace = observer.trace(name="booking_flow")
    trace.update(metadata={
        "previous_booking_stage": soft_state.get("booking_stage"),
        "next_booking_stage": "confirmed",
        "review_generated": True,
        "receipt_generated": True,
        "registration_id_present": bool(registration_id),
    })
    trace.end()
    soft_state["booking_stage"] = "confirmed"
    soft_state["active_flow"] = "booking"
    soft_state["booking_status"] = cfg.booking_confirmed_status
    soft_state["booking_registration_id"] = registration_id
    soft_state["booking_receipt"] = dict(receipt)
    soft_state["last_presented_view"] = "booking_receipt"
    clear_awaiting_field(soft_state)

    return {
        "status": Status.BOOKING_CONFIRMED,
        "receipt": receipt,
        "deterministic_reply": _receipt_reply(receipt),
    }

async def handle_active_booking_turn(
    message: str,
    soft_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    observer = get_observer()
    previous_stage = str(soft_state.get("booking_stage") or "").strip()
    
    with observer.trace(name="booking_flow", metadata={
        "previous_booking_stage": previous_stage,
        "soft_state_summary": summarize_booking_state(soft_state)
    }) as trace:
        stage = str(soft_state.get("booking_stage") or "").strip()
    if stage not in _ACTIVE_BOOKING_STAGES:
        return None
    soft_state["active_flow"] = "booking"

    if has_post_confirmation_amendment_context(message, soft_state):
        amendment_payload = await handle_booking_amendment_turn(message, soft_state)
        if amendment_payload:
            return amendment_payload
        if stage in _POST_CONFIRMATION_AMENDMENT_STAGES:
            return None

    if _is_booking_faq(message):
        tool_context = SimpleNamespace(state={"soft_state": soft_state})
        payload = await check_faq(question=message, tool_context=tool_context)
        return _enrich_booking_faq_payload(payload, soft_state)

    clear_faq_interruption(soft_state)
    sync_alias_keys(soft_state)

    if stage in {"awaiting_confirmation", "awaiting_modification_choice"}:
        normalized = _normalize(message)
        should_interpret_as_modification = (
            stage == "awaiting_modification_choice"
            or re.search(r"\b(change|update|modify|edit)\b", normalized) is not None
        )
        if should_interpret_as_modification:
            field = _modification_field_from_message(message)
            if field == "property_id":
                return handle_review_modification_request(message, soft_state)
            if field:
                inline_updates, _mentioned_fields, inline_errors = _extract_updates_from_message(
                    message,
                    field,
                )
                soft_state["booking_stage"] = "modifying_details"
                set_awaiting_field(soft_state, [field])
                if field in inline_updates and field not in inline_errors:
                    stage = "modifying_details"
                else:
                    return {
                        "status": Status.GATHERING_INFO,
                        "missing_fields": [field],
                        "deterministic_reply": _friendly_prompt_for_field(field),
                    }
            elif stage == "awaiting_modification_choice":
                return {
                    "status": Status.GATHERING_INFO,
                    "deterministic_reply": str(getattr(cfg, "booking_modification_prompt_template", "") or "").strip(),
                }
            else:
                return handle_review_modification_request(message, soft_state)
        elif stage == "awaiting_confirmation":
            return None

    if stage == "awaiting_property_reselection":
        return {
            "status": Status.GATHERING_INFO,
            "deterministic_reply": "Please choose a different property from the current results.",
        }

    awaiting_field = soft_state.get("awaiting_field")
    awaiting_field_name = str(awaiting_field).strip() if isinstance(awaiting_field, str) else None
    updates, mentioned_fields, extraction_errors = _extract_updates_from_message(
        message,
        awaiting_field_name,
    )
    state, next_field, reply = _validate_and_commit_state(
        soft_state,
        updates=updates,
        mentioned_fields=mentioned_fields,
        extraction_errors=extraction_errors,
    )

    if next_field:
        soft_state["active_flow"] = "booking"
        soft_state["booking_stage"] = "modifying_details" if stage in {"modifying_details", "awaiting_modification_choice"} else "collecting_details"
        soft_state["last_presented_view"] = "booking_details_request"
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": [next_field],
            "deterministic_reply": reply,
        }

    return _review_payload_from_state(soft_state, state)

def list_available_cities_payload() -> Dict[str, Any]:
    payload = get_all_available_cities()
    cities = payload.get("cities") or []
    if not isinstance(cities, list):
        cities = []
    city_text = ", ".join(str(city) for city in cities)
    payload["deterministic_reply"] = (
        f"Our service is currently available in: {city_text}."
        if city_text
        else "I couldn't load the available cities right now."
    )
    return payload

