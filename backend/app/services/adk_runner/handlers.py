from __future__ import annotations
"""ADK runner submodule."""

import asyncio
import hashlib
import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from google.adk.runners import Runner
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.genai.types import Content, Part

from app.config.agent_config_loader import cfg as _cfg
from app.services.redis_store import (
    clear_session_snapshot,
    get_redis_client,
    get_session_snapshot,
    save_session_snapshot,
)

logger = logging.getLogger(__name__)
ADK_TURN_TIMEOUT = float(getattr(_cfg, "runtime_turn_timeout_seconds", 45))
MEM0_ENABLED = os.getenv("MEM0_ENABLED", "1").strip().lower() in ("1", "true", "yes")
APP_NAME = "ai_concierge"


from app.services.adk_runner.rendering import (
    _render_booking_details_request,
    _render_property_details_from_router_output,
    _render_property_results_from_router_output,
    _start_booking_for_selected_property,
)
from app.services.adk_runner.state_helpers import _state_with_persisted_soft_state
from app.services.booking_flow import (
    confirm_booking_review as _confirm_booking_review,
    handle_active_booking_turn as _handle_active_booking_turn,
    handle_booking_amendment_turn as _handle_booking_amendment_turn,
    handle_booking_cancellation_turn as _handle_booking_cancellation_turn,
    handle_booking_status_check as _handle_booking_status_check,
    has_post_confirmation_amendment_context as _has_post_confirmation_amendment_context,
    list_available_cities_payload as _list_available_cities_payload,
    resume_booking_flow as _resume_booking_flow,
    start_booking_for_selected_property as _start_booking_for_selected_property_flow,
)
from app.agents.tools.search import search_properties, select_property, get_property_details
from app.agents.tools.search.cities import get_all_available_cities
from app.services.booking.constants import _NO_TOKENS, _YES_TOKENS

from app.services.faq_interruption import (
    clear_faq_interruption,
    detect_policy_question,
    detect_resume_cue,
    get_faq_interruption,
    is_active as faq_interruption_active,
    resolve_resume_target,
    sync_alias_keys,
)
from app.config.conversation_shortcuts_loader import match_shortcut
from app.config.service_coverage_loader import evaluate_message_coverage
from app.services.observability.langfuse_observer import get_observer

def _coverage_normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _coverage_is_yes(message: str) -> bool:
    normalized = _coverage_normalize(message)
    tokens = set(normalized.split())
    yes_tokens = {str(token).strip().lower() for token in _YES_TOKENS}
    yes_tokens.update({"yes", "y", "yeah", "yep", "sure", "ok", "okay", "yes please"})
    return normalized in yes_tokens or bool(tokens & yes_tokens)


def _coverage_is_no(message: str) -> bool:
    normalized = _coverage_normalize(message)
    tokens = set(normalized.split())
    no_tokens = {str(token).strip().lower() for token in _NO_TOKENS}
    no_tokens.update({"no", "n", "nope", "nah", "no thanks", "no thank you", "not now"})
    return normalized in no_tokens or normalized.startswith("no ") or bool(tokens & no_tokens)


def _coverage_goodbye_reply() -> str:
    return "Okay, bye. See you soon."


def _coverage_available_cities() -> list[str]:
    payload = get_all_available_cities()
    cities = payload.get("cities") if isinstance(payload, dict) else []
    if not isinstance(cities, list):
        return []
    return [str(city).strip() for city in cities if str(city).strip()]


def _coverage_match_city(message: str, cities: list[str]) -> str | None:
    normalized = _coverage_normalize(message)
    if not normalized:
        return None

    for city in cities:
        if _coverage_normalize(city) == normalized:
            return city

    for city in cities:
        city_norm = _coverage_normalize(city)
        if city_norm and city_norm in normalized:
            return city

    return None


def _coverage_property_types_for_city(city: str) -> list[str]:
    try:
        from app.components.search import _DATASET
    except Exception:
        _DATASET = []

    city_norm = _coverage_normalize(city)
    seen: set[str] = set()
    types: list[str] = []

    for row in _DATASET or []:
        if not isinstance(row, dict):
            continue
        if _coverage_normalize(str(row.get("city") or "")) != city_norm:
            continue
        prop_type = str(row.get("property_type") or "").strip()
        if not prop_type:
            continue
        key = _coverage_normalize(prop_type)
        if key and key not in seen:
            seen.add(key)
            types.append(prop_type)

    return sorted(types, key=lambda value: value.lower())


def _coverage_match_property_type(message: str, property_types: list[str]) -> str | None:
    normalized = _coverage_normalize(message)
    if not normalized:
        return None

    for prop_type in property_types:
        if _coverage_normalize(prop_type) == normalized:
            return prop_type

    for prop_type in property_types:
        prop_norm = _coverage_normalize(prop_type)
        if prop_norm and prop_norm in normalized:
            return prop_type

    return None


def _clear_service_coverage_followup(soft_state: dict) -> None:
    for key in (
        "service_coverage_stage",
        "last_unsupported_region",
        "service_coverage_selected_city",
    ):
        soft_state.pop(key, None)


async def _persist_service_coverage_state(
    *,
    session_id: str,
    snapshot: dict,
    state: dict,
    soft_state: dict,
) -> None:
    persisted_state = _state_with_persisted_soft_state(state, soft_state)
    meta = snapshot.get("meta") or {}
    await save_session_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=persisted_state,
        metadata={
            key: meta[key]
            for key in ("app_name", "user_id", "last_update_time")
            if key in meta
        },
    )


async def _maybe_handle_service_coverage_followup(
    *,
    session_id: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    snapshot = await get_session_snapshot(session_id)
    if not isinstance(snapshot, dict):
        return None

    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return None

    soft_state = state.get("soft_state")
    if not isinstance(soft_state, dict):
        return None

    stage = str(soft_state.get("service_coverage_stage") or "").strip()
    if not stage:
        return None

    if _coverage_is_no(message):
        _clear_service_coverage_followup(soft_state)
        await _persist_service_coverage_state(
            session_id=session_id,
            snapshot=snapshot,
            state=state,
            soft_state=soft_state,
        )
        return {
            "status": "service_coverage_ended",
            "deterministic_reply": _coverage_goodbye_reply(),
        }

    cities = _coverage_available_cities()

    if stage == "awaiting_city_list_confirmation":
        # A negative reply is handled before this block. Any other reply here
        # should move forward by listing available cities instead of looping.
        soft_state["service_coverage_stage"] = "awaiting_supported_city_choice"
        await _persist_service_coverage_state(
            session_id=session_id,
            snapshot=snapshot,
            state=state,
            soft_state=soft_state,
        )

        city_text = ", ".join(cities)
        return {
            "status": "service_coverage_city_list",
            "cities": cities,
            "deterministic_reply": f"{city_text}\nWhich city do you want to book with?",
        }

    if stage == "awaiting_supported_city_choice":
        city = _coverage_match_city(message, cities)
        if not city:
            city_text = ", ".join(cities)
            return {
                "status": "service_coverage_awaiting_supported_city_choice",
                "cities": cities,
                "deterministic_reply": f"Please choose one of these cities: {city_text}\nWhich city do you want to book with?",
            }

        property_types = _coverage_property_types_for_city(city)
        soft_state["service_coverage_stage"] = "awaiting_property_type_choice"
        soft_state["service_coverage_selected_city"] = city
        await _persist_service_coverage_state(
            session_id=session_id,
            snapshot=snapshot,
            state=state,
            soft_state=soft_state,
        )

        type_text = ", ".join(property_types)
        return {
            "status": "service_coverage_property_type_list",
            "city": city,
            "property_types": property_types,
            "deterministic_reply": f"{type_text}\nWhich property type do you want to book?",
        }

    if stage == "awaiting_property_type_choice":
        city = str(soft_state.get("service_coverage_selected_city") or "").strip()
        property_types = _coverage_property_types_for_city(city)
        property_type = _coverage_match_property_type(message, property_types)

        if not city or not property_type:
            type_text = ", ".join(property_types)
            return {
                "status": "service_coverage_awaiting_property_type_choice",
                "city": city,
                "property_types": property_types,
                "deterministic_reply": f"Please choose one of these property types: {type_text}",
            }

        _clear_service_coverage_followup(soft_state)
        tool_context = SimpleNamespace(state={"soft_state": soft_state})
        payload = await search_properties(
            city=city,
            property_type=property_type,
            action_intent="new_search",
            tool_context=tool_context,
        )

        if isinstance(payload, dict):
            payload["deterministic_reply"] = _render_property_results_from_router_output(payload)

        await _persist_service_coverage_state(
            session_id=session_id,
            snapshot=snapshot,
            state=state,
            soft_state=soft_state,
        )
        return payload

    return None


def _property_refinement_city_from_state(soft_state: dict) -> str | None:
    candidates: list[str] = []

    last_search = soft_state.get("last_search")
    if isinstance(last_search, dict):
        query_context = last_search.get("query_context") or {}
        if isinstance(query_context, dict):
            candidates.append(str(query_context.get("city") or "").strip())

    for key in ("last_search_filters", "last_filters"):
        filters = soft_state.get(key)
        if isinstance(filters, dict):
            candidates.append(str(filters.get("city") or "").strip())

    for city in candidates:
        if city:
            return city

    return None


def _property_refinement_prompt(city: str, property_types: list[str]) -> str:
    type_text = ", ".join(property_types)
    return (
        f"Sure, which property type do you want to search in {city}? "
        f"Available property types: {type_text}."
    )


def _clear_property_refinement_state(soft_state: dict) -> None:
    soft_state.pop("property_refinement_stage", None)


async def _maybe_handle_property_refinement_followup(
    *,
    session_id: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    snapshot = await get_session_snapshot(session_id)
    if not isinstance(snapshot, dict):
        return None

    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return None

    soft_state = state.get("soft_state")
    if not isinstance(soft_state, dict):
        return None

    city = _property_refinement_city_from_state(soft_state)
    if not city:
        return None

    stage = str(soft_state.get("property_refinement_stage") or "").strip()
    last_view = str(soft_state.get("last_presented_view") or "").strip()
    if stage != "awaiting_property_type" and last_view != "property_list":
        return None

    property_types = _coverage_property_types_for_city(city)
    if not property_types:
        return None

    if _coverage_is_no(message):
        _clear_property_refinement_state(soft_state)
        await _persist_service_coverage_state(
            session_id=session_id,
            snapshot=snapshot,
            state=state,
            soft_state=soft_state,
        )
        return {
            "status": "property_refinement_ended",
            "deterministic_reply": _coverage_goodbye_reply(),
        }

    normalized = _coverage_normalize(message)
    tokens = set(normalized.split())
    property_type = _coverage_match_property_type(message, property_types)

    wants_property_type_change = (
        "property type" in normalized
        or {"change", "property"} <= tokens
        or {"change", "type"} <= tokens
        or {"modify", "property"} <= tokens
        or {"modify", "type"} <= tokens
        or {"update", "property"} <= tokens
        or {"update", "type"} <= tokens
    )

    if stage == "awaiting_property_type":
        if not property_type:
            return {
                "status": "property_refinement_awaiting_property_type",
                "city": city,
                "property_types": property_types,
                "deterministic_reply": _property_refinement_prompt(city, property_types),
            }

        _clear_property_refinement_state(soft_state)
        tool_context = SimpleNamespace(state={"soft_state": soft_state})
        payload = await search_properties(
            city=city,
            property_type=property_type,
            action_intent="new_search",
            tool_context=tool_context,
        )
        if isinstance(payload, dict):
            payload["deterministic_reply"] = _render_property_results_from_router_output(payload)

        await _persist_service_coverage_state(
            session_id=session_id,
            snapshot=snapshot,
            state=state,
            soft_state=soft_state,
        )
        return payload

    if wants_property_type_change:
        if property_type and normalized != "property type":
            _clear_property_refinement_state(soft_state)
            tool_context = SimpleNamespace(state={"soft_state": soft_state})
            payload = await search_properties(
                city=city,
                property_type=property_type,
                action_intent="new_search",
                tool_context=tool_context,
            )
            if isinstance(payload, dict):
                payload["deterministic_reply"] = _render_property_results_from_router_output(payload)

            await _persist_service_coverage_state(
                session_id=session_id,
                snapshot=snapshot,
                state=state,
                soft_state=soft_state,
            )
            return payload

        soft_state["property_refinement_stage"] = "awaiting_property_type"
        await _persist_service_coverage_state(
            session_id=session_id,
            snapshot=snapshot,
            state=state,
            soft_state=soft_state,
        )
        return {
            "status": "property_refinement_awaiting_property_type",
            "city": city,
            "property_types": property_types,
            "deterministic_reply": _property_refinement_prompt(city, property_types),
        }

    return None


async def _maybe_handle_faq_resume_turn(
    *,
    session_id: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    snapshot = await get_session_snapshot(session_id)
    if not isinstance(snapshot, dict):
        return None

    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return None

    soft_state = state.get("soft_state")
    if not isinstance(soft_state, dict) or not faq_interruption_active(soft_state):
        return None

    if detect_policy_question(message):
        from app.agents.tools.support import check_faq

        tool_context = SimpleNamespace(state={"soft_state": soft_state})
        payload = await check_faq(question=message, tool_context=tool_context)
    elif detect_resume_cue(message):
        interruption = get_faq_interruption(soft_state)
        resume_target = interruption.get("resume_target") or resolve_resume_target(soft_state)
        resume_payload = interruption.get("resume_payload") if isinstance(interruption, dict) else {}

        if resume_target == "property_menu":
            payload = resume_payload if isinstance(resume_payload, dict) else {}
            if not payload:
                payload = {"status": str(getattr(_cfg.status, "properties_found", "properties_found"))}
            if not payload.get("properties"):
                sync_alias_keys(soft_state)
                payload = {
                    "status": str(getattr(_cfg.status, "properties_found", "properties_found")),
                    "properties": soft_state.get("last_visible_results") or soft_state.get("visible_results") or [],
                    "all_search_results": soft_state.get("all_search_results") or [],
                    "shown_count": len(soft_state.get("last_visible_results") or soft_state.get("visible_results") or []),
                    "total_found": soft_state.get("active_property_options_total_found") or len(soft_state.get("all_search_results") or []),
                    "query_context": soft_state.get("last_search_filters") or soft_state.get("last_filters") or {},
                    "pagination": (soft_state.get("last_search") or {}).get("pagination") or {},
                    "summary_mode": bool((soft_state.get("last_search") or {}).get("summary_mode", False)),
                }
            payload["deterministic_reply"] = _render_property_results_from_router_output(payload)
        elif resume_target == "selected_property":
            payload = resume_payload if isinstance(resume_payload, dict) else {}
            if not payload.get("property"):
                selected_property = (
                    soft_state.get("selected_property")
                    or soft_state.get("booking_selected_property")
                    or {}
                )
                payload = {
                    "status": str(getattr(_cfg.status, "property_details", "property_details")),
                    "property": selected_property if isinstance(selected_property, dict) else {},
                }
            payload["deterministic_reply"] = _render_property_details_from_router_output(payload)
        elif resume_target == "booking_flow":
            payload = _resume_booking_flow(soft_state) or {}
            if payload:
                clear_faq_interruption(soft_state)
        else:
            return None

        clear_faq_interruption(soft_state)
    else:
        return None

    if not payload:
        return None

    sync_alias_keys(soft_state)
    persisted_state = _state_with_persisted_soft_state(state, soft_state)
    meta = snapshot.get("meta") or {}
    await save_session_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=persisted_state,
        metadata={
            key: meta[key]
            for key in ("app_name", "user_id", "last_update_time")
            if key in meta
        },
    )
    return payload

async def _maybe_handle_search_state_shortcut(
    *,
    session_id: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    """
    Check the message for a stored search shortcut and, if matched, execute the corresponding search tool and persist any updated soft_state.
    
    The function:
    - Loads the Redis session snapshot for the given session_id.
    - Robustly extracts soft_state (supports both nested 'soft_state' and flat/compatibility shapes).
    - Calls match_shortcut(message, soft_state); if no shortcut matches, returns None.
    - If the shortcut action is "select_property" and a selection_number is provided, calls select_property(...) with a tool context containing the current soft_state.
    - If the shortcut action is "paginate_results", calls paginate_stored_results(...) with the requested direction ("next" by default).
    - If the shortcut action is "return_to_previous_results", calls return_to_previous_results(soft_state).
    - If the shortcut action is "start_booking_for_selected_property", seeds booking state from the selected property.
    - If the shortcut action is "confirm_booking_review", finalizes the deterministic receipt.
    - If the shortcut action is "modify_booking_review", enters deterministic review modification.
    - If the shortcut action is "resume_booking_flow", resumes the active deterministic booking stage.
    - If the shortcut action is "list_available_cities", returns the dataset-backed city list.
    - If a tool payload is produced, updates snapshot.state.soft_state, saves the session snapshot, and returns the payload.
    - Returns None when validation fails, no shortcut matches, or the tool produced no payload.
    
    Returns:
        Optional[Dict[str, Any]]: The tool payload produced by the shortcut when handled, or `None` if no shortcut was applied or no payload was produced.
    """
    from app.agents.tools.search import (
        paginate_stored_results,
        return_to_previous_results,
        select_property,
    )

    snapshot = await get_session_snapshot(session_id)
    if not isinstance(snapshot, dict):
        return None

    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return None


    soft_state: Dict[str, Any]
    if "soft_state" in state and isinstance(state["soft_state"], dict):
        soft_state = state["soft_state"]
    else:
        soft_state = dict(state)

    shortcut = match_shortcut(message, soft_state)
    if shortcut is None:
        return None

    active_booking_session = _has_active_booking_session(soft_state)
    booking_stage = str(soft_state.get("booking_stage") or "").strip()

    if active_booking_session and shortcut.action in {"paginate_results", "return_to_previous_results"}:
        return None

    if shortcut.action == "select_property":
        if active_booking_session and booking_stage != "awaiting_property_reselection":
            return None

    payload: Optional[Dict[str, Any]] = None

    if shortcut.action == "select_property" and shortcut.selection_number is not None:
        tool_context = SimpleNamespace(state={"soft_state": soft_state})
        payload = await select_property(
            option_number=shortcut.selection_number,
            tool_context=tool_context,
        )
    elif shortcut.action == "paginate_results":
        payload = paginate_stored_results(
            soft_state,
            direction=shortcut.direction or "next",
        )
    elif shortcut.action == "return_to_previous_results":
        payload = return_to_previous_results(soft_state)
    elif shortcut.action == "start_booking_for_selected_property":
        payload = _start_booking_for_selected_property(soft_state)
    elif shortcut.action == "confirm_booking_review":
        payload = await _confirm_booking_review(soft_state)
    elif shortcut.action == "modify_booking_review":
        payload = _handle_review_modification_request(message, soft_state)
    elif shortcut.action == "resume_booking_flow":
        payload = _resume_booking_flow(soft_state)
    elif shortcut.action == "list_available_cities":
        payload = _list_available_cities_payload()

    if not payload:
        return None

    persisted_state = _state_with_persisted_soft_state(state, soft_state)
    meta = snapshot.get("meta") or {}
    await save_session_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=persisted_state,
        metadata={
            key: meta[key]
            for key in ("app_name", "user_id", "last_update_time")
            if key in meta
        },
    )
    return payload

async def _maybe_record_unsupported_region(
    *,
    session_id: str,
    country: str | None,
) -> None:
    """Persist unsupported-region follow-up state without touching booking/search state."""
    if not country:
        return
    try:
        snapshot = await get_session_snapshot(session_id)
        if not isinstance(snapshot, dict):
            snapshot = {"state": {}, "history": [], "meta": {}}

        state = snapshot.get("state")
        if not isinstance(state, dict):
            state = {}

        soft_state = state.get("soft_state")
        if not isinstance(soft_state, dict):
            soft_state = {}
        else:
            soft_state = dict(soft_state)

        soft_state["last_unsupported_region"] = country
        soft_state["service_coverage_stage"] = "awaiting_city_list_confirmation"

        await _persist_service_coverage_state(
            session_id=session_id,
            snapshot=snapshot,
            state=state,
            soft_state=soft_state,
        )
    except Exception as exc:
        logger.debug("[service_coverage] Could not persist unsupported-region follow-up: %s", exc)

async def _maybe_handle_active_booking_turn(
    *,
    session_id: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    """
    Handle a deterministic booking step for an active booking flow and persist any updated soft state.
    
    Attempts to load the session snapshot for session_id, derives a mutable soft_state (from state["soft_state"] when present or a compatibility copy), and delegates handling to the booking flow handler. If that handler returns a payload, the function persists the session state with a normalized, JSON-safe `soft_state` and returns the payload. If the snapshot is missing or the handler produces no payload, nothing is persisted and the function returns None.
    
    Parameters:
        session_id (str): Identifier of the session whose snapshot will be read and updated.
        message (str): The user's message to process for the active booking turn.
    
    Returns:
        payload (Optional[Dict[str, Any]]): The booking handler's payload when a deterministic action was produced; `None` if no action was taken or the snapshot was invalid.
    """
    snapshot = await get_session_snapshot(session_id)
    if not isinstance(snapshot, dict):
        return None

    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return None

    if "soft_state" in state and isinstance(state["soft_state"], dict):
        soft_state = state["soft_state"]
    else:
        soft_state = dict(state)

    payload = await _handle_active_booking_turn(message, soft_state)
    if not payload:
        return None

    persisted_state = _state_with_persisted_soft_state(state, soft_state)
    meta = snapshot.get("meta") or {}
    await save_session_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=persisted_state,
        metadata={
            key: meta[key]
            for key in ("app_name", "user_id", "last_update_time")
            if key in meta
        },
    )
    return payload

async def _maybe_handle_booking_amendment_turn(
    *,
    session_id: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    snapshot = await get_session_snapshot(session_id)
    if not isinstance(snapshot, dict):
        return None

    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return None

    if "soft_state" in state and isinstance(state["soft_state"], dict):
        soft_state = state["soft_state"]
    else:
        soft_state = dict(state)

    if not _has_post_confirmation_amendment_context(message, soft_state):
        return None

    payload = await _handle_booking_amendment_turn(message, soft_state)
    if not payload:
        return None

    persisted_state = _state_with_persisted_soft_state(state, soft_state)
    meta = snapshot.get("meta") or {}
    await save_session_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=persisted_state,
        metadata={
            key: meta[key]
            for key in ("app_name", "user_id", "last_update_time")
            if key in meta
        },
    )
    return payload

async def _maybe_handle_booking_status_check(
    *,
    session_id: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    """
    Handle a deterministic booking-status lookup from the current session soft_state.

    Loads the Redis session snapshot, delegates to
    :func:`app.services.booking_flow.handle_booking_status_check`, and persists
    any updated soft_state when a reply is produced.

    Parameters:
        session_id (str): Redis session identifier.
        message (str): The raw user message to inspect.

    Returns:
        Optional[Dict[str, Any]]: A payload with ``deterministic_reply`` when
        booking-status intent is detected and handled, or ``None`` to let the
        turn continue through the remaining pipeline stages.
    """
    snapshot = await get_session_snapshot(session_id)
    if not isinstance(snapshot, dict):
        return None

    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return None

    if "soft_state" in state and isinstance(state["soft_state"], dict):
        soft_state = state["soft_state"]
    else:
        soft_state = dict(state)

    payload = await _handle_booking_status_check(message, soft_state)
    if not payload:
        return None

    persisted_state = _state_with_persisted_soft_state(state, soft_state)
    meta = snapshot.get("meta") or {}
    await save_session_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=persisted_state,
        metadata={
            key: meta[key]
            for key in ("app_name", "user_id", "last_update_time")
            if key in meta
        },
    )
    return payload

async def _maybe_handle_booking_cancellation_turn(
    *,
    session_id: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    """
    Handle deterministic booking cancellation/deletion turn from the current session soft_state.
    """
    snapshot = await get_session_snapshot(session_id)
    if not isinstance(snapshot, dict):
        return None

    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return None

    if "soft_state" in state and isinstance(state["soft_state"], dict):
        soft_state = state["soft_state"]
    else:
        soft_state = dict(state)

    payload = await _handle_booking_cancellation_turn(message, soft_state)
    if not payload:
        return None

    persisted_state = _state_with_persisted_soft_state(state, soft_state)
    meta = snapshot.get("meta") or {}
    await save_session_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=persisted_state,
        metadata={
            key: meta[key]
            for key in ("app_name", "user_id", "last_update_time")
            if key in meta
        },
    )
    return payload

def _has_active_booking_session(soft_state: dict | None) -> bool:
    """Return True only when a booking collection/amendment flow is active."""
    if not isinstance(soft_state, dict):
        return False

    stage = (
        soft_state.get("booking_stage")
        or soft_state.get("active_booking_stage")
        or soft_state.get("stage")
    )

    if not stage:
        return False

    normalized_stage = str(stage).strip().lower()
    inactive_stages = {
        "completed",
        "confirmed",
        "cancelled",
        "canceled",
        "idle",
        "none",
        "property_reselection",
    }

    return normalized_stage not in inactive_stages

def _handle_review_modification_request(message: str, soft_state: dict):
    """Handle review-screen modification shortcuts before ADK routing."""
    if not isinstance(soft_state, dict):
        return None

    text = " ".join((message or "").strip().lower().split())
    stage = str(soft_state.get("booking_stage") or "").strip().lower()

    if stage not in {
        "awaiting_confirmation",
        "awaiting_modification_choice",
        "review_pending",
    }:
        return None

    if text in {"no", "nope", "not correct", "change", "modify", "edit"}:
        soft_state["booking_stage"] = "awaiting_modification_choice"
        soft_state["last_presented_view"] = "booking_review"
        return {
            "status": "gathering_info",
            "deterministic_reply": "What would you like to change? You can say name, email, phone, dates, guests, or property.",
        }

    if "property" in text:
        soft_state["booking_stage"] = "awaiting_property_reselection"
        soft_state["last_presented_view"] = "property_list"

        visible = (
            soft_state.get("visible_results")
            or soft_state.get("active_property_options")
            or soft_state.get("all_search_results")
            or []
        )

        lines = ["Please choose another property:"]
        for idx, item in enumerate(visible[:5], start=1):
            if isinstance(item, dict):
                title = item.get("title") or item.get("property_title") or item.get("name") or f"Option {idx}"
                price = item.get("price_per_night")
                if price is not None:
                    lines.append(f"{idx}. {title} - ${price}/night")
                else:
                    lines.append(f"{idx}. {title}")

        if len(lines) == 1:
            lines.append("Please search again or tell me what kind of property you prefer.")

        return {
            "status": "property_reselection",
            "deterministic_reply": "\n".join(lines),
        }

    return None
