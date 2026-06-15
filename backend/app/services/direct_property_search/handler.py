from __future__ import annotations
import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from app.agents.tools.search import search_properties
from app.components.nlp_engine import (
    is_property_search,
    is_status_query,
    wants_property_search_request,
)
from app.config.conversation_shortcuts_loader import match_shortcut
from app.config.service_coverage_loader import check_region_supported, detect_region_in_message
from app.services.booking_flow import has_active_booking_session
from app.services.direct_property_search.city import (
    _city_clarification_reply,
    _exact_supported_city_from_message,
    _normalize,
    resolve_supported_city_from_message,
)
from app.services.direct_property_search.constraints import _has_search_phrase
from app.services.direct_property_search.extraction import (
    _canonical_property_type_key,
    extract_property_type_from_message,
)
from app.services.faq_interruption import detect_policy_question
from app.services.redis_store import get_session_snapshot, save_session_snapshot

logger = logging.getLogger(__name__)

def is_clear_direct_property_search(message: str, soft_state: Optional[Dict[str, Any]]) -> bool:
    """
    True when the message is a new explicit property search that should bypass ADK.
    """
    if not message or not message.strip():
        return False
    if is_status_query(message):
        return False
    if not is_property_search(message):
        return False

    state = soft_state if isinstance(soft_state, dict) else {}
    if match_shortcut(message, state) is not None:
        return False
    if wants_property_search_request(message) and state.get("all_search_results"):
        return False

    city_match = resolve_supported_city_from_message(message)
    if not city_match.city:
        return False

    region = detect_region_in_message(message) or detect_region_in_message(city_match.city)
    if region:
        decision = check_region_supported(region)
        if decision.blocked:
            return False

    property_type = extract_property_type_from_message(message)
    if property_type or _has_search_phrase(message):
        return True
    return False

def extract_soft_state_from_snapshot(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return {}
    if "soft_state" in state and isinstance(state["soft_state"], dict):
        return dict(state["soft_state"])
    return dict(state)

async def maybe_handle_direct_property_search(
    message: str,
    session_id: str,
    *,
    get_snapshot: Optional[Callable[[str], Any]] = None,
    save_snapshot: Optional[Callable[..., Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run search_properties for clear search intents and persist soft_state to Redis.
    Applies deterministic post-filtering for exact bedroom / price / occupancy
    / amenity constraints derived from the structured query.
    """
    if get_snapshot is None:
        get_snapshot = get_session_snapshot
    if save_snapshot is None:
        save_snapshot = save_session_snapshot

    snapshot = await get_snapshot(session_id)
    if not isinstance(snapshot, dict):
        snapshot = {}

    state = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else {}
    soft_state = state.get("soft_state") if isinstance(state.get("soft_state"), dict) else {}
    normalized_message = _normalize(message)
    if not normalized_message or is_status_query(message):
        return None

    if has_active_booking_session(soft_state):
        logger.info(
            "[direct_search.trace] skipped due to active booking session: stage=%r view=%r",
            soft_state.get("booking_stage"),
            soft_state.get("last_presented_view"),
        )
        return None

    if match_shortcut(message, soft_state) is not None:
        return None

    from app.services.constraint_extractor import (
        dynamic_constraints_from_soft_state,
        extract_dynamic_constraints,
    )

    active_search_context = bool(soft_state.get("all_search_results") or soft_state.get("last_filters"))
    session_constraints = dynamic_constraints_from_soft_state(soft_state) if active_search_context else None
    session_sort_preferences = list(soft_state.get("last_sort_preferences") or []) if active_search_context else []
    dynamic_constraints, plan = extract_dynamic_constraints(
        message,
        session_constraints=session_constraints,
        session_id=session_id,
    )
    property_type = extract_property_type_from_message(message)
    has_search_phrase = _has_search_phrase(message)
    has_new_signal = bool(plan.trace.extracted_constraints) or bool(plan.sort_preferences)
    if active_search_context and not has_new_signal:
        if not is_property_search(message):
            return None
        if wants_property_search_request(message) and soft_state.get("all_search_results"):
            return None
        try:
            if detect_policy_question(message):
                return None
        except Exception as exc:
            logger.debug("[direct_search] policy detection skipped after error: %s", exc)
    elif not (active_search_context and has_new_signal):
        try:
            if detect_policy_question(message):
                return None
        except Exception as exc:
            logger.debug("[direct_search] policy detection skipped after error: %s", exc)
    city_match = resolve_supported_city_from_message(message)
    exact_city = _exact_supported_city_from_message(message)
    logger.info(
        "[direct_search.trace] exact_city=%r city_resolution_status=%s canonical_city=%r "
        "active_search_context=%s extracted=%s merged=%s session_sort_preferences=%s sort_preferences=%s",
        exact_city,
        city_match.status,
        city_match.city,
        active_search_context,
        plan.trace.extracted_constraints,
        plan.trace.merged_constraints,
        session_sort_preferences,
        plan.sort_preferences,
    )

    city = dynamic_constraints.get("city") or city_match.city
    if not city and not active_search_context and not property_type and not has_search_phrase and not has_new_signal:
        return None

    if not city:
        return {
            "status": "needs_clarification",
            "missing": ["city"],
            "query_context": {
                "city": None,
                "property_type": dynamic_constraints.get("property_type") or property_type,
            },
            "deterministic_reply": _city_clarification_reply(city_match),
        }

    if not dynamic_constraints.has("city"):
        dynamic_constraints.set("city", city, "exact")

    property_type = _canonical_property_type_key(
        dynamic_constraints.get("property_type") or property_type
    )
    if property_type and not dynamic_constraints.has("property_type"):
        dynamic_constraints.set("property_type", property_type, "exact")

    tool_context = SimpleNamespace(state={"soft_state": dict(soft_state)})

    effective_sort_preferences = list(plan.sort_preferences or session_sort_preferences)

    search_kwargs: Dict[str, Any] = {
        "city": city,
        "property_type": property_type,
        "beds": dynamic_constraints.get("bedrooms"),
        "beds_operator": dynamic_constraints.get_operator("bedrooms", "exact"),
        "bathrooms": dynamic_constraints.get("bathrooms"),
        "bathrooms_operator": dynamic_constraints.get_operator("bathrooms", "exact"),
        "guests": dynamic_constraints.get("occupancy_max"),
        "guests_operator": dynamic_constraints.get_operator("occupancy_max", "min"),
        "budget": dynamic_constraints.get("price_per_night"),
        "amenities": ",".join(dynamic_constraints.get("amenities") or []),
        "sort_preferences": effective_sort_preferences,
        "search_path": "direct",
        "tool_context": tool_context,
    }

    try:
        payload = await search_properties(**search_kwargs)
    except Exception as exc:
        logger.warning("[direct_search] search_properties failed: %s", exc)
        return None

    new_soft_state = tool_context.state.get("soft_state")
    if not isinstance(new_soft_state, dict):
        new_soft_state = {}
    if not isinstance(payload, dict):
        return None

    state = dict(state)
    state["soft_state"] = new_soft_state

    meta = snapshot.get("meta") if isinstance(snapshot.get("meta"), dict) else {}
    await save_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=state,
        metadata={
            key: meta[key]
            for key in ("app_name", "user_id", "last_update_time")
            if key in meta
        },
    )
    logger.debug(
        "[direct_search] handled city=%r property_type=%r status=%s total_found=%s",
        city,
        property_type,
        payload.get("status"),
        payload.get("total_found"),
    )
    return payload

# Compatibility wrapper: direct-search payloads must persist nested state["soft_state"].
_maybe_handle_direct_property_search_impl = maybe_handle_direct_property_search

async def maybe_handle_direct_property_search(*args, **kwargs):
    payload = await _maybe_handle_direct_property_search_impl(*args, **kwargs)

    soft_state = kwargs.get("soft_state")
    if soft_state is None:
        for arg in reversed(args):
            if isinstance(arg, dict):
                soft_state = arg.get("soft_state") if isinstance(arg.get("soft_state"), dict) else arg
                break

    if isinstance(payload, dict) and isinstance(soft_state, dict):
        payload.setdefault("soft_state", soft_state)
        payload.setdefault("state", {}).setdefault("soft_state", soft_state)
        payload.setdefault("state_delta", {}).setdefault("soft_state", soft_state)

    return payload
