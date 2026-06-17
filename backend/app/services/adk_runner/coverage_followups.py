from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

from app.agents.tools.search import search_properties
from app.agents.tools.search.cities import get_all_available_cities
from app.services.adk_runner.rendering import _render_property_results_from_router_output
from app.services.adk_runner.state_helpers import _state_with_persisted_soft_state
from app.services.booking.constants import _NO_TOKENS, _YES_TOKENS
from app.services.redis_store import get_session_snapshot, save_session_snapshot


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
