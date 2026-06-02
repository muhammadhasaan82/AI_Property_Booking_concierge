"""
Deterministic pre-ADK property search for clear search intents.

Detection terms (cities, property types, search phrases) are loaded from YAML
config and dataset helpers — no hardcoded city/type lists in Python.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from app.agents.tools.search import get_all_available_cities, search_properties
from app.components.nlp_engine import (
    is_property_search,
    is_status_query,
    wants_property_search_request,
)
from app.config.conversation_shortcuts_loader import match_shortcut
from app.config.service_coverage_loader import (
    check_region_supported,
    detect_region_in_message,
)
from app.services.dynamic_config import get_vocabulary
from app.services.property_query_constraints import (
    PropertySearchQuery,
    extract_property_search_query,
)
from app.services.property_type_normalizer import normalize_property_type
from app.services.redis_store import get_session_snapshot, save_session_snapshot
from app.services.observability.langfuse_observer import get_observer, sanitize_for_observability

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "property_type_taxonomy.yaml"
)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _contains_term(normalized_message: str, term: str) -> bool:
    needle = _normalize(term)
    if not needle:
        return False
    if " " in needle:
        return needle in normalized_message
    return bool(re.search(rf"\b{re.escape(needle)}\b", normalized_message))


@lru_cache(maxsize=1)
def _property_type_terms() -> Tuple[Tuple[str, str], ...]:
    terms: List[Tuple[str, str]] = []
    try:
        with open(_TAXONOMY_PATH, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        for canonical, spec in (raw.get("property_types") or {}).items():
            canonical_key = str(canonical).strip().lower()
            if not canonical_key:
                continue
            terms.append((canonical_key, canonical_key))
            for alias in spec.get("aliases") or []:
                alias_key = str(alias).strip().lower()
                if alias_key:
                    terms.append((alias_key, canonical_key))
    except Exception as exc:
        logger.warning("[direct_search] taxonomy load failed: %s", exc)

    vocab = get_vocabulary()
    for seed in vocab.seed_property_types:
        seed_key = str(seed).strip().lower()
        if seed_key:
            canonical = normalize_property_type(seed_key) or seed_key
            terms.append((seed_key, canonical))

    deduped = sorted(set(terms), key=lambda item: len(item[0]), reverse=True)
    return tuple(deduped)


@lru_cache(maxsize=1)
def _city_terms() -> Tuple[str, ...]:
    cities: set[str] = set()
    try:
        payload = get_all_available_cities()
        for city in payload.get("cities") or []:
            if isinstance(city, str) and city.strip():
                cities.add(city.strip())
    except Exception as exc:
        logger.debug("[direct_search] dataset cities unavailable: %s", exc)

    vocab = get_vocabulary()
    for city in vocab.fallback_cities:
        if isinstance(city, str) and city.strip():
            cities.add(city.strip())
    for alias, canonical in (vocab.city_aliases or {}).items():
        if isinstance(alias, str) and alias.strip():
            cities.add(alias.strip())
        if isinstance(canonical, str) and canonical.strip():
            cities.add(canonical.strip())

    return tuple(sorted(cities, key=len, reverse=True))


def extract_city_from_message(message: str) -> Optional[str]:
    """Return the best-matching configured city name from the message."""
    normalized = _normalize(message)
    if not normalized:
        return None

    vocab = get_vocabulary()
    alias_map = {
        _normalize(alias): canonical.strip()
        for alias, canonical in (vocab.city_aliases or {}).items()
        if str(alias).strip() and str(canonical).strip()
    }

    best: Optional[str] = None
    best_len = 0
    for city in _city_terms():
        if _contains_term(normalized, city):
            if len(city) > best_len:
                best = city
                best_len = len(city)

    if not best:
        return None

    resolved = alias_map.get(_normalize(best), best)
    return resolved.strip() if resolved else None


def extract_property_type_from_message(message: str) -> Optional[str]:
    """Return canonical property type key when a configured alias appears."""
    normalized = _normalize(message)
    if not normalized:
        return None

    for term, canonical in _property_type_terms():
        if _contains_term(normalized, term):
            return normalize_property_type(term) or canonical
    return None


def extract_bedrooms_from_message(message: str) -> Optional[int]:
    """Extract exact bedroom count from message."""
    normalized = _normalize(message)
    match = re.search(r'\b(\d+)\s+(?:bedroom|bed|bd|br)\b', normalized)
    if match:
        return int(match.group(1))
    return None


def _constraint_values_from_query(
    query: Optional[PropertySearchQuery],
) -> Dict[str, Any]:
    """Extract structured constraint values from a PropertySearchQuery."""
    values: Dict[str, Any] = {
        "bedroom_exact": None,
        "bedroom_min": None,
        "bedroom_max": None,
        "price_max": None,
        "occupancy_min": None,
        "amenities": [],
    }
    if query is None:
        return values
    for c in query.constraints or []:
        if c.field == "bedrooms":
            try:
                num = int(c.value)
            except (TypeError, ValueError):
                continue
            if c.operator == "exact":
                values["bedroom_exact"] = num
            elif c.operator == "min":
                values["bedroom_min"] = num
            elif c.operator == "max":
                values["bedroom_max"] = num
        elif c.field == "price_per_night" and c.operator == "max":
            try:
                values["price_max"] = float(c.value)
            except (TypeError, ValueError):
                continue
        elif c.field == "occupancy_max" and c.operator == "min":
            try:
                values["occupancy_min"] = int(c.value)
            except (TypeError, ValueError):
                continue
        elif c.field == "amenities":
            values["amenities"].append(str(c.value))
    return values


def _property_matches_constraints(prop: Any, constraints: Dict[str, Any]) -> bool:
    if not isinstance(prop, dict):
        return False
    bedroom_exact = constraints.get("bedroom_exact")
    bedroom_min = constraints.get("bedroom_min")
    bedroom_max = constraints.get("bedroom_max")
    price_max = constraints.get("price_max")
    occupancy_min = constraints.get("occupancy_min")
    amenities = constraints.get("amenities") or []

    if bedroom_exact is not None:
        try:
            if int(prop.get("bedrooms") or 0) != int(bedroom_exact):
                return False
        except (TypeError, ValueError):
            return False
    if bedroom_min is not None:
        try:
            if int(prop.get("bedrooms") or 0) < int(bedroom_min):
                return False
        except (TypeError, ValueError):
            return False
    if bedroom_max is not None:
        try:
            if int(prop.get("bedrooms") or 0) > int(bedroom_max):
                return False
        except (TypeError, ValueError):
            return False
    if price_max is not None:
        try:
            if float(prop.get("price_per_night") or 0) > float(price_max):
                return False
        except (TypeError, ValueError):
            return False
    if occupancy_min is not None:
        try:
            if int(prop.get("occupancy_max") or 0) < int(occupancy_min):
                return False
        except (TypeError, ValueError):
            return False
    if amenities:
        prop_amenities = {
            str(a).strip().lower()
            for a in (prop.get("amenities") or [])
            if a is not None
        }
        for required in amenities:
            token = str(required).strip().lower()
            if token and token not in prop_amenities:
                return False
    return True


def _has_active_constraints(constraints: Dict[str, Any]) -> bool:
    if not constraints:
        return False
    for key in ("bedroom_exact", "bedroom_min", "bedroom_max", "price_max", "occupancy_min"):
        if constraints.get(key) is not None:
            return True
    if constraints.get("amenities"):
        return True
    return False


def _filter_properties_by_constraints(
    properties: Optional[List[Any]],
    constraints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not properties:
        return []
    if not _has_active_constraints(constraints):
        return [p for p in properties if isinstance(p, dict)]
    return [
        p
        for p in properties
        if isinstance(p, dict) and _property_matches_constraints(p, constraints)
    ]


def _build_option_map(properties: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    option_map: Dict[str, Dict[str, Any]] = {}
    for prop in properties:
        if not isinstance(prop, dict):
            continue
        number = prop.get("number")
        if number is None:
            continue
        prop_id = prop.get("id")
        if prop_id is None:
            continue
        option_map[str(number)] = {
            "property_id": str(prop_id),
            "title": prop.get("title"),
        }
    return option_map


def _renumber_properties(properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for idx, prop in enumerate(properties, start=1):
        if isinstance(prop, dict):
            prop["number"] = idx
    return properties


def _sort_properties_for_display(properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        properties,
        key=lambda p: (
            float(p.get("rating") or 0),
            int(p.get("reviews_count") or 0),
        ),
        reverse=True,
    )


def _has_search_phrase(message: str) -> bool:
    normalized = _normalize(message)
    if not normalized:
        return False
    nlp = get_vocabulary().nlp_fallback
    for phrase in nlp.search_phrases:
        if phrase and phrase.lower() in normalized:
            return True
    for signal in nlp.search_signals:
        if signal and _contains_term(normalized, signal):
            return True
    return False


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

    city = extract_city_from_message(message)
    if not city:
        return False

    region = detect_region_in_message(message) or detect_region_in_message(city)
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

    if not is_clear_direct_property_search(message, soft_state):
        return None

    structured_query = extract_property_search_query(message)
    constraints = _constraint_values_from_query(structured_query)

    city = structured_query.city or extract_city_from_message(message)
    if not city:
        return None

    property_type = (
        structured_query.property_type or extract_property_type_from_message(message)
    )

    bedrooms_exact = constraints["bedroom_exact"]
    bedrooms_min = constraints["bedroom_min"]
    bedrooms_max = constraints["bedroom_max"]
    price_max = constraints["price_max"]
    occupancy_min = constraints["occupancy_min"]
    amenities = constraints["amenities"]

    tool_context = SimpleNamespace(state={"soft_state": dict(soft_state)})

    search_kwargs: Dict[str, Any] = {
        "city": city,
        "property_type": property_type,
        "tool_context": tool_context,
    }
    if bedrooms_exact is not None:
        search_kwargs["beds"] = bedrooms_exact
    elif bedrooms_min is not None:
        search_kwargs["beds"] = bedrooms_min
    if price_max is not None:
        search_kwargs["budget"] = price_max
    if amenities:
        search_kwargs["amenities"] = ",".join(amenities)

    try:
        payload = await search_properties(**search_kwargs)
    except Exception as exc:
        logger.warning("[direct_search] search_properties failed: %s", exc)
        return None

    new_soft_state = tool_context.state.get("soft_state")
    if not isinstance(new_soft_state, dict):
        new_soft_state = {}

    has_active = _has_active_constraints(constraints)
    if not has_active:
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
            payload.get("status") if isinstance(payload, dict) else None,
            payload.get("total_found") if isinstance(payload, dict) else None,
        )
        return payload if isinstance(payload, dict) else None

    if not isinstance(payload, dict):
        return None

    original_props = payload.get("properties") or []
    original_all_results = (
        new_soft_state.get("all_search_results")
        if isinstance(new_soft_state, dict)
        else None
    ) or []
    original_last_search_props: List[Dict[str, Any]] = []
    if isinstance(new_soft_state, dict):
        last_search = new_soft_state.get("last_search")
        if isinstance(last_search, dict):
            original_last_search_props = list(last_search.get("properties") or [])

    filtered_props = _filter_properties_by_constraints(original_props, constraints)
    filtered_all = _filter_properties_by_constraints(original_all_results, constraints)
    filtered_last_search = _filter_properties_by_constraints(
        original_last_search_props, constraints
    )

    filtered_all = _sort_properties_for_display(filtered_all)
    filtered_props = _renumber_properties(filtered_props)
    option_map = _build_option_map(filtered_props)

    payload["properties"] = filtered_props
    payload["shown_count"] = len(filtered_props)
    payload["total_found"] = len(filtered_all) or len(filtered_props)
    payload["option_map"] = option_map

    qctx = dict(payload.get("query_context") or {})
    qctx["city"] = city
    if property_type:
        qctx["property_type"] = str(property_type).strip().lower()
    if bedrooms_exact is not None:
        qctx["bedrooms"] = bedrooms_exact
        qctx["bedrooms_operator"] = "exact"
    elif bedrooms_min is not None:
        qctx["bedrooms"] = bedrooms_min
        qctx["bedrooms_operator"] = "min"
    elif bedrooms_max is not None:
        qctx["bedrooms"] = bedrooms_max
        qctx["bedrooms_operator"] = "max"
    if price_max is not None:
        qctx["budget"] = price_max
    payload["query_context"] = qctx

    if bedrooms_exact is not None and not filtered_props:
        no_results_payload = {
            "status": "no_results",
            "city": city,
            "property_type": str(property_type).strip().lower() if property_type else None,
            "bedrooms": bedrooms_exact,
            "bedrooms_operator": "exact",
            "query_context": dict(qctx),
            "filters_applied": {
                "city": city,
                "property_type": str(property_type).strip().lower() if property_type else None,
                "bedrooms": bedrooms_exact,
                "bedrooms_operator": "exact",
                "budget": price_max,
            },
        }
        logger.debug(
            "[direct_search] no_results city=%r property_type=%r bedrooms=%s",
            city,
            property_type,
            bedrooms_exact,
        )
        return no_results_payload

    new_soft_state["all_search_results"] = filtered_all
    new_soft_state["visible_results"] = filtered_props
    new_soft_state["option_map"] = option_map
    new_soft_state["active_property_options_map"] = option_map
    new_soft_state["active_property_options_shown_count"] = len(filtered_props)
    new_soft_state["active_property_options_total_found"] = (
        len(filtered_all) or len(filtered_props)
    )
    new_soft_state["active_flow"] = "search"
    new_soft_state["last_filters"] = {
        "city": city,
        "property_type": str(property_type).strip().lower() if property_type else None,
        "bedrooms": bedrooms_exact,
        "bedrooms_operator": "exact" if bedrooms_exact is not None else None,
        "budget": price_max,
    }

    if isinstance(new_soft_state.get("last_search"), dict):
        new_soft_state["last_search"] = dict(new_soft_state["last_search"])
        new_soft_state["last_search"]["properties"] = filtered_last_search or filtered_all
        new_soft_state["last_search"]["option_map"] = option_map
        new_soft_state["last_search"]["shown_count"] = len(filtered_props)
        new_soft_state["last_search"]["total_found"] = (
            len(filtered_all) or len(filtered_props)
        )
        if qctx:
            new_soft_state["last_search"]["query_context"] = dict(qctx)

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
