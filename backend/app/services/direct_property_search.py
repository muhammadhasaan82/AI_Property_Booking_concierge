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

    Returns the tool payload when handled; None when the message should fall through.
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

    city = extract_city_from_message(message)
    if not city:
        return None

    property_type = extract_property_type_from_message(message)
    tool_context = SimpleNamespace(state={"soft_state": dict(soft_state)})

    observer = get_observer()
    with observer.trace(name="classified_search", metadata={
        "extracted_city": city,
        "extracted_property_type": property_type,
    }) as trace:
        try:
            payload = await search_properties(
                city=city,
                property_type=property_type,
                tool_context=tool_context,
            )
    except Exception as exc:
        logger.error("[direct_search] search_properties failed: %s", exc, exc_info=True)
        return None

    new_soft_state = tool_context.state.get("soft_state")
    if not isinstance(new_soft_state, dict):
        new_soft_state = {}
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
        trace.update(metadata={
            "result_count": len(payload.get("properties", [])) if isinstance(payload, dict) else 0,
            "status": payload.get("status") if isinstance(payload, dict) else None,
        })
        return payload if isinstance(payload, dict) else None
