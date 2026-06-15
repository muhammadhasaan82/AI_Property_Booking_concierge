from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.config.booking_schema_loader import (
    get_field_aliases,
    get_required_fields,
    get_required_numeric_fields,
    get_ask_order,
)
from app.agents.state.booking_state import get_booking_state
from app.agents.tools.helpers import _coerce_float
from app.services.booking.constants import (
    _ACTIVE_BOOKING_STAGES,
    _POST_CONFIRMATION_AMENDMENT_STAGES,
    _extract_booking_id,
)
import re
import time

def has_post_confirmation_amendment_context(
    message: str,
    soft_state: Dict[str, Any],
) -> bool:
    if _extract_booking_id(message):
        return True

    if not isinstance(soft_state, dict):
        return False

    stage = str(soft_state.get("booking_stage") or "").strip()
    if stage in _POST_CONFIRMATION_AMENDMENT_STAGES:
        return True

    if isinstance(soft_state.get("booking_receipt"), dict):
        return True

    if soft_state.get("booking_registration_id"):
        return True

    return False

def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())

def has_active_booking_session(soft_state: Dict[str, Any]) -> bool:
    """
    Return True when session state indicates an in-progress booking workflow.

    This is intentionally based on workflow state and persisted booking artifacts,
    not on lexical message heuristics.
    """
    if not isinstance(soft_state, dict):
        return False

    stage = str(soft_state.get("booking_stage") or "").strip()
    if stage in _ACTIVE_BOOKING_STAGES:
        return True

    view = str(soft_state.get("last_presented_view") or "").strip()
    if view in {"booking_details_request", "booking_review"}:
        if soft_state.get("booking_property_id") or soft_state.get("booking_selected_property"):
            return True

    if soft_state.get("pending_booking") or soft_state.get("booking_review"):
        if soft_state.get("booking_property_id") or soft_state.get("booking_selected_property"):
            return True

    return False

def _field_alias_map() -> Dict[str, List[str]]:
    aliases = get_field_aliases()
    merged: Dict[str, List[str]] = {}
    for field in set(get_required_fields() + get_required_numeric_fields()):
        values = [field]
        values.extend(aliases.get(field, []))
        merged[field] = [alias for alias in values if alias]
    return merged

def _property_title_from_soft_state(soft_state: Dict[str, Any]) -> str:
    selected = soft_state.get("booking_selected_property")
    if isinstance(selected, dict):
        title = selected.get("title")
        if title:
            return str(title)
    state = soft_state.get("booking_state")
    if isinstance(state, dict):
        title = state.get("property_title")
        if title:
            return str(title)
    return "this property"

def _property_id_matches(item: Any, selected_id: str) -> bool:
    if not isinstance(item, dict):
        return False
    for key in ("id", "property_id"):
        value = item.get(key)
        if value is not None and str(value) == selected_id:
            return True
    return False

def _resolve_selected_property_from_soft_state(
    soft_state: Dict[str, Any],
    selected_id: str,
) -> Dict[str, Any]:
    for key in ("booking_selected_property",):
        candidate = soft_state.get(key)
        if _property_id_matches(candidate, selected_id):
            return dict(candidate)

    for state_key in ("visible_results", "all_search_results"):
        items = soft_state.get(state_key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if _property_id_matches(item, selected_id):
                return dict(item)

    last_search = soft_state.get("last_search")
    if isinstance(last_search, dict):
        for item in last_search.get("properties", []) or []:
            if _property_id_matches(item, selected_id):
                return dict(item)

    option_map = soft_state.get("active_property_options_map")
    if isinstance(option_map, dict):
        for item in option_map.values():
            if _property_id_matches(item, selected_id):
                resolved = dict(item)
                resolved.setdefault("id", resolved.get("property_id"))
                return resolved

    return {}

def _seed_property_fields(soft_state: Dict[str, Any], state: Dict[str, Any]) -> None:
    property_id = soft_state.get("booking_property_id") or soft_state.get("last_selected_property_id")
    selected = soft_state.get("booking_selected_property")
    if property_id:
        state["property_id"] = str(property_id)
    if isinstance(selected, dict):
        if selected.get("title"):
            state["property_title"] = str(selected.get("title"))
        if selected.get("price_per_night") is not None:
            state["price_per_night"] = _coerce_float(selected.get("price_per_night"))

def _selected_property(soft_state: Dict[str, Any]) -> Dict[str, Any]:
    selected = soft_state.get("booking_selected_property")
    return dict(selected) if isinstance(selected, dict) else {}

def _replace_booking_state(soft_state: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    target = get_booking_state(soft_state)
    target.clear()
    target.update(state)
    soft_state["booking_state_updated_at"] = time.time()
    return target

def _sync_review_state(soft_state: Dict[str, Any], summary: Dict[str, Any]) -> None:
    soft_state["booking_review"] = dict(summary)
    soft_state["pending_booking"] = dict(summary)
    soft_state["pending_booking_updated_at"] = time.time()

def _ensure_property_seeded(soft_state: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(get_booking_state(soft_state))
    _seed_property_fields(soft_state, state)
    return _replace_booking_state(soft_state, state)

def _modification_field_from_message(message: str) -> Optional[str]:
    normalized = _normalize(message)
    alias_map = _field_alias_map()
    field_order = ["property_id"] + get_ask_order()
    for field in field_order:
        aliases = alias_map.get(field, [])
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return field
    return None

