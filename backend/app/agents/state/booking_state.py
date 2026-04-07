"""
app/agents/state/booking_state.py
---------------------------------
Canonical booking state helpers for soft-state persistence.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.config.agent_config_loader import cfg
from app.agents.tools.helpers import _coerce_float, _coerce_int, _is_blank

_ALLOWED_FIELDS = set(cfg.booking_required_fields + cfg.booking_required_numeric_fields)


def get_booking_state(soft_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(soft_state, dict):
        return {}
    state = soft_state.get("booking_state")
    if isinstance(state, dict):
        return state
    state = {}
    soft_state["booking_state"] = state
    return state


def _coerce_numeric(field: str, value: Any) -> Any:
    if field == "guests":
        return _coerce_int(value)
    if field == "price_per_night":
        return _coerce_float(value)
    return value


def extract_booking_updates(**kwargs: Any) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key not in _ALLOWED_FIELDS:
            continue
        if _is_blank(value):
            continue
        coerced = _coerce_numeric(key, value) if key in cfg.booking_required_numeric_fields else value
        if coerced is None:
            continue
        updates[key] = coerced
    return updates


def update_booking_state(
    soft_state: Optional[Dict[str, Any]],
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(soft_state, dict):
        return {}
    state = get_booking_state(soft_state)
    if not updates:
        return state
    for key, value in updates.items():
        state[key] = value
    soft_state["booking_state_updated_at"] = time.time()
    return state


def compute_missing_booking_fields(state: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for field in cfg.booking_required_fields:
        if _is_blank(state.get(field)):
            missing.append(field)
    for field in cfg.booking_required_numeric_fields:
        value = _coerce_numeric(field, state.get(field))
        if value is None:
            missing.append(field)
    # Preserve order while removing duplicates
    return list(dict.fromkeys(missing))


def set_awaiting_field(soft_state: Optional[Dict[str, Any]], missing_fields: List[str]) -> None:
    if not isinstance(soft_state, dict):
        return
    soft_state["awaiting_field"] = missing_fields[0] if missing_fields else None


def clear_awaiting_field(soft_state: Optional[Dict[str, Any]]) -> None:
    if not isinstance(soft_state, dict):
        return
    soft_state.pop("awaiting_field", None)


def clear_booking_state(soft_state: Optional[Dict[str, Any]]) -> None:
    if not isinstance(soft_state, dict):
        return
    soft_state.pop("booking_state", None)
    soft_state.pop("booking_state_updated_at", None)
    soft_state.pop("awaiting_field", None)
