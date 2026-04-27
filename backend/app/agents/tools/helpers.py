"""
Shared utility helpers used across all tool modules.

Contains:
- Type coercers  (_coerce_int, _coerce_float, _coerce_bool)
- Blank check    (_is_blank)
- Soft-state accessors (_get_soft_state, _get_unresolved_turns, _set_unresolved_turns)
- Cache helpers  (_get_cached_last_search, _set_cached_last_search)
- Payload helpers (_missing_critical_data, _finalize_payload)
- Misc utilities (_normalize_action_intent, _normalize_extracted_parameters,
                  _sanitize_soft_state_for_model, _build_active_options,
                  _diff_booking_summary, _extract_json_dict)
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from google.adk.tools import ToolContext

from ..status_codes import Status, BOOKING_REQUIRED_FIELDS
from app.config.agent_config_loader import cfg
DATE_FORMAT: str = cfg.date_format
SOFT_SESSION_TTL_SECONDS: int = cfg.session_ttl
HISTORY_ACTION_INTENTS: frozenset = cfg.history_action_intents
NEW_SEARCH_ACTION_INTENTS: frozenset = cfg.new_search_action_intents
def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False
def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False

def _get_soft_state(tool_context: Optional[ToolContext]) -> Optional[Dict[str, Any]]:
    """Return per-session soft state from ADK ToolContext, or None when unavailable."""
    if tool_context is None:
        return None
    state = getattr(tool_context, "state", None)
    if not isinstance(state, dict):
        return None
    soft_state = state.get("soft_state")
    if isinstance(soft_state, dict):
        return soft_state
    try:
        state["soft_state"] = {}
        return state["soft_state"]
    except Exception:
        return None


def _get_unresolved_turns(soft_state: Optional[Dict[str, Any]]) -> int:
    if not isinstance(soft_state, dict):
        return 0
    value = _coerce_int(soft_state.get("unresolved_turns"))
    return max(value or 0, 0)


def _set_unresolved_turns(soft_state: Optional[Dict[str, Any]], value: int) -> int:
    resolved = max(_coerce_int(value) or 0, 0)
    if isinstance(soft_state, dict):
        soft_state["unresolved_turns"] = resolved
    return resolved
def _get_cached_last_search(store: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(store, dict):
        return None
    last = store.get("last_search")
    ts = store.get("last_search_at")
    if last and ts and isinstance(ts, (int, float)):
        if time.time() - ts > SOFT_SESSION_TTL_SECONDS:
            store.pop("last_search", None)
            store.pop("last_search_at", None)
            return None
    return last if isinstance(last, dict) else None


def _set_cached_last_search(store: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    if not isinstance(store, dict):
        return
    store["last_search"] = payload
    store["last_search_at"] = time.time()
def _missing_critical_data(
    missing: List[str],
    context: str,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    unique_missing = list(dict.fromkeys([m for m in missing if m]))
    payload: Dict[str, Any] = {
        "status": Status.MISSING_CRITICAL_DATA,
        "missing": unique_missing,
        "context": context,
    }
    if action_intent:
        payload["action_intent"] = action_intent
    if context_flag:
        payload["context_flag"] = context_flag
    if extra:
        payload.update(extra)
    return payload


def _finalize_payload(
    payload: Dict[str, Any],
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> Dict[str, Any]:
    """Append optional routing fields to any tool payload and return it."""
    if action_intent:
        payload["action_intent"] = action_intent
    if context_flag:
        payload["context_flag"] = context_flag
    return payload

def _normalize_action_intent(action_intent: Optional[str], context_flag: Optional[str]) -> str:
    raw = (action_intent or context_flag or "").strip()
    return raw.lower()


def _normalize_extracted_parameters(data: Any) -> Dict[str, Any]:
    raw = data if isinstance(data, dict) else {}
    return {
        "city": str(raw.get("city")).strip() if raw.get("city") not in (None, "") else None,
        "budget": _coerce_float(raw.get("budget")),
        "beds": _coerce_int(raw.get("beds")),
        "check_in": str(raw.get("check_in")).strip() if raw.get("check_in") not in (None, "") else None,
        "check_out": str(raw.get("check_out")).strip() if raw.get("check_out") not in (None, "") else None,
    }


def _sanitize_soft_state_for_model(soft_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Provide a compact soft-state view to the model without duplicating active options."""
    if not isinstance(soft_state, dict):
        return {}
    allowed = {}
    for key, value in soft_state.items():
        if key in {"last_search", "last_search_at"}:
            continue
        if value in (None, "", [], {}):
            continue
        allowed[key] = value
    return allowed


def _build_active_options(last_search: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the currently active options from the most recent search memory."""
    if not isinstance(last_search, dict):
        return []
    options: List[Dict[str, Any]] = []
    for item in last_search.get("properties", []):
        if not isinstance(item, dict):
            continue
        option = {key: value for key, value in item.items() if value is not None and value != ""}
        if option:
            options.append(option)
    return options


def _diff_booking_summary(
    previous: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(previous, dict):
        return {
            "was_update": False,
            "changed_fields": [],
            "changed_values": {},
        }
    changed_fields: List[str] = []
    changed_values: Dict[str, Any] = {}
    for key, value in current.items():
        if previous.get(key) != value:
            changed_fields.append(str(key))
            changed_values[str(key)] = value
    return {
        "was_update": True,
        "changed_fields": changed_fields,
        "changed_values": changed_values,
    }


def _extract_json_dict(raw_text: Any) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from model output, tolerating code fences."""
    if not isinstance(raw_text, str):
        return None
    candidate = raw_text.strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        lines = [line for line in candidate.splitlines() if not line.strip().startswith("```")]
        candidate = "\n".join(lines).strip()
    for payload in (
        candidate,
        candidate[candidate.find("{") : candidate.rfind("}") + 1]
        if "{" in candidate and "}" in candidate
        else "",
    ):
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None

def _classify_engagement_state(unresolved_turns: int) -> str:
    """
    Config-driven engagement state classifier.
    Thresholds are loaded from agent_config.yaml — no hardcoded numbers.
    Change behaviour by editing: session.unresolved_turns_fatigued
    and session.unresolved_turns_exhausted in agent_config.yaml.
    """
    return cfg.classify_engagement(unresolved_turns)


def _validate_booking_fields(
    property_id: Optional[str],
    property_title: Optional[str],
    guest_name: Optional[str],
    guest_email: Optional[str],
    guest_phone: Optional[str],
    check_in: Optional[str],
    check_out: Optional[str],
    guests: Any,
    price_per_night: Any,
) -> Tuple[List[str], Optional[int], Optional[float]]:
    """
    Validate all required booking fields.

    Returns:
        (missing_fields, guests_value, price_value)
        missing_fields is empty when all fields are present and valid.
    """

    guests_value = _coerce_int(guests)
    price_value = _coerce_float(price_per_night)

    field_map = {
        "property_id": property_id,
        "property_title": property_title,
        "guest_name": guest_name,
        "guest_email": guest_email,
        "guest_phone": guest_phone,
        "check_in": check_in,
        "check_out": check_out,
    }

    missing = [
        field_name
        for field_name, _ in BOOKING_REQUIRED_FIELDS
        if _is_blank(field_map.get(field_name))
    ]
    numeric_values = {"guests": guests_value, "price_per_night": price_value}
    for nf in cfg.booking_required_numeric_fields:
        if numeric_values.get(nf) is None:
            missing.append(nf)

    return missing, guests_value, price_value


def _compute_nights_and_total(
    check_in: str,
    check_out: str,
    price_per_night: float,
) -> Tuple[int, float]:
    """Compute nights and total price. Date format driven by cfg.date_format."""
    from datetime import datetime
    try:
        d1 = datetime.strptime(check_in, DATE_FORMAT)
        d2 = datetime.strptime(check_out, DATE_FORMAT)
        nights = max((d2 - d1).days, 1)
    except Exception:
        nights = 1
    return nights, round(nights * price_per_night, 2)
