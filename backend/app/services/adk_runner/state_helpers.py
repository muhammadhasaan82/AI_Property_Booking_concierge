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
from app.services.config import (
    ADK_MAX_COGNITIVE_CONTEXT_CHARS,
    ADK_SESSION_MAX_CONTEXT_CHARS,
    ADK_SESSION_MAX_EVENTS,
)
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


def _snapshot_has_context(snapshot: Dict[str, Any]) -> bool:
    history = snapshot.get("history", [])
    state = snapshot.get("state", {})
    meta = snapshot.get("meta", {})
    return bool(history) or bool(state) or bool(meta.get("saved_at"))

def _filter_persistent_state(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    return {
        str(key): value
        for key, value in state.items()
        if not str(key).startswith("temp:")
    }

def _merge_soft_state(existing: Any, updates: Any) -> Dict[str, Any]:
    """
    Merge soft_state updates with constraint-aware logic.
    
    - last_filters: Merge individual fields, not entire dict
    - last_search_filters: Merge individual fields, not entire dict
    - all_search_results: Replace (new search)
    - visible_results: Replace (new page)
    - option_map: Replace (new results)
    - Other fields: Standard update
    """
    base = dict(existing) if isinstance(existing, dict) else {}
    
    if not isinstance(updates, dict):
        return base
    
    if "last_filters" in updates and isinstance(updates["last_filters"], dict):
        existing_filters = base.get("last_filters", {})
        if isinstance(existing_filters, dict):
            merged_filters = dict(existing_filters)
            for key, value in updates["last_filters"].items():
                if value is not None:
                    merged_filters[key] = value
            base["last_filters"] = merged_filters
            updates = dict(updates)
            updates.pop("last_filters", None)
    
    if "last_search_filters" in updates and isinstance(updates["last_search_filters"], dict):
        existing_filters = base.get("last_search_filters", {})
        if isinstance(existing_filters, dict):
            merged_filters = dict(existing_filters)
            for key, value in updates["last_search_filters"].items():
                if value is not None:
                    merged_filters[key] = value
            base["last_search_filters"] = merged_filters
            updates = dict(updates)
            updates.pop("last_search_filters", None)
    
    if updates:
        base.update(updates)
    
    return base

def _merge_state(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """
    Merge keys from `updates` into `target` in place, with special merging behavior for the `soft_state` key.
    
    If either `target` or `updates` is not a dict, the function does nothing. When `updates` contains a `soft_state` mapping, that mapping is merged with `target`'s existing `soft_state` (preserving and combining entries rather than replacing the whole `soft_state`) and the merged `soft_state` is stored under the `soft_state` key. All other keys from `updates` are applied to `target` via an in-place update.
     
    Parameters:
        target (Dict[str, Any]): The dictionary to be mutated with the merged values.
        updates (Dict[str, Any]): The dictionary of updates to apply; may include a `soft_state` dict which will be merged rather than replaced.
    """
    if not isinstance(target, dict) or not isinstance(updates, dict):
        return
    soft_updates = updates.get("soft_state")
    if isinstance(soft_updates, dict):
        merged_soft = _merge_soft_state(target.get("soft_state"), soft_updates)
        updates = dict(updates)
        updates["soft_state"] = merged_soft
    target.update(updates)

def _state_with_persisted_soft_state(state: Any, soft_state: Any) -> Dict[str, Any]:
    """
    Build a JSON-safe persisted state payload that guarantees a dictionary `soft_state` and filters out non-persistent keys.
    
    Parameters:
        state (Any): Existing session state (may be dict or other). If a dict, keys whose names start with "temp:" are excluded from the persisted result.
        soft_state (Any): In-memory soft state to persist; will be normalized into a JSON-serializable dict. Non-dict or non-serializable inputs become an empty dict.
    
    Returns:
        Dict[str, Any]: A new state dictionary suitable for persistence that contains the filtered persistent keys from `state` and a `soft_state` key whose value is a JSON-safe dict.
    """
    persisted_state = _filter_persistent_state(state)
    persisted_state.pop("soft_state", None)

    normalized_soft_state = _jsonable(soft_state if isinstance(soft_state, dict) else {})
    if not isinstance(normalized_soft_state, dict):
        normalized_soft_state = {}

    persisted_state["soft_state"] = normalized_soft_state
    return persisted_state

def _soft_state_from_router_output(router_output: Dict[str, Any]) -> Dict[str, Any]:
    """Extract navigable search/selection context from a triage_router tool payload.

    When the ADK in-memory session does not propagate soft_state mutations back
    to Redis, this helper rebuilds a best-effort soft_state from the tool return
    value so that follow-up shortcuts (select, paginate, reject) always have
    navigable context.

    Mapping rules
    -------------
    status == "properties_found"
        visible_results                    <- properties list
        all_search_results                 <- all_search_results/results_full fallback to properties
        option_map                         <- {"1": {"property_id": id}, ...} built from properties
        active_property_options_map        <- option_map
        active_property_options_shown_count <- shown_count or len(properties)
        active_property_options_total_found <- total_found or len(all_search_results)
        last_filters                       <- query_context or filters
        last_presented_view                <- "property_list"

    status == "property_details"
        last_presented_view       <- "property_details"
        last_selected_property_id <- property_id or property["id"]

    All other statuses: returns empty dict (no soft_state derived).
    """
    if not isinstance(router_output, dict):
        return {}

    status = str(router_output.get("status") or "").lower()
    if status == "properties_found":
        props = [
            item
            for item in (router_output.get("properties") or [])
            if isinstance(item, dict)
        ]
        all_results = (
            router_output.get("all_search_results")
            or router_output.get("results_full")
            or props
        )
        option_map = router_output.get("option_map")
        if not isinstance(option_map, dict) or not option_map:
            option_map = {}
            for idx, item in enumerate(props, start=1):
                number = item.get("number") or idx
                prop_id = item.get("id")
                if prop_id is not None:
                    option_map[str(number)] = {"property_id": str(prop_id)}

        return {
            "active_flow": "search",
            "visible_results": props,
            "last_visible_results": props,
            "all_search_results": all_results,
            "option_map": option_map,
            "active_property_options_map": option_map,
            "active_property_options_shown_count": router_output.get("shown_count") or len(props),
            "active_property_options_total_found": router_output.get("total_found") or len(all_results),
            "last_search": router_output,
            "last_filters": router_output.get("query_context") or router_output.get("filters") or {},
            "last_search_filters": router_output.get("query_context") or router_output.get("filters") or {},
            "selected_property": None,
            "last_presented_view": "property_list",
        }

    if status == "property_details":
        prop = router_output.get("property") or {}
        prop_id = router_output.get("property_id") or prop.get("id")
        updates: Dict[str, Any] = {
            "last_presented_view": "property_details",
            "selected_property": prop if isinstance(prop, dict) else {},
        }
        if prop_id:
            updates["last_selected_property_id"] = prop_id
            updates["selected_property_id"] = prop_id
        return updates

    return {}

def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]

    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _jsonable(value.model_dump(mode="json", by_alias=False))
        except Exception:
            try:
                return _jsonable(value.model_dump())
            except Exception:
                pass

    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            public = {key: val for key, val in vars(value).items() if not key.startswith("_")}
            return _jsonable(public)
        except Exception:
            pass

    return str(value)

def _deserialize_event(event_payload: Any) -> Optional[Any]:
    if event_payload is None:
        return None

    try:
        from google.adk.events import Event as AdkEvent
    except Exception:
        try:
            from google.adk.events.event import Event as AdkEvent
        except Exception:
            return None

    if isinstance(event_payload, AdkEvent):
        return event_payload

    if hasattr(AdkEvent, "model_validate"):
        try:
            return AdkEvent.model_validate(event_payload)
        except Exception:
            pass

    if isinstance(event_payload, dict):
        try:
            return AdkEvent(**event_payload)
        except Exception:
            pass

    return None

def _extract_text_parts(event: Any) -> str:
    """Extract concatenated text parts from an ADK event content payload."""
    try:
        if event.content and event.content.parts:
            return "".join(
                part.text for part in event.content.parts
                if hasattr(part, "text") and part.text
            )
    except Exception:
        pass
    return ""

def _event_timestamp(event: Any) -> float:
    try:
        return float(getattr(event, "timestamp", 0.0) or 0.0)
    except Exception:
        return 0.0

def _normalize_cognitive_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return ""
    return str(value).strip()

def _truncate_text_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."

def _estimate_event_chars(event: Any) -> int:
    try:
        return len(json.dumps(_jsonable(event), ensure_ascii=False, sort_keys=True))
    except Exception:
        return len(str(event))

def _trim_events_for_context(
    events: List[Any],
    *,
    max_events: int = ADK_SESSION_MAX_EVENTS,
    max_chars: int = ADK_SESSION_MAX_CONTEXT_CHARS,
) -> List[Any]:
    """Keep only the most recent events within the configured context budget."""
    if not events:
        return []

    bounded_max_events = max(int(max_events or 0), 1)
    bounded_max_chars = max(int(max_chars or 0), 1)

    recent_events = list(events[-bounded_max_events:])
    kept_reversed: List[Any] = []
    running_chars = 0

    for event in reversed(recent_events):
        event_chars = max(_estimate_event_chars(event), 1)
        if kept_reversed and running_chars + event_chars > bounded_max_chars:
            break
        kept_reversed.append(event)
        running_chars += event_chars

    trimmed = list(reversed(kept_reversed))
    return trimmed or [recent_events[-1]]

def _already_exists_error(message: str) -> Exception:
    try:
        from google.adk.errors.already_exists_error import AlreadyExistsError

        return AlreadyExistsError(message)
    except Exception:
        return ValueError(message)

