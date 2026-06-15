from __future__ import annotations
"""Search tool submodule."""

from collections.abc import Iterable
import csv
import logging
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from app.agents.status_codes import Source, Status
from app.agents.tools.helpers import (
    _build_active_options,
    _classify_engagement_state,
    _coerce_float,
    _coerce_int,
    _finalize_payload,
    _get_cached_last_search,
    _get_soft_state,
    _get_unresolved_turns,
    _is_blank,
    _missing_critical_data,
    _normalize_action_intent,
    _set_cached_last_search,
    _set_unresolved_turns,
    HISTORY_ACTION_INTENTS,
    NEW_SEARCH_ACTION_INTENTS,
)
from app.config.agent_config_loader import cfg
from app.services.dynamic_constraints import DynamicConstraints
from app.services.faq_interruption import clear_faq_interruption, sync_alias_keys
from app.services.observability.langfuse_observer import (
    get_observer,
    sanitize_for_observability,
    summarize_property_results,
)
from app.services.property_type_normalizer import normalize_property_type as _normalize_property_type
from app.services.search_planner import SearchPlan, SearchTrace, get_search_planner
from app.agents.tools.search.constants import (
    CITY_COLUMN_CANDIDATES,
    DATASET_PATH,
    PROPERTY_RERANK_LIMIT,
    PROPERTY_RERANK_TIMEOUT_SECONDS,
    PROPERTY_RESULT_LIMIT_DEFAULT,
    PROPERTY_RESULT_LIMIT_MAX,
    PROPERTY_SUMMARY_THRESHOLD,
)

logger = logging.getLogger(__name__)
from app.agents.tools.search.normalize import _normalize_search_value

def _resolve_property_id_from_selection(
    selection_value: Optional[int],
    soft_state: Optional[Dict[str, Any]],
    last_search: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Resolve a property identifier from a user's numeric selection using session mappings and cached search results.
    
    Checks, in order:
      1. soft_state["option_map"] (preferred) for a mapping keyed by the selection number,
      2. soft_state["active_property_options_map"] (legacy) as a fallback,
      3. last_search["properties"] for an item whose `number` equals the selection.
    
    Parameters:
        selection_value (Optional[int]): The numeric selection value provided by the user.
        soft_state (Optional[Dict[str, Any]]): Session soft state that may contain option maps.
        last_search (Optional[Dict[str, Any]]): Cached last search payload that may contain a `properties` list.
    
    Returns:
        Optional[str]: The resolved property id as a string if found, `None` otherwise.
    """
    if selection_value is None:
        return None

    if isinstance(soft_state, dict):
        option_map = soft_state.get("option_map")
        if isinstance(option_map, dict):
            option = option_map.get(str(selection_value))
            if isinstance(option, dict) and option.get("property_id") is not None:
                return str(option.get("property_id"))

        legacy_map = soft_state.get("active_property_options_map")
        if isinstance(legacy_map, dict):
            option = legacy_map.get(str(selection_value))
            if isinstance(option, dict) and option.get("property_id") is not None:
                return str(option.get("property_id"))

    if isinstance(last_search, dict):
        for item in last_search.get("properties", []):
            if isinstance(item, dict) and item.get("number") == selection_value:
                resolved_id = item.get("id")
                if resolved_id is not None:
                    return str(resolved_id)

    return None

