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
from app.agents.tools.search.normalize import _normalize_search_value, _normalize_city_key

def _resolve_result_limit(requested_limit: Optional[int]) -> int:
    floor = 1
    ceiling = max(PROPERTY_RESULT_LIMIT_MAX, floor)
    default_limit = max(PROPERTY_RESULT_LIMIT_DEFAULT, floor)
    if requested_limit is None:
        return min(default_limit, ceiling)
    return min(max(requested_limit, floor), ceiling)

def _resolve_page_size_max() -> int:
    """
    Resolve the maximum page size from config.

    A configured zero/negative value is clamped to 1. A missing value falls
    back to the legacy-safe maximum.
    """
    configured = _coerce_int(getattr(cfg, "page_size_max", None))
    if configured is None:
        configured = _DEFAULT_SEARCH_PAGE_SIZE_MAX
    return max(configured, 1)

def _resolve_page_size() -> int:
    """
    Resolve default page size.

    Config still controls this dynamically. When config is missing or invalid,
    we fall back to the legacy default expected by pagination callers.
    """
    configured = _coerce_int(getattr(cfg, "page_size", None))
    max_size = _resolve_page_size_max()
    if configured is None or configured <= 0:
        configured = _DEFAULT_SEARCH_PAGE_SIZE
    return max(1, min(configured, max_size))

def _resolve_page_size_from(value: Any) -> int:
    """
    Resolve a requested/stored page size, falling back to the configured default.
    """
    configured = _coerce_int(value)
    if configured is None or configured <= 0:
        return _resolve_page_size()
    return max(1, min(configured, _resolve_page_size_max()))

def _search_display_cfg() -> Any:
    return getattr(cfg, "search_display", None)

def _search_display_mode() -> str:
    display = _search_display_cfg()
    return str(getattr(display, "mode", "paginated") or "paginated").strip().lower()

def _search_display_pagination_enabled() -> bool:
    display = _search_display_cfg()
    return bool(getattr(display, "pagination_enabled", True))

def _search_display_max_inline_results() -> Optional[int]:
    display = _search_display_cfg()
    raw = getattr(display, "max_inline_results", None)
    value = _coerce_int(raw)
    return value if value and value > 0 else None

def _search_display_sort_rules() -> List[Dict[str, Any]]:
    display = _search_display_cfg()
    rules = getattr(display, "sort", []) or []
    return [dict(rule) for rule in rules if isinstance(rule, dict)]

def _uses_all_matching_display() -> bool:
    return (
        _search_display_mode() == "all_matching"
        and not _search_display_pagination_enabled()
    )

def _is_missing_sort_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())

def _sort_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return (0, value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return (0, float(stripped.replace("$", "").replace(",", "")))
        except ValueError:
            return (1, stripped.lower())
    return (1, str(value).lower())

def _sort_results_for_display(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_results = list(results)
    for rule in reversed(_search_display_sort_rules()):
        field = str(rule.get("field") or "").strip()
        if not field:
            continue
        descending = str(rule.get("direction") or "asc").strip().lower() == "desc"
        missing_last = bool(rule.get("missing_last", True))
        present = [
            item for item in sorted_results
            if not _is_missing_sort_value(item.get(field))
        ]
        missing = [
            item for item in sorted_results
            if _is_missing_sort_value(item.get(field))
        ]
        present.sort(
            key=lambda item, sort_field=field: _sort_value(item.get(sort_field)),
            reverse=descending,
        )
        sorted_results = present + missing if missing_last else missing + present
    return sorted_results

def _build_option_map_from_formatted(
    formatted:List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Builds a lookup map from formatted property entries keyed by their displayed number.
    
    Parameters:
        formatted (List[Dict[str, Any]]): List of formatted property dictionaries; each item is expected to include a `number` (display index) and property fields such as `id`, `title`, `city`, `price_per_night`, `rating`, `bedrooms`, `bathrooms`, and `property_type`.
    
    Returns:
        Dict[str, Dict[str, Any]]: Mapping where each key is the stringified `number` and each value is a dictionary containing the property's `property_id`, `title`, `city`, `price_per_night`, `rating`, `bedrooms`, `bathrooms`, and `property_type`. Entries with no `number` are omitted.
    """
    option_map: Dict[str, Dict[str, Any]] = {}
    for item in formatted:
        number = item.get("number")
        if number is None:
            continue
        option_map[str(number)] = {
            "property_id": item.get("id"),
            "title": item.get("title"),
            "city": item.get("city"),
            "price_per_night": item.get("price_per_night"),
            "rating": item.get("rating"),
            "reviews_count": item.get("reviews_count"),
            "bedrooms": item.get("bedrooms"),
            "bathrooms": item.get("bathrooms"),
            "property_type": item.get("property_type"),
        }
    return option_map

_DEFAULT_SEARCH_PAGE_SIZE = 5
_DEFAULT_SEARCH_PAGE_SIZE_MAX = 25
