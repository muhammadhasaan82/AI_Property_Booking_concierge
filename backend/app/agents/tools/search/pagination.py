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

from app.agents.tools.search.display import (
    _build_option_map_from_formatted,
    _resolve_page_size,
    _resolve_page_size_from,
    _resolve_page_size_max,
    _search_display_cfg,
    _search_display_max_inline_results,
    _search_display_mode,
    _search_display_pagination_enabled,
    _search_display_sort_rules,
    _sort_results_for_display,
    _uses_all_matching_display,
)
from app.agents.tools.search.normalize import _normalize_search_value
from app.agents.tools.search.selection import _resolve_property_id_from_selection

def _get_active_option_window(
    soft_state: Optional[Dict[str, Any]],
    last_search: Optional[Dict[str, Any]],
) -> tuple[int, int]:
    """
    Return the active option window counts (number shown and total found) for the current session or last search.
    
    Checks `soft_state` and `last_search` dictionaries for numeric counts and falls back to the length of `last_search["properties"]` when counts are missing or non-positive.
    
    Parameters:
        soft_state (Optional[Dict[str, Any]]): Session soft state; reads
            `active_property_options_shown_count` and `active_property_options_total_found` if present.
        last_search (Optional[Dict[str, Any]]): Cached last search payload; reads
            `shown_count`, `total_found`, and `properties` (used for fallback).
    
    Returns:
        tuple[int, int]: A pair `(shown_count, total_found)` where each value is an integer >= 0.
    """
    shown_count = 0
    total_found = 0

    if isinstance(soft_state, dict):
        shown_count = _coerce_int(soft_state.get("active_property_options_shown_count")) or 0
        total_found = _coerce_int(soft_state.get("active_property_options_total_found")) or 0

    if isinstance(last_search, dict):
        if shown_count <= 0:
            shown_count = _coerce_int(last_search.get("shown_count")) or len(last_search.get("properties", []))
        if total_found <= 0:
            total_found = _coerce_int(last_search.get("total_found")) or len(last_search.get("properties", []))

    return max(shown_count, 0), max(total_found, 0)

def _build_search_page_payload(
    *,
    results: List[Dict[str, Any]],
    filters: Dict[str, Any],
    page: int,
    page_size: Optional[int] = None,
    search_limit: int = PROPERTY_RESULT_LIMIT_DEFAULT,
    summary_threshold: int = PROPERTY_SUMMARY_THRESHOLD,
    all_matching_display: bool = False,
) -> tuple[Dict[str, Any], list[Dict[str,Any]], Dict[str, Dict[str, Any]]]:
    """
    Builds a paginated search payload, the visible results for the requested page, and an option map for quick lookup.
    
    Parameters:
        results (List[Dict[str, Any]]): Full list of search result records.
        filters (Dict[str, Any]): Query context used for the payload (e.g., city, budget, beds, property_type).
        page (int): 1-based page number to return; values outside valid range are clamped.
        page_size (Optional[int]): Desired page size; when None or invalid, the configured default is used and the value is clamped to allowed bounds.
        search_limit (int): Maximum number of results considered for reporting (`max_results` in the payload).
        summary_threshold (int): Threshold at which the payload enables summary mode when total results exceed this value.
    
    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
            - payload: A dictionary containing search metadata and the formatted properties for the page. Key fields include:
                - status, total_found, shown_count, has_more, remaining_count, max_results
                - summary_mode, summary_mode_threshold
                - properties: list of formatted property entries (with keys like number, id, title, city, price_per_night, bedrooms, bathrooms, property_type, rating, amenities)
                - query_context: echo of relevant filters
                - pagination: {current_page, page_size, page_start, page_end, total_pages}
            - visible_results: The raw slice of `results` included on the returned page.
            - option_map: Mapping from stringified option number to a compact dict of property fields for each visible item.
    """
    total_found = len(results)
    pagination_enabled = _search_display_pagination_enabled()
    max_inline_results = _search_display_max_inline_results()
    all_matching_display = bool(all_matching_display)

    if all_matching_display:
        safe_page = 1
        total_pages = 1
        start = 0
        end = total_found if max_inline_results is None else min(max_inline_results, total_found)
        safe_page_size = end
    else:
        safe_page_size = max_inline_results or _resolve_page_size_from(page_size)
        safe_page = max(_coerce_int(page) or 1, 1)
        total_pages = max((total_found + safe_page_size - 1) // safe_page_size, 1)
        safe_page = min(safe_page, total_pages)
        start = (safe_page - 1) * safe_page_size
        end = min(start + safe_page_size, total_found)

    visible_results = results[start:end]
    formatted: List[Dict[str, Any]] = []
    for i, r in enumerate(visible_results, start + 1):
        raw_amenities = r.get("amenities") or []
        top_amenities = raw_amenities[:3] if isinstance(raw_amenities, list) else []
        formatted.append(
            {
                "number": i,
                "id": r.get("id"),
                "title": r.get("title", "Property"),
                "city": (r.get("city") or "").title(),
                "price_per_night": r.get("price_per_night"),
                "bedrooms": r.get("bedrooms"),
                "bathrooms": r.get("bathrooms"),
                "property_type": r.get("property_type", ""),
                "rating": r.get("rating"),
                "reviews_count": r.get("reviews_count"),
                "amenities": top_amenities,
            }
        )
 
    option_map = _build_option_map_from_formatted(formatted)
    shown_count = len(formatted)
    has_more = bool(pagination_enabled and end < total_found)
    has_prev = bool(pagination_enabled and safe_page > 1)
    remaining_count = max(total_found - end, 0)
 
    payload = {
        "status": Status.PROPERTIES_FOUND,
        "total_found": total_found,
        "shown_count": shown_count,
        "has_more": has_more,
        "remaining_count": remaining_count,
        "max_results": search_limit,
        "summary_mode": total_found > max(summary_threshold, 1),
        "summary_mode_threshold": max(summary_threshold, 1),
        "properties": formatted,
        "query_context": {
            "city": filters.get("city"),
            "budget": filters.get("budget"),
            "beds": filters.get("beds"),
            "property_type": filters.get("property_type"),
        },
        "pagination": {
            "current_page": safe_page,
            "page_size": safe_page_size,
            "page_start": start + 1 if total_found else 0,
            "page_end": end,
            "total_pages": total_pages,
            "has_more": has_more,
            "has_next": has_more,
            "has_prev": has_prev,
            "pagination_enabled": pagination_enabled,
        },
    }
    return payload, formatted, option_map

def paginate_stored_results(
    soft_state: Optional[Dict[str, Any]],
    *,
    direction: str = "next",
) -> Optional[Dict[str, Any]]:
    """
    Advance or rewind the current paginated search results stored in session soft state.
    """
    if not isinstance(soft_state, dict):
        return None

    all_results = soft_state.get("all_search_results") or []
    if not isinstance(all_results, list) or not all_results:
        return None

    current_page = max(_coerce_int(soft_state.get("current_page")) or 1, 1)
    page_size = _resolve_page_size_from(soft_state.get("page_size"))
    filters = soft_state.get("last_filters") or {}

    if direction == "previous":
        target_page = max(current_page - 1, 1)
    else:
        target_page = current_page + 1

    payload, visible_results, option_map = _build_search_page_payload(
        results=all_results,
        filters=filters,
        page=target_page,
        page_size=page_size,
        search_limit=_resolve_result_limit(None),
        summary_threshold=PROPERTY_SUMMARY_THRESHOLD,
        all_matching_display=False,
    )

    if (
        direction != "previous"
        and payload.get("pagination", {}).get("current_page") == current_page
        and not payload.get("pagination", {}).get("has_next")
    ):
        payload["deterministic_reply"] = "All matching properties are already shown."

    soft_state["active_flow"] = "search"
    soft_state["current_page"] = payload["pagination"]["current_page"]
    soft_state["page_size"] = payload["pagination"]["page_size"]
    soft_state["visible_results"] = visible_results
    soft_state["last_visible_results"] = list(visible_results)
    soft_state["option_map"] = option_map
    soft_state["active_property_options_map"] = option_map
    soft_state["active_property_options_shown_count"] = payload["shown_count"]
    soft_state["active_property_options_total_found"] = payload["total_found"]
    soft_state["active_property_options_generated_at"] = time.time()
    sync_alias_keys(soft_state)

    _set_cached_last_search(soft_state, dict(payload))

    payload["source"] = Source.MEMORY
    payload["memory"] = {
        "read_from": "soft_state.all_search_results",
        "written_to": "soft_state.visible_results",
        "state_available": True,
    }
    return payload

def return_to_previous_results(
    soft_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(soft_state, dict):
        return None

    visible_results = soft_state.get("visible_results") or []
    all_results = soft_state.get("all_search_results") or []
    if not isinstance(visible_results, list) or not visible_results:
        if not isinstance(all_results, list) or not all_results:
            return None
        filters = soft_state.get("last_filters") or {}
        sorted_results = _sort_results_for_display(list(all_results))
        payload, visible_results, option_map = _build_search_page_payload(
            results=sorted_results,
            filters=filters,
            page=1,
            page_size=len(sorted_results),
            search_limit=len(sorted_results),
            summary_threshold=PROPERTY_SUMMARY_THRESHOLD,
        )
    else:
        last_search = _get_cached_last_search(soft_state) or {}
        payload = dict(last_search) if isinstance(last_search, dict) else {}
        if not payload:
            filters = soft_state.get("last_filters") or {}
            payload, visible_results, option_map = _build_search_page_payload(
                results=list(visible_results),
                filters=filters,
                page=1,
                page_size=len(visible_results),
                search_limit=len(visible_results),
                summary_threshold=PROPERTY_SUMMARY_THRESHOLD,
            )
        else:
            option_map = soft_state.get("option_map") or {}
            payload["properties"] = list(visible_results)
            payload["shown_count"] = len(visible_results)
            payload["total_found"] = _coerce_int(payload.get("total_found")) or len(visible_results)
            payload["has_more"] = False
            payload["remaining_count"] = 0
            payload["pagination"] = {
                "current_page": 1,
                "page_size": len(visible_results),
                "page_start": 1 if visible_results else 0,
                "page_end": len(visible_results),
                "total_pages": 1,
                "has_more": False,
                "has_next": False,
                "has_prev": False,
                "pagination_enabled": False,
            }

    option_map = _build_option_map_from_formatted(payload.get("properties") or [])
    rejected_id = soft_state.get("last_selected_property_id")
    if rejected_id:
        soft_state["last_rejected_property_id"] = rejected_id
    soft_state["last_presented_view"] = "property_list"
    soft_state["selected_property_id"] = None
    soft_state["selected_property"] = None
    soft_state["visible_results"] = list(payload.get("properties") or [])
    soft_state["last_visible_results"] = list(payload.get("properties") or [])
    soft_state["option_map"] = option_map
    soft_state["active_property_options_map"] = option_map
    soft_state["active_property_options_shown_count"] = payload.get("shown_count")
    soft_state["active_property_options_total_found"] = payload.get("total_found")
    filters = soft_state.get("last_filters") or {}
    if isinstance(filters, dict):
        soft_state["last_search_filters"] = dict(filters)
    clear_faq_interruption(soft_state)
    _set_cached_last_search(soft_state, dict(payload))

    payload["source"] = Source.MEMORY
    payload["memory"] = {
        "read_from": "soft_state.visible_results",
        "written_to": "soft_state.last_presented_view",
        "state_available": True,
    }
    return payload

