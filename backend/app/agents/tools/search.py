"""
---------------------------
Tools: search_properties, get_property_details, select_property, get_all_available_cities
"""
from __future__ import annotations

from collections.abc import Iterable
import csv
import logging
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from ..status_codes import Source, Status
from app.config.agent_config_loader import cfg
from app.services.dynamic_constraints import DynamicConstraints
from app.services.faq_interruption import clear_faq_interruption, sync_alias_keys
from app.services.search_planner import SearchPlan, SearchTrace, get_search_planner
from .helpers import (
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
from app.services.property_type_normalizer import normalize_property_type as _normalize_property_type
from app.services.observability.langfuse_observer import get_observer, sanitize_for_observability, summarize_property_results
logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = _BACKEND_ROOT / cfg.dataset_relative_path
CITY_COLUMN_CANDIDATES = cfg.city_column_candidates
PROPERTY_RERANK_LIMIT: int = cfg.rerank_limit
PROPERTY_RERANK_TIMEOUT_SECONDS: float = cfg.rerank_timeout
PROPERTY_RESULT_LIMIT_DEFAULT: int = cfg.search_result_limit
PROPERTY_RESULT_LIMIT_MAX: int = cfg.search_result_limit_max
PROPERTY_SUMMARY_THRESHOLD: int = cfg.search_summary_mode_threshold


def _normalize_search_value(value: Any) -> str:
    """
    Generic deterministic normalization for schema/entity comparison.

    This does not encode business rules.
    It only normalizes casing, surrounding punctuation, and whitespace.
    """
    text = str(value or "").strip().lower()
    text = text.strip(string.whitespace + string.punctuation)
    return " ".join(text.split())


def _split_amenity_input(value: Any) -> list[str]:
    """
    Normalize structured amenity candidate input.

    This intentionally expects already-extracted candidate terms.
    Natural-language understanding belongs to the planner/model, not here.
    """
    if value is None:
        return []

    if isinstance(value, str):
        text = value.replace(";", ",")
        rows = csv.reader([text], skipinitialspace=True)
        parts = next(rows, [])
        return [
            normalized
            for item in parts
            if (normalized := _normalize_search_value(item))
        ]

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        terms: list[str] = []
        for item in value:
            terms.extend(_split_amenity_input(item))
        return _dedupe_preserve_order(terms)

    normalized = _normalize_search_value(value)
    return [normalized] if normalized else []


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = _normalize_search_value(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)

    return result


def _split_amenities_by_known(
    candidate_terms: Any,
    *,
    known_amenities: Iterable[str],
) -> tuple[list[str], list[str]]:
    """
    Split model/planner-extracted amenity candidates into hard and soft terms.

    Contract:
    - known_amenities comes from schema/config/taxonomy
    - matching known terms become hard filters
    - unknown terms remain soft ranking terms
    - no natural-language phrase parsing
    - no regex
    - no fixed amenity list
    """
    known = {
        _normalize_search_value(item)
        for item in known_amenities
        if _normalize_search_value(item)
    }

    hard_terms: list[str] = []
    soft_terms: list[str] = []

    for term in _split_amenity_input(candidate_terms):
        normalized = _normalize_search_value(term)

        if normalized in known:
            hard_terms.append(normalized)
        else:
            soft_terms.append(normalized)

    return (
        _dedupe_preserve_order(hard_terms),
        _dedupe_preserve_order(soft_terms),
    )


def _known_amenities_from_dataset(dataset: Optional[List[Dict[str, Any]]]) -> list[str]:
    """
    Derive known amenity values dynamically from the loaded dataset.

    This keeps hard-filter eligibility data-driven:
    - no fixed amenity list
    - no natural-language parsing
    - no regex
    """
    if not dataset:
        return []

    terms: list[str] = []
    for row in dataset:
        if isinstance(row, dict):
            terms.extend(_split_amenity_input(row.get("amenities")))

    return _dedupe_preserve_order(terms)


def _build_vibe_query(soft_terms: List[str], free_text: Optional[str]) -> str:
    parts: List[str] = []
    if free_text and free_text.strip():
        parts.append(free_text.strip())
    if soft_terms:
        parts.append(", ".join(soft_terms))
    return " ".join(parts).strip()


def _normalize_city_key(raw: Optional[str]) -> str:
    return " ".join((raw or "").strip().lower().split())


def _city_words(raw: Optional[str]) -> set[str]:
    return {token for token in _normalize_city_key(raw).split(" ") if token}


def _resolve_city_from_catalog(city: Optional[str], dataset: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not city:
        return None
    requested = city.strip()
    if not requested:
        return None

    known_cities: set[str] = set()
    for row in dataset or []:
        current_city = row.get("city") or row.get("location")
        if isinstance(current_city, str) and current_city.strip():
            known_cities.add(current_city.strip())

    if not known_cities:
        return requested

    normalized_to_city: Dict[str, str] = {
        _normalize_city_key(c): c for c in sorted(known_cities)
    }
    exact = normalized_to_city.get(_normalize_city_key(requested))
    if exact:
        return exact

    requested_words = _city_words(requested)
    subset_candidates: List[str] = []
    for candidate in known_cities:
        candidate_words = _city_words(candidate)
        if candidate_words and candidate_words.issubset(requested_words):
            subset_candidates.append(candidate)

    if subset_candidates:
        subset_candidates.sort(key=lambda c: (len(_city_words(c)), len(c)), reverse=True)
        top = subset_candidates[0]
        top_score = (len(_city_words(top)), len(top))
        tied = [c for c in subset_candidates if (len(_city_words(c)), len(c)) == top_score]
        if len(tied) == 1:
            return top

    return requested


def _resolve_result_limit(requested_limit: Optional[int]) -> int:
    floor = 1
    ceiling = max(PROPERTY_RESULT_LIMIT_MAX, floor)
    default_limit = max(PROPERTY_RESULT_LIMIT_DEFAULT, floor)
    if requested_limit is None:
        return min(default_limit, ceiling)
    return min(max(requested_limit, floor), ceiling)


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


def _resolve_page_size_max() -> int:
    """
    Determine the maximum allowed page size from configuration.
    
    Coerces `cfg.page_size_max` to an integer, uses 25 if the configured value is missing or invalid, and enforces a minimum value of 1.
    
    Returns:
        int: The maximum page size (always >= 1).
    """
    configured = _coerce_int(getattr(cfg, "page_size_max", None)) or 25
    return max(configured, 1)


def _resolve_page_size() -> int:
    """
    Determine the effective page size for pagination, clamped to allowed bounds.
    
    Reads the configured page size and falls back to the configured maximum if
    the default is missing or invalid, then clamps the result to the range
    [1, page_size_max].
    
    Returns:
        int: An integer page size between 1 and the configured maximum.
    """
    configured = _coerce_int(getattr(cfg, "page_size", None))
    max_size = _resolve_page_size_max()
    if configured is None or configured <= 0:
        configured = max_size
    return max(1, min(configured, max_size))


def _resolve_page_size_from(value: Any) -> int:
    """
    Resolve an input into a valid page size bounded by configured defaults and maximum.
    
    Parameters:
        value (Any): Candidate page size; will be coerced to an integer.
    
    Returns:
        int: A page size integer at least 1 and at most the configured maximum. If `value` is missing, invalid, or <= 0, the configured default page size is returned.
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

def _build_search_page_payload(
    *,
    results: List[Dict[str, Any]],
    filters: Dict[str, Any],
    page: int,
    page_size: Optional[int] = None,
    search_limit: int = PROPERTY_RESULT_LIMIT_DEFAULT,
    summary_threshold: int = PROPERTY_SUMMARY_THRESHOLD,
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
    all_matching_display = _uses_all_matching_display()

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
    
    Updates the provided `soft_state` to reflect the new page and returns a payload describing the page.
    
    Parameters:
        soft_state (Optional[Dict[str, Any]]): Session soft state containing `all_search_results` and pagination keys; must be a dict with a non-empty `all_search_results` list.
        direction (str): "next" to advance a page or "previous" to go back one page.
    
    Returns:
        Optional[Dict[str, Any]]: A payload dictionary with pagination metadata, visible `properties`, and memory instructions, or `None` if `soft_state` is invalid or has no stored results.
    
    Side effects:
        - Mutates `soft_state` setting keys such as `active_flow`, `current_page`, `page_size`, `visible_results`, `option_map`, `active_property_options_*`, and `active_property_options_generated_at`.
        - Caches the updated last search via `_set_cached_last_search`.
    """
    if not isinstance(soft_state, dict):
        return None
 
    all_results = soft_state.get("all_search_results") or []
    if not isinstance(all_results, list) or not all_results:
        return None

    last_payload = soft_state.get("last_search")
    pagination_state = (
        last_payload.get("pagination", {}) if isinstance(last_payload, dict) else {}
    )
    if not _search_display_pagination_enabled() or (
        direction != "previous" and not bool(pagination_state.get("has_more", False))
    ):
        return {
            "status": Status.PROPERTIES_FOUND,
            "deterministic_reply": cfg.msg_all_results_already_shown,
            "message": cfg.msg_all_results_already_shown,
            "properties": [],
            "total_found": len(all_results),
            "shown_count": len(soft_state.get("visible_results") or []),
            "pagination": {
                "current_page": _coerce_int(soft_state.get("current_page")) or 1,
                "page_size": len(soft_state.get("visible_results") or []),
                "page_start": 1 if all_results else 0,
                "page_end": len(soft_state.get("visible_results") or []),
                "total_pages": 1,
                "has_more": False,
                "has_next": False,
                "has_prev": False,
                "pagination_enabled": False,
            },
            "source": Source.MEMORY,
            "memory": {
                "read_from": "soft_state.all_search_results",
                "state_available": True,
            },
        }
 
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
    )
 
    soft_state["active_flow"] = "search"
    soft_state["current_page"] = payload["pagination"]["current_page"]
    soft_state["page_size"] = page_size
    soft_state["visible_results"] = visible_results
    soft_state["option_map"] = option_map
    soft_state["active_property_options_map"] = option_map
    soft_state["active_property_options_shown_count"] = payload["shown_count"]
    soft_state["active_property_options_total_found"] = payload["total_found"]
    soft_state["active_property_options_generated_at"] = time.time()
 
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



async def _rerank_properties_by_vibe(
    results: List[Dict[str, Any]],
    vibe_query: str,
) -> List[Dict[str, Any]]:
    import asyncio
    if not results or not vibe_query:
        return results
    try:
        from ...components.retrieval import build_doc_text
        from ...services.rag_pipeline import rerank

        class _RerankDoc:
            __slots__ = ("page_content", "metadata")
            def __init__(self, page_content: str, metadata: Dict[str, Any]):
                self.page_content = page_content
                self.metadata = metadata

        docs: List[_RerankDoc] = []
        id_to_prop: Dict[str, Dict[str, Any]] = {}
        for idx, prop in enumerate(results):
            pid = str(prop.get("id") or idx)
            id_to_prop[pid] = prop
            docs.append(_RerankDoc(build_doc_text(prop), {"id": pid}))

        limit = min(len(docs), max(PROPERTY_RERANK_LIMIT, 1))
        reranked_docs = await asyncio.wait_for(
            asyncio.to_thread(rerank, vibe_query, docs[:limit], top_n=limit),
            timeout=PROPERTY_RERANK_TIMEOUT_SECONDS,
        )
        ranked: List[Dict[str, Any]] = []
        for doc in reranked_docs or []:
            meta = getattr(doc, "metadata", {}) or {}
            pid = meta.get("id")
            if pid is None:
                continue
            prop = id_to_prop.get(str(pid))
            if prop and prop not in ranked:
                ranked.append(prop)
        for prop in results:
            if prop not in ranked:
                ranked.append(prop)
        return ranked
    except Exception as exc:
        logger.warning("Property re-ranking failed; using default order: %s", exc)
        return results

def get_all_available_cities(
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Use this tool when the user asks for a list of available cities or locations."""
    try:
        cities: set[str] = set()
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col_name = next(
                (c for c in CITY_COLUMN_CANDIDATES if c in (reader.fieldnames or [])),
                "city",
            )
            for row in reader:
                val = row.get(col_name)
                if val:
                    cities.add(val.strip())
        city_list = sorted(cities)
        payload = {
            "status": Status.CITIES_FOUND,
            "total_cities": len(city_list),
            "cities": city_list,
        }
        return _finalize_payload(payload, action_intent, context_flag)
    except Exception as e:
        return {"status": Status.ERROR, "error": str(e)}


def _build_dynamic_constraints_from_inputs(
    *,
    city: Optional[str] = None,
    budget: Optional[float] = None,
    beds: Optional[int] = None,
    beds_operator: str = "exact",
    bathrooms: Optional[int] = None,
    bathrooms_operator: str = "exact",
    guests: Optional[int] = None,
    guests_operator: str = "min",
    property_type: Optional[str] = None,
    amenities: Optional[List[str]] = None,
) -> DynamicConstraints:
    planner = get_search_planner()
    constraints = DynamicConstraints(schema=planner._schema)
    if city:
        constraints.set("city", city, "exact")
    if property_type:
        constraints.set("property_type", property_type, "exact")
    if beds is not None:
        constraints.set("bedrooms", beds, beds_operator or "exact")
    if bathrooms is not None:
        constraints.set("bathrooms", bathrooms, bathrooms_operator or "exact")
    if guests is not None:
        constraints.set("occupancy_max", guests, guests_operator or "min")
    if budget is not None:
        constraints.set("price_per_night", budget, "max")
    if amenities:
        constraints.set("amenities", list(amenities), "contains")
    return constraints


def _search_plan_from_constraints(
    constraints: DynamicConstraints,
    *,
    search_path: str,
    user_message: str = "",
    sort_preferences: Optional[List[Dict[str, str]]] = None,
) -> SearchPlan:
    trace = SearchTrace(
        session_id="search-tool",
        user_message=user_message,
        extracted_constraints=constraints.to_dict(),
        merged_constraints=constraints.to_dict(),
        search_path=search_path,
        sort_preferences=list(sort_preferences or []),
    )
    trace.log(f"Executing schema search path={search_path} constraints={constraints.to_full_dict()}")
    hard_filters = constraints.get_hard_filters()
    soft_filters = constraints.get_soft_filters()
    trace.hard_filters = hard_filters
    trace.soft_filters = soft_filters
    return SearchPlan(
        constraints=constraints,
        trace=trace,
        session_id="search-tool",
        user_message=user_message,
        sort_preferences=list(sort_preferences or []),
    )


def _apply_planner_to_results(
    results: List[Dict[str, Any]],
    plan: SearchPlan,
) -> List[Dict[str, Any]]:
    planner = get_search_planner()
    filtered, _trace = planner.execute_search(list(results or []), plan)
    return filtered

async def search_properties(
    city: Optional[str] = None,
    budget: Optional[float] = None,
    beds: Optional[int] = None,
    beds_operator: str = "exact",
    bathrooms: Optional[int] = None,
    bathrooms_operator: str = "exact",
    guests: Optional[int] = None,
    guests_operator: str = "min",
    property_type: Optional[str] = None,
    amenities: Optional[str] = None,
    free_text: Optional[str] = None,
    max_results: Optional[int] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    sort_preferences: Optional[List[Dict[str, str]]] = None,
    search_path: str = "tool",
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Search rental properties using optional filters and session-aware behavior.
    
    Performs a catalog-aware city resolution, applies budget/beds/property-type/amenity filters,
    optionally reranks results by a free-text "vibe" query, paginates the first page of results,
    and updates session soft state and cached last search. If critical inputs are missing the
    function returns a payload describing the missing data instead of raising.
    
    Returns:
        dict: A response payload describing the outcome. On success the payload includes
        keys such as `status` (e.g., `Status.PROPERTIES_FOUND`), `properties` (formatted
        results for the returned page), `pagination` (current page, total pages, etc.),
        `shown_count`, `total_found`, and `memory` metadata. If no matches are found the
        payload contains `status: Status.NO_RESULTS` and `filters_applied`. If required
        inputs are absent the payload indicates missing critical data (e.g., `status:
        Status.MISSING_CRITICAL_DATA`) and lists the missing fields. The payload also
        carries `user_engagement_state` and `unresolved_turns` when applicable.
    """
    import asyncio
    from ..tools.rust_client import search_properties as rust_search
    from ...components.search import property_search, _DATASET

    normalized_action = _normalize_action_intent(action_intent, context_flag)
    soft_state = _get_soft_state(tool_context)

    if normalized_action in NEW_SEARCH_ACTION_INTENTS and isinstance(soft_state, dict):
        soft_state.pop("last_search", None)
        soft_state.pop("last_search_at", None)
        _set_unresolved_turns(soft_state, 0)

    last_search = _get_cached_last_search(soft_state)
    has_filters = any([budget is not None, beds is not None, bool(property_type), bool(amenities)])

    if not city:
        if normalized_action in HISTORY_ACTION_INTENTS and last_search:
            cached_city = (last_search.get("query_context") or {}).get("city")
            if cached_city:
                city = cached_city
                if not has_filters:
                    payload = dict(last_search)
                    payload["source"] = Source.MEMORY
                    payload["memory"] = {
                        "read_from": "soft_state.last_search",
                        "state_available": isinstance(soft_state, dict),
                    }
                    return _finalize_payload(payload, normalized_action or action_intent, context_flag)
            else:
                return _missing_critical_data(
                    ["city"],
                    "User asked to revisit previous results but no prior city is stored.",
                    normalized_action or action_intent, context_flag,
                )
        elif normalized_action in HISTORY_ACTION_INTENTS and not last_search:
            return _missing_critical_data(
                ["search_history"],
                "User asked to revisit previous results but no search history is available.",
                normalized_action or action_intent, context_flag,
            )
        else:
            return _missing_critical_data(
                ["city"],
                "User wants to search but has not specified a city.",
                normalized_action or action_intent, context_flag,
            )

    budget_value = _coerce_float(budget)
    beds_value = _coerce_int(beds)
    bathrooms_value = _coerce_int(bathrooms)
    guests_value = _coerce_int(guests)
    normalized_property_type = _normalize_property_type(property_type)
    resolved_city = _resolve_city_from_catalog(city, _DATASET or None)
    if resolved_city:
        city = resolved_city

    raw_amenities = _split_amenity_input(amenities)
    known_amenities = _known_amenities_from_dataset(_DATASET or None)
    hard_amenities, soft_terms = _split_amenities_by_known(
        raw_amenities,
        known_amenities=known_amenities,
    )
    amenity_list = hard_amenities or None
    vibe_query = _build_vibe_query(soft_terms, free_text)
    should_rerank = bool(vibe_query)

    constraints = _build_dynamic_constraints_from_inputs(
        city=city,
        budget=budget_value,
        beds=beds_value,
        beds_operator=beds_operator,
        bathrooms=bathrooms_value,
        bathrooms_operator=bathrooms_operator,
        guests=guests_value,
        guests_operator=guests_operator,
        property_type=normalized_property_type,
        amenities=hard_amenities,
    )
    plan = _search_plan_from_constraints(
        constraints,
        search_path=search_path,
        user_message=free_text or "",
        sort_preferences=sort_preferences,
    )

    requested_limit = _coerce_int(max_results)
    search_limit = _resolve_result_limit(requested_limit)
    summary_threshold = max(PROPERTY_SUMMARY_THRESHOLD, 1)
    base_beds_value = beds_value if (beds_operator or "exact") in {"exact", "min"} else None


    results = None
    if not _uses_all_matching_display():
        try:
            rust_result = await rust_search(
                location=city,
                budget=budget_value,
                beds=base_beds_value,
                amenities=amenity_list or [],
                property_type=normalized_property_type or "",
                max_results=search_limit,
                summary_mode_threshold=summary_threshold,
                properties=_DATASET or None,
            )
            if rust_result and not rust_result.get("fallback"):
                inner = rust_result.get("result", rust_result) or {}
                rust_results = inner.get("results", [])
                if isinstance(rust_results, list):
                    results = rust_results
        except Exception as e:
            logger.warning("Rust property search failed: %s, using Python fallback", e)

    if results is None:
        results = await asyncio.to_thread(
            property_search,
            query_text=f"{normalized_property_type or ''} {city}".strip(),
            budget=int(budget_value) if budget_value is not None else None,
            amenities=amenity_list,
            location=city,
            beds=base_beds_value,
            property_type=normalized_property_type,
        )

    results = _apply_planner_to_results(list(results or []), plan)
    if not plan.sort_preferences:
        results = _sort_results_for_display(list(results or []))

    if not results:
        unresolved_turns = _set_unresolved_turns(soft_state, _get_unresolved_turns(soft_state) + 1)
        payload = {
            "status": Status.NO_RESULTS,
            "city": city,
            "filters_applied": {
                "budget": budget_value,
                "beds": beds_value,
                "beds_operator": beds_operator,
                "bathrooms": bathrooms_value,
                "bathrooms_operator": bathrooms_operator,
                "guests": guests_value,
                "guests_operator": guests_operator,
                "property_type": property_type,
                "amenities": amenities,
            },
            "query_context": {
                "city": city,
                "property_type": normalized_property_type,
                "budget": budget_value,
                "bedrooms": beds_value,
                "bedrooms_operator": beds_operator if beds_value is not None else None,
                "bathrooms": bathrooms_value,
                "bathrooms_operator": bathrooms_operator if bathrooms_value is not None else None,
                "guests": guests_value,
                "guests_operator": guests_operator if guests_value is not None else None,
                "amenities": list(raw_amenities),
            },
            "user_engagement_state": _classify_engagement_state(unresolved_turns),
            "unresolved_turns": unresolved_turns,
        }
        return _finalize_payload(payload, normalized_action or action_intent, context_flag)

    if should_rerank:
        results = await _rerank_properties_by_vibe(results, vibe_query)


    filters = constraints.to_dict()
    filters["city"] = city
    filters["property_type"] = normalized_property_type
    filters["budget"] = budget_value
    if beds_value is not None:
        filters["bedrooms"] = beds_value
        filters["bedrooms_operator"] = beds_operator
    if bathrooms_value is not None:
        filters["bathrooms"] = bathrooms_value
        filters["bathrooms_operator"] = bathrooms_operator
    if guests_value is not None:
        filters["guests"] = guests_value
        filters["guests_operator"] = guests_operator
    if hard_amenities:
        filters["amenities"] = list(hard_amenities)
    if soft_terms:
        filters["soft_amenity_terms"] = list(soft_terms)
    page_size = _search_display_max_inline_results() or _resolve_page_size()
    payload, visible_results, option_map = _build_search_page_payload(
        results=results,
        filters=filters,
        page=1,
        page_size=page_size,
        search_limit=len(results) if _uses_all_matching_display() else search_limit,
        summary_threshold=summary_threshold,
    )
    payload["query_context"] = {
        **dict(payload.get("query_context") or {}),
        "city": city,
        "property_type": normalized_property_type,
        "budget": budget_value,
        "bedrooms": beds_value,
        "bedrooms_operator": beds_operator if beds_value is not None else None,
        "bathrooms": bathrooms_value,
        "bathrooms_operator": bathrooms_operator if bathrooms_value is not None else None,
        "guests": guests_value,
        "guests_operator": guests_operator if guests_value is not None else None,
        "amenities": list(raw_amenities),
        "sort_preferences": list(plan.sort_preferences),
    }

    if isinstance(soft_state, dict):
        soft_state["active_flow"] = "search"
        soft_state["last_filters"] = filters
        soft_state["last_search_filters"] = dict(filters)
        soft_state["last_dynamic_constraints"] = constraints.to_full_dict()
        soft_state["last_sort_preferences"] = list(plan.sort_preferences)
        soft_state["all_search_results"] = list(results)
        soft_state["current_page"] = payload["pagination"]["current_page"]
        soft_state["page_size"] = payload["pagination"]["page_size"]
        soft_state["visible_results"] = visible_results
        soft_state["last_visible_results"] = list(visible_results)
        soft_state["option_map"] = option_map
        soft_state["selected_property_id"] = None
        soft_state["selected_property"] = None
        soft_state["last_presented_view"] = "property_list"
        soft_state["active_property_options_map"] = option_map
        soft_state["active_property_options_shown_count"] = payload["shown_count"]
        soft_state["active_property_options_total_found"] = payload["total_found"]
        clear_faq_interruption(soft_state)

    if isinstance(soft_state, dict):
        soft_state["active_property_options_generated_at"] = time.time()
        sync_alias_keys(soft_state)

    _set_unresolved_turns(soft_state, 0)
    _set_cached_last_search(soft_state, dict(payload))

    payload["memory"] = {
        "written_to": "soft_state.last_search",
        "state_available": isinstance(soft_state, dict),
    }
    payload["user_engagement_state"] = _classify_engagement_state(_get_unresolved_turns(soft_state))
    payload["unresolved_turns"] = _get_unresolved_turns(soft_state)

    try:
        observer = get_observer()
        with observer.trace(
            name="search_tool",
            metadata={
                "city": city,
                "property_type": normalized_property_type,
                "budget": budget_value,
                "beds": beds_value,
                "bathrooms": bathrooms_value,
                "guests": guests_value,
                "amenities": amenity_list,
                "search_path": search_path,
                "hard_filters": plan.trace.hard_filters,
                "sort_preferences": plan.sort_preferences,
            },
        ) as trace:
            trace.update(
                metadata={
                    "result_count": len(results) if results else 0,
                    "filters_applied": {
                        "city": city,
                        "budget": budget_value,
                        "beds": beds_value,
                        "bathrooms": bathrooms_value,
                        "guests": guests_value,
                        "property_type": normalized_property_type,
                        "amenities": raw_amenities,
                    },
                    "summary": summarize_property_results(results),
                }
            )
    except Exception:
        pass

    return _finalize_payload(payload, normalized_action or action_intent, context_flag)

async def select_property(
    option_number: Optional[int] = None,
    property_reference: Optional[str] = None,
    user_engagement_state: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Resolve a user-selected shortlist option and return full property details.

    Use this tool when a user says "option 2", "the second one", or similar.
    The actual ID mapping is resolved from session state.
    """
    return await get_property_details(
        selection_number=option_number,
        property_reference=property_reference,
        user_engagement_state=user_engagement_state,
        action_intent=action_intent,
        context_flag=context_flag,
        tool_context=tool_context,
    )

async def get_property_details(
    property_id: Optional[str] = None,
    selection_number: Optional[int] = None,
    property_reference: Optional[str] = None,
    user_engagement_state: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Get full details of a specific property by its ID, index, or natural language reference."""

    import time
    from ...components.search import _DATASET
    from ..resolvers.property_resolver import resolve_property_reference

    DISPATCHER_MODEL = cfg.dispatcher_model
    soft_state = _get_soft_state(tool_context)
    resolved_from_history = False
    resolution = None
    selection_value = _coerce_int(selection_number)
    last_search = _get_cached_last_search(soft_state)
    selected_item = None


    if selection_value is not None and _is_blank(property_id):
        property_id = _resolve_property_id_from_selection(selection_value, soft_state, last_search)

    if isinstance(soft_state, dict) and property_id:
        for item in soft_state.get("visible_results", []) or []:
            if str(item.get("id")) == str(property_id):
                selected_item = item
                resolved_from_history = True
                break

    if selection_value is not None and not selected_item and last_search:
        for item in last_search.get("properties", []):
            if item.get("number") == selection_value:
                selected_item = item
                resolved_from_history = True
                break

        if not selected_item:
            shown_count, total_found = _get_active_option_window(soft_state, last_search)
            if shown_count > 0 and selection_value > shown_count:
                unresolved_turns = _set_unresolved_turns(soft_state, _get_unresolved_turns(soft_state) + 1)
                engagement_state = str(user_engagement_state).strip() if user_engagement_state else _classify_engagement_state(unresolved_turns)
                payload = {
                    "status": Status.PROPERTY_SELECTION_UNRESOLVED,
                    "resolution": {
                        "internal_reasoning_log": cfg.msg_resolution_not_matched_log,
                        "agent_response": getattr(cfg, "msg_selection_out_of_range", "Option out of range."),
                    },
                    "query_context": (last_search or {}).get("query_context", {}),
                    "shown_count": shown_count,
                    "user_engagement_state": engagement_state,
                    "unresolved_turns": unresolved_turns,
                }
                return _finalize_payload(payload, action_intent, context_flag)

    if not selected_item and not _is_blank(property_reference) and last_search:
        active_options = _build_active_options(last_search)
        if active_options:
            engagement_state = str(user_engagement_state).strip() if user_engagement_state else _classify_engagement_state(_get_unresolved_turns(soft_state))
            resolution = resolve_property_reference(
                user_input=str(property_reference),
                active_options=active_options,
                user_engagement_state=engagement_state,
                dispatcher_model=DISPATCHER_MODEL,
                unresolved_turns=_get_unresolved_turns(soft_state),
                soft_state=soft_state,
                backend_tool_payload=last_search,
            )
            res_id = resolution.get("resolved_property_id")
            if res_id is not None:
                for item in last_search.get("properties", []):
                    if str(item.get("id")) == str(res_id) or str(item.get("number")) == str(res_id):
                        selected_item = item
                        resolved_from_history = True
                        _set_unresolved_turns(soft_state, 0)
                        break
            if not selected_item:
                unresolved_turns = _set_unresolved_turns(soft_state, _get_unresolved_turns(soft_state) + 1)
                payload = {
                    "status": Status.PROPERTY_SELECTION_UNRESOLVED,
                    "resolution": resolution,
                    "active_options": active_options,
                    "user_engagement_state": resolution.get("user_engagement_state", engagement_state),
                    "unresolved_turns": unresolved_turns,
                }
                return _finalize_payload(payload, action_intent, context_flag)

    if selected_item:
        property_id = str(selected_item.get("id") or selected_item.get("title"))

    if _is_blank(property_id):
        missing = ["property_id"]
        if selection_value is None: missing.append("selection_number")
        if _is_blank(property_reference): missing.append("property_reference")
        return _missing_critical_data(
            missing, "User wants property details but no identifier was provided.",
            action_intent, context_flag,
        )

    property_id = str(property_id)
    matched_prop = None
    fallback_prop = None
    for r in _DATASET:
        r_id = str(r.get("id")) if r.get("id") is not None else str(r.get("title"))
        if r_id == property_id:
            matched_prop = r
            break

        if (
            fallback_prop is None
            and selected_item
            and r.get("title") == selected_item.get("title")
            and r.get("city") == selected_item.get("city")
        ):
            fallback_prop = r

    if not matched_prop and fallback_prop is not None:
        matched_prop = fallback_prop

    if not matched_prop and selected_item:
        matched_prop = selected_item

    if matched_prop:
        payload = {
            "status": Status.PROPERTY_DETAILS,
            "property": {
                "id": str(matched_prop.get("id") or matched_prop.get("title", "")),
                "title": matched_prop.get("title"),
                "city": matched_prop.get("city"),
                "price_per_night": matched_prop.get("price_per_night"),
                "bedrooms": matched_prop.get("bedrooms"),
                "bathrooms": matched_prop.get("bathrooms"),
                "amenities": matched_prop.get("amenities"),
                "description": matched_prop.get("description"),
                "rating": matched_prop.get("rating"),
                "reviews_count": matched_prop.get("reviews_count"),
            },
        }
        if isinstance(soft_state, dict):
            soft_state["last_selected_property_id"] = payload["property"]["id"]
            soft_state["last_selected_property_at"] = time.time()
            soft_state["selected_property_id"] = payload["property"]["id"]
            soft_state["selected_property"] = dict(payload["property"])
            soft_state["last_presented_view"] = "property_details"
            _set_unresolved_turns(soft_state, 0)
            sync_alias_keys(soft_state)
            clear_faq_interruption(soft_state)
        payload["memory"] = {
            "read_from": "soft_state.last_search" if resolved_from_history else None,
            "written_to": "soft_state.last_selected_property_id",
            "state_available": isinstance(soft_state, dict),
        }
        if resolution:
            payload["selection_resolution"] = resolution
            payload["user_engagement_state"] = resolution.get("user_engagement_state")
            payload["unresolved_turns"] = _get_unresolved_turns(soft_state)
        return _finalize_payload(payload, action_intent, context_flag)

    return _finalize_payload(
        {"status": Status.NOT_FOUND, "property_id": property_id},
        action_intent, context_flag,

    )

    
