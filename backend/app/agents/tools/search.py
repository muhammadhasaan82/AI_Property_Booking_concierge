"""
app/agents/tools/search.py
---------------------------
Tools: search_properties, get_property_details, select_property, get_all_available_cities
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from ..status_codes import Source, Status
from app.config.agent_config_loader import cfg
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

logger = logging.getLogger(__name__)

# Dataset path and search config — all driven from agent_config.yaml
# To change the path or rerank limits: edit agent_config.yaml, not this file.
_BACKEND_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = _BACKEND_ROOT / cfg.dataset_relative_path
CITY_COLUMN_CANDIDATES = cfg.city_column_candidates
PROPERTY_RERANK_LIMIT: int = cfg.rerank_limit
PROPERTY_RERANK_TIMEOUT_SECONDS: float = cfg.rerank_timeout
PROPERTY_RESULT_LIMIT_DEFAULT: int = cfg.search_result_limit
PROPERTY_RESULT_LIMIT_MAX: int = cfg.search_result_limit_max
PROPERTY_SUMMARY_THRESHOLD: int = cfg.search_summary_mode_threshold


# ---------------------------------------------------------------------------
# Internal search helpers
# ---------------------------------------------------------------------------

def _split_amenities_by_known(
    amenities: Optional[List[str]],
    dataset: Optional[List[Dict[str, Any]]],
) -> tuple[List[str], List[str]]:
    if not amenities:
        return [], []
    if not dataset:
        return list(amenities), []
    known: set[str] = set()
    for row in dataset:
        for item in row.get("amenities") or []:
            if isinstance(item, str) and item.strip():
                known.add(item.strip().lower())
    hard_terms: List[str] = []
    soft_terms: List[str] = []
    for term in amenities:
        cleaned = (term or "").strip()
        if not cleaned:
            continue
        if cleaned.lower() in known:
            hard_terms.append(cleaned)
        else:
            soft_terms.append(cleaned)
    return hard_terms, soft_terms


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
    if selection_value is None:
        return None

    if isinstance(soft_state, dict):
        option_map = soft_state.get("active_property_options_map")
        if isinstance(option_map, dict):
            option = option_map.get(str(selection_value))
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


# ---------------------------------------------------------------------------
# Tool: get_all_available_cities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool: search_properties
# ---------------------------------------------------------------------------

async def search_properties(
    city: Optional[str] = None,
    budget: Optional[float] = None,
    beds: Optional[int] = None,
    property_type: Optional[str] = None,
    amenities: Optional[str] = None,
    free_text: Optional[str] = None,
    max_results: Optional[int] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Search for rental properties with soft-coded inputs.

    Use this tool when the user wants to find, browse, or compare properties.
    All parameters are optional. If critical data is missing, this tool returns
    status=missing_critical_data rather than failing.

    Args:
        city: The city name if known.
        budget: Maximum nightly price in USD (optional).
        beds: Minimum number of bedrooms (optional).
        property_type: Type of property like apartment, house, villa, etc (optional).
        amenities: Comma-separated list of required amenities (optional).
        free_text: Optional vibe or descriptive constraints for semantic matching.
        max_results: Optional result window size (bounded by config).
        action_intent: Optional context flag like "re_evaluate_history" or "new_search".
        context_flag: Optional secondary context flag.
        tool_context: ADK tool context for session state.
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
    resolved_city = _resolve_city_from_catalog(city, _DATASET or None)
    if resolved_city:
        city = resolved_city

    requested_limit = _coerce_int(max_results)
    search_limit = _resolve_result_limit(requested_limit)
    summary_threshold = max(PROPERTY_SUMMARY_THRESHOLD, 1)

    raw_amenities = [a.strip() for a in (amenities or "").split(",") if a.strip()]
    hard_amenities, soft_terms = _split_amenities_by_known(raw_amenities, _DATASET or None)
    amenity_list = hard_amenities or None
    vibe_query = _build_vibe_query(soft_terms, free_text)
    should_rerank = bool(vibe_query)

    results = None
    try:
        rust_result = await rust_search(
            location=city,
            budget=budget_value,
            beds=beds_value,
            amenities=amenity_list or [],
            property_type=property_type or "",
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
            query_text=f"{property_type or ''} {city}".strip(),
            budget=int(budget_value) if budget_value is not None else None,
            amenities=amenity_list,
            location=city,
            beds=beds_value,
            property_type=property_type,
        )

    if results and property_type:
        results = [
            r for r in results
            if r.get("property_type")
            and property_type.lower() in str(r.get("property_type")).lower()
        ]

    if not results:
        unresolved_turns = _set_unresolved_turns(soft_state, _get_unresolved_turns(soft_state) + 1)
        payload = {
            "status": Status.NO_RESULTS,
            "city": city,
            "filters_applied": {
                "budget": budget_value,
                "beds": beds_value,
                "property_type": property_type,
                "amenities": amenities,
            },
            "user_engagement_state": _classify_engagement_state(unresolved_turns),
            "unresolved_turns": unresolved_turns,
        }
        return _finalize_payload(payload, normalized_action or action_intent, context_flag)

    if should_rerank:
        results = await _rerank_properties_by_vibe(results, vibe_query)

    total_found = len(results)
    shown_results = results[:search_limit]
    summary_mode = total_found > summary_threshold

    formatted: List[Dict[str, Any]] = []
    for i, r in enumerate(shown_results, 1):
        item = {
            "number": i,
            "id": r.get("id"),
            "title": r.get("title", "Property"),
            "city": (r.get("city") or "").title(),
            "price_per_night": r.get("price_per_night"),
            "bedrooms": r.get("bedrooms"),
            "bathrooms": r.get("bathrooms"),
            "property_type": r.get("property_type", ""),
            "rating": r.get("rating"),
        }
        if not summary_mode:
            item["amenities"] = r.get("amenities")
            item["description"] = r.get("description")
        formatted.append(item)

    shown_count = len(formatted)
    has_more = total_found > shown_count
    remaining_count = max(total_found - shown_count, 0)

    payload = {
        "status": Status.PROPERTIES_FOUND,
        "total_found": total_found,
        "shown_count": shown_count,
        "has_more": has_more,
        "remaining_count": remaining_count,
        "max_results": search_limit,
        "summary_mode": summary_mode,
        "summary_mode_threshold": summary_threshold,
        "properties": formatted,
        "query_context": {
            "city": city,
            "budget": budget_value,
            "beds": beds_value,
            "property_type": property_type,
        },
    }

    if isinstance(soft_state, dict):
        soft_state["active_property_options_map"] = {
            str(item["number"]): {
                "property_id": item.get("id"),
                "title": item.get("title"),
                "city": item.get("city"),
                "price_per_night": item.get("price_per_night"),
                "rating": item.get("rating"),
            }
            for item in formatted
            if item.get("number") is not None
        }
        soft_state["active_property_options_shown_count"] = shown_count
        soft_state["active_property_options_total_found"] = total_found
        soft_state["active_property_options_generated_at"] = time.time()

    _set_unresolved_turns(soft_state, 0)
    _set_cached_last_search(soft_state, dict(payload))
    payload["memory"] = {
        "written_to": "soft_state.last_search",
        "state_available": isinstance(soft_state, dict),
    }
    payload["user_engagement_state"] = _classify_engagement_state(_get_unresolved_turns(soft_state))
    payload["unresolved_turns"] = _get_unresolved_turns(soft_state)
    return _finalize_payload(payload, normalized_action or action_intent, context_flag)


# ---------------------------------------------------------------------------
# Tool: get_property_details
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool: get_property_details
# ---------------------------------------------------------------------------


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
    import os
    import time
    from ...components.search import _DATASET
    from ..resolvers.property_resolver import resolve_property_reference

    DISPATCHER_MODEL = os.getenv("ADK_DISPATCHER_MODEL", "openai/gpt-5-nano")

    soft_state = _get_soft_state(tool_context)
    resolved_from_history = False
    resolution = None
    selection_value = _coerce_int(selection_number)
    last_search = _get_cached_last_search(soft_state)
    selected_item = None

    # 1. DETERMINISTIC RESOLUTION: Match exact option number from memory
    if selection_value is not None and last_search:
        for item in last_search.get("properties", []):
            if item.get("number") == selection_value:
                selected_item = item
                resolved_from_history = True
                break

        # Handle edge case where number is out of bounds
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

    # 2. PROBABILISTIC RESOLUTION: Match fuzzy descriptions/text using the LLM Router
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

    # 3. Establish the base ID for dataset lookup
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

    # 4. FULL DETAILS LOOKUP: Find full description & amenities from _DATASET
    property_id = str(property_id)
    matched_prop = None
    for r in _DATASET:
        r_id = str(r.get("id")) if r.get("id") is not None else str(r.get("title"))
        if r_id == property_id:
            matched_prop = r
            break
        # DYNAMIC FALLBACK: If dataset lacks an explicit ID column, match by title + city
        if selected_item and r.get("title") == selected_item.get("title") and r.get("city") == selected_item.get("city"):
            matched_prop = r
            break

    # 5. ABSOLUTE FALLBACK: Use whatever properties we have in memory
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
            },
        }
        if isinstance(soft_state, dict):
            soft_state["last_selected_property_id"] = payload["property"]["id"]
            soft_state["last_selected_property_at"] = time.time()
            _set_unresolved_turns(soft_state, 0)
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