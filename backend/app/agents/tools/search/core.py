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

from app.agents.tools.search.cities import get_all_available_cities
from app.agents.tools.search.display import (
    _resolve_result_limit,
    _search_display_max_inline_results,
    _search_display_mode,
    _search_display_pagination_enabled,
    _sort_results_for_display,
    _uses_all_matching_display,
)
from app.agents.tools.search.normalize import (
    _dedupe_preserve_order,
    _known_amenities_from_dataset,
    _normalize_city_key,
    _normalize_search_value,
    _resolve_city_from_catalog,
    _split_amenities_by_known,
    _split_amenity_input,
)
from app.agents.tools.search.pagination import _build_search_page_payload, paginate_stored_results
from app.agents.tools.search.planner import (
    _apply_planner_to_results,
    _build_dynamic_constraints_from_inputs,
    _rerank_properties_by_vibe,
    _search_plan_from_constraints,
)
from app.agents.tools.search.selection import _resolve_property_id_from_selection

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
    from app.agents.tools.rust_client import search_properties as rust_search
    from app.components.search import property_search, _DATASET

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
        all_matching_display=_uses_all_matching_display(),
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
    from app.components.search import _DATASET
    from app.agents.resolvers.property_resolver import resolve_property_reference

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

from app.agents.tools.search.normalize import _build_vibe_query  # noqa: F401

from app.agents.tools.search.display import _resolve_page_size  # noqa: F401
