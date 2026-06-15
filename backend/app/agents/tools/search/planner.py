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

from app.agents.tools.search.constants import PROPERTY_RERANK_LIMIT, PROPERTY_RERANK_TIMEOUT_SECONDS
from app.agents.tools.search.normalize import _build_vibe_query, _normalize_search_value

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

