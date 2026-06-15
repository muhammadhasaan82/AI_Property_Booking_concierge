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

