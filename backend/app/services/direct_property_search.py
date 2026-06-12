"""
Deterministic pre-ADK property search for clear search intents.

Detection terms (cities, property types, search phrases) are loaded from YAML
config and dataset helpers — no hardcoded city/type lists in Python.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import yaml

from app.agents.tools.search import get_all_available_cities, search_properties
from app.components.nlp_engine import (
    is_property_search,
    is_status_query,
    wants_property_search_request,
)
from app.config.conversation_shortcuts_loader import match_shortcut
from app.config.service_coverage_loader import (
    check_region_supported,
    detect_region_in_message,
)
from app.services.dynamic_config import get_vocabulary
from app.services.dynamic_config import get_thresholds
from app.services.property_query_constraints import (
    PropertySearchQuery,
    extract_property_search_query,
)
from app.services.property_type_normalizer import (
    fuzzy_resolve_property_type,
    normalize_property_type,
)
from app.services.booking_flow import has_active_booking_session
from app.services.faq_interruption import detect_policy_question
from app.services.redis_store import get_session_snapshot, save_session_snapshot
from app.services.observability.langfuse_observer import get_observer, sanitize_for_observability

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "property_type_taxonomy.yaml"
)


@dataclass(frozen=True)
class SupportedCityMatch:
    city: Optional[str]
    status: Literal["exact", "alias", "fuzzy", "ambiguous", "missing"]
    confidence: float = 0.0
    raw_candidate: Optional[str] = None
    suggestions: Tuple[str, ...] = ()


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _contains_term(normalized_message: str, term: str) -> bool:
    needle = _normalize(term)
    if not needle:
        return False
    if " " in needle:
        return needle in normalized_message
    return bool(re.search(rf"\b{re.escape(needle)}\b", normalized_message))


def _normalize_compact(text: str) -> str:
    return _normalize(text).replace(" ", "")


def _city_token_set(text: str) -> set[str]:
    return {token for token in _normalize(text).split(" ") if token}


def _city_similarity(left: str, right: str) -> float:
    normalized_left = _normalize(left)
    normalized_right = _normalize(right)
    if not normalized_left or not normalized_right:
        return 0.0
    spaced = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    compact = SequenceMatcher(
        None,
        _normalize_compact(normalized_left),
        _normalize_compact(normalized_right),
    ).ratio()
    return max(spaced, compact)


def _city_match_threshold(candidate: str, supported_city: str) -> float:
    thresholds = get_thresholds().nlp
    candidate_tokens = _city_token_set(candidate)
    city_tokens = _city_token_set(supported_city)
    shared_tokens = candidate_tokens & city_tokens
    if len(city_tokens) > 1 and shared_tokens:
        return float(thresholds.fuzzy_match_low)
    if len(candidate_tokens) > 1 and shared_tokens:
        return float(thresholds.fuzzy_match_medium)
    return float(thresholds.fuzzy_match_high)


def _city_match_margin() -> float:
    high = float(get_thresholds().nlp.fuzzy_match_high)
    medium = float(get_thresholds().nlp.fuzzy_match_medium)
    return max(high - medium, 0.03)


@lru_cache(maxsize=1)
def _property_type_terms() -> Tuple[Tuple[str, str], ...]:
    terms: List[Tuple[str, str]] = []
    try:
        with open(_TAXONOMY_PATH, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        for canonical, spec in (raw.get("property_types") or {}).items():
            canonical_key = str(canonical).strip().lower()
            if not canonical_key:
                continue
            terms.append((canonical_key, canonical_key))
            for alias in spec.get("aliases") or []:
                alias_key = str(alias).strip().lower()
                if alias_key:
                    terms.append((alias_key, canonical_key))
    except Exception as exc:
        logger.warning("[direct_search] taxonomy load failed: %s", exc)

    vocab = get_vocabulary()
    for seed in vocab.seed_property_types:
        seed_key = str(seed).strip().lower()
        if seed_key:
            canonical = normalize_property_type(seed_key) or seed_key
            terms.append((seed_key, canonical))

    deduped = sorted(set(terms), key=lambda item: len(item[0]), reverse=True)
    return tuple(deduped)


@lru_cache(maxsize=1)
def _city_terms() -> Tuple[str, ...]:
    cities: set[str] = set()
    dataset_loaded = False
    try:
        payload = get_all_available_cities()
        for city in payload.get("cities") or []:
            if isinstance(city, str) and city.strip():
                cities.add(city.strip())
                dataset_loaded = True
    except Exception as exc:
        logger.debug("[direct_search] dataset cities unavailable: %s", exc)

    if not dataset_loaded:
        vocab = get_vocabulary()
        for city in vocab.fallback_cities:
            if isinstance(city, str) and city.strip():
                cities.add(city.strip())
        for canonical in (vocab.city_aliases or {}).values():
            if isinstance(canonical, str) and canonical.strip():
                cities.add(canonical.strip())

    return tuple(sorted(cities, key=len, reverse=True))


@lru_cache(maxsize=1)
def _supported_city_lookup() -> Dict[str, str]:
    return {
        _normalize(city): city.strip()
        for city in _city_terms()
        if isinstance(city, str) and city.strip()
    }


@lru_cache(maxsize=1)
def _city_alias_lookup() -> Dict[str, str]:
    supported = _supported_city_lookup()
    aliases: Dict[str, str] = {}
    vocab = get_vocabulary()
    for alias, canonical in (vocab.city_aliases or {}).items():
        alias_key = _normalize(alias)
        canonical_key = _normalize(canonical)
        if not alias_key or not canonical_key:
            continue
        resolved = supported.get(canonical_key)
        if resolved:
            aliases[alias_key] = resolved
    return aliases


def _city_candidate_phrases(message: str) -> List[str]:
    normalized = _normalize(message)
    if not normalized:
        return []

    vocab = get_vocabulary().nlp_fallback
    prefix_candidates: List[str] = []
    suffix_candidates: List[str] = []
    candidates: List[str] = []
    prefix_pattern = str(vocab.city_candidate_prefix_pattern or "").strip()
    split_pattern = str(vocab.city_candidate_split_pattern or "").strip()
    block_words = {
        _normalize(word)
        for word in (vocab.city_candidate_block_words or [])
        if _normalize(word)
    }
    property_type_terms = {
        _normalize(term)
        for term, _canonical in _property_type_terms()
        if _normalize(term)
    }
    filler_terms = {
        _normalize(term)
        for term in (
            list(vocab.phrase_fillers or [])
            + ["in", "at", "near", "around", "for"]
        )
        if _normalize(term)
    }

    if prefix_pattern:
        try:
            for match in re.finditer(prefix_pattern, normalized):
                candidate = match.group(1) if match.groups() else match.group(0)
                if split_pattern:
                    candidate = re.split(split_pattern, candidate, maxsplit=1)[0]
                cleaned = _normalize(candidate)
                if cleaned:
                    prefix_candidates.append(cleaned)
        except re.error as exc:
            logger.warning("[direct_search] invalid city prefix pattern: %s", exc)

    for term in sorted(property_type_terms, key=len, reverse=True):
        pattern = rf"\b{re.escape(term)}\b(?P<suffix>.*)$"
        match = re.search(pattern, normalized)
        if not match:
            continue
        suffix = _normalize(match.group("suffix"))
        if split_pattern:
            suffix = _normalize(re.split(split_pattern, suffix, maxsplit=1)[0])
        suffix_tokens = [token for token in suffix.split(" ") if token and token not in filler_terms]
        if suffix_tokens:
            suffix_candidates.append(" ".join(suffix_tokens[:4]))

    focused_candidates = prefix_candidates + suffix_candidates
    if focused_candidates:
        candidates.extend(focused_candidates)

    tokens = [token for token in normalized.split(" ") if token]
    if not candidates:
        max_city_tokens = max(
            1,
            max((len(_normalize(city).split(" ")) for city in _city_terms()), default=1),
        )
        max_ngram = min(max_city_tokens + 1, max(len(tokens), 1))
        for size in range(1, max_ngram + 1):
            for start in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[start : start + size]).strip()
                if not phrase:
                    continue
                phrase_tokens = phrase.split(" ")
                if any(token in block_words or token in filler_terms for token in phrase_tokens):
                    continue
                if phrase in property_type_terms:
                    continue
                if not any(char.isalpha() for char in phrase):
                    continue
                candidates.append(phrase)

    deduped: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalize(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    deduped.sort(key=len, reverse=True)
    return deduped


def resolve_supported_city_from_message(message: str) -> SupportedCityMatch:
    normalized = _normalize(message)
    if not normalized:
        return SupportedCityMatch(city=None, status="missing")

    supported_lookup = _supported_city_lookup()
    alias_lookup = _city_alias_lookup()

    best_exact: Optional[Tuple[str, str]] = None
    for supported_key, canonical in supported_lookup.items():
        if _contains_term(normalized, supported_key):
            score = len(supported_key)
            if best_exact is None or score > len(best_exact[0]):
                best_exact = (supported_key, canonical)
    if best_exact is not None:
        return SupportedCityMatch(
            city=best_exact[1],
            status="exact",
            confidence=1.0,
            raw_candidate=best_exact[0],
        )

    best_alias: Optional[Tuple[str, str]] = None
    for alias_key, canonical in alias_lookup.items():
        if _contains_term(normalized, alias_key):
            score = len(alias_key)
            if best_alias is None or score > len(best_alias[0]):
                best_alias = (alias_key, canonical)
    if best_alias is not None:
        return SupportedCityMatch(
            city=best_alias[1],
            status="alias",
            confidence=1.0,
            raw_candidate=best_alias[0],
        )

    candidates = _city_candidate_phrases(message)
    if not candidates:
        return SupportedCityMatch(city=None, status="missing")

    scored: List[Tuple[float, str, str]] = []
    for candidate in candidates:
        for canonical in supported_lookup.values():
            score = _city_similarity(candidate, canonical)
            scored.append((score, candidate, canonical))
    if not scored:
        return SupportedCityMatch(city=None, status="missing", raw_candidate=candidates[0])

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate, best_city = scored[0]
    runner_up_score = next(
        (
            score
            for score, _candidate, city in scored[1:]
            if city != best_city
        ),
        0.0,
    )
    threshold = _city_match_threshold(best_candidate, best_city)
    margin = _city_match_margin()
    if best_score >= threshold and (
        best_score >= float(get_thresholds().nlp.fuzzy_match_strict)
        or (best_score - runner_up_score) >= margin
    ):
        return SupportedCityMatch(
            city=best_city,
            status="fuzzy",
            confidence=best_score,
            raw_candidate=best_candidate,
        )

    suggestions = tuple(
        city
        for _score, _candidate, city in scored
        if city != best_city and _score >= float(get_thresholds().nlp.fuzzy_match_low)
    )
    ordered_suggestions = []
    for city in (best_city, *suggestions):
        if city not in ordered_suggestions:
            ordered_suggestions.append(city)
        if len(ordered_suggestions) >= 3:
            break
    return SupportedCityMatch(
        city=None,
        status="ambiguous",
        confidence=best_score,
        raw_candidate=best_candidate,
        suggestions=tuple(ordered_suggestions),
    )


def extract_city_from_message(message: str) -> Optional[str]:
    """Return the best-matching configured city name from the message."""
    resolved = resolve_supported_city_from_message(message)
    return resolved.city


def _exact_supported_city_from_message(message: str) -> Optional[str]:
    normalized = _normalize(message)
    if not normalized:
        return None

    supported_lookup = _supported_city_lookup()
    alias_lookup = _city_alias_lookup()
    for supported_key, canonical in supported_lookup.items():
        if _contains_term(normalized, supported_key):
            return canonical
    for alias_key, canonical in alias_lookup.items():
        if _contains_term(normalized, alias_key):
            return canonical
    return None


def _property_type_block_words() -> frozenset[str]:
    vocab = get_vocabulary().nlp_fallback
    configured = {
        str(word).strip().lower()
        for word in (getattr(vocab, "property_type_block_words", None) or [])
        if str(word).strip()
    }
    defaults = {
        "looking", "search", "find", "show", "need", "want", "some", "with", "and",
        "for", "the", "an", "a", "in", "at", "of", "me", "am", "i", "my", "please",
        "bedroom", "bedrooms", "bed", "beds", "br", "bathroom", "bathrooms", "bath",
        "guest", "guests", "people", "person", "wifi", "parking", "pool", "gym",
        "pet", "friendly", "allowed", "under", "below", "above", "over", "city",
    }
    return frozenset(configured | defaults)


def _fuzzy_property_type_from_message(normalized_message: str) -> Optional[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z]+", normalized_message)
        if len(token) >= 3 and token not in _property_type_block_words()
    ]
    resolved: List[str] = []
    for token in tokens:
        canonical = fuzzy_resolve_property_type(token)
        if canonical:
            resolved.append(canonical)

    if not resolved:
        for term, canonical in _property_type_terms():
            if " " not in term or _contains_term(normalized_message, term):
                continue
            if term in normalized_message:
                resolved.append(normalize_property_type(term) or canonical)
                continue
            for token in tokens:
                if _type_similarity(token, term) >= _fuzzy_match_threshold(token, term):
                    resolved.append(normalize_property_type(term) or canonical)
                    break

    if not resolved:
        return None

    unique = sorted(set(resolved))
    if len(unique) > 1:
        return None
    return unique[0]


def _type_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _fuzzy_match_threshold(candidate: str, alias: str) -> float:
    return _city_match_threshold(candidate, alias)


def extract_property_type_from_message(message: str) -> Optional[str]:
    """Return canonical property type key when a configured alias appears."""
    normalized = _normalize(message)
    if not normalized:
        return None

    for term, canonical in _property_type_terms():
        if _contains_term(normalized, term):
            return normalize_property_type(term) or canonical
    return _fuzzy_property_type_from_message(normalized)


def extract_bedrooms_from_message(message: str) -> Optional[int]:
    """Extract exact bedroom count from message."""
    normalized = _normalize(message)
    match = re.search(r'\b(\d+)\s+(?:bedroom|bed|bd|br)\b', normalized)
    if match:
        return int(match.group(1))
    return None


def _canonical_property_type_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    return normalize_property_type(raw) or fuzzy_resolve_property_type(raw) or raw


def _constraint_values_from_query(
    query: Optional[PropertySearchQuery],
    *,
    property_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract structured constraint values from a PropertySearchQuery."""
    values: Dict[str, Any] = {
        "property_type": _canonical_property_type_key(property_type),
        "bedroom_exact": None,
        "bedroom_min": None,
        "bedroom_max": None,
        "bathroom_exact": None,
        "price_max": None,
        "occupancy_min": None,
        "amenities": [],
    }
    if query is not None and query.property_type:
        values["property_type"] = _canonical_property_type_key(query.property_type)
    if query is None:
        return values
    for c in query.constraints or []:
        if c.field == "bedrooms":
            try:
                num = int(c.value)
            except (TypeError, ValueError):
                continue
            if c.operator == "exact":
                values["bedroom_exact"] = num
            elif c.operator == "min":
                values["bedroom_min"] = num
            elif c.operator == "max":
                values["bedroom_max"] = num
        elif c.field == "bathrooms" and c.operator == "exact":
            try:
                values["bathroom_exact"] = int(c.value)
            except (TypeError, ValueError):
                continue
        elif c.field == "price_per_night" and c.operator == "max":
            try:
                values["price_max"] = float(c.value)
            except (TypeError, ValueError):
                continue
        elif c.field == "occupancy_max" and c.operator == "min":
            try:
                values["occupancy_min"] = int(c.value)
            except (TypeError, ValueError):
                continue
        elif c.field == "amenities":
            values["amenities"].append(str(c.value))
    return values


def _property_matches_constraints(prop: Any, constraints: Dict[str, Any]) -> bool:
    if not isinstance(prop, dict):
        return False
    property_type = constraints.get("property_type")
    bedroom_exact = constraints.get("bedroom_exact")
    bedroom_min = constraints.get("bedroom_min")
    bedroom_max = constraints.get("bedroom_max")
    bathroom_exact = constraints.get("bathroom_exact")
    price_max = constraints.get("price_max")
    occupancy_min = constraints.get("occupancy_min")
    amenities = constraints.get("amenities") or []

    if property_type:
        row_type = _canonical_property_type_key(prop.get("property_type"))
        if row_type != property_type:
            return False
    if bedroom_exact is not None:
        try:
            if int(prop.get("bedrooms") or 0) != int(bedroom_exact):
                return False
        except (TypeError, ValueError):
            return False
    if bedroom_min is not None:
        try:
            if int(prop.get("bedrooms") or 0) < int(bedroom_min):
                return False
        except (TypeError, ValueError):
            return False
    if bedroom_max is not None:
        try:
            if int(prop.get("bedrooms") or 0) > int(bedroom_max):
                return False
        except (TypeError, ValueError):
            return False
    if bathroom_exact is not None:
        try:
            if int(prop.get("bathrooms") or 0) != int(bathroom_exact):
                return False
        except (TypeError, ValueError):
            return False
    if price_max is not None:
        try:
            if float(prop.get("price_per_night") or 0) > float(price_max):
                return False
        except (TypeError, ValueError):
            return False
    if occupancy_min is not None:
        try:
            occupancy_value = prop.get("occupancy_max")
            if occupancy_value is None:
                bedrooms_value = prop.get("bedrooms")
                occupancy_value = int(bedrooms_value) * 2 if bedrooms_value is not None else 0
            if int(occupancy_value) < int(occupancy_min):
                return False
        except (TypeError, ValueError):
            return False
    if amenities:
        prop_amenities = {
            str(a).strip().lower()
            for a in (prop.get("amenities") or [])
            if a is not None
        }
        for required in amenities:
            token = str(required).strip().lower()
            if token and token not in prop_amenities:
                return False
    return True


def _has_active_constraints(constraints: Dict[str, Any]) -> bool:
    if not constraints:
        return False
    for key in (
        "property_type",
        "bedroom_exact",
        "bedroom_min",
        "bedroom_max",
        "bathroom_exact",
        "price_max",
        "occupancy_min",
    ):
        if constraints.get(key) is not None:
            return True
    if constraints.get("amenities"):
        return True
    return False


def _build_query_context(
    *,
    city: str,
    property_type: Optional[str],
    constraints: Dict[str, Any],
    price_max: Optional[float],
) -> Dict[str, Any]:
    qctx: Dict[str, Any] = {"city": city}
    if property_type:
        qctx["property_type"] = property_type
    bedrooms_exact = constraints.get("bedroom_exact")
    bedrooms_min = constraints.get("bedroom_min")
    bedrooms_max = constraints.get("bedroom_max")
    if bedrooms_exact is not None:
        qctx["bedrooms"] = bedrooms_exact
        qctx["bedrooms_operator"] = "exact"
    elif bedrooms_min is not None:
        qctx["bedrooms"] = bedrooms_min
        qctx["bedrooms_operator"] = "min"
    elif bedrooms_max is not None:
        qctx["bedrooms"] = bedrooms_max
        qctx["bedrooms_operator"] = "max"
    if constraints.get("bathroom_exact") is not None:
        qctx["bathrooms"] = constraints["bathroom_exact"]
        qctx["bathrooms_operator"] = "exact"
    if constraints.get("occupancy_min") is not None:
        qctx["guests"] = constraints["occupancy_min"]
    if constraints.get("amenities"):
        qctx["amenities"] = list(constraints["amenities"])
    if price_max is not None:
        qctx["budget"] = price_max
    return qctx


def _build_no_results_payload(
    *,
    city: str,
    property_type: Optional[str],
    constraints: Dict[str, Any],
    price_max: Optional[float],
) -> Dict[str, Any]:
    qctx = _build_query_context(
        city=city,
        property_type=property_type,
        constraints=constraints,
        price_max=price_max,
    )
    return {
        "status": "no_results",
        "city": city,
        "property_type": property_type,
        "bedrooms": constraints.get("bedroom_exact"),
        "bedrooms_operator": "exact" if constraints.get("bedroom_exact") is not None else None,
        "bathrooms": constraints.get("bathroom_exact"),
        "amenities": list(constraints.get("amenities") or []),
        "query_context": qctx,
        "filters_applied": dict(qctx),
    }


def _filter_properties_by_constraints(
    properties: Optional[List[Any]],
    constraints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not properties:
        return []
    if not _has_active_constraints(constraints):
        return [p for p in properties if isinstance(p, dict)]
    return [
        p
        for p in properties
        if isinstance(p, dict) and _property_matches_constraints(p, constraints)
    ]


def _build_option_map(properties: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    option_map: Dict[str, Dict[str, Any]] = {}
    for prop in properties:
        if not isinstance(prop, dict):
            continue
        number = prop.get("number")
        if number is None:
            continue
        prop_id = prop.get("id")
        if prop_id is None:
            continue
        option_map[str(number)] = {
            "property_id": str(prop_id),
            "title": prop.get("title"),
        }
    return option_map


def _renumber_properties(properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for idx, prop in enumerate(properties, start=1):
        if isinstance(prop, dict):
            prop["number"] = idx
    return properties


def _sort_properties_for_display(properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        properties,
        key=lambda p: (
            float(p.get("rating") or 0),
            int(p.get("reviews_count") or 0),
        ),
        reverse=True,
    )


def _has_search_phrase(message: str) -> bool:
    normalized = _normalize(message)
    if not normalized:
        return False
    nlp = get_vocabulary().nlp_fallback
    for phrase in nlp.search_phrases:
        if phrase and phrase.lower() in normalized:
            return True
    for signal in nlp.search_signals:
        if signal and _contains_term(normalized, signal):
            return True
    return False


def _city_clarification_reply(city_match: SupportedCityMatch) -> str:
    raw_candidate = (city_match.raw_candidate or "").strip()
    if raw_candidate:
        opening = (
            f"I couldn't confidently match '{raw_candidate}' to a supported city in our listings."
        )
    else:
        opening = "Which city should I search in?"

    suggestions = [city for city in city_match.suggestions if city]
    if suggestions:
        return f"{opening} Did you mean {', '.join(suggestions)}?"
    return f"{opening} Please share a supported city so I can search the real property dataset."


def is_clear_direct_property_search(message: str, soft_state: Optional[Dict[str, Any]]) -> bool:
    """
    True when the message is a new explicit property search that should bypass ADK.
    """
    if not message or not message.strip():
        return False
    if is_status_query(message):
        return False
    if not is_property_search(message):
        return False

    state = soft_state if isinstance(soft_state, dict) else {}
    if match_shortcut(message, state) is not None:
        return False
    if wants_property_search_request(message) and state.get("all_search_results"):
        return False

    city_match = resolve_supported_city_from_message(message)
    if not city_match.city:
        return False

    region = detect_region_in_message(message) or detect_region_in_message(city_match.city)
    if region:
        decision = check_region_supported(region)
        if decision.blocked:
            return False

    property_type = extract_property_type_from_message(message)
    if property_type or _has_search_phrase(message):
        return True
    return False


def extract_soft_state_from_snapshot(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    state = snapshot.get("state") or {}
    if not isinstance(state, dict):
        return {}
    if "soft_state" in state and isinstance(state["soft_state"], dict):
        return dict(state["soft_state"])
    return dict(state)


async def maybe_handle_direct_property_search(
    message: str,
    session_id: str,
    *,
    get_snapshot: Optional[Callable[[str], Any]] = None,
    save_snapshot: Optional[Callable[..., Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run search_properties for clear search intents and persist soft_state to Redis.
    Applies deterministic post-filtering for exact bedroom / price / occupancy
    / amenity constraints derived from the structured query.
    """
    if get_snapshot is None:
        get_snapshot = get_session_snapshot
    if save_snapshot is None:
        save_snapshot = save_session_snapshot

    snapshot = await get_snapshot(session_id)
    if not isinstance(snapshot, dict):
        snapshot = {}

    state = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else {}
    soft_state = state.get("soft_state") if isinstance(state.get("soft_state"), dict) else {}
    normalized_message = _normalize(message)
    if not normalized_message or is_status_query(message):
        return None

    if has_active_booking_session(soft_state):
        logger.info(
            "[direct_search.trace] skipped due to active booking session: stage=%r view=%r",
            soft_state.get("booking_stage"),
            soft_state.get("last_presented_view"),
        )
        return None

    if match_shortcut(message, soft_state) is not None:
        return None

    from app.services.constraint_extractor import (
        dynamic_constraints_from_soft_state,
        extract_dynamic_constraints,
    )

    active_search_context = bool(soft_state.get("all_search_results") or soft_state.get("last_filters"))
    session_constraints = dynamic_constraints_from_soft_state(soft_state) if active_search_context else None
    session_sort_preferences = list(soft_state.get("last_sort_preferences") or []) if active_search_context else []
    dynamic_constraints, plan = extract_dynamic_constraints(
        message,
        session_constraints=session_constraints,
        session_id=session_id,
    )
    property_type = extract_property_type_from_message(message)
    has_search_phrase = _has_search_phrase(message)
    has_new_signal = bool(plan.trace.extracted_constraints) or bool(plan.sort_preferences)
    if not (active_search_context and has_new_signal):
        try:
            if detect_policy_question(message):
                return None
        except Exception as exc:
            logger.debug("[direct_search] policy detection skipped after error: %s", exc)
    city_match = resolve_supported_city_from_message(message)
    exact_city = _exact_supported_city_from_message(message)
    logger.info(
        "[direct_search.trace] exact_city=%r city_resolution_status=%s canonical_city=%r "
        "active_search_context=%s extracted=%s merged=%s session_sort_preferences=%s sort_preferences=%s",
        exact_city,
        city_match.status,
        city_match.city,
        active_search_context,
        plan.trace.extracted_constraints,
        plan.trace.merged_constraints,
        session_sort_preferences,
        plan.sort_preferences,
    )

    city = dynamic_constraints.get("city") or city_match.city
    if not city and not active_search_context and not property_type and not has_search_phrase and not has_new_signal:
        return None

    if not city:
        return {
            "status": "needs_clarification",
            "missing": ["city"],
            "query_context": {
                "city": None,
                "property_type": dynamic_constraints.get("property_type") or property_type,
            },
            "deterministic_reply": _city_clarification_reply(city_match),
        }

    if not dynamic_constraints.has("city"):
        dynamic_constraints.set("city", city, "exact")

    property_type = _canonical_property_type_key(
        dynamic_constraints.get("property_type") or property_type
    )
    if property_type and not dynamic_constraints.has("property_type"):
        dynamic_constraints.set("property_type", property_type, "exact")

    tool_context = SimpleNamespace(state={"soft_state": dict(soft_state)})

    effective_sort_preferences = list(plan.sort_preferences or session_sort_preferences)

    search_kwargs: Dict[str, Any] = {
        "city": city,
        "property_type": property_type,
        "beds": dynamic_constraints.get("bedrooms"),
        "beds_operator": dynamic_constraints.get_operator("bedrooms", "exact"),
        "bathrooms": dynamic_constraints.get("bathrooms"),
        "bathrooms_operator": dynamic_constraints.get_operator("bathrooms", "exact"),
        "guests": dynamic_constraints.get("occupancy_max"),
        "guests_operator": dynamic_constraints.get_operator("occupancy_max", "min"),
        "budget": dynamic_constraints.get("price_per_night"),
        "amenities": ",".join(dynamic_constraints.get("amenities") or []),
        "sort_preferences": effective_sort_preferences,
        "search_path": "direct",
        "tool_context": tool_context,
    }

    try:
        payload = await search_properties(**search_kwargs)
    except Exception as exc:
        logger.warning("[direct_search] search_properties failed: %s", exc)
        return None

    new_soft_state = tool_context.state.get("soft_state")
    if not isinstance(new_soft_state, dict):
        new_soft_state = {}
    if not isinstance(payload, dict):
        return None

    state = dict(state)
    state["soft_state"] = new_soft_state

    meta = snapshot.get("meta") if isinstance(snapshot.get("meta"), dict) else {}
    await save_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=state,
        metadata={
            key: meta[key]
            for key in ("app_name", "user_id", "last_update_time")
            if key in meta
        },
    )
    logger.debug(
        "[direct_search] handled city=%r property_type=%r status=%s total_found=%s",
        city,
        property_type,
        payload.get("status"),
        payload.get("total_found"),
    )
    return payload
