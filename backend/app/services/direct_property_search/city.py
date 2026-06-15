from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.agents.tools.search import get_all_available_cities
from app.config.service_coverage_loader import detect_region_in_message
from app.services.dynamic_config import get_thresholds, get_vocabulary

logger = logging.getLogger(__name__)

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

def _supported_city_lookup() -> Dict[str, str]:
    return {
        _normalize(city): city.strip()
        for city in _city_terms()
        if isinstance(city, str) and city.strip()
    }

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

from app.services.direct_property_search.extraction import _property_type_terms   
