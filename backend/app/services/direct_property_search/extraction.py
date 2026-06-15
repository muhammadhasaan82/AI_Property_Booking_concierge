from __future__ import annotations
from app.services.direct_property_search.city import _city_match_threshold

import logging
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import yaml

from app.services.direct_property_search.constants import _TAXONOMY_PATH
from app.services.direct_property_search.city import _normalize, _contains_term
from app.services.dynamic_config import get_thresholds, get_vocabulary
from app.services.property_type_normalizer import (
    fuzzy_resolve_property_type,
    normalize_property_type,
)

logger = logging.getLogger(__name__)

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

