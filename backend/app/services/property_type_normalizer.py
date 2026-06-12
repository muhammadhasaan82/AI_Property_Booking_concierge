"""
Resolves raw user-supplied property type strings (including aliases and
plurals) to canonical keys defined in config/property_type_taxonomy.yaml.
 
Usage:
    from app.services.property_type_normalizer import normalize_property_type
 
    normalize_property_type("apartments")   
    normalize_property_type("Flat")         
    normalize_property_type("villa")        
    normalize_property_type("warehouse")    
    normalize_property_type(None)           
"""
from __future__ import annotations
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional, Set

import yaml

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "config" / "property_type_taxonomy.yaml"

def _load_alias_map() -> Dict[str, str]:
    """Build {alias_lower: canonical_key} lookup table from YAML."""
    alias_map: Dict[str, str] = {}
    try:
        with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}
        for canonical, spec in (raw.get("property_types") or {}).items():
            canonical_key = canonical.strip().lower()
            for alias in (spec.get("aliases") or []):
                alias_map[alias.strip().lower()] = canonical_key
    except Exception as exc:
        logger.warning("property_type_normalizer: failed to load taxonomy: %s", exc)
    return alias_map

def _get_pass_through_policy() -> str:
    try:
        with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}
        return str(raw.get("unknown_type_policy", "pass_through"))
    except Exception:
        return "pass_through"

_ALIAS_MAP: Dict[str, str] = _load_alias_map()
_PASS_THROUGH_POLICY: str = _get_pass_through_policy()
_CANONICAL_KEYS: Set[str] = set(_ALIAS_MAP.values())

def _type_similarity(left: str, right: str) -> float:
    normalized_left = left.strip().lower()
    normalized_right = right.strip().lower()
    if not normalized_left or not normalized_right:
        return 0.0
    spaced = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    compact_left = normalized_left.replace(" ", "")
    compact_right = normalized_right.replace(" ", "")
    compact = SequenceMatcher(None, compact_left, compact_right).ratio()
    return max(spaced, compact)


def _fuzzy_match_threshold(candidate: str, alias: str) -> float:
    try:
        from app.services.dynamic_config import get_thresholds

        thresholds = get_thresholds().nlp
        if " " in alias or len(alias.split()) > 1:
            return float(thresholds.fuzzy_match_medium)
        if len(candidate) <= 4:
            return float(thresholds.fuzzy_match_medium)
        return float(thresholds.fuzzy_match_high)
    except Exception:
        return 0.88


def fuzzy_resolve_property_type(raw: Optional[str]) -> Optional[str]:
    """Resolve a single token/phrase to a canonical property type via fuzzy matching."""
    if not raw or not str(raw).strip():
        return None

    key = re.sub(r"\s+", " ", str(raw).strip().lower())
    if len(key) < 3:
        return None

    exact = _ALIAS_MAP.get(key)
    if exact:
        return exact
    if key in _CANONICAL_KEYS:
        return key

    best_score = 0.0
    best_canonical: Optional[str] = None
    second_best_score = 0.0
    second_best_canonical: Optional[str] = None

    candidates: Set[str] = {key}
    if key.endswith("s") and len(key) > 3:
        candidates.add(key[:-1])

    for candidate in candidates:
        for alias, canonical in _ALIAS_MAP.items():
            threshold = _fuzzy_match_threshold(candidate, alias)
            score = max(
                _type_similarity(candidate, alias),
                _type_similarity(candidate, canonical),
            )
            if score < threshold:
                continue
            if score > best_score:
                second_best_score = best_score
                second_best_canonical = best_canonical
                best_score = score
                best_canonical = canonical
            elif score > second_best_score and canonical != best_canonical:
                second_best_score = score
                second_best_canonical = canonical

    if not best_canonical:
        return None

    try:
        from app.services.dynamic_config import get_thresholds

        high = float(get_thresholds().nlp.fuzzy_match_high)
        medium = float(get_thresholds().nlp.fuzzy_match_medium)
        margin = max(high - medium, 0.03)
    except Exception:
        margin = 0.03

    if (
        second_best_canonical
        and second_best_canonical != best_canonical
        and second_best_score >= best_score - margin
    ):
        return None

    return best_canonical


def normalize_property_type(raw: Optional[str]) -> Optional[str]:
    """Return the canonical property type key, or None if unknown/empty.
 
    None means "no type filter" — callers must treat it as "all types allowed".
    Never returns an alias or plural form.
    """
    if not raw or not raw.strip():
        return None
    key = raw.strip().lower()
    canonical = _ALIAS_MAP.get(key)
    if canonical:
        return canonical
    if key in _CANONICAL_KEYS:
        return key
    if _PASS_THROUGH_POLICY == "strict":
        logger.debug("property_type_normalizer: unknown type '%s' (policy=strict → None)", raw)
        return None
    logger.debug("property_type_normalizer: unknown type '%s' (policy=pass_through → raw)", raw)
    return key

def get_all_canonical_types() -> Set[str]:
    """Return all known canonical property type keys."""
    return set(_CANONICAL_KEYS)

def reload() -> None:
    """Hot-reload the taxonomy without restarting the server."""
    global _ALIAS_MAP, _PASS_THROUGH_POLICY, _CANONICAL_KEYS
    _ALIAS_MAP = _load_alias_map()
    _PASS_THROUGH_POLICY = _get_pass_through_policy()
    _CANONICAL_KEYS = set(_ALIAS_MAP.values())
    logger.info("property_type_normalizer: taxonomy reloaded (%d aliases)", len(_ALIAS_MAP))


