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


