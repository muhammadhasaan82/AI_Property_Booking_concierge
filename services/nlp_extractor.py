# services/nlp_extractor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract filters (city, budget, beds, amenities, property_type) from natural language.

City extraction uses word-boundary 1–3 gram matching + strict fuzzy on n-grams
(>0.93) to avoid false positives like selecting 'newport news' when the user
typed 'new york'.
"""

from __future__ import annotations
import os
import re
import csv
import difflib
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple

from .config import (
    FALLBACK_CITIES as _CFG_FALLBACK_CITIES,
    FALLBACK_CITY_ALIASES as _CFG_FALLBACK_ALIASES,
    SEED_PROPERTY_TYPES as _CFG_SEED_PROPERTY_TYPES,
    BASE_AMENITY_SYNONYMS as _CFG_BASE_AMENITY_SYNONYMS,
)

# ------------------------------- Utils -------------------------------

def _norm(s: str) -> str:
    """Lowercase + collapse internal whitespace."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _nospace(s: str) -> str:
    return (s or "").replace(" ", "").lower().strip()

def _split_amenities(val: str) -> List[str]:
    if not val:
        return []
    sep = ";" if ";" in val else ","
    return [a.strip() for a in val.split(sep) if a.strip()]

def _initials(city_name: str) -> Optional[str]:
    parts = [p for p in re.split(r"[\s\-]+", city_name.strip()) if p]
    if len(parts) >= 2:
        return "".join(p[0] for p in parts).lower()
    return None

def _word_tokens(text: str) -> List[str]:
    # alphabetic+numeric words (avoid punctuation)
    return re.findall(r"[a-z0-9]+", text.lower())

def _ngrams(tokens: List[str], n_min: int = 1, n_max: int = 3) -> List[str]:
    out: List[str] = []
    L = len(tokens)
    for n in range(n_min, n_max + 1):
        for i in range(0, L - n + 1):
            out.append(" ".join(tokens[i:i+n]))
    return out

# ----------------------- Find dataset.csv ----------------------------

def _discover_dataset_path() -> Optional[str]:
    env_path = os.getenv("DATASET_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    here = Path(__file__).parent
    candidate = here / "dataset.csv"
    if candidate.exists():
        return str(candidate)

    p = here
    for _ in range(3):
        p = p.parent
        cand = p / "services" / "dataset.csv"
        if cand.exists():
            return str(cand)
    return None

DATASET_PATH = _discover_dataset_path()

# ----------------------- Dynamic vocabulary -------------------------

KNOWN_CITIES: Set[str] = set()        # canonical (normalized) city names
CITY_ALIASES: Dict[str, str] = {}     # alias -> canonical city
DATASET_AMENITIES: Set[str] = set()

def _add_city_alias(alias: str, canonical: str) -> None:
    alias_n = _norm(alias)
    if alias_n and canonical:
        CITY_ALIASES[alias_n] = canonical

def _load_from_dataset() -> None:
    global KNOWN_CITIES, CITY_ALIASES, DATASET_AMENITIES
    if not DATASET_PATH or not Path(DATASET_PATH).exists():
        print(f"[nlp_extractor] WARN: dataset.csv not found. Set DATASET_PATH or place dataset.csv in services/.")
        return
    try:
        with open(DATASET_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                city_raw = (row.get("city") or "").strip()
                if city_raw:
                    KNOWN_CITIES.add(_norm(city_raw))
                am_raw = (row.get("amenities") or "")
                if am_raw:
                    for a in _split_amenities(am_raw):
                        a_n = _norm(a)
                        if a_n:
                            DATASET_AMENITIES.add(a_n)

        # Aliases (from dataset cities)
        for c in list(KNOWN_CITIES):
            _add_city_alias(f"{c} city", c)   # e.g., "new york city"
            init = _initials(c)
            if init:
                _add_city_alias(init, c)

        # ---- Fallback: add common cities even if dataset lacks them ----
        # Merge fallback into vocab
        KNOWN_CITIES.update(_CFG_FALLBACK_CITIES)
        for c in _CFG_FALLBACK_CITIES:
            _add_city_alias(f"{c} city", c)
            init = _initials(c)
            if init:
                _add_city_alias(init, c)
        for k, v in _CFG_FALLBACK_ALIASES.items():
            # don't override explicit dataset-derived aliases
            if k not in CITY_ALIASES:
                CITY_ALIASES[k] = v

        print(f"[nlp_extractor] loaded {len(KNOWN_CITIES)} cities, {len(DATASET_AMENITIES)} amenities from {DATASET_PATH}")
    except Exception as e:
        print(f"[nlp_extractor] WARN: could not load vocab from dataset: {e}")

_load_from_dataset()

# ----------------------- Amenity vocabulary -------------------------

AMENITY_KEYWORDS: Dict[str, List[str]] = {k: list(v) for k, v in _CFG_BASE_AMENITY_SYNONYMS.items()}
for a in sorted(DATASET_AMENITIES):
    if a and not any(a in syns for syns in AMENITY_KEYWORDS.values()):
        AMENITY_KEYWORDS.setdefault(a, [a])

# ----------------------- Property types (dataset-driven) ----------------

# Seed types that are always recognized (from central config)
_SEED_PROPERTY_TYPES = _CFG_SEED_PROPERTY_TYPES

# Augment from dataset if available
PROPERTY_TYPES: list[str] = sorted(_SEED_PROPERTY_TYPES)

def _load_property_types_from_dataset() -> None:
    """Dynamically add property types discovered in the dataset."""
    global PROPERTY_TYPES
    if not DATASET_PATH or not Path(DATASET_PATH).exists():
        return
    try:
        found: set[str] = set(_SEED_PROPERTY_TYPES)
        with open(DATASET_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pt = _norm(row.get("property_type", ""))
                if pt and len(pt) >= 3:
                    found.add(pt)
        PROPERTY_TYPES = sorted(found)
    except Exception:
        pass

_load_property_types_from_dataset()

def _fuzzy_property_type(text: str) -> Optional[str]:
    tokens = re.findall(r"[a-zA-Z]+", text)
    candidates = set(tokens + text.split())
    for tok in candidates:
        tok_n = _norm(tok)
        if not tok_n:
            continue
        for p in PROPERTY_TYPES:
            if p in tok_n or tok_n in p:
                return p
        match = difflib.get_close_matches(tok_n, PROPERTY_TYPES, n=1, cutoff=0.78)
        if match:
            return match[0]
    return None

# ----------------------- City detection (NEW) -----------------------

def _detect_city(user_text: str) -> Optional[str]:
    """
    Detect city using:
      1) Alias match with word boundaries (and nospace variants)
      2) Exact 1–3 gram match against KNOWN_CITIES
      3) Strict fuzzy (>=0.93) on 1–3 gram candidates only
    """
    t = _norm(user_text)
    tokens = _word_tokens(t)
    grams = _ngrams(tokens, 1, 3)

    # Build fresh maps each call to reflect dynamic vocab updates
    cities_ns = { _nospace(c): c for c in KNOWN_CITIES }
    aliases_ns = { _nospace(a): v for a, v in CITY_ALIASES.items() }

    # 1) alias (word-boundary)
    for alias, canonical in CITY_ALIASES.items():
        # \b handles multiword alias segments word-by-word
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return canonical
    # alias nospace (e.g., "newyork")
    t_ns = _nospace(user_text)
    for alias_ns, canonical in aliases_ns.items():
        # avoid short aliases (e.g., 'nn', 'la', 'ny') causing substring hits like 'innew'
        if not alias_ns or len(alias_ns) < 4:
            continue
        if alias_ns in t_ns:
            return canonical

    # 2) exact n-gram -> city (prefer longer n-grams first)
    # Sort by number of words (tokens) in the gram, not character length
    grams_sorted = sorted(set(grams), key=lambda x: len(x.split()), reverse=True)
    for g in grams_sorted:
        g_norm = _norm(g)
        if g_norm in KNOWN_CITIES:
            return g_norm
        # nospace equality
        g_ns = _nospace(g_norm)
        if g_ns in cities_ns:
            return cities_ns[g_ns]

    # 3) fuzzy on MULTIWORD grams only (avoid 1-token false positives)
    best: Tuple[float, Optional[str]] = (0.0, None)
    for g in grams_sorted:
        if " " not in g:
            continue
        g_norm = _norm(g)
        g_tokens = g_norm.split()
        # broader candidate pool; we'll apply our own stricter acceptance below
        matches = difflib.get_close_matches(g_norm, list(KNOWN_CITIES), n=3, cutoff=0.88)
        if not matches:
            continue
        for cand in matches:
            cand_tokens = cand.split()
            # dynamic acceptance: if first tokens match, allow slightly lower ratio
            tokens_share_first = len(g_tokens) > 0 and len(cand_tokens) > 0 and g_tokens[0] == cand_tokens[0]
            required_ratio = 0.94 if not tokens_share_first else 0.90
            # require that at least half the tokens overlap (rounded up)
            overlap = len(set(g_tokens) & set(cand_tokens))
            min_overlap = max(1, (len(g_tokens) + 1) // 2)
            if overlap < min_overlap:
                continue
            score = difflib.SequenceMatcher(None, g_norm, cand).ratio()
            if score >= required_ratio and score > best[0]:
                best = (score, cand)
    if best[1]:
        return best[1]

    return None

# ----------------------- Main extractors ----------------------------

def extract_filters(user_text: str, existing_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract city, budget, beds, and amenities from user text.
    """
    txt_norm = _norm(user_text)
    filters = (existing_filters.copy() if existing_filters else {})

    # ---- City / Location (robust) ----
    # If a city is mentioned in the current user text, use it (override stale session city)
    new_city = _detect_city(user_text)
    if new_city:
        filters["location"] = new_city
        filters["city"] = new_city

    # ---- Budget ----
    if not filters.get("budget"):
        budget_patterns = [
            r'(?:under|below|less than|up to|max|maximum|budget of?)\s*[\$€£]?\s*(\d+(?:\.\d+)?)',
            r'[\$€£]\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:dollars?|euros?|pounds?|bucks?|per night|/night|nightly)',
        ]
        for pattern in budget_patterns:
            m = re.search(pattern, txt_norm)
            if m:
                try:
                    filters["budget"] = float(m.group(1))
                    break
                except Exception:
                    pass

    # ---- Beds / Bedrooms ----
    if not filters.get("beds"):
        m_num = re.search(r'(\d+)\s*(?:bed(?:room)?s?|br)\b', txt_norm)
        if m_num:
            try:
                filters["beds"] = int(m_num.group(1))
            except Exception:
                pass
        else:
            m_word = re.search(r'\b(one|two|three|four|five)\s*(?:bed(?:room)?s?)\b', txt_norm)
            if m_word:
                word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
                filters["beds"] = word_to_num.get(m_word.group(1))

    # ---- Amenities ----
    found_amenities: Set[str] = set()
    # match with word boundaries for multiword tokens as well
    for key, keywords in AMENITY_KEYWORDS.items():
        for kw in keywords:
            if not kw:
                continue
            kw_norm = _norm(kw)
            if re.search(rf"\b{re.escape(kw_norm)}\b", txt_norm):
                found_amenities.add(key)
                break
    if found_amenities:
        existing = filters.get("amenities", []) or []
        filters["amenities"] = list(dict.fromkeys([*(a.lower() for a in existing), *sorted(found_amenities)]))

    return filters

def extract_property_type(user_text: str) -> Optional[str]:
    """
    Extract (with typo tolerance) a property type: condo, loft, apartment, house, studio, villa, townhouse.
    """
    t = _norm(user_text)
    for p in PROPERTY_TYPES:
        if p in t:
            return p
    return _fuzzy_property_type(t)
