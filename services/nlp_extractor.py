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

import services.config as config
from .dynamic_config import get_thresholds

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

# ----------------------- Dynamic vocabulary loading -----------------

KNOWN_CITIES: Set[str] = set()        # canonical (normalized) city names
CITY_ALIASES: Dict[str, str] = {}     # alias -> canonical city
DATASET_AMENITIES: Set[str] = set()
PROPERTY_TYPES: list[str] = []
AMENITY_KEYWORDS: Dict[str, List[str]] = {}

_vocab_loaded = False


def _nlp_thresholds():
    return get_thresholds().nlp


def _add_city_alias(alias: str, canonical: str) -> None:
    alias_n = _norm(alias)
    if alias_n and canonical:
        CITY_ALIASES[alias_n] = canonical


def _ensure_vocab_loaded() -> None:
    """Lazy-load dataset and config values to avoid import-time loops."""
    global KNOWN_CITIES, CITY_ALIASES, DATASET_AMENITIES, PROPERTY_TYPES, AMENITY_KEYWORDS, _vocab_loaded
    if _vocab_loaded:
        return

    # 1. Start with config defaults
    KNOWN_CITIES = set(config.FALLBACK_CITIES)
    CITY_ALIASES = dict(config.FALLBACK_CITY_ALIASES)
    PROPERTY_TYPES = sorted(config.SEED_PROPERTY_TYPES)
    AMENITY_KEYWORDS = {k: list(v) for k, v in config.BASE_AMENITY_SYNONYMS.items()}
    dataset_property_types = set(PROPERTY_TYPES)

    # 2. Add aliases for fallback cities
    for c in config.FALLBACK_CITIES:
        _add_city_alias(f"{c} city", c)
        init = _initials(c)
        if init:
            _add_city_alias(init, c)

    # 3. Load from dataset if available
    if DATASET_PATH and Path(DATASET_PATH).exists():
        try:
            with open(DATASET_PATH, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # City
                    city_raw = (row.get("city") or "").strip()
                    if city_raw:
                        c_norm = _norm(city_raw)
                        KNOWN_CITIES.add(c_norm)
                        _add_city_alias(f"{c_norm} city", c_norm)
                        init = _initials(c_norm)
                        if init:
                            _add_city_alias(init, c_norm)

                    # Amenities
                    am_raw = (row.get("amenities") or "")
                    if am_raw:
                        for a in _split_amenities(am_raw):
                            a_n = _norm(a)
                            if a_n:
                                DATASET_AMENITIES.add(a_n)

                    # Property Type
                    pt = _norm(row.get("property_type", ""))
                    if pt and len(pt) >= 3:
                        dataset_property_types.add(pt)

            PROPERTY_TYPES = sorted(dataset_property_types)

            # Update amenity keywords with dataset amenities
            for a in sorted(DATASET_AMENITIES):
                if a and not any(a in syns for syns in AMENITY_KEYWORDS.values()):
                    AMENITY_KEYWORDS.setdefault(a, [a])

            print(f"[nlp_extractor] loaded {len(KNOWN_CITIES)} cities, {len(DATASET_AMENITIES)} amenities from {DATASET_PATH}")
        except Exception as e:
            print(f"[nlp_extractor] WARN: could not load vocab from dataset: {e}")

    _vocab_loaded = True


# ----------------------- Helpers ------------------------------------

def _fuzzy_property_type(text: str) -> Optional[str]:
    cutoff = _nlp_thresholds().fuzzy_match_low
    tokens = re.findall(r"[a-zA-Z]+", text)
    candidates = set(tokens)
    for tok in candidates:
        tok_n = _norm(tok)
        # Ignore extremely short words like 'a', 'in', 'is' to prevent false positive substring matches
        if not tok_n or len(tok_n) < 4:
            continue
        for p in PROPERTY_TYPES:
            # Only match if the property type is in the user's word (e.g. 'townhouse' in 'townhouses')
            if p in tok_n:
                return p
        match = difflib.get_close_matches(tok_n, PROPERTY_TYPES, n=1, cutoff=cutoff)
        if match:
            return match[0]
    return None

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
    thresholds = _nlp_thresholds()
    best: Tuple[float, Optional[str]] = (0.0, None)
    for g in grams_sorted:
        if " " not in g:
            continue
        g_norm = _norm(g)
        g_tokens = g_norm.split()
        # broader candidate pool; we'll apply our own stricter acceptance below
        matches = difflib.get_close_matches(g_norm, list(KNOWN_CITIES), n=3, cutoff=thresholds.fuzzy_match_medium)
        if not matches:
            continue
        for cand in matches:
            cand_tokens = cand.split()
            # dynamic acceptance: if first tokens match, allow slightly lower ratio
            tokens_share_first = len(g_tokens) > 0 and len(cand_tokens) > 0 and g_tokens[0] == cand_tokens[0]
            required_ratio = thresholds.fuzzy_match_strict if not tokens_share_first else thresholds.fuzzy_match_high
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
    _ensure_vocab_loaded()
    txt_norm = _norm(user_text)
    filters = (existing_filters.copy() if existing_filters else {})

    # ---- City / Location (robust) ----
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
    Extract a property type using dynamically loaded config and dataset vocabulary.
    """
    _ensure_vocab_loaded()
    t = _norm(user_text)
    for p in PROPERTY_TYPES:
        if p in t:
            return p
    return _fuzzy_property_type(t)
