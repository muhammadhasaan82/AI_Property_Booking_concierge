from __future__ import annotations
import logging
import os
import re
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from app.services.dynamic_config import get_intent_catalog as _get_catalog
from app.services.dynamic_config import get_retrieval_config as _get_retrieval_config
from app.services.dynamic_config import get_thresholds as _get_thresholds
from app.services.dynamic_config import get_vocabulary as _get_vocabulary

logger = logging.getLogger(__name__)

from app.components.nlp_engine.config import _get_field_prototypes, _get_vocab, _name_full_pattern
from app.components.nlp_engine.models import _get_spacy, _get_st_model, _max_semantic_similarity
from app.components.nlp_engine.constants import ISO_DATE_PATTERN, UUID_PATTERN
import re
from typing import List, Optional

def extract_person_name(text: str) -> Optional[str]:
    if not text:
        return None

    thresholds = _get_nlp_thresholds()
    normalized = text.lower().strip()
    vocabulary = _get_vocab()

    if any(term in normalized for term in vocabulary.name_search_guards):
        return None

    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        for entity in doc.ents:
            if entity.label_ == "PERSON":
                candidate = entity.text.strip().rstrip(".").strip()
                if len(candidate) >= thresholds.name_min_length:
                    return candidate

    words = set(re.findall(r"[a-z]+", normalized))
    has_conversational_words = bool(words & set(vocabulary.name_conversational_guards))

    for pattern in vocabulary.name_explicit_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            candidate = match.group(1).strip().rstrip(".").strip()
            if (
                len(candidate.split()) <= thresholds.name_max_words
                and len(candidate) >= thresholds.name_min_length
                and not _looks_like_email_username(candidate)
            ):
                return candidate

    if not has_conversational_words:
        stripped = text.strip()
        word_count = len(stripped.split())
        pattern = _name_full_pattern(thresholds.name_pattern_max_chars)
        if 1 <= word_count <= thresholds.name_max_words:
            match = pattern.match(stripped)
            if match:
                candidate = match.group(1).strip().rstrip(".").strip()
                if len(candidate) >= thresholds.name_min_length and not _looks_like_email_username(candidate):
                    return candidate

    return None

def _looks_like_email_username(value: str) -> bool:
    if not value:
        return False

    common = set(_get_vocab().email_username_common)
    return (
        len(value) <= 3
        or value.lower() in common
        or bool(re.search(r"\d", value))
        or bool(re.search(r"[^a-zA-Z\s\-.'']", value))
    )

def extract_dates(text: str) -> List[str]:
    if not text:
        return []

    results = ISO_DATE_PATTERN.findall(text)
    if results:
        return results

    nlp = _get_spacy()
    if not nlp:
        return []

    output: List[str] = []
    doc = nlp(text)
    for entity in doc.ents:
        if entity.label_ == "DATE":
            output.append(entity.text)
    return output

def extract_cardinal(text: str) -> Optional[int]:
    if not text or not text.strip():
        return None

    normalized = text.strip().lower()
    if normalized.isdigit():
        value = int(normalized)
        return value if value >= 1 else None

    vocabulary = _get_vocab()

    for pattern in vocabulary.selection_patterns:
        try:
            match = re.search(pattern, normalized, re.I)
        except re.error:
            continue
        if not match:
            continue
        raw = next((group for group in match.groups() if group), "")
        if raw.isdigit():
            value = int(raw)
            if value >= 1:
                return value

    for token, value in vocabulary.selection_ordinals.items():
        if re.search(r"\b" + re.escape(str(token)) + r"\b", normalized):
            if value >= 1:
                return int(value)

    for token, value in vocabulary.selection_cardinals.items():
        if re.search(r"\b" + re.escape(str(token)) + r"\b", normalized):
            if value >= 1:
                return int(value)

    if vocabulary.selection_cardinal_context_pattern and vocabulary.selection_cardinals:
        alternatives = "|".join(re.escape(token) for token in vocabulary.selection_cardinals.keys())
        context_pattern = vocabulary.selection_cardinal_context_pattern.replace("{cardinals}", alternatives)
        try:
            match = re.search(context_pattern, normalized, re.I)
        except re.error:
            match = None
        if match:
            candidate = match.group(1).strip().lower()
            if candidate in vocabulary.selection_cardinals:
                value = int(vocabulary.selection_cardinals[candidate])
                if value >= 1:
                    return value

    nlp = _get_spacy()
    if nlp and vocabulary.selection_entity_labels:
        labels = set(vocabulary.selection_entity_labels)
        doc = nlp(normalized)
        for entity in doc.ents:
            if entity.label_ not in labels:
                continue
            token = entity.text.strip().lower()
            if token.isdigit() and int(token) >= 1:
                return int(token)
            if token in vocabulary.selection_ordinals and vocabulary.selection_ordinals[token] >= 1:
                return int(vocabulary.selection_ordinals[token])
            if token in vocabulary.selection_cardinals and vocabulary.selection_cardinals[token] >= 1:
                return int(vocabulary.selection_cardinals[token])

    return None

def has_cardinal_extraction(text: str) -> bool:
    return extract_cardinal(text) is not None

def is_low_semantic_density(text: str) -> bool:
    if not text or not text.strip():
        return True

    normalized = text.strip().lower()
    if has_cardinal_extraction(normalized) and not re.search(r"[a-z]", normalized):
        return True
    return False

def extract_guests(text: str) -> Optional[int]:
    if not text:
        return None

    normalized = text.lower().strip()
    unit_terms = _get_vocab().guest_unit_terms
    unit_pattern = "|".join(re.escape(unit) for unit in unit_terms)

    if unit_pattern:
        match = re.search(rf"(\d{{1,3}})\s*(?:{unit_pattern})?\b", normalized) or re.search(r"^(\d{1,3})$", normalized)
    else:
        match = re.search(r"(\d{1,3})\b", normalized) or re.search(r"^(\d{1,3})$", normalized)

    if not match:
        return None

    try:
        value = int(match.group(1))
    except (ValueError, IndexError):
        return None

    return value if 1 <= value <= 100 else None

def extract_phone(text: str) -> Optional[str]:
    if not text:
        return None

    match = re.search(r"(\+?[\d\s\-]{8,15}\d)", text)
    if not match:
        return None

    normalized = re.sub(r"[\s\-]", "", match.group(1))
    if ISO_DATE_PATTERN.search(text):
        return None
    if re.match(r"^\d{8}$", normalized):
        return None
    return normalized

def extract_email(text: str) -> Optional[str]:
    if not text:
        return None

    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None

def extract_booking_id(text: str) -> Optional[str]:
    if not text:
        return None

    normalized = text.lower()
    match = UUID_PATTERN.search(normalized)
    if match:
        return match.group(0)

    short_match = re.search(r"\b([0-9a-f]{8})\b", normalized)
    if short_match:
        return short_match.group(1)

    return None

def detect_requested_fields(text: str) -> List[str]:
    if not text:
        return []
    if is_low_semantic_density(text):
        return []

    normalized = text.lower()
    fields: List[str] = []

    model = _get_st_model()
    if model:
        try:
            import numpy as np

            semantic_threshold = _get_nlp_thresholds().field_detection_semantic_threshold
            text_embedding = model.encode([normalized], convert_to_numpy=True)[0]
            for field, phrases in _get_field_prototypes().items():
                if not phrases:
                    continue
                embeddings = model.encode(phrases, convert_to_numpy=True)
                mean_embedding = np.mean(embeddings, axis=0)
                similarity = float(
                    np.dot(text_embedding, mean_embedding)
                    / ((np.linalg.norm(text_embedding) * np.linalg.norm(mean_embedding)) + 1e-8)
                )
                if similarity >= semantic_threshold and field not in fields:
                    fields.append(field)
        except Exception:
            fields = []

    if fields:
        return fields

    fillers = set(_get_vocab().phrase_fillers)

    def phrase_hit(phrase: str) -> bool:
        clean = (phrase or "").strip().lower()
        if not clean:
            return False
        if clean in normalized:
            return True
        tokens = [token for token in re.findall(r"[a-z0-9_+-]+", clean) if token not in fillers]
        return bool(tokens) and all(token in normalized for token in tokens)

    for field, phrases in _get_field_prototypes().items():
        if any(phrase_hit(phrase) for phrase in phrases) and field not in fields:
            fields.append(field)

    return fields

