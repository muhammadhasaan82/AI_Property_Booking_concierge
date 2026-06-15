from __future__ import annotations
from app.components.nlp_engine.config import _get_nlp_thresholds

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

from app.components.nlp_engine.config import (
    _get_affirm_no_prototypes,
    _get_affirm_yes_prototypes,
    _get_intent_prototypes,
    _get_intent_threshold,
    _get_modification_prototypes,
    _get_property_search_request_prototypes,
    _get_receipt_request_prototypes,
    _get_resume_request_prototypes,
    _get_vocab,
)
from app.components.nlp_engine.models import (
    _get_intent_embeddings,
    _get_spacy,
    _get_st_model,
    _get_vader,
    _max_semantic_similarity,
)
from app.components.nlp_engine.constants import ISO_DATE_PATTERN, UUID_PATTERN
import re

def classify_affirmation(text: str) -> str:
    if not text or not text.strip():
        return "neutral"

    thresholds = _get_nlp_thresholds()
    normalized = text.strip().lower()

    yes_similarity = _max_semantic_similarity(normalized, _get_affirm_yes_prototypes())
    no_similarity = _max_semantic_similarity(normalized, _get_affirm_no_prototypes())

    if (
        yes_similarity >= thresholds.affirmation_semantic_threshold
        and yes_similarity >= (no_similarity + thresholds.affirmation_margin)
    ):
        return "yes"
    if (
        no_similarity >= thresholds.affirmation_semantic_threshold
        and no_similarity >= (yes_similarity + thresholds.affirmation_margin)
    ):
        return "no"

    vocabulary = _get_vocab()
    if normalized in set(vocabulary.affirm_yes_tokens):
        return "yes"
    if normalized in set(vocabulary.affirm_no_tokens):
        return "no"

    compound = _get_vader().polarity_scores(normalized).get("compound", 0.0)
    token_len = len(normalized.split())
    if compound >= thresholds.affirmation_compound_positive and token_len <= thresholds.affirmation_max_tokens:
        return "yes"
    if compound <= thresholds.affirmation_compound_negative and token_len <= thresholds.affirmation_max_tokens:
        return "no"

    return "neutral"

def is_greeting(text: str) -> bool:
    if not text or not text.strip():
        return False

    normalized = text.strip().lower()
    if is_acknowledgment(text):
        return False

    tokens = normalized.split()
    if len(tokens) > _get_nlp_thresholds().affirmation_max_tokens:
        return False

    vocabulary = _get_vocab()
    if tokens and tokens[0] in set(vocabulary.greeting_seeds):
        return True
    if any(re.search(r"\b" + re.escape(phrase) + r"\b", normalized) for phrase in vocabulary.greeting_phrases):
        return True

    intents = list(_get_catalog().intents.keys())
    if not intents:
        return False
    if classify_intent_sync(normalized, intents) != "greeting":
        return False

    return _semantic_confidence(normalized, "greeting") >= _get_intent_threshold("greeting")

def is_acknowledgment(text: str) -> bool:
    if not text or not text.strip():
        return False

    normalized = text.strip().lower()
    vocabulary = _get_vocab()
    if normalized in set(vocabulary.acknowledgment_tokens):
        return True
    return any(phrase in normalized for phrase in vocabulary.acknowledgment_phrases)

def is_handoff_request(text: str) -> bool:
    if not text:
        return False

    normalized = text.strip().lower()
    vocabulary = _get_vocab()
    if any(seed in normalized for seed in vocabulary.handoff_seeds):
        return True
    return any(phrase in normalized for phrase in vocabulary.handoff_phrases)

def is_availability_query(text: str) -> bool:
    if not text:
        return False
    normalized = text.strip().lower()
    return any(phrase in normalized for phrase in _get_vocab().availability_phrases)

def is_end_request(text: str) -> bool:
    if not text:
        return False

    normalized = text.strip().lower()
    vocabulary = _get_vocab()
    if normalized in set(vocabulary.end_exact):
        return True
    return any(phrase in normalized for phrase in vocabulary.end_phrases)

def is_status_query(text: str) -> bool:
    if not text:
        return False

    normalized = text.strip().lower()
    if UUID_PATTERN.search(normalized):
        return True

    intents = list(_get_catalog().intents.keys())
    if intents and classify_intent_sync(normalized, intents) == "status_update":
        confidence = _semantic_confidence(normalized, "status_update")
        if confidence >= _get_intent_threshold("status_update"):
            return True

    return any(seed in normalized for seed in _get_vocab().status_seeds)

def is_property_search(text: str) -> bool:
    if not text:
        return False

    normalized = text.strip().lower()
    if is_status_query(text):
        return False

    vocabulary = _get_vocab()
    if any(marker in normalized for marker in vocabulary.status_booking_id_markers):
        return False
    if UUID_PATTERN.search(normalized):
        return False

    intents = list(_get_catalog().intents.keys())
    if intents and classify_intent_sync(normalized, intents) == "property_search":
        confidence = _semantic_confidence(normalized, "property_search")
        if confidence >= _get_intent_threshold("property_search"):
            return True

    vocab_cfg = _get_vocabulary()
    property_types = set(vocab_cfg.seed_property_types)
    known_cities = set(vocab_cfg.fallback_cities)
    city_aliases = set(vocab_cfg.city_aliases.keys()) | set(vocab_cfg.city_aliases.values())

    money_pattern = None
    if vocabulary.money_intent_pattern:
        try:
            money_pattern = re.compile(vocabulary.money_intent_pattern, re.I)
        except re.error:
            money_pattern = None

    if any(re.search(r"\b" + re.escape(item) + r"\b", normalized) for item in property_types):
        return True
    if any(re.search(r"\b" + re.escape(item) + r"\b", normalized) for item in known_cities):
        return True
    if any(re.search(r"\b" + re.escape(item) + r"\b", normalized) for item in city_aliases):
        return True
    if money_pattern and money_pattern.search(normalized):
        return True
    if any(re.search(r"\b" + re.escape(item) + r"\b", normalized) for item in vocabulary.search_signals):
        return True
    if any(re.search(r"\b" + re.escape(item) + r"\b", normalized) for item in vocabulary.search_phrases):
        return True

    return False

def wants_modification(text: str) -> bool:
    if not text or not text.strip():
        return False

    normalized = text.strip().lower()
    threshold = _get_nlp_thresholds().modification_semantic_threshold
    if _max_semantic_similarity(normalized, _get_modification_prototypes()) >= threshold:
        return True

    return any(seed in normalized for seed in _get_vocab().modification_seeds)

def wants_property_search_request(text: str) -> bool:
    if not text or not text.strip():
        return False

    normalized = text.strip().lower()
    threshold = _get_nlp_thresholds().property_search_request_semantic_threshold
    if _max_semantic_similarity(normalized, _get_property_search_request_prototypes()) >= threshold:
        return True

    return any(seed in normalized for seed in _get_vocab().property_search_request_seeds)

def is_receipt_request(text: str) -> bool:
    if not text or not text.strip():
        return False

    normalized = text.strip().lower()
    threshold = _get_nlp_thresholds().receipt_semantic_threshold
    if _max_semantic_similarity(normalized, _get_receipt_request_prototypes()) >= threshold:
        return True

    vocabulary = _get_vocab()
    if any(seed in normalized for seed in vocabulary.receipt_seeds):
        return True
    if any(phrase in normalized for phrase in vocabulary.receipt_phrases):
        return True
    return (
        any(term in normalized for term in vocabulary.receipt_quantity_terms)
        and any(term in normalized for term in vocabulary.receipt_amount_terms)
    )

def is_resume_request(text: str) -> bool:
    if not text or not text.strip():
        return False

    normalized = text.strip().lower()
    threshold = _get_nlp_thresholds().resume_semantic_threshold
    if _max_semantic_similarity(normalized, _get_resume_request_prototypes()) >= threshold:
        return True

    vocabulary = _get_vocab()
    if normalized in set(vocabulary.resume_exact_phrases):
        return True
    return any(phrase in normalized for phrase in vocabulary.resume_phrases)

def wants_previous_results_sync(text: str) -> bool:
    if not text or not text.strip():
        return False

    previous_cfg = _get_catalog().previous_results_prototypes
    model = _get_st_model()
    if model is None or not previous_cfg.prototypes:
        normalized = text.lower()
        return any(keyword in normalized for keyword in previous_cfg.fallback_keywords)

    try:
        import numpy as np

        text_vec = model.encode([text], convert_to_numpy=True)[0]
        prototype_vectors = model.encode(previous_cfg.prototypes, convert_to_numpy=True)
        sims = np.dot(prototype_vectors, text_vec) / (
            (np.linalg.norm(prototype_vectors, axis=1) * np.linalg.norm(text_vec)) + 1e-8
        )
        configured = float(previous_cfg.threshold or 0.0)
        threshold = configured if configured > 0 else _get_nlp_thresholds().previous_results_semantic_threshold
        return float(np.max(sims)) >= threshold
    except Exception:
        normalized = text.lower()
        return any(keyword in normalized for keyword in previous_cfg.fallback_keywords)

def detect_faq_intent(text: str) -> bool:
    if not text or not text.strip():
        return False

    normalized = text.strip().lower()
    intents = list(_get_catalog().intents.keys())
    if intents and classify_intent_sync(normalized, intents) == "faq":
        if _semantic_confidence(normalized, "faq") >= _get_intent_threshold("faq"):
            return True

    vocabulary = _get_vocab()
    if any(re.search(r"\b" + re.escape(keyword) + r"\b", normalized) for keyword in vocabulary.faq_strong_keywords):
        return True

    has_faq_seed = any(re.search(r"\b" + re.escape(seed) + r"\b", normalized) for seed in vocabulary.faq_seeds)
    has_question = (
        "?" in normalized
        or any(normalized.startswith(starter) for starter in vocabulary.faq_question_starts)
        or any(cue in normalized for cue in vocabulary.faq_question_cues)
    )
    return has_faq_seed and has_question

def _semantic_confidence(text: str, intent: str) -> float:
    model = _get_st_model()
    if model is None:
        return 0.0

    embeddings = _get_intent_embeddings()
    if embeddings is None or intent not in embeddings:
        return 0.0

    try:
        import numpy as np

        text_vector = model.encode([text], convert_to_numpy=True)[0]
        intent_vector = embeddings[intent]
        similarity = float(
            np.dot(text_vector, intent_vector)
            / ((np.linalg.norm(text_vector) * np.linalg.norm(intent_vector)) + 1e-8)
        )
        return similarity
    except Exception:
        return 0.0

def classify_intent_sync(text: str, candidates: List[str]) -> str:
    if not candidates:
        return "other"

    model = _get_st_model()
    embeddings = _get_intent_embeddings()
    if model is None or embeddings is None:
        return _classify_intent_keyword_fallback(text, candidates)

    try:
        import numpy as np

        text_vector = model.encode([text], convert_to_numpy=True)[0]
        best_intent = candidates[-1]
        best_score = -1.0

        for intent in candidates:
            if intent not in embeddings:
                continue
            similarity = float(
                np.dot(text_vector, embeddings[intent])
                / ((np.linalg.norm(text_vector) * np.linalg.norm(embeddings[intent])) + 1e-8)
            )
            if similarity > best_score:
                best_score = similarity
                best_intent = intent

        return best_intent
    except Exception:
        return _classify_intent_keyword_fallback(text, candidates)

def _classify_intent_keyword_fallback(text: str, candidates: List[str]) -> str:
    normalized = text.lower()
    keyword_map = _get_catalog().keyword_fallback_map

    best_intent = candidates[-1] if candidates else "other"
    best_count = 0

    for intent in candidates:
        keywords = keyword_map.get(intent, [])
        count = sum(1 for keyword in keywords if keyword in normalized)
        if count > best_count:
            best_count = count
            best_intent = intent

    return best_intent

def is_greeting_sync(text: str) -> bool:
    return is_greeting(text)

