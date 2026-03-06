# services/nlp_engine.py
# -*- coding: utf-8 -*-
"""
Unified NLP engine — replaces all hardcoded regex/arrays/dicts with
dynamic analysis powered by VADER, spaCy, and sentence-transformers.

Heavy NLP calls are wrapped with asyncio.to_thread to avoid blocking
the FastAPI event loop.

Lazy-loads models on first use. Gracefully degrades if spaCy or
sentence-transformers are unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Lazy singletons
# ─────────────────────────────────────────────────────────────────────

_vader_analyzer = None
_spacy_nlp = None
_st_model = None
_intent_embeddings: Optional[Dict[str, Any]] = None

# ─────────────────────────────────────────────────────────────────────
# Config-driven prototypes (loaded from config/intent_catalog.yaml)
# ─────────────────────────────────────────────────────────────────────
from services.dynamic_config import (
    get_intent_catalog as _get_catalog,
    get_vocabulary as _get_vocabulary,
)


def _get_intent_prototypes() -> Dict[str, List[str]]:
    """Load intent prototypes from config."""
    cat = _get_catalog()
    return {k: v.prototypes for k, v in cat.intents.items()}


def _get_field_prototypes() -> Dict[str, List[str]]:
    """Load field detection prototypes from config."""
    return _get_catalog().field_prototypes


def _get_modification_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().modification_prototypes)


def _get_property_search_request_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().property_search_request_prototypes)


def _get_receipt_request_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().receipt_request_prototypes)


def _get_resume_request_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().resume_request_prototypes)


def _get_affirm_yes_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().affirm_yes_prototypes)


def _get_affirm_no_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().affirm_no_prototypes)


def _get_vocab():
    """Load lexical fallback vocabulary from config/vocabulary.yaml."""
    return _get_vocabulary().nlp_fallback


def _get_vader():
    """Lazy-init VADER sentiment analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("[nlp_engine] VADER initialized")
        except ImportError:
            logger.warning("[nlp_engine] vaderSentiment not installed, using fallback")
            _vader_analyzer = _FallbackVader()
    return _vader_analyzer


def _get_spacy():
    """Lazy-init spaCy pipeline."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("[nlp_engine] spaCy en_core_web_sm loaded")
        except (ImportError, OSError):
            logger.warning("[nlp_engine] spaCy model not available, NER disabled")
            _spacy_nlp = False  # sentinel so we don't retry
    return _spacy_nlp if _spacy_nlp is not False else None


def _get_st_model():
    """Lazy-init sentence-transformers model for zero-shot classification."""
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.info("[nlp_engine] sentence-transformers BAAI/bge-small-en-v1.5 loaded")
        except Exception as exc:  # noqa: BLE001 - degrade gracefully in restricted envs
            logger.warning(
                "[nlp_engine] sentence-transformers unavailable (%s), falling back to keyword matching",
                exc,
            )
            _st_model = False
    return _st_model if _st_model is not False else None


def _get_intent_embeddings() -> Optional[Dict[str, Any]]:
    """Pre-compute prototype embeddings (cached after first call)."""
    global _intent_embeddings
    if _intent_embeddings is not None:
        return _intent_embeddings
    model = _get_st_model()
    if model is None:
        return None
    try:
        import numpy as np

        _intent_embeddings = {}
        for intent, phrases in _get_intent_prototypes().items():
            vecs = model.encode(phrases, convert_to_numpy=True)
            _intent_embeddings[intent] = np.mean(vecs, axis=0)
        logger.info("[nlp_engine] Intent embeddings pre-computed for %d intents",
                    len(_intent_embeddings))
        return _intent_embeddings
    except Exception as exc:  # noqa: BLE001 - degrade gracefully in restricted envs
        logger.warning(
            "[nlp_engine] could not precompute intent embeddings (%s); using keyword fallback",
            exc,
        )
        _intent_embeddings = None
        return None


# ─────────────────────────────────────────────────────────────────────
# Fallback VADER (when vaderSentiment is not installed)
# ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def _encode_prototypes_cached(prototypes: Tuple[str, ...]):
    """Encode prototype sentences once per process for semantic matching."""
    model = _get_st_model()
    if model is None:
        return None
    try:
        return model.encode(list(prototypes), convert_to_numpy=True)
    except Exception:  # noqa: BLE001 - degrade gracefully in restricted envs
        return None


def _max_semantic_similarity(text: str, prototypes: Tuple[str, ...]) -> float:
    """Return max cosine similarity of text to a prototype set."""
    if not text or not prototypes:
        return 0.0
    model = _get_st_model()
    if model is None:
        return 0.0
    proto_vecs = _encode_prototypes_cached(prototypes)
    if proto_vecs is None:
        return 0.0
    try:
        import numpy as np

        text_vec = model.encode([text], convert_to_numpy=True)[0]
        denom = (np.linalg.norm(proto_vecs, axis=1) * np.linalg.norm(text_vec) + 1e-8)
        sims = np.dot(proto_vecs, text_vec) / denom
        return float(np.max(sims))
    except Exception:  # noqa: BLE001 - degrade gracefully in restricted envs
        return 0.0


class _FallbackVader:
    """Minimal polarity scorer used when vaderSentiment is not installed."""

    @property
    def _POS(self):
        cat = _get_catalog()
        return set(cat.vader_fallback.get("positive", []))
    @property
    def _NEG(self):
        cat = _get_catalog()
        return set(cat.vader_fallback.get("negative", []))

    def polarity_scores(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-z']+", text.lower())
        pos = sum(1 for t in tokens if t in self._POS)
        neg = sum(1 for t in tokens if t in self._NEG)
        total = max(pos + neg, 1)
        compound = (pos - neg) / total
        return {"pos": pos / total, "neg": neg / total,
                "neu": 1.0 - (pos + neg) / max(len(tokens), 1),
                "compound": compound}


# ─────────────────────────────────────────────────────────────────────
# Core NLP Functions (sync — wrap with asyncio.to_thread for async)
# ─────────────────────────────────────────────────────────────────────

def classify_affirmation(text: str) -> str:
    """Classify text as 'yes', 'no', or 'neutral' using VADER + lexicon.

    Returns
    -------
    'yes' | 'no' | 'neutral'
    """
    if not text or not text.strip():
        return "neutral"

    tl = text.strip().lower()

    # Semantic-first classification for extended confirmations.
    yes_sim = _max_semantic_similarity(tl, _get_affirm_yes_prototypes())
    no_sim = _max_semantic_similarity(tl, _get_affirm_no_prototypes())
    if yes_sim >= 0.65 and yes_sim >= (no_sim + 0.04):
        return "yes"
    if no_sim >= 0.65 and no_sim >= (yes_sim + 0.04):
        return "no"

    # Minimal lexical fallback for explicit confirmations.
    vocab = _get_vocab()
    if tl in set(vocab.affirm_yes_tokens):
        return "yes"
    if tl in set(vocab.affirm_no_tokens):
        return "no"

    # VADER compound score
    vader = _get_vader()
    scores = vader.polarity_scores(tl)
    compound = scores["compound"]

    # Polarity fallback for short-to-medium confirmation utterances.
    token_len = len(tl.split())
    if compound >= 0.28 and token_len <= 10:
        return "yes"
    if compound <= -0.28 and token_len <= 10:
        return "no"

    return "neutral"


def is_greeting(text: str) -> bool:
    """Detect conversational greetings dynamically."""
    if not text or not text.strip():
        return False

    tl = text.strip().lower()
    if is_acknowledgment(text):
        return False
    
    tokens = tl.split()

    # Short utterance check (greetings are usually 1-4 words)
    if len(tokens) > 6:
        return False

    vocab = _get_vocab()
    if tokens[0] in set(vocab.greeting_seeds):
        return True
    if any(re.search(r'\b' + re.escape(p) + r'\b', tl) for p in vocab.greeting_phrases):
        return True

    # Semantic classification fallback
    all_intents = list(_get_catalog().intents.keys())
    intent = classify_intent_sync(tl, all_intents)
    if intent != "greeting":
        return False
    threshold = _get_catalog().intents["greeting"].threshold if "greeting" in _get_catalog().intents else 0.60
    return _semantic_confidence(tl, "greeting") >= threshold


def is_acknowledgment(text: str) -> bool:
    """Detect conversational acknowledgment (ok, sounds good, got it, etc.)."""
    if not text or not text.strip():
        return False
    tl = text.strip().lower()
    vocab = _get_vocab()
    if tl in set(vocab.acknowledgment_tokens):
        return True
    return any(p in tl for p in vocab.acknowledgment_phrases)


def is_handoff_request(text: str) -> bool:
    """Detect request to speak with a human agent."""
    if not text:
        return False
    tl = text.strip().lower()
    vocab = _get_vocab()
    if any(s in tl for s in vocab.handoff_seeds):
        return True
    return any(p in tl for p in vocab.handoff_phrases)


def is_availability_query(text: str) -> bool:
    """Detect questions about date availability."""
    if not text:
        return False
    tl = text.strip().lower()
    return any(p in tl for p in _get_vocab().availability_phrases)


def is_end_request(text: str) -> bool:
    """Detect conversation end requests."""
    if not text:
        return False
    tl = text.strip().lower()
    vocab = _get_vocab()
    if tl in set(vocab.end_exact):
        return True
    return any(p in tl for p in vocab.end_phrases)


def is_status_query(text: str) -> bool:
    """Detect booking status or check-in/out inquiries."""
    if not text:
        return False
    tl = text.strip().lower()

    # UUID presence is a strong signal
    if re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", tl):
        return True

    # Semantic classification
    all_intents = list(_get_catalog().intents.keys())
    intent = classify_intent_sync(tl, all_intents)
    if intent == "status_update":
        threshold = _get_catalog().intents["status_update"].threshold if "status_update" in _get_catalog().intents else 0.55
        conf = _semantic_confidence(tl, "status_update")
        if conf >= threshold:
            return True

    # Keyword fallback
    return any(s in tl for s in _get_vocab().status_seeds)


def is_property_search(text: str) -> bool:
    """Detect property search intent, excluding status queries."""
    if not text:
        return False
    tl = text.strip().lower()

    # Exclude status queries
    if is_status_query(text):
        return False

    # Exclude booking ID mentions
    vocab = _get_vocab()
    booking_id_markers = list(vocab.status_booking_id_markers)
    if any(x in tl for x in booking_id_markers):
        return False
    if re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", tl):
        return False

    # Semantic classification
    all_intents = list(_get_catalog().intents.keys())
    intent = classify_intent_sync(tl, all_intents)
    if intent == "property_search":
        threshold = _get_catalog().intents["property_search"].threshold if "property_search" in _get_catalog().intents else 0.50
        conf = _semantic_confidence(tl, "property_search")
        if conf >= threshold:
            return True

    # Keyword + NER fallback
    from .nlp_extractor import KNOWN_CITIES, CITY_ALIASES, PROPERTY_TYPES

    money_pat = None
    if vocab.money_intent_pattern:
        try:
            money_pat = re.compile(vocab.money_intent_pattern, re.I)
        except re.error:
            money_pat = None

    if any(re.search(r'\b' + re.escape(p) + r'\b', tl) for p in PROPERTY_TYPES):
        return True
    if any(re.search(r'\b' + re.escape(c) + r'\b', tl) for c in KNOWN_CITIES):
        return True
    if any(re.search(r'\b' + re.escape(a) + r'\b', tl) for a in CITY_ALIASES):
        return True
    if money_pat and money_pat.search(tl):
        return True
    if any(re.search(r'\b' + re.escape(w) + r'\b', tl) for w in vocab.search_signals):
        return True
    if any(re.search(r'\b' + re.escape(p) + r'\b', tl) for p in vocab.search_phrases):
        return True

    return False


def wants_modification(text: str) -> bool:
    """Detect intent to modify booking details."""
    if not text or not text.strip():
        return False
    tl = text.strip().lower()
    if _max_semantic_similarity(tl, _get_modification_prototypes()) >= 0.70:
        return True

    # Keyword fallback when semantic model is unavailable or uncertain.
    return any(w in tl for w in _get_vocab().modification_seeds)


def wants_property_search_request(text: str) -> bool:
    """Detect intent to search for different/more properties."""
    if not text or not text.strip():
        return False
    tl = text.strip().lower()
    if _max_semantic_similarity(tl, _get_property_search_request_prototypes()) >= 0.70:
        return True

    # Keyword fallback when semantic model is unavailable or uncertain.
    return any(p in tl for p in _get_vocab().property_search_request_seeds)


def is_receipt_request(text: str) -> bool:
    """Detect requests to view booking total/receipt."""
    if not text or not text.strip():
        return False
    tl = text.strip().lower()
    if _max_semantic_similarity(tl, _get_receipt_request_prototypes()) >= 0.65:
        return True
    
    # Keyword fallback when semantic model is unavailable or uncertain.
    vocab = _get_vocab()
    if any(seed in tl for seed in vocab.receipt_seeds):
        return True
    if any(p in tl for p in vocab.receipt_phrases):
        return True
    return (
        any(q in tl for q in vocab.receipt_quantity_terms)
        and any(term in tl for term in vocab.receipt_amount_terms)
    )


def is_resume_request(text: str) -> bool:
    """Detect whether user wants to resume/continue the previous flow."""
    if not text or not text.strip():
        return False
    tl = text.strip().lower()
    if _max_semantic_similarity(tl, _get_resume_request_prototypes()) >= 0.65:
        return True
    
    # Keyword fallback when semantic model is unavailable or uncertain.
    if tl in set(_get_vocab().resume_exact_phrases):
        return True
    return any(p in tl for p in _get_vocab().resume_phrases)


def wants_previous_results_sync(text: str) -> bool:
    """Semantic detection: does user want to return to previous search results?"""
    if not text or not text.strip():
        return False
    
    model = _get_st_model()
    if model is None:
        # Fallback keyword logic
        tl = text.lower()
        _prev_cfg = _get_catalog().previous_results_prototypes
        return any(p in tl for p in _prev_cfg.fallback_keywords)

    import numpy as np
    _prev_cfg = _get_catalog().previous_results_prototypes
    prototypes = _prev_cfg.prototypes
    
    text_vec = model.encode([text], convert_to_numpy=True)[0]
    proto_vecs = model.encode(prototypes, convert_to_numpy=True)
    
    # Compute similarity against all prototypes
    scores = np.dot(proto_vecs, text_vec) / (np.linalg.norm(proto_vecs, axis=1) * np.linalg.norm(text_vec) + 1e-8)
    threshold = _get_catalog().previous_results_prototypes.threshold
    return float(np.max(scores)) > threshold


def detect_faq_intent(text: str) -> bool:
    """Detect FAQ/policy questions using semantic classification."""
    if not text or not text.strip():
        return False
    tl = text.strip().lower()

    # Semantic classification
    all_intents = list(_get_catalog().intents.keys())
    intent = classify_intent_sync(tl, all_intents)
    if intent == "faq":
        threshold = _get_catalog().intents["faq"].threshold if "faq" in _get_catalog().intents else 0.50
        conf = _semantic_confidence(tl, "faq")
        if conf >= threshold:
            return True

    # Strong trigger keywords (policy/terms are always FAQ — regardless of booking context)
    vocab = _get_vocab()
    if any(re.search(r'\b' + re.escape(s) + r'\b', tl) for s in vocab.faq_strong_keywords):
        return True

    # Broader keyword + question pattern check
    has_faq_seed = any(re.search(r'\b' + re.escape(s) + r'\b', tl) for s in vocab.faq_seeds)
    has_question = (
        "?" in tl
        or any(tl.startswith(q) for q in vocab.faq_question_starts)
        or any(cue in tl for cue in vocab.faq_question_cues)
    )
    return has_faq_seed and has_question


# ─────────────────────────────────────────────────────────────────────
# NER Extraction (spaCy-based with fallback)
# ─────────────────────────────────────────────────────────────────────

def extract_person_name(text: str) -> Optional[str]:
    """Extract person name via spaCy PERSON NER, with regex fallback."""
    if not text:
        return None

    t_lower = text.lower().strip()
    # Guard: skip if text looks like a search query
    vocab = _get_vocab()
    if any(term in t_lower for term in vocab.name_search_guards):
        return None

    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip().rstrip('.').strip()
                if len(name) >= 2:
                    return name

    # Regex fallback — STRICT to prevent capturing conversational sentences
    # Guard: reject text that contains common conversational/functional words
    words = set(re.findall(r"[a-z]+", t_lower))
    _has_conversational = bool(words & set(vocab.name_conversational_guards))

    # Explicit "my name is ..." / "I am ..." patterns — always safe
    for pat_str in vocab.name_explicit_patterns:
        m = re.search(pat_str, text, re.I)
        if m:
            cand = m.group(1).strip().rstrip('.').strip()
            # Limit explicit captures to 1-3 meaningful words
            if len(cand.split()) <= 3 and len(cand) >= 2 and not _looks_like_email_username(cand):
                return cand

    # Full-string fallback — ONLY if no conversational words detected
    # and text is very short (1-3 words, looks like a raw name)
    if not _has_conversational:
        stripped = text.strip()
        word_count = len(stripped.split())
        if 1 <= word_count <= 3:
            _NAME_FULL_PAT = re.compile(r"^([A-Za-z][A-Za-z .'-]{1,60})$", re.I)
            m = _NAME_FULL_PAT.match(stripped)
            if m:
                cand = m.group(1).strip().rstrip('.').strip()
                if len(cand) >= 2 and not _looks_like_email_username(cand):
                    return cand

    return None


def _looks_like_email_username(s: str) -> bool:
    if not s:
        return False
    common = set(_get_vocab().email_username_common)
    return (len(s) <= 3 or s.lower() in common or
            bool(re.search(r'\d', s)) or
            bool(re.search(r"[^a-zA-Z\s\-.'']", s)))


def extract_dates(text: str) -> List[str]:
    """Extract dates from text using spaCy DATE NER + regex fallback."""
    results: List[str] = []
    if not text:
        return results

    # Primary: ISO date regex (always reliable for YYYY-MM-DD)
    iso_dates = re.findall(r"\b(\d{4}-\d{1,2}-\d{1,2})\b", text)
    results.extend(iso_dates)

    if results:
        return results

    # spaCy DATE entity extraction (for natural language dates)
    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                results.append(ent.text)


def wants_previous_results_sync(text: str) -> bool:
    """Semantic detection: does user want to return to previous search results?"""
    if not text or not text.strip():
        return False
    
    model = _get_st_model()
    if model is None:
        # Fallback keyword logic
        tl = text.lower()
        _prev_cfg = _get_catalog().previous_results_prototypes
        return any(p in tl for p in _prev_cfg.fallback_keywords)

    import numpy as np
    _prev_cfg = _get_catalog().previous_results_prototypes
    prototypes = _prev_cfg.prototypes
    
    text_vec = model.encode([text], convert_to_numpy=True)[0]
    proto_vecs = model.encode(prototypes, convert_to_numpy=True)
    
    # Compute similarity against all prototypes
    scores = np.dot(proto_vecs, text_vec) / (np.linalg.norm(proto_vecs, axis=1) * np.linalg.norm(text_vec) + 1e-8)
    threshold = _get_catalog().previous_results_prototypes.threshold
    return float(np.max(scores)) > threshold


def detect_faq_intent(text: str) -> bool:
    """Detect FAQ/policy questions using semantic classification."""
    if not text or not text.strip():
        return False
    tl = text.strip().lower()

    # Semantic classification
    all_intents = list(_get_catalog().intents.keys())
    intent = classify_intent_sync(tl, all_intents)
    if intent == "faq":
        threshold = _get_catalog().intents["faq"].threshold if "faq" in _get_catalog().intents else 0.50
        conf = _semantic_confidence(tl, "faq")
        if conf >= threshold:
            return True

    # Strong trigger keywords (policy/terms are always FAQ — regardless of booking context)
    vocab = _get_vocab()
    if any(re.search(r'\b' + re.escape(s) + r'\b', tl) for s in vocab.faq_strong_keywords):
        return True

    # Broader keyword + question pattern check
    has_faq_seed = any(re.search(r'\b' + re.escape(s) + r'\b', tl) for s in vocab.faq_seeds)
    has_question = (
        "?" in tl
        or any(tl.startswith(q) for q in vocab.faq_question_starts)
        or any(cue in tl for cue in vocab.faq_question_cues)
    )
    return has_faq_seed and has_question


# ─────────────────────────────────────────────────────────────────────
# NER Extraction (spaCy-based with fallback)
# ─────────────────────────────────────────────────────────────────────

def extract_person_name(text: str) -> Optional[str]:
    """Extract person name via spaCy PERSON NER, with regex fallback."""
    if not text:
        return None

    t_lower = text.lower().strip()
    # Guard: skip if text looks like a search query
    vocab = _get_vocab()
    if any(term in t_lower for term in vocab.name_search_guards):
        return None

    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip().rstrip('.').strip()
                if len(name) >= 2:
                    return name

    # Regex fallback — STRICT to prevent capturing conversational sentences
    # Guard: reject text that contains common conversational/functional words
    words = set(re.findall(r"[a-z]+", t_lower))
    _has_conversational = bool(words & set(vocab.name_conversational_guards))

    # Explicit "my name is ..." / "I am ..." patterns — always safe
    for pat_str in vocab.name_explicit_patterns:
        m = re.search(pat_str, text, re.I)
        if m:
            cand = m.group(1).strip().rstrip('.').strip()
            # Limit explicit captures to 1-3 meaningful words
            if len(cand.split()) <= 3 and len(cand) >= 2 and not _looks_like_email_username(cand):
                return cand

    # Full-string fallback — ONLY if no conversational words detected
    # and text is very short (1-3 words, looks like a raw name)
    if not _has_conversational:
        stripped = text.strip()
        word_count = len(stripped.split())
        if 1 <= word_count <= 3:
            _NAME_FULL_PAT = re.compile(r"^([A-Za-z][A-Za-z .'-]{1,60})$", re.I)
            m = _NAME_FULL_PAT.match(stripped)
            if m:
                cand = m.group(1).strip().rstrip('.').strip()
                if len(cand) >= 2 and not _looks_like_email_username(cand):
                    return cand

    return None


def _looks_like_email_username(s: str) -> bool:
    if not s:
        return False
    common = set(_get_vocab().email_username_common)
    return (len(s) <= 3 or s.lower() in common or
            bool(re.search(r'\d', s)) or
            bool(re.search(r"[^a-zA-Z\s\-.'']", s)))


def extract_dates(text: str) -> List[str]:
    """Extract dates from text using spaCy DATE NER + regex fallback."""
    results: List[str] = []
    if not text:
        return results

    # Primary: ISO date regex (always reliable for YYYY-MM-DD)
    iso_dates = re.findall(r"\b(\d{4}-\d{1,2}-\d{1,2})\b", text)
    results.extend(iso_dates)

    if results:
        return results

    # spaCy DATE entity extraction (for natural language dates)
    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                results.append(ent.text)

    return results


def _llm_route_intent(user_text: str, filters: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Use LLM structured output to classify intent with minimal hardcoded rules."""
    if not (OPENAI_API_KEY and LLM_STRUCTURED and SOFT_INTENT_ROUTER):
        return None

    text = (user_text or "").strip()
    if not text:
        return None

    active_filters = filters or {}
    
    # 1. Build the dynamic system prompt
    system_prompt = (
        "You are an intent classifier for a hotel booking chatbot. "
        "Classify the user's message into ONE intent. Return strict JSON: {intent, confidence, brief_reason}. "
        "Intent must be one of: greeting, faq, confirmation, property_search, booking, "
        "status_update, payment_link, handoff, availability, end.\n\n"
        "CRITICAL RULES:\n"
        "- 'faq' = user is asking about rules, policy, refund, cancellation, pets, etc. A policy question always overrides booking context.\n"
        "- 'confirmation' = user is selecting a numbered option, providing booking details, or affirming/declining a step.\n"
        "- 'property_search' = user is looking for a place or asking about properties.\n"
        "- 'greeting' = ONLY pure greetings with NO other intent.\n"
    )

    # 2. INJECT STATE DIRECTLY INTO SYSTEM PROMPT (The Super Soft-Coded Fix)
    has_last_results = bool(active_filters.get("last_results"))
    if has_last_results:
        count = len(active_filters.get("last_results") or [])
        system_prompt += f"\n[CRITICAL STATE OVERRIDE]: You just showed the user a numbered list of {count} properties. If their message is a number (e.g. '1', '{count}'), an ordinal, or a selection phrase like 'option 7', they are making a selection. You MUST classify this intent strictly as 'confirmation'. Do NOT classify as property_search."

    if active_filters.get(SK.awaiting_field):
        system_prompt += f"\n[CRITICAL STATE OVERRIDE]: You are currently awaiting the user to provide their '{active_filters.get(SK.awaiting_field)}'. Treat their input as a 'confirmation' of this data."

    try:
        valid_types = get_vocabulary().seed_property_types
        if valid_types:
            system_prompt += f"\n\nValid property types in our database: {', '.join(valid_types)}."
    except Exception:
        pass

    # 3. Send ONLY the user text, no confusing JSON context block
    payload: Dict[str, Any] = {
        "model": OPENAI_CHAT_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}, 
        ],
    }

    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }, json=payload)
        if r.status_code != 200:
            return None
        body = r.json()
        content = (((body or {}).get("choices") or [{}])[0] or {}).get("message", {}).get("content", "")
        if not content:
            return None
        parsed = json.loads(content)
        intent = str(parsed.get("intent", "")).strip()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        if intent in _ALLOWED_INTENTS and confidence >= 0.45:
            return intent
    except Exception:
        return None
    return None


def has_cardinal_extraction(text: str) -> bool:
    """Return True when the utterance contains a valid selection cardinal."""
    return extract_cardinal(text) is not None


def is_low_semantic_density(text: str) -> bool:
    """Treat cardinal-only replies as low-information until state gives them meaning."""
    if not text or not text.strip():
        return True
    tl = text.strip().lower()
    if has_cardinal_extraction(tl) and not re.search(r"[a-z]", tl):
        return True
    return False


def extract_guests(text: str) -> Optional[int]:
    """Extract guest count from text."""
    if not text:
        return None
    tl = text.lower().strip()
    units_alt = "|".join(re.escape(u) for u in _get_vocab().guest_unit_terms)
    if units_alt:
        m = re.search(rf"(\d{{1,3}})\s*(?:{units_alt})?\b", tl) or re.search(r"^(\d{1,3})$", tl)
    else:
        m = re.search(r"(\d{1,3})\b", tl) or re.search(r"^(\d{1,3})$", tl)
    if m:
        try:
            n = int(m.group(1))
            return n if 1 <= n <= 100 else None
        except (ValueError, IndexError):
            pass
    return None


def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text using regex."""
    if not text:
        return None
    m = re.search(r"(\+?[\d\s\-]{8,15}\d)", text)
    if m:
        num = re.sub(r"[\s\-]", "", m.group(1))
        # Reject if it looks like a date
        if re.search(r"\d{4}-\d{1,2}-\d{1,2}", text):
            return None
        if re.match(r"^\d{8}$", num):
            return None
        return num
    return None


def extract_email(text: str) -> Optional[str]:
    """Extract email address from text."""
    if not text:
        return None
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def extract_booking_id(text: str) -> Optional[str]:
    """Extract UUID or hex booking ID from text."""
    if not text:
        return None
    tl = text.lower()
    uuid_m = re.search(
        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b", tl
    )
    if uuid_m:
        return uuid_m.group(1)
    short_m = re.search(r"\b([0-9a-f]{8})\b", tl)
    if short_m:
        return short_m.group(1)
    return None


def detect_requested_fields(text: str) -> List[str]:
    """Detect which booking fields the user wants to modify."""
    if not text:
        return []
    if is_low_semantic_density(text):
        return []
    tl = text.lower()
    fields: List[str] = []

    model = _get_st_model()
    if model:
        import numpy as np
        text_emb = model.encode([tl], convert_to_numpy=True)[0]
        for field, phrases in _get_field_prototypes().items():
            field_embs = model.encode(phrases, convert_to_numpy=True)
            mean_emb = np.mean(field_embs, axis=0)
            sim = float(np.dot(text_emb, mean_emb) /
                        (np.linalg.norm(text_emb) * np.linalg.norm(mean_emb) + 1e-8))
            if sim > 0.50:
                if field not in fields:
                    fields.append(field)

    if fields:
        return fields

    # Keyword fallback from config-driven field prototypes.
    def _phrase_hit(phrase: str) -> bool:
        p = (phrase or "").strip().lower()
        if not p:
            return False
        if p in tl:
            return True
        # Soft lexical match: allow light filler words like "my".
        filler = set(_get_vocab().phrase_fillers)
        tokens = [t for t in re.findall(r"[a-z0-9_+-]+", p) if t not in filler]
        return bool(tokens) and all(tok in tl for tok in tokens)

    for field, phrases in _get_field_prototypes().items():
        if any(_phrase_hit(p) for p in phrases) and field not in fields:
            fields.append(field)
    return fields


# ─────────────────────────────────────────────────────────────────────
# Semantic classification (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────

def _semantic_confidence(text: str, intent: str) -> float:
    """Return cosine similarity between text and a specific intent prototype."""
    model = _get_st_model()
    if model is None:
        return 0.0
    embs = _get_intent_embeddings()
    if embs is None or intent not in embs:
        return 0.0
    import numpy as np
    text_vec = model.encode([text], convert_to_numpy=True)[0]
    intent_vec = embs[intent]
    sim = float(np.dot(text_vec, intent_vec) /
                (np.linalg.norm(text_vec) * np.linalg.norm(intent_vec) + 1e-8))
    return sim


def classify_intent_sync(text: str, candidates: List[str]) -> str:
    """Classify text against candidate intents using sentence-transformers.

    Falls back to keyword-based heuristics if sentence-transformers unavailable.
    """
    model = _get_st_model()
    if model is None:
        return _classify_intent_keyword_fallback(text, candidates)

    embs = _get_intent_embeddings()
    if embs is None:
        return _classify_intent_keyword_fallback(text, candidates)

    import numpy as np
    text_vec = model.encode([text], convert_to_numpy=True)[0]

    best_intent = candidates[-1] if candidates else "other"
    best_score = -1.0
    for intent in candidates:
        if intent not in embs:
            continue
        sim = float(np.dot(text_vec, embs[intent]) /
                    (np.linalg.norm(text_vec) * np.linalg.norm(embs[intent]) + 1e-8))
        if sim > best_score:
            best_score = sim
            best_intent = intent
    return best_intent


def _classify_intent_keyword_fallback(text: str, candidates: List[str]) -> str:
    """Keyword-based intent classification fallback."""
    tl = text.lower()
    _KEYWORD_MAP = _get_catalog().keyword_fallback_map
    best = candidates[-1] if candidates else "other"
    best_count = 0
    for intent in candidates:
        kws = _KEYWORD_MAP.get(intent, [])
        count = sum(1 for k in kws if k in tl)
        if count > best_count:
            best_count = count
            best = intent
    return best


# ─────────────────────────────────────────────────────────────────────
# Async wrappers (for use in FastAPI handlers)
# ─────────────────────────────────────────────────────────────────────

def is_greeting_sync(text: str) -> bool:
    """Sync alias for greeting detection (used by graph/session guards)."""
    return is_greeting(text)

async def classify_affirmation_async(text: str) -> str:
    return await asyncio.to_thread(classify_affirmation, text)

async def is_greeting_async(text: str) -> bool:
    return await asyncio.to_thread(is_greeting, text)

async def extract_person_name_async(text: str) -> Optional[str]:
    return await asyncio.to_thread(extract_person_name, text)

async def extract_dates_async(text: str) -> List[str]:
    return await asyncio.to_thread(extract_dates, text)

async def extract_cardinal_async(text: str) -> Optional[int]:
    return await asyncio.to_thread(extract_cardinal, text)


async def has_cardinal_extraction_async(text: str) -> bool:
    return await asyncio.to_thread(has_cardinal_extraction, text)


async def is_low_semantic_density_async(text: str) -> bool:
    return await asyncio.to_thread(is_low_semantic_density, text)

async def classify_intent_async(text: str, candidates: List[str]) -> str:
    return await asyncio.to_thread(classify_intent_sync, text, candidates)

async def detect_faq_intent_async(text: str) -> bool:
    return await asyncio.to_thread(detect_faq_intent, text)

async def detect_requested_fields_async(text: str) -> List[str]:
    return await asyncio.to_thread(detect_requested_fields, text)

async def is_property_search_async(text: str) -> bool:
    return await asyncio.to_thread(is_property_search, text)

async def is_status_query_async(text: str) -> bool:
    return await asyncio.to_thread(is_status_query, text)

async def is_receipt_request_async(text: str) -> bool:
    return await asyncio.to_thread(is_receipt_request, text)

async def is_resume_request_async(text: str) -> bool:
    return await asyncio.to_thread(is_resume_request, text)

async def wants_previous_results_async(text: str) -> bool:
    return await asyncio.to_thread(wants_previous_results_sync, text)
