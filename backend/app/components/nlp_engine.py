"""
Unified NLP engine.

Intent and lexical behavior are loaded from dynamic YAML configuration.
Model loading is lazy and degrades gracefully when optional NLP dependencies
are unavailable.
"""
from __future__ import annotations
import asyncio
from contextlib import contextmanager
import logging
from dotenv import load_dotenv
from huggingface_hub import login
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

load_dotenv()
# login(token=os.getenv("HF_TOKEN"))

from app.services.dynamic_config import get_intent_catalog as _get_catalog
from app.services.dynamic_config import get_retrieval_config as _get_retrieval_config
from app.services.dynamic_config import get_thresholds as _get_thresholds
from app.services.dynamic_config import get_vocabulary as _get_vocabulary

logger = logging.getLogger(__name__)

_vader_analyzer = None
_spacy_nlp = None
_st_model = None
_intent_embeddings: Optional[Dict[str, Any]] = None

RAG_LOCAL_MODELS_ONLY = os.getenv("RAG_LOCAL_MODELS_ONLY", "1").lower() not in {"0", "false", "no"}
UUID_PATTERN = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
ISO_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2})\b")


@contextmanager
def _local_model_load(enabled: bool):
    if not enabled:
        yield
        return

    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _is_local_model_reference(model_name: str) -> bool:
    try:
        return os.path.exists(os.path.expanduser(model_name))
    except OSError:
        return False


def _get_intent_prototypes() -> Dict[str, List[str]]:
    catalog = _get_catalog()
    return {name: cfg.prototypes for name, cfg in catalog.intents.items() if cfg.prototypes}


def _get_field_prototypes() -> Dict[str, List[str]]:
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
    return _get_vocabulary().nlp_fallback


def _get_nlp_thresholds():
    return _get_thresholds().nlp


def _get_intent_threshold(intent: str) -> float:
    catalog = _get_catalog()
    if intent in catalog.intents:
        threshold = float(catalog.intents[intent].threshold or 0.0)
        if threshold > 0:
            return threshold
    catalog_default = float(catalog.default_threshold or 0.0)
    if catalog_default > 0:
        return catalog_default
    return float(_get_nlp_thresholds().intent_threshold_default)


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            _vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("[nlp_engine] VADER initialized")
        except ImportError:
            logger.warning("[nlp_engine] vaderSentiment unavailable; using fallback")
            _vader_analyzer = _FallbackVader()
    return _vader_analyzer


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy

            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("[nlp_engine] spaCy en_core_web_sm loaded")
        except (ImportError, OSError):
            logger.warning("[nlp_engine] spaCy unavailable; disabling NER")
            _spacy_nlp = False
    return _spacy_nlp if _spacy_nlp is not False else None


def _get_st_model():
    global _st_model
    if _st_model is None:
        try:
            model_name = os.getenv("EMBED_MODEL", _get_retrieval_config().embeddings.model_name)
            if RAG_LOCAL_MODELS_ONLY and not _is_local_model_reference(model_name):
                _st_model = False
                return None
            from sentence_transformers import SentenceTransformer

            with _local_model_load(RAG_LOCAL_MODELS_ONLY):
                hf_token = os.getenv("HF_TOKEN")
                # print(whoami())
                if hf_token:
                    login(token=hf_token)
                cache_folder = os.getenv("cache_folder")
                _st_model = SentenceTransformer(model_name, cache_folder=cache_folder)
            logger.info("[nlp_engine] sentence-transformers loaded: %s", model_name)
        except Exception as exc:
            logger.warning("[nlp_engine] sentence-transformers unavailable (%s)", exc)
            _st_model = False
    return _st_model if _st_model is not False else None


def _get_intent_embeddings() -> Optional[Dict[str, Any]]:
    global _intent_embeddings
    if _intent_embeddings is not None:
        return _intent_embeddings
    model = _get_st_model()
    if model is None:
        return None

    prototypes = _get_intent_prototypes()
    if not prototypes:
        return None

    try:
        import numpy as np

        embeddings: Dict[str, Any] = {}
        for intent, phrases in prototypes.items():
            if not phrases:
                continue
            vectors = model.encode(phrases, convert_to_numpy=True)
            embeddings[intent] = np.mean(vectors, axis=0)
        _intent_embeddings = embeddings or None
        return _intent_embeddings
    except Exception as exc:
        logger.warning("[nlp_engine] could not build intent embeddings (%s)", exc)
        _intent_embeddings = None
        return None


@lru_cache(maxsize=64)
def _encode_prototypes_cached(prototypes: Tuple[str, ...]):
    model = _get_st_model()
    if model is None:
        return None
    if not prototypes:
        return None
    try:
        return model.encode(list(prototypes), convert_to_numpy=True)
    except Exception:
        return None


def _max_semantic_similarity(text: str, prototypes: Tuple[str, ...]) -> float:
    if not text or not prototypes:
        return 0.0
    model = _get_st_model()
    if model is None:
        return 0.0
    prototype_vectors = _encode_prototypes_cached(prototypes)
    if prototype_vectors is None:
        return 0.0

    try:
        import numpy as np

        text_vector = model.encode([text], convert_to_numpy=True)[0]
        denom = (np.linalg.norm(prototype_vectors, axis=1) * np.linalg.norm(text_vector)) + 1e-8
        sims = np.dot(prototype_vectors, text_vector) / denom
        return float(np.max(sims))
    except Exception:
        return 0.0


@lru_cache(maxsize=16)
def _name_full_pattern(max_chars: int):
    return re.compile(rf"^([A-Za-z][A-Za-z .'-]{{1,{max_chars}}})$", re.I)


class _FallbackVader:
    @property
    def _pos(self):
        return set(_get_catalog().vader_fallback.get("positive", []))

    @property
    def _neg(self):
        return set(_get_catalog().vader_fallback.get("negative", []))

    def polarity_scores(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-z']+", text.lower())
        pos = sum(1 for token in tokens if token in self._pos)
        neg = sum(1 for token in tokens if token in self._neg)
        total = max(pos + neg, 1)
        token_count = max(len(tokens), 1)
        compound = (pos - neg) / total
        return {
            "pos": pos / total,
            "neg": neg / total,
            "neu": 1.0 - (pos + neg) / token_count,
            "compound": compound,
        }


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
