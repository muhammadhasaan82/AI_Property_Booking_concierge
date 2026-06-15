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

from app.components.nlp_engine.config import (
    _get_intent_prototypes,
    _get_nlp_thresholds,
    _is_local_model_reference,
    _local_model_load,
)
from app.components.nlp_engine.constants import (
    RAG_LOCAL_MODELS_ONLY,
    _intent_embeddings,
    _spacy_nlp,
    _st_model,
    _vader_analyzer,
)
import app.components.nlp_engine.constants as _nlp_constants

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

def _get_vader():
    if _nlp_constants._vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            _nlp_constants._vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("[nlp_engine] VADER initialized")
        except ImportError:
            logger.warning("[nlp_engine] vaderSentiment unavailable; using fallback")
            _nlp_constants._vader_analyzer = _FallbackVader()
    return _nlp_constants._vader_analyzer


def _get_spacy():
    if _nlp_constants._spacy_nlp is None:
        try:
            import spacy

            _nlp_constants._spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("[nlp_engine] spaCy en_core_web_sm loaded")
        except (ImportError, OSError):
            logger.warning("[nlp_engine] spaCy unavailable; disabling NER")
            _nlp_constants._spacy_nlp = False
    return _nlp_constants._spacy_nlp if _nlp_constants._spacy_nlp is not False else None


def _get_st_model():
    if _nlp_constants._st_model is None:
        try:
            model_name = os.getenv("EMBED_MODEL", _get_retrieval_config().embeddings.model_name)
            if RAG_LOCAL_MODELS_ONLY and not _is_local_model_reference(model_name):
                _nlp_constants._st_model = False
                return None
            from huggingface_hub import login
            from sentence_transformers import SentenceTransformer

            with _local_model_load(RAG_LOCAL_MODELS_ONLY):
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    login(token=hf_token)
                cache_folder = os.getenv("cache_folder")
                _nlp_constants._st_model = SentenceTransformer(model_name, cache_folder=cache_folder)
            logger.info("[nlp_engine] sentence-transformers loaded: %s", model_name)
        except Exception as exc:
            logger.warning("[nlp_engine] sentence-transformers unavailable (%s)", exc)
            _nlp_constants._st_model = False
    return _nlp_constants._st_model if _nlp_constants._st_model is not False else None


def _get_intent_embeddings() -> Optional[Dict[str, Any]]:
    if _nlp_constants._intent_embeddings is not None:
        return _nlp_constants._intent_embeddings
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
        _nlp_constants._intent_embeddings = embeddings or None
        return _nlp_constants._intent_embeddings
    except Exception as exc:
        logger.warning("[nlp_engine] could not build intent embeddings (%s)", exc)
        _nlp_constants._intent_embeddings = None
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

