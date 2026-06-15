from __future__ import annotations
import logging
import os
from contextlib import contextmanager
from typing import Any, List

from huggingface_hub import login

from app.components.faq_enhanced.constants import RAG_LOCAL_MODELS_ONLY

logger = logging.getLogger(__name__)


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


class _SentenceTransformerEmbeddings:
    """Minimal LangChain-compatible embeddings adapter."""

    def __init__(self, model_name: str, *, device: str = "cpu", normalize_embeddings: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ImportError("sentence-transformers is unavailable") from exc
        with _local_model_load(RAG_LOCAL_MODELS_ONLY):
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
            cache_folder = os.getenv("cache_folder")
            self._model = SentenceTransformer(model_name, device=device, cache_folder=cache_folder)
        self._normalize_embeddings = normalize_embeddings

    @staticmethod
    def _as_list(vector: Any) -> List[float]:
        if hasattr(vector, "tolist"):
            return vector.tolist()
        return list(vector)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=self._normalize_embeddings)
        return [self._as_list(vector) for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode(text, normalize_embeddings=self._normalize_embeddings)
        return self._as_list(vector)
