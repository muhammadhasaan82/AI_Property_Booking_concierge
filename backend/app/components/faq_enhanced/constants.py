from __future__ import annotations
"""Shared FAQ enhanced configuration."""

import os
from pathlib import Path

import litellm

from app.services.dynamic_config import get_retrieval_config

litellm.drop_params = True

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano")

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_RETRIEVAL_CFG = get_retrieval_config()
_chroma_dir_value = os.getenv("FAQ_CHROMA_PATH", _RETRIEVAL_CFG.chroma.persist_dir)
_chroma_dir_path = Path(_chroma_dir_value)
if not _chroma_dir_path.is_absolute():
    _chroma_dir_path = _BACKEND_ROOT / _chroma_dir_path

CHROMA_PATH = _chroma_dir_path
CHROMA_PATH.mkdir(parents=True, exist_ok=True)
EMBED_MODEL = os.getenv("EMBED_MODEL", _RETRIEVAL_CFG.embeddings.model_name)
EMBED_NORMALIZE = bool(_RETRIEVAL_CFG.embeddings.normalize_embeddings)
FAQ_COLLECTION_NAME = _RETRIEVAL_CFG.chroma.collection_name
RAG_LOCAL_MODELS_ONLY = os.getenv("RAG_LOCAL_MODELS_ONLY", "1").lower() not in {"0", "false", "no"}
