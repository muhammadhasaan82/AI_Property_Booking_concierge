from __future__ import annotations
"""Shared NLP engine constants."""

import os
import re

RAG_LOCAL_MODELS_ONLY = os.getenv("RAG_LOCAL_MODELS_ONLY", "1").lower() not in {"0", "false", "no"}
UUID_PATTERN = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
ISO_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2})\b")

_vader_analyzer = None
_spacy_nlp = None
_st_model = None
_intent_embeddings = None
