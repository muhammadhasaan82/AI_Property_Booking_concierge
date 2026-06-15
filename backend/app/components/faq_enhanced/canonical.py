from __future__ import annotations
import logging, os, re, time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import litellm, yaml
from huggingface_hub import login
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
try:
    from langchain_huggingface import HuggingFaceBgeEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    except Exception:
        HuggingFaceBgeEmbeddings = None
from app.components.faq_enhanced.constants import (
    CHROMA_PATH, EMBED_MODEL, EMBED_NORMALIZE, FAQ_COLLECTION_NAME,
    OPENAI_API_KEY, OPENAI_CHAT_MODEL, RAG_LOCAL_MODELS_ONLY, _BACKEND_ROOT,
)
from app.services.dynamic_config import get_retrieval_config, get_vocabulary
logger = logging.getLogger(__name__)

def _merge_ranked_docs(*batches: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """Merge ranked retrieval batches while preserving rank priority and removing duplicates."""
    merged: List[Tuple[Document, float]] = []
    seen: set[str] = set()

    for batch in batches:
        for item in batch or []:
            if not isinstance(item, tuple) or len(item) < 1:
                continue
            doc = item[0]
            if not hasattr(doc, "page_content"):
                continue
            score = float(item[1]) if len(item) > 1 else 0.0

            page = ""
            metadata = getattr(doc, "metadata", {}) or {}
            if isinstance(metadata, dict):
                page = str(metadata.get("page", ""))

            content_prefix = str(getattr(doc, "page_content", "") or "")[:180]
            key = f"{page}|{content_prefix}"
            if key in seen:
                continue
            seen.add(key)
            merged.append((doc, score))

    return merged

def _load_faq_canonical() -> dict:
    if not _FAQ_CANONICAL_PATH.exists():
        return{"settings": {}, "policies": []}
    cached = _FAQ_CANONICAL_CACHE.get("data")
    loaded_at = float(_FAQ_CANONICAL_CACHE.get("loaded_at") or 0.0)
    if cached:
        settings = cached.get("settings") or {}
        ttl = float(settings.get("ttl_seconds", 3600))
        if time.time() - loaded_at <= ttl:
            return cached
    with _FAQ_CANONICAL_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _FAQ_CANONICAL_CACHE["data"] = data
    _FAQ_CANONICAL_CACHE["loaded_at"] = time.time()
    return data

def _string_list(values: Any) -> List[str]:
    if values in (None, ""):
        return []
    if isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [values]
    normalized: List[str] = []
    for value in raw_values:
        text = str(value or "").strip()
        if text:
            normalized.append(text)
    return normalized

def _normalize_faq_text(text: Any)  -> str:
    t = str(text or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

def _keyword_overlap_score(query: str, keywords: Any) -> float:
    q_tokens = set(_normalize_faq_text(query).split())
    k_token = {
        token
        for keyword in _string_list(keywords)
        for token in _normalize_faq_text(keyword).split()
        if token
    }
    if not q_tokens or not k_token:
        return 0.0
    overlap = len(q_tokens & k_token)
    return overlap / max(len(k_token), 1)

def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_faq_text(a), _normalize_faq_text(b)).ratio()

def _policy_bias(policy: Dict[str, Any], question_norm: str, question_tokens: set[str]) -> float:
    policy_id = str(policy.get("id") or "").strip().lower()
    bias = 0.0

    if "deposit" in question_tokens:
        if policy_id == "damage_deposit":
            bias += 0.30
        if policy_id == "refund_policy":
            bias -= 0.10

    if {"pet", "pets", "dog", "cat"} & question_tokens and policy_id == "pet_policy":
        bias += 0.20

    if {"refund", "cancel", "cancellation", "check", "checkin"} & question_tokens and policy_id == "refund_policy":
        bias += 0.15

    if "refund of deposit" in question_norm and policy_id == "damage_deposit":
        bias += 0.20

    return bias

def _match_canonical_faq(question: str)  -> Optional[Dict[str, Any]]:
    cfg = _load_faq_canonical()
    settings = cfg.get("settings") or {}
    _fuzzy_th = float(settings.get("fuzzy_threshold", 0.82))
    keyword_th = float(settings.get("keyword_threshold", 0.60))
    combined_th = float(settings.get("combined_threshold", 0.72))

    question_norm = _normalize_faq_text(question)
    question_tokens = set(question_norm.split())

    best = None
    best_score = 0.0

    for policy in cfg.get("policies") or []:
        canonical = str(policy.get("canonical_question") or "").strip()
        answer = str(policy.get("answer") or "").strip()
        if not canonical or not answer:
            continue

        paraphrases = _string_list(policy.get("paraphrases") or [])
        keywords = _string_list(policy.get("keywords") or [])
        candidates = [canonical] + paraphrases
        _fuzzy_score = max((_fuzzy_ratio(question, c) for c in candidates), default=0.0)
        keyword_score = _keyword_overlap_score(question, keywords)
        phrase_score = max(
            (
                1.0
                if (_normalize_faq_text(candidate) and _normalize_faq_text(candidate) in question_norm)
                else 0.0
            )
            for candidate in candidates
        ) if candidates else 0.0
        token_overlap_score = 0.0
        if question_tokens:
            candidate_tokens = {
                token
                for candidate in candidates + keywords
                for token in _normalize_faq_text(candidate).split()
                if token
            }
            if candidate_tokens:
                token_overlap_score = len(question_tokens & candidate_tokens) / max(len(candidate_tokens), 1)

        score = max(_fuzzy_score, keyword_score, phrase_score, token_overlap_score) + _policy_bias(
            policy,
            question_norm,
            question_tokens,
        )
        if score > best_score:
            best_score = score
            best = {
                "id": policy.get("id"),
                "answer": answer,
                "canonical_question": canonical,
                "paraphrases": paraphrases,
                "keywords": keywords,
                "fuzzy_score": _fuzzy_score,
                "keyword_score": keyword_score,
                "phrase_score": phrase_score,
                "token_overlap_score": token_overlap_score,
                "combined_score": score,
            }
    if best and (
        best["fuzzy_score"] >= _fuzzy_th
        or best["keyword_score"] >= keyword_th
        or best["phrase_score"] >= 1.0
        or best["combined_score"] >= combined_th
    ):
        return best
    return None

def lookup_canonical_faq(question: str) -> Optional[Dict[str, Any]]:
    return _match_canonical_faq(question)

def _build_canonical_documents() -> List[Document]:
    cfg = _load_faq_canonical()
    docs: List[Document] = []
    for idx, policy in enumerate(cfg.get("policies") or []):
        canonical = str(policy.get("canonical_question") or "").strip()
        answer = str(policy.get("answer") or "").strip()
        if not canonical or not answer:
            continue
        paraphrases = _string_list(policy.get("paraphrases") or [])
        keywords = _string_list(policy.get("keywords") or [])
        text = "\n".join([
            f"Q: {canonical}",
            f"A: {answer}",
            f"Paraphrases: {', '.join(paraphrases)}",
            f"Keywords: {', '.join(keywords)}",
        ])
        docs.append(Document(     
            page_content=text,
            metadata={
                "source": "Canonical_faq",
                "document": "faq_canonical.yaml",
                "chunk_index": idx,
            },
        ))
    return docs

from pathlib import Path
_FAQ_CANONICAL_PATH = Path('app/config/faq_canonical.yaml')

_FAQ_CANONICAL_CACHE: Dict[str, Any] = {}
_FAQ_CANONICAL_PATH = _BACKEND_ROOT / "data" / "faq_canonical.yaml"
