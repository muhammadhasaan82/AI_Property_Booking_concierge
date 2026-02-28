"""
Advanced RAG Pipeline Utilities
- Query rewriting
- BM25 keyword search
- Hybrid retrieval (vector + BM25 via Reciprocal Rank Fusion)
- Cross-encoder re-ranking
- Context compression
- Answer grounding verification
- CAG (Cache-Augmented Generation)
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    env_path_svc = Path(__file__).parent / ".env"
    if env_path_svc.exists():
        load_dotenv(env_path_svc)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Lazy-loaded models (avoid import-time downloads)
# ---------------------------------------------------------------------------
_cross_encoder = None
_cross_encoder_lock = threading.Lock()
_rerank_pool = ThreadPoolExecutor(max_workers=2)


def _get_cross_encoder():
    """Lazy-load cross-encoder for re-ranking."""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
                except Exception as e:
                    print(f"[RAG] Cross-encoder unavailable: {e}")
    return _cross_encoder


# ---------------------------------------------------------------------------
# 1. Query Rewriting
# ---------------------------------------------------------------------------
def rewrite_query(user_text: str) -> str:
    """Rewrite a casual user question into a precise policy/FAQ query.

    Uses a single lightweight LLM call.  Falls back to the original text
    on any error so the pipeline never breaks.
    """
    if not OPENAI_API_KEY or not user_text.strip():
        return user_text

    try:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "temperature": 0,
                "max_tokens": 80,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You rewrite informal user questions into clear, precise queries "
                            "for searching a company policy document. Output ONLY the rewritten query, nothing else."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
            },
            timeout=6.0,
        )
        if r.status_code == 200:
            rewritten = r.json()["choices"][0]["message"]["content"].strip()
            if rewritten:
                print(f"[RAG] Query rewritten: {user_text!r} -> {rewritten!r}")
                return rewritten
    except Exception as e:
        print(f"[RAG] Query rewrite failed (using original): {e}")

    return user_text


# ---------------------------------------------------------------------------
# 2. BM25 Keyword Search
# ---------------------------------------------------------------------------
def bm25_search(query: str, corpus: List[str], k: int = 6) -> List[Tuple[int, float]]:
    """Run BM25 over a list of text chunks.

    Returns list of (index, score) sorted descending by score.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[RAG] rank_bm25 not installed, skipping BM25")
        return []

    if not corpus:
        return []

    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# 3. Hybrid Retrieval (Vector + BM25 via Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------
def hybrid_retrieve(
    vector_store,
    query: str,
    k: int = 4,
    vector_k: int = 6,
    bm25_k: int = 6,
    rrf_constant: int = 60,
) -> List[Tuple[Any, float]]:
    """Combine Chroma vector search with BM25 keyword search via RRF.

    Returns list of (Document, fused_score) tuples, highest score first.
    """
    # --- Vector search ---
    try:
        vector_results = vector_store.similarity_search_with_score(query, k=vector_k)
    except Exception as e:
        print(f"[RAG] Vector search failed: {e}")
        vector_results = []

    # --- Build corpus from vector store for BM25 ---
    try:
        collection = vector_store._collection
        all_docs = collection.get(include=["documents"])
        corpus = all_docs.get("documents", []) or []
        doc_ids = all_docs.get("ids", []) or []
    except Exception:
        corpus = []
        doc_ids = []

    # --- BM25 search ---
    bm25_results = bm25_search(query, corpus, k=bm25_k) if corpus else []

    # --- Reciprocal Rank Fusion ---
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Any] = {}

    # Score vector results
    for rank, (doc, score) in enumerate(vector_results):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_constant + rank + 1)
        doc_map[key] = doc

    # Score BM25 results
    for rank, (idx, score) in enumerate(bm25_results):
        if idx < len(corpus):
            text = corpus[idx]
            key = text[:100]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_constant + rank + 1)
            if key not in doc_map:
                from langchain_core.documents import Document
                meta = {}
                if idx < len(doc_ids):
                    meta["id"] = doc_ids[idx]
                doc_map[key] = Document(page_content=text, metadata=meta)

    # Sort by fused score descending
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = [(doc_map[key], score) for key, score in ranked if key in doc_map]

    return results[:k]


# ---------------------------------------------------------------------------
# 4. Cross-Encoder Re-ranking
# ---------------------------------------------------------------------------
def _predict_sync(encoder, pairs: List[Tuple[str, str]]) -> List[float]:
    """Run encoder.predict in a worker thread (CPU-bound)."""
    return encoder.predict(pairs).tolist()


def rerank(query: str, documents: List[Any], top_n: int = 3) -> List[Any]:
    """Re-rank documents using a cross-encoder model.

    The heavy ``encoder.predict()`` call is dispatched to a thread-pool so it
    never blocks the async event loop.
    Falls back to original order if the model is unavailable.
    """
    if not documents:
        return documents

    encoder = _get_cross_encoder()
    if encoder is None:
        return documents[:top_n]

    try:
        pairs = [(query, doc.page_content if hasattr(doc, "page_content") else str(doc)) for doc in documents]
        scores = _rerank_pool.submit(_predict_sync, encoder, pairs).result()
        scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_n]]
    except Exception as e:
        print(f"[RAG] Re-ranking failed (using original order): {e}")
        return documents[:top_n]


# ---------------------------------------------------------------------------
# 5. Context Compression
# ---------------------------------------------------------------------------
def compress_context(question: str, chunks: List[str], max_chars: int = 1500) -> str:
    """Extract only the sentences most relevant to the question.

    Uses keyword overlap scoring to pick the best sentences from all chunks.
    """
    if not chunks:
        return ""

    q_words = set(re.sub(r"[^\w\s]", "", question.lower()).split())
    # Remove stop words
    stop = {"the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
            "what", "how", "can", "i", "my", "me", "we", "our", "to", "of", "in",
            "for", "and", "or", "on", "it", "this", "that", "with", "be", "at"}
    q_words -= stop

    scored_sentences: List[Tuple[str, float]] = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            s_lower = sent.lower()
            hits = sum(1 for w in q_words if w in s_lower)
            if hits > 0:
                score = hits / max(len(q_words), 1)
                scored_sentences.append((sent, score))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    result = ""
    for sent, _ in scored_sentences:
        if len(result) + len(sent) + 2 > max_chars:
            break
        result += sent + " "

    # If compression yielded too little, fall back to truncated raw chunks
    if len(result.strip()) < 50:
        raw = " ".join(chunks)
        return raw[:max_chars]

    return result.strip()


# ---------------------------------------------------------------------------
# 6. Answer Grounding Verification
# ---------------------------------------------------------------------------
def verify_grounding(answer: str, source_chunks: List[str]) -> Tuple[str, float]:
    """Check that each sentence in the answer has supporting evidence.

    Returns (answer, grounding_score).  Score is 0.0-1.0 indicating
    the fraction of answer sentences that are grounded in sources.
    """
    if not answer or not source_chunks:
        return answer, 0.0

    combined_source = " ".join(source_chunks).lower()
    source_words = set(re.sub(r"[^\w\s]", "", combined_source).split())

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    if not sentences:
        return answer, 0.0

    grounded_count = 0
    for sent in sentences:
        sent_words = set(re.sub(r"[^\w\s]", "", sent.lower()).split())
        # Remove common stop words
        meaningful = sent_words - {"the", "a", "an", "is", "are", "was", "were",
                                    "this", "that", "it", "to", "of", "and", "or",
                                    "in", "for", "on", "with", "be", "can", "you",
                                    "your", "our", "we", "i", "my"}
        if not meaningful:
            grounded_count += 1
            continue
        overlap = meaningful & source_words
        ratio = len(overlap) / max(len(meaningful), 1)
        if ratio >= 0.4:
            grounded_count += 1

    score = grounded_count / max(len(sentences), 1)
    return answer, score


# ---------------------------------------------------------------------------
# 7. CAG Cache (Cache-Augmented Generation)
# ---------------------------------------------------------------------------
class CAGCache:
    """Thread-safe in-memory cache for generated FAQ answers.

    Avoids repeated OpenAI calls for identical or near-identical questions.
    """

    def __init__(self, max_entries: int = 500, default_ttl: int = 900):
        self._store: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, created, ttl)
        self._lock = threading.Lock()
        self._max = max_entries
        self._default_ttl = default_ttl

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize query for cache key."""
        t = text.lower().strip()
        t = re.sub(r"[^\w\s]", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        key = self._hash(self._normalize(query))
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, created, ttl = entry
            if (time.time() - created) > ttl:
                del self._store[key]
                return None
            print(f"[CAG] Cache HIT for query")
            return value

    def set(self, query: str, value: Any, ttl: Optional[int] = None):
        key = self._hash(self._normalize(query))
        with self._lock:
            if len(self._store) >= self._max:
                # Evict oldest entry
                oldest_key = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest_key]
            self._store[key] = (value, time.time(), ttl or self._default_ttl)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            now = time.time()
            active = sum(1 for _, (_, c, t) in self._store.items() if (now - c) <= t)
            return {"total": len(self._store), "active": active}


# Module-level singleton
_cag_cache = CAGCache()


def get_cag_cache() -> CAGCache:
    return _cag_cache
