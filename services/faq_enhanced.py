"""
Enhanced FAQ Service with PDF Processing and Vector Search
Handles company policy questions using semantic search
"""

from __future__ import annotations
from contextlib import contextmanager
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# PDF Processing
from PyPDF2 import PdfReader

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceBgeEmbeddings
except Exception:  # noqa: BLE001 - package may not be installed in all envs
    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    except Exception:  # noqa: BLE001 - degrade with explicit runtime error in initializer
        HuggingFaceBgeEmbeddings = None  # type: ignore[assignment]

# Load environment variables
from dotenv import load_dotenv
from .dynamic_config import get_vocabulary

env_path_root = Path(__file__).parent.parent / ".env"
env_path_services = Path(__file__).parent / ".env"

if env_path_root.exists():
    load_dotenv(env_path_root)
elif env_path_services.exists():
    load_dotenv(env_path_services)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Initialize ChromaDB path
CHROMA_PATH = Path(__file__).parent.parent / "chroma_faq"
CHROMA_PATH.mkdir(exist_ok=True)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RAG_LOCAL_MODELS_ONLY = os.getenv("RAG_LOCAL_MODELS_ONLY", "1").lower() not in {"0", "false", "no"}


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
        return Path(model_name).expanduser().exists()
    except OSError:
        return False


class _SentenceTransformerEmbeddings:
    """Minimal LangChain-compatible embeddings adapter."""

    def __init__(self, model_name: str, *, device: str = "cpu", normalize_embeddings: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001 - optional dependency in some envs
            raise ImportError("sentence-transformers is unavailable") from exc

        with _local_model_load(RAG_LOCAL_MODELS_ONLY):
            self._model = SentenceTransformer(model_name, device=device)
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

# ---------------------------------------------------------------------------
# FAQService class — replaces global state with dependency injection
# ---------------------------------------------------------------------------

class FAQService:
    """
    Encapsulates vector store, embeddings, and FAQ operations.
    Inject via FastAPI app.state or pass explicitly.
    """

    def __init__(self, chroma_path: Optional[Path] = None, openai_api_key: Optional[str] = None):
        self._openai_api_key = openai_api_key or OPENAI_API_KEY
        self._chroma_path = chroma_path or CHROMA_PATH
        self._chroma_path.mkdir(exist_ok=True)
        self._vector_store: Optional[Chroma] = None
        self._embeddings = None
        self._documents: List[Document] = []
        self._healthy = False

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    # --- PDF ingestion ---

    @staticmethod
    def load_pdf_document(pdf_path: str) -> str:
        """Load and extract text from PDF document."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num}]\n{page_text}\n"
            return text
        except Exception as e:
            logger.error("Error loading PDF: %s", e)
            return ""

    @staticmethod
    def _detect_device() -> str:
        device = "cpu"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"
        return device

    @staticmethod
    def _build_documents_from_text(pdf_text: str) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            length_function=len, separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_text(pdf_text)
        documents = []
        for i, chunk in enumerate(chunks):
            page_match = re.search(r"\[Page (\d+)\]", chunk)
            page_num = page_match.group(1) if page_match else "Unknown"
            clean_chunk = re.sub(r"\[Page \d+\]", "", chunk).strip()
            if not clean_chunk:
                continue
            doc = Document(
                page_content=clean_chunk,
                metadata={"source": "Company Policy", "page": page_num, "chunk_index": i, "document": "company_policy.pdf"},
            )
            documents.append(doc)
        return documents

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return self._embeddings

        if RAG_LOCAL_MODELS_ONLY and not _is_local_model_reference(EMBED_MODEL):
            raise ImportError(
                f"Embedding model '{EMBED_MODEL}' is not a local path. "
                "Set RAG_LOCAL_MODELS_ONLY=0 to allow remote model downloads."
            )

        device = self._detect_device()
        backend_error: Exception | None = None

        if HuggingFaceBgeEmbeddings is not None:
            try:
                with _local_model_load(RAG_LOCAL_MODELS_ONLY):
                    self._embeddings = HuggingFaceBgeEmbeddings(
                        model_name=EMBED_MODEL,
                        model_kwargs={"device": device},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                return self._embeddings
            except Exception as exc:  # noqa: BLE001 - fallback to direct sentence-transformers adapter
                backend_error = exc
                logger.warning("BGE embedding backend unavailable, falling back: %s", exc)

        try:
            self._embeddings = _SentenceTransformerEmbeddings(
                EMBED_MODEL,
                device=device,
                normalize_embeddings=True,
            )
            return self._embeddings
        except Exception as exc:  # noqa: BLE001 - final fallback is lexical retrieval
            backend_error = exc
            logger.warning("SentenceTransformer embedding backend unavailable: %s", exc)

        raise ImportError(
            "No FAQ embedding backend is available. Install langchain-huggingface or ensure "
            "sentence-transformers is installed."
        ) from backend_error

    def _keyword_retrieve(self, query: str, k: int) -> List[Tuple[Document, float]]:
        docs = list(self._documents)
        if not docs and self._vector_store is not None:
            try:
                payload = self._vector_store._collection.get(include=["documents", "metadatas"])
                raw_docs = payload.get("documents", []) or []
                metadatas = payload.get("metadatas", []) or []
                docs = [
                    Document(page_content=text, metadata=metadatas[idx] or {})
                    for idx, text in enumerate(raw_docs)
                    if text
                ]
            except Exception as exc:  # noqa: BLE001 - store introspection is best effort
                logger.warning("Could not load lexical FAQ corpus from Chroma: %s", exc)

        if not docs:
            return []

        ranked: List[Tuple[int, float]] = []
        try:
            from .rag_pipeline import bm25_search

            ranked = bm25_search(query, [doc.page_content for doc in docs], k=max(k, 1))
        except Exception as exc:  # noqa: BLE001 - final fallback uses token overlap
            logger.warning("BM25 FAQ retrieval failed, using token overlap: %s", exc)

        if not ranked:
            tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2}
            scored = []
            for idx, doc in enumerate(docs):
                content = doc.page_content.lower()
                overlap = sum(1 for token in tokens if token in content)
                if overlap:
                    scored.append((idx, float(overlap)))
            ranked = sorted(scored, key=lambda item: item[1], reverse=True)[: max(k, 1)]

        total = max(len(ranked), 1)
        results: List[Tuple[Document, float]] = []
        for rank, (idx, _score) in enumerate(ranked[: max(k, 1)]):
            if idx >= len(docs):
                continue
            results.append((docs[idx], max(0.0, 1.0 - (rank / total))))
        return results

    def process_policy_document(self, pdf_path: str, force_reload: bool = False) -> Optional[Chroma]:
        if self._vector_store is not None and not force_reload:
            return self._vector_store

        persist_directory = str(self._chroma_path)
        embeddings = None
        try:
            embeddings = self._ensure_embeddings()
        except Exception as exc:  # noqa: BLE001 - lexical fallback keeps FAQ usable
            logger.warning("Embeddings unavailable, using lexical FAQ retrieval: %s", exc)

        if embeddings is not None and os.path.exists(persist_directory) and not force_reload:
            try:
                self._vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings,
                    collection_name="company_policies",
                )
                self._healthy = True
                logger.info("Loaded existing vector store")
                return self._vector_store
            except Exception as e:
                logger.warning("Error loading existing store: %s, creating new one", e)

        logger.info("Processing PDF document: %s", pdf_path)
        pdf_text = self.load_pdf_document(pdf_path)
        if not pdf_text:
            raise ValueError("Could not extract text from PDF")
        self._documents = self._build_documents_from_text(pdf_text)
        logger.info("Created %d document chunks", len(self._documents))

        if embeddings is None:
            self._vector_store = None
            self._healthy = bool(self._documents)
            return None

        try:
            self._vector_store = Chroma.from_documents(
                documents=self._documents,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name="company_policies",
            )
            self._healthy = True
            logger.info("Vector store created and persisted")
            return self._vector_store
        except Exception as exc:  # noqa: BLE001 - do not take FAQ offline if vector indexing fails
            logger.warning("Vector store creation failed, using lexical FAQ retrieval: %s", exc)
            self._vector_store = None
            self._healthy = bool(self._documents)
            return None

    # --- Semantic search (enhanced with full RAG pipeline) ---

    def semantic_search(self, question: str, k: int = 3, score_threshold: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
        from .rag_pipeline import (
            rewrite_query, hybrid_retrieve, rerank,
            compress_context, verify_grounding, get_cag_cache,
        )
        from .dynamic_config import get_retrieval_config

        # --- CAG: check cache first ---
        cache = get_cag_cache()
        cached = cache.get(question)
        if cached is not None:
            return cached  # (answer, sources) tuple

        if self._vector_store is None and not self._documents:
            pdf_path = Path(__file__).parent.parent / "Company policy.pdf"
            if not pdf_path.exists():
                return "Company policy document not found. Please ensure the PDF is uploaded.", []
            self.process_policy_document(str(pdf_path))

        # --- Query rewriting ---
        rewritten = rewrite_query(question)

        # --- Hybrid retrieval (vector + BM25 via RRF) ---
        rag_cfg = get_retrieval_config().rag
        hybrid_results: List[Tuple[Document, float]] = []
        if self._vector_store is not None:
            hybrid_results = hybrid_retrieve(self._vector_store, rewritten, k=rag_cfg.vector_k)

        if not hybrid_results:
            if self._vector_store is not None:
                try:
                    hybrid_results = self._vector_store.similarity_search_with_score(rewritten, k=k)
                except Exception as exc:
                    logger.warning("Vector similarity search unavailable, using lexical FAQ retrieval: %s", exc)
            if not hybrid_results:
                hybrid_results = self._keyword_retrieve(rewritten, k=max(k, rag_cfg.bm25_k))

        # Keep top candidates by rank. Raw scores differ by retriever type
        # (RRF, cosine distance, etc.), so rank is more stable than absolute thresholds.
        relevant_docs = list(hybrid_results[: max(k, 1)])
        if not relevant_docs:
            return "I couldn't find specific information about that in our policies. Would you like to speak with a human agent?", []

        # --- Cross-encoder re-ranking ---
        docs_only = [doc for doc, _ in relevant_docs]
        reranked = rerank(rewritten, docs_only, top_n=k)

        # --- Build sources from reranked docs ---
        sources: List[Dict[str, Any]] = []
        raw_chunks: List[str] = []
        total_docs = max(len(relevant_docs), 1)
        for doc in reranked:
            content = doc.page_content.strip()
            page = doc.metadata.get("page", "Unknown")
            # Convert rank to a normalized relevance score in [0,1].
            orig_rank = next(
                (idx for idx, (d, _) in enumerate(relevant_docs) if d.page_content == doc.page_content),
                total_docs - 1,
            )
            normalized_score = max(0.0, 1.0 - (orig_rank / total_docs))
            raw_chunks.append(content)
            sources.append(
                {
                    "content": content,
                    "page": page,
                    "score": normalized_score,
                    "metadata": doc.metadata,
                }
            )

        # --- Context compression ---
        compressed = compress_context(rewritten, raw_chunks)

        # --- Generate answer ---
        answer = generate_concise_answer(question, compressed)

        # --- Answer grounding verification ---
        answer, grounding_score = verify_grounding(answer, raw_chunks)
        _grounding_threshold = rag_cfg.grounding_threshold
        if grounding_score < _grounding_threshold:
            answer += "\n\n[Note: Some details may need verification. Please contact support for confirmation.]"

        # --- Page references ---
        # We are turning off the page reference tags to make the chat feel more natural.
        # pages = list(set(doc.metadata.get("page", "Unknown") for doc in reranked))
        # if pages and pages != ["Unknown"]:
        #     page_refs = ", ".join(f"Page {p}" for p in pages if p != "Unknown")
        #     answer += f"\n\n[Reference: {page_refs} of Company Policy]"

        # --- CAG: store in cache ---
        cache.set(question, (answer, sources))

        return answer, sources

    # --- Health & initialization ---

    def initialize(self) -> bool:
        """Initialize FAQ system. Returns True on success."""
        try:
            pdf_path = Path(__file__).parent.parent / "Company policy.pdf"
            if pdf_path.exists():
                self.process_policy_document(str(pdf_path))
                logger.info("System initialized successfully")
                return True
            else:
                logger.critical("Company policy.pdf not found — FAQ will not function")
                return False
        except Exception as e:
            logger.critical("Initialization failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# Module-level singleton for backwards compatibility
# ---------------------------------------------------------------------------
_faq_service = FAQService()


def load_pdf_document(pdf_path: str) -> str:
    return FAQService.load_pdf_document(pdf_path)


def process_policy_document(pdf_path: str, force_reload: bool = False) -> Optional[Chroma]:
    return _faq_service.process_policy_document(pdf_path, force_reload)


def semantic_faq_search(question: str, k: int = 3, score_threshold: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
    return _faq_service.semantic_search(question, k, score_threshold)


def detect_faq_intent(user_text: str) -> bool:
    """
    Detect if the user's message is asking about policies, terms, or FAQs.

    Uses NLP-powered semantic classification via nlp_engine instead of
    hardcoded keyword arrays.

    Args:
        user_text: User's input text

    Returns:
        True if this appears to be a FAQ/policy question
    """
    from . import nlp_engine
    return nlp_engine.detect_faq_intent(user_text)


def enhanced_faq_agent(user_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhanced FAQ agent that uses semantic search on policy documents
    
    Args:
        user_text: User's question
        context: Optional context from the conversation state
    
    Returns:
        Dictionary with reply and metadata
    """
    if not user_text or not user_text.strip():
        return {
            "reply": "Please ask me a specific question about our policies or services.",
            "tool_result": {"ok": False, "error": "Empty question"}
        }
    
    try:
        # Perform semantic search
        answer, sources = semantic_faq_search(user_text)
        
        # Check if we found a good answer
        top_score = float((sources[0] or {}).get("score", 0.0)) if sources else 0.0
        from .dynamic_config import get_thresholds
        _faq_thresholds = get_thresholds().faq
        if sources and answer and top_score >= _faq_thresholds.high_confidence:
            # High confidence answer
            result = {
                "reply": answer,
                "tool_result": {
                    "ok": True,
                    "sources": sources,
                    "confidence": "high"
                }
            }
        elif sources and answer and top_score >= _faq_thresholds.low_confidence:
            # Medium confidence - add disclaimer
            result = {
                "reply": f"{answer}\n\n[Note: If this doesn't fully answer your question, I can connect you with our support team.]",
                "tool_result": {
                    "ok": True,
                    "sources": sources,
                    "confidence": "medium"
                }
            }
        elif sources and answer:
            # Still provide best effort answer when retrieval returned evidence.
            result = {
                "reply": f"{answer}\n\n[Note: If you want, I can also connect you with support for confirmation.]",
                "tool_result": {
                    "ok": True,
                    "sources": sources,
                    "confidence": "low"
                }
            }
        else:
            # Low confidence or no results
            result = {
                "reply": "I couldn't find specific information about that in our policies. Would you like me to:\n1. Try rephrasing your question\n2. Connect you with our support team\n3. Continue with your booking",
                "tool_result": {
                    "ok": False,
                    "confidence": "low",
                    "need_clarification": True
                }
            }
        
        # Add context preservation and continuation prompt if in booking flow
        if context and context.get("in_booking_flow"):
            result["preserve_context"] = True
            result["return_to"] = context.get("return_to", "booking")
            result["reply"] += (
                "\n\nWould you like to continue your booking now, or ask another FAQ? "
                "Feel free to ask more policy questions."
            )
        
        return result
        
    except Exception as e:
        logger.error("Error in FAQ agent: %s", e)
        fallback_answer = best_effort_policy_answer(user_text)
        if fallback_answer:
            return {
                "reply": fallback_answer,
                "tool_result": {
                    "ok": True,
                    "confidence": "fallback",
                    "fallback": "lexical",
                },
            }
        return {
            "reply": "I'm having trouble accessing the policy information right now. Please try again or contact support directly.",
            "tool_result": {
                "ok": False,
                "error": str(e)
            }
        }


# Initialize the vector store on module load
def initialize_faq_system():
    """Initialize the FAQ system with the company policy document."""
    return _faq_service.initialize()


def best_effort_policy_answer(question: str) -> Optional[str]:
    pdf_path = Path(__file__).parent.parent / "Company policy.pdf"
    if pdf_path.exists():
        text = load_pdf_document(str(pdf_path))
        if text:
            answer = extract_key_sentences(text, question)
            if answer:
                return answer

    try:
        from .faq import faq_lookup

        return faq_lookup(question)
    except Exception as exc:  # noqa: BLE001 - FAQ table may be offline too
        logger.warning("Basic FAQ fallback failed: %s", exc)
        return None


def generate_concise_answer(question: str, context: str) -> str:
    """
    Generate a concise, summarized answer using OpenAI
    
    Args:
        question: User's question
        context: Retrieved policy text
    
    Returns:
        Concise answer (5-20 lines based on complexity)
    """
    if not OPENAI_API_KEY:
        # Fallback to extracting key sentences if no OpenAI key
        return extract_key_sentences(context, question)
    
    try:
        import httpx
        
        # Determine question complexity using VADER + heuristics
        from . import nlp_engine
        vader = nlp_engine._get_vader()
        q_lower = question.lower()
        q_words = len(q_lower.split())
        scores = vader.polarity_scores(q_lower)
        # Simple: short, direct questions; Complex: longer, analytical questions
        is_simple = q_words <= 6 and "?" in question
        is_complex = q_words > 10 or any(w in q_lower for w in get_vocabulary().nlp_fallback.faq_complex_indicators)
        
        # Set length guidance
        if is_simple and not is_complex:
            length_guide = "Provide a brief, direct answer in 3-5 lines."
        elif is_complex:
            length_guide = "Provide a comprehensive but concise answer in 10-15 lines, covering key points."
        else:
            length_guide = "Provide a clear, concise answer in 5-10 lines."
        
        system_prompt = f"""You are a polite, helpful property rental assistant. Answer questions based ONLY on the provided policy text.
Think step by step: 1) Identify the relevant policy section, 2) Extract the specific answer, 3) Provide a clear response.
{length_guide}
CRITICAL FORMATTING RULES:
- Use clean Markdown formatting with bullet points and bold text where appropriate.
- DO NOT include raw PDF artifacts, document titles, or headers (e.g., do not say "xyz company Internal Policies & Terms v1.0").
- Speak directly and naturally to the user.
- Do not add information not present in the context."""
        
        user_prompt = f"""Based on the following policy text, answer this question concisely:

Question: {question}

Policy Text:
{context[:2000]}  # Limit context to avoid token limits

Provide a clear, direct answer:"""
        
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            return _clean_pdf_artifacts(answer)
        else:
            logger.error("OpenAI API error: %s", response.status_code)
            return extract_key_sentences(context, question)
            
    except Exception as e:
        logger.error("Error generating concise answer: %s", e)
        return extract_key_sentences(context, question)


def _clean_pdf_artifacts(text: str) -> str:
    """Remove common PDF artifacts like headers, footers, and artifact text."""
    if not text:
        return text
    
    # Remove common PDF header patterns (company names, document titles)
    patterns_to_remove = [
        r'xyz\s*company\s*Internal\s*Policies?\s*&\s*Terms?\s*v?\d+\.\d+',
        r'Company\s*Policy',
        r'Internal\s*Policies?\s*&\s*Terms?',
        r'Property\s*Rental\s*Company\s*-\s*Policies?\s*&\s*Terms?',
    ]
    
    cleaned = text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
    # Remove any remaining page-like artifacts at the start
    cleaned = re.sub(r'^\s*[-=]+\s*$', '', cleaned, flags=re.MULTILINE)
    
    return cleaned.strip()


def extract_key_sentences(context: str, question: str, max_lines: int = 10) -> str:
    """
    Fallback method to extract key sentences when OpenAI is not available
    
    Args:
        context: Full policy text
        question: User's question
        max_lines: Maximum number of lines to return
    
    Returns:
        Key sentences related to the question
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', context)
    
    # Find sentences with keywords from the question
    question_words = set(question.lower().split())
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Count matching words
        matches = sum(1 for word in question_words if word in sentence_lower)
        fallback_kws = get_vocabulary().nlp_fallback.faq_seeds + get_vocabulary().nlp_fallback.faq_strong_keywords
        if matches >= 2 or any(kw in sentence_lower for kw in fallback_kws):
            relevant_sentences.append(sentence.strip())
    
    # Return the most relevant sentences
    result = " ".join(relevant_sentences[:5])
    
    # Clean up the text
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'[©\[\]]', '', result)
    
    # Limit to reasonable length
    if len(result) > 800:
        result = result[:800] + "..."
    
    return _clean_pdf_artifacts(result)


# Optional: Auto-initialize when module is imported
# Uncomment the line below if you want automatic initialization
# initialize_faq_system()

