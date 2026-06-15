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

from app.components.faq_enhanced.canonical import _build_canonical_documents, _merge_ranked_docs
from app.components.faq_enhanced.embeddings import _SentenceTransformerEmbeddings, _local_model_load

class FAQService:
    """
    Encapsulates vector store, embeddings, and FAQ operations.
    Inject via FastAPI app.state or pass explicitly.
    """

    def __init__(self, chroma_path: Optional[Path] = None, openai_api_key: Optional[str] = None):
        self._openai_api_key = openai_api_key or OPENAI_API_KEY
        self._chroma_path = chroma_path or CHROMA_PATH
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        self._vector_store: Optional[Chroma] = None
        self._embeddings = None
        self._documents: List[Document] = []
        self._healthy = False

    @property
    def is_healthy(self) -> bool:
        return self._healthy


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
        configured = str(_RETRIEVAL_CFG.embeddings.device or "auto").strip().lower()
        if configured in {"cpu", "cuda"}:
            return configured

        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"
        return device

    @staticmethod
    def _build_documents_from_text(pdf_text: str) -> List[Document]:
        chunk_cfg = _RETRIEVAL_CFG.chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            length_function=len,
            separators=list(chunk_cfg.separators),
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
                        encode_kwargs={"normalize_embeddings": EMBED_NORMALIZE},
                    )
                return self._embeddings
            except Exception as exc:
                backend_error = exc
                logger.warning("BGE embedding backend unavailable, falling back: %s", exc)

        try:
            self._embeddings = _SentenceTransformerEmbeddings(
                EMBED_MODEL,
                device=device,
                normalize_embeddings=EMBED_NORMALIZE,
            )
            return self._embeddings
        except Exception as exc:
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
            except Exception as exc:
                logger.warning("Could not load lexical FAQ corpus from Chroma: %s", exc)
        if self._vector_store is None and not self._documents:
            pdf_path = Path(__file__).resolve().parents[2] / "data" / "Company policy.pdf"
            if pdf_path.exists():
                self.process_policy_document(str(pdf_path))
            else:
                self._documents = _build_canonical_documents()

        ranked: List[Tuple[int, float]] = []
        try:
            from ..services.rag_pipeline import bm25_search

            ranked = bm25_search(query, [doc.page_content for doc in docs], k=max(k, 1))
        except Exception as exc:
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
        except Exception as exc:
            logger.warning("Embeddings unavailable, using lexical FAQ retrieval: %s", exc)

        if embeddings is not None and os.path.exists(persist_directory) and not force_reload:
            try:
                temp_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings,
                    collection_name=FAQ_COLLECTION_NAME,
                )
                if temp_store._collection.count() > 0:
                    self._vector_store = temp_store
                    self._healthy = True
                    logger.info("Loaded existing vector store")
                    return self._vector_store
                else:
                    logger.info("Existing vector store is empty. Forcing rebuild...")
            except Exception as e:
                logger.warning("Error loading existing store: %s, creating new one", e)

        logger.info("Processing PDF document: %s", pdf_path)
        pdf_text = self.load_pdf_document(pdf_path)
        if not pdf_text:
            raise ValueError("Could not extract text from PDF")
        self._documents = self._build_documents_from_text(pdf_text)
        canonical_docs = _build_canonical_documents()
        if canonical_docs:
            self._documents.extend(canonical_docs)
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
                collection_name=FAQ_COLLECTION_NAME,
            )
            self._healthy = True
            logger.info("Vector store created and persisted")
            return self._vector_store
        except Exception as exc:
            logger.warning("Vector store creation failed, using lexical FAQ retrieval: %s", exc)
            self._vector_store = None
            self._healthy = bool(self._documents)
            return None


    def semantic_search(self, question: str, k: int = 3, score_threshold: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
        from ..services.rag_pipeline import (
            rewrite_query, hybrid_retrieve, rerank,
            compress_context, verify_grounding, get_cag_cache,
        )
        from ..services.dynamic_config import get_retrieval_config

        cache = get_cag_cache()
        cached = cache.get(question)
        if cached is not None:
            return cached

        if self._vector_store is None and not self._documents:
            pdf_path = Path(__file__).resolve().parents[2] / "data" / "Company policy.pdf"
            if not pdf_path.exists():
                return "Company policy document not found. Please ensure the PDF is uploaded.", []
            self.process_policy_document(str(pdf_path))

        rewritten = rewrite_query(question)

        rag_cfg = get_retrieval_config().rag
        retrieval_k = max(k, rag_cfg.vector_k, rag_cfg.bm25_k)
        hybrid_results: List[Tuple[Document, float]] = []
        secondary_results: List[Tuple[Document, float]] = []
        if self._vector_store is not None:
            hybrid_results = hybrid_retrieve(self._vector_store, rewritten, k=retrieval_k)
            if rewritten.strip().lower() != question.strip().lower():
                secondary_results = hybrid_retrieve(self._vector_store, question, k=retrieval_k)

        if not hybrid_results:
            if self._vector_store is not None:
                try:
                    hybrid_results = self._vector_store.similarity_search_with_score(rewritten, k=retrieval_k)
                except Exception as exc:
                    logger.warning("Vector similarity search unavailable, using lexical FAQ retrieval: %s", exc)
            if not hybrid_results:
                hybrid_results = self._keyword_retrieve(rewritten, k=retrieval_k)
            if rewritten.strip().lower() != question.strip().lower() and not secondary_results:
                secondary_results = self._keyword_retrieve(question, k=retrieval_k)

        merged_results = _merge_ranked_docs(hybrid_results, secondary_results)
        relevant_limit = max(k * 2, rag_cfg.vector_k, rag_cfg.bm25_k)
        relevant_docs = list(merged_results[: max(relevant_limit, 1)])
        if not relevant_docs:
            return "I couldn't find specific information about that in our policies. Would you like to speak with a human agent?", []

        docs_only = [doc for doc, _ in relevant_docs]
        rerank_top_n = min(len(docs_only), max(k, rag_cfg.vector_k))
        reranked = rerank(rewritten, docs_only, top_n=rerank_top_n)

        sources: List[Dict[str, Any]] = []
        raw_chunks: List[str] = []
        total_docs = max(len(relevant_docs), 1)
        for doc in reranked:
            content = doc.page_content.strip()
            page = doc.metadata.get("page", "Unknown")
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

        compressed = compress_context(rewritten, raw_chunks)

        from app.components.faq_enhanced.agent import generate_concise_answer
        answer = generate_concise_answer(question, compressed)

        answer, grounding_score = verify_grounding(answer, raw_chunks)
        _grounding_threshold = rag_cfg.grounding_threshold
        if grounding_score < _grounding_threshold:
            answer += "\n\n[Note: Some details may need verification. Please contact support for confirmation.]"


        cache.set(question, (answer, sources))

        return answer, sources


    def initialize(self) -> bool:
        """Initialize FAQ system. Returns True on success."""
        try:
            pdf_path = Path(__file__).resolve().parents[2] / "data" / "Company policy.pdf"
            if pdf_path.exists():
                self.process_policy_document(str(pdf_path))
                logger.info("System initialized successfully")
                return True
            else:
                logger.critical("Company policy.pdf not found - FAQ will not function")
                return False
        except Exception as e:
            logger.critical("Initialization failed: %s", e)
            return False

