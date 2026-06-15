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

from app.components.faq_enhanced.service import FAQService


def _faq_service():
    from app.components import faq_enhanced as pkg
    return pkg._faq_service


def load_pdf_document(pdf_path: str) -> str:
    return FAQService.load_pdf_document(pdf_path)

def process_policy_document(pdf_path: str, force_reload: bool = False) -> Optional[Chroma]:
    return _faq_service().process_policy_document(pdf_path, force_reload)

def semantic_faq_search(question: str, k: int = 3, score_threshold: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
    return _faq_service().semantic_search(question, k, score_threshold)

def _clean_pdf_artifacts(text: str) -> str:
    """Remove common PDF artifacts like headers, footers, and artifact text."""
    if not text:
        return text
    
    patterns_to_remove = [
        r'xyz\s*company\s*Internal\s*Policies?\s*&\s*Terms?\s*v?\d+\.\d+',
        r'Company\s*Policy',
        r'Internal\s*Policies?\s*&\s*Terms?',
        r'Property\s*Rental\s*Company\s*-\s*Policies?\s*&\s*Terms?',
    ]
    
    cleaned = text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
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
    sentences = re.split(r'(?<=[.!?])\s+', context)
    
    question_words = set(question.lower().split())
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matches = sum(1 for word in question_words if word in sentence_lower)
        fallback_kws = get_vocabulary().nlp_fallback.faq_seeds + get_vocabulary().nlp_fallback.faq_strong_keywords
        if matches >= 2 or any(kw in sentence_lower for kw in fallback_kws):
            relevant_sentences.append(sentence.strip())
    
    result = " ".join(relevant_sentences[:5])
    
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'[\[\]]', '', result)
    
    if len(result) > 800:
        result = result[:800] + "..."
    
    return _clean_pdf_artifacts(result)

