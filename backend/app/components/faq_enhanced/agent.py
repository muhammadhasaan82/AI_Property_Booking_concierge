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

from app.components.faq_enhanced.canonical import lookup_canonical_faq
from app.components.faq_enhanced.documents import (
    extract_key_sentences,
    load_pdf_document,
    semantic_faq_search,
    _clean_pdf_artifacts,
)

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
    from app.components import nlp_engine
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
    match = lookup_canonical_faq(user_text)
    if match:
        result = {
            "reply": match["answer"],
            "tool_result": {
                "ok": True,
                "confidence": "deterministic",
                "source": "canonical_faq",
                "match": {
                    "id": match["id"],
                    "canonical_question": match["canonical_question"],
                    "fuzzy_score": match["fuzzy_score"],
                    "keyword_score": match["keyword_score"],
                }
            },
        }
        if context and context.get("in_booking_flow"):
            result["preserve_context"] = True
            result["return_to"] = context.get("return_to", "booking")
        return result

    try:
        answer, sources = semantic_faq_search(user_text)
        
        top_score = float((sources[0] or {}).get("score", 0.0)) if sources else 0.0
        from ..services.dynamic_config import get_thresholds
        _faq_thresholds = get_thresholds().faq
        if sources and answer and top_score >= _faq_thresholds.high_confidence:
            result = {
                "reply": answer,
                "tool_result": {
                    "ok": True,
                    "sources": sources,
                    "confidence": "high"
                }
            }
        elif sources and answer and top_score >= _faq_thresholds.low_confidence:
            result = {
                "reply": f"{answer}\n\n[Note: If this doesn't fully answer your question, I can connect you with our support team.]",
                "tool_result": {
                    "ok": True,
                    "sources": sources,
                    "confidence": "medium"
                }
            }
        elif sources and answer:
            result = {
                "reply": f"{answer}\n\n[Note: If you want, I can also connect you with support for confirmation.]",
                "tool_result": {
                    "ok": True,
                    "sources": sources,
                    "confidence": "low"
                }
            }
        else:
            result = {
                "reply": "I couldn't find specific information about that in our policies. Would you like me to:\n1. Try rephrasing your question\n2. Connect you with our support team\n3. Continue with your booking",
                "tool_result": {
                    "ok": False,
                    "confidence": "low",
                    "need_clarification": True
                }
            }
        
        if context and context.get("in_booking_flow"):
            result["preserve_context"] = True
            result["return_to"] = context.get("return_to", "booking")
        
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

def initialize_faq_system():
    """Initialize the FAQ system with the company policy document."""
    from app.components import faq_enhanced as pkg
    return pkg._faq_service.initialize()

def best_effort_policy_answer(question: str) -> Optional[str]:
    pdf_path = Path(__file__).resolve().parents[2] / "data" / "Company policy.pdf"
    if pdf_path.exists():
        text = load_pdf_document(str(pdf_path))
        if text:
            answer = extract_key_sentences(text, question)
            if answer:
                return answer

    try:
        from app.services.faq import faq_lookup

        return faq_lookup(question)
    except Exception as exc:
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
        return extract_key_sentences(context, question)
    
    try:
        from app.components import nlp_engine
        vader = nlp_engine.models._get_vader()
        q_lower = question.lower()
        q_words = len(q_lower.split())
        scores = vader.polarity_scores(q_lower)
        is_simple = q_words <= 6 and "?" in question
        is_complex = q_words > 10 or any(
            w in q_lower for w in get_vocabulary().nlp_fallback.faq_complex_indicators
        )

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
- DO NOT include raw PDF artifacts, document titles, or headers.
- Speak directly and naturally to the user.
- Do not add information not present in the context."""

        user_prompt = (
            "Based on the following policy text, answer this question concisely:\n\n"
            f"Question: {question}\n\n"
            f"Policy Text:\n{context[:2000]}\n\n"
            "Provide a clear, direct answer:"
        )

        response = litellm.completion(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1500,
        )

        content = ""
        if getattr(response, "choices", None):
            message = getattr(response.choices[0], "message", None)
            content = str(getattr(message, "content", "") or "").strip()
        if not content and isinstance(response, dict):
            content = str(
                response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            ).strip()

        if not content:
            return extract_key_sentences(context, question)

        answer = content
        return _clean_pdf_artifacts(answer)

    except Exception as e:
        logger.error("Error generating concise answer: %s", e)
        return extract_key_sentences(context, question)

