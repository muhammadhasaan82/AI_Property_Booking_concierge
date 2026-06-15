from __future__ import annotations
"""Enhanced FAQ service package — backward-compatible public API."""

from app.components.faq_enhanced.agent import (
    best_effort_policy_answer,
    detect_faq_intent,
    enhanced_faq_agent,
    generate_concise_answer,
    initialize_faq_system,
)
from app.components.faq_enhanced.canonical import lookup_canonical_faq
from app.components.faq_enhanced.documents import (
    extract_key_sentences,
    load_pdf_document,
    process_policy_document,
    semantic_faq_search,
)
from app.components.faq_enhanced.service import FAQService

_faq_service = FAQService()

__all__ = [
    "FAQService",
    "_faq_service",
    "best_effort_policy_answer",
    "detect_faq_intent",
    "enhanced_faq_agent",
    "extract_key_sentences",
    "generate_concise_answer",
    "initialize_faq_system",
    "load_pdf_document",
    "lookup_canonical_faq",
    "process_policy_document",
    "semantic_faq_search",
]
