import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import faq_enhanced


def test_process_policy_document_falls_back_to_lexical_retrieval(monkeypatch, tmp_path):
    service = faq_enhanced.FAQService(chroma_path=tmp_path / "chroma")

    def raise_embeddings():
        raise ImportError("missing embeddings backend")

    monkeypatch.setattr(service, "_ensure_embeddings", raise_embeddings)
    monkeypatch.setattr(
        service,
        "load_pdf_document",
        lambda pdf_path: "\n[Page 2]\nRefunds are allowed within 24 hours of booking.\n",
    )

    vector_store = service.process_policy_document("Company policy.pdf", force_reload=True)

    assert vector_store is None
    assert service.is_healthy is True
    assert service._documents

    results = service._keyword_retrieve("refund policy", k=1)
    assert results
    assert "Refunds are allowed within 24 hours" in results[0][0].page_content


def test_enhanced_faq_agent_uses_best_effort_fallback(monkeypatch):
    def raise_semantic_search(*_args, **_kwargs):
        raise RuntimeError("semantic search offline")

    monkeypatch.setattr(faq_enhanced, "semantic_faq_search", raise_semantic_search)
    monkeypatch.setattr(
        faq_enhanced,
        "best_effort_policy_answer",
        lambda question: "Refunds are allowed within 24 hours of booking.",
    )

    result = faq_enhanced.enhanced_faq_agent("please let me know the refund policy")

    assert result["tool_result"]["ok"] is True
    assert result["tool_result"]["fallback"] == "lexical"
    assert "Refunds are allowed within 24 hours" in result["reply"]
