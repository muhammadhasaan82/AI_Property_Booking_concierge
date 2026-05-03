"""
Force-rebuilds the Chroma RAG index from the company policy PDF.
 
Idempotent. Safe to run on every deploy. Use whenever the policy PDF changes.
 
Usage:
    python scripts/rebuild_rag_index.py
    python scripts/rebuild_rag_index.py --pdf /path/to/policy.pdf
    python scripts/rebuild_rag_index.py --no-force
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("rebuild_rag_index")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _default_pdf_path() -> Path:
    return ROOT / "data" / "Company policy.pdf"

def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the RAG vector index.")
    parser.add_argument("--pdf", type=Path, default=_default_pdf_path())
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="only build if no existing index is found",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        logger.error("PDF not found: %s", args.pdf)
        return 2
    
    from app.components.faq_enhanced import process_policy_document, _faq_service

    force = not args.no_force
    logger.info("Rebuilding RAG index from %s (force=%s)", args.pdf, force)
    t0 = time.time()
    try:
        store = process_policy_document(str(args.pdf), force_reload=force)
    except Exception as exc:
        logger.exception("Rebuild Failed: %s", exc)
        return 1

    elapsed = time.time() - t0
    persist_dir  = getatte(_faq_service, "_chroma_path", "<unknown>")
    chunk_count = len(getattr(_faq_service, "_documents",[]) or [])

    logger.info(
        "Rebuild complete: chunks=%d persist_dir=%s elapsed=%.2fs healthy=%s",
        chunk_count, persist_dir, elapsed,
        bool(getattr(_faq_srvices, "_healthy", False)),
    )
    return 0 if store is not None or chunk_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())