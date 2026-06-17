from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.agents.status_codes import Status
from app.services.booking.constants import _FAQ_KEYWORDS
from app.services.booking.state import _normalize
from app.services.faq_interruption import (
    build_answer_with_resume_prompt,
    capture_faq_interruption,
)

def _is_booking_faq(message: str) -> bool:
    normalized = _normalize(message)
    if not normalized:
        return False
    if any(keyword in normalized for keyword in _FAQ_KEYWORDS):
        return True
    if "?" in message and not re.search(r"\b(email|phone|guest|guests|check[- ]?in|check[- ]?out|name)\b", normalized):
        return True
    return False

def _enrich_booking_faq_payload(
    payload: Optional[Dict[str, Any]],
    soft_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return payload

    status = str(payload.get("status") or "").strip().lower()
    if status != str(Status.ANSWERED).lower():
        return payload

    answer = str(payload.get("answer") or "").strip()
    if not answer:
        return payload

    deterministic_reply = str(payload.get("deterministic_reply") or "").strip()
    if deterministic_reply:
        return payload

    faq_intent = str(payload.get("faq_intent") or "").strip() or None
    capture_faq_interruption(soft_state, last_faq_intent=faq_intent)

    enriched = dict(payload)
    enriched["deterministic_reply"] = build_answer_with_resume_prompt(answer, soft_state)
    return enriched

