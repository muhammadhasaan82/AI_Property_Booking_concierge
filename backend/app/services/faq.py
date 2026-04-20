from __future__ import annotations
import os
from typing import Optional
from supabase import create_client, Client

from app.services.config import MOCK_MODE

_SUPABASE_URL = os.getenv("SUPABASE_URL", "")
_SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_sb: Client | None = None
def _sb_client() -> Client:
    global _sb
    if _sb is None:
        if not (_SUPABASE_URL and _SUPABASE_SERVICE_KEY):
            raise RuntimeError("Supabase env not set (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required)")
        _sb = create_client(_SUPABASE_URL, _SUPABASE_SERVICE_KEY)
    return _sb

def faq_lookup(question: str) -> Optional[str]:
    """
    Tries exact-ish match first, then a loose ILIKE contains.
    Returns the answer string or None.
    """
    if MOCK_MODE:
        return "Based on our mock policy, the answer to your question is: All reservations are subject to standard policies. Check out by 11am, and full refund available up to 48 hours before check-in."

    try:
        sb = _sb_client()
    except RuntimeError:
        return "I'm sorry, I cannot look up that policy right now because the FAQ database isn't connected."

    q = (question or "").strip()
    if not q:
        return None

    try:

        res = sb.table("faqs").select("answer").ilike("question", q).limit(1).execute()
        if res.data:
            return res.data[0]["answer"]

        res2 = sb.table("faqs").select("answer").ilike("question", f"%{q}%").limit(1).execute()
        if res2.data:
            return res2.data[0]["answer"]
    except Exception as e:
        print(f"[FAQ Database Error] {e}")
        return "I'm sorry, my FAQ knowledge base is currently inaccessible."

    return None