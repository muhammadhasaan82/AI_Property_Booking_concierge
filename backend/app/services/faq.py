##services/faq.py
from __future__ import annotations
import os
from typing import Optional
from supabase import create_client, Client

_SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# Use service role key for server-side operations (bypasses RLS appropriately).
# The anon key is scoped to public/unauthenticated access and must NOT be used
# for server-side writes.
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
    sb = _sb_client()
    q = (question or "").strip()
    if not q:
        return None

    # 1) Try exact case-insensitive
    res = sb.table("faqs").select("answer").ilike("question", q).limit(1).execute()
    if res.data:
        return res.data[0]["answer"]

    # 2) Try contains match (wrap with %...%)
    res2 = sb.table("faqs").select("answer").ilike("question", f"%{q}%").limit(1).execute()
    if res2.data:
        return res2.data[0]["answer"]

    return None