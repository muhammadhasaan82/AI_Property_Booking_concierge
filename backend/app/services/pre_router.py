from __future__ import annotations
import hashlib
import re
from typing import Any, Optional, List
from app.config.agent_config_loader import cfg

def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _pick_reply(replies: List[str], user_id: str, session_id: str, message: str) -> str:
    if not replies:
        return ""
    seed = f"{user_id}:{session_id}:{message}".encode("utf-8")
    idx = int(hashlib.md5(seed).hexdigest(), 16) % len(replies)
    return replies[idx]

def route_pre_adk(
    *,
    message: str,
    user_id: str,
    session_id: str,
) -> Optional[dict[str, Any]]:
    config = getattr(cfg, "pre_router", None)
    if not config or not getattr(config, "enabled", False):
        return None
    
    normalized = _normalize(message)
    
    intents = getattr(config, "intents", None)
    if not intents:
        return None

    if not normalized:
        fallback = getattr(intents, "empty_or_unclear", None)
        if not fallback:
            return None
        replies = getattr(fallback, "replies", []) or []
        return {
            "intent": "empty_or_unclear",
            "reply": _pick_reply(replies, user_id, session_id, message),
            "source": "pre_router",
        }

    for intent_name, intent_cfg in vars(intents).items():
        match_cfg = getattr(intent_cfg, "match", None)
        if not match_cfg:
            continue
        exact_values = getattr(match_cfg, "normalized_exact", []) or []
        normalized_exact = {_normalize(value) for value in exact_values}
        if normalized in normalized_exact:
            replies = getattr(intent_cfg,"replies", []) or []
            return {
                "intent": intent_name,
                "reply": _pick_reply(replies, user_id, session_id, message),
                "source": "pre_router",
            }
    
    return None