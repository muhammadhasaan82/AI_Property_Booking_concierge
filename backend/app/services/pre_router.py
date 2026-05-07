from __future__ import annotations
import asyncio
import logging
import os
import re
from typing import Any, Optional
import litellm
from app.config.agent_config_loader import cfg

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_intent(message: str) -> Optional[str]:
    """Deterministic classification — returns intent name or None."""
    config = getattr(cfg, "pre_router", None)
    if not config or not getattr(config, "enabled", False):
        return None

    intents = getattr(config, "intents", None)
    if not intents:
        return None

    normalized = _normalize(message)
    if not normalized:
        return "empty_or_unclear" if hasattr(intents, "empty_or_unclear") else None

    for intent_name, intent_cfg in vars(intents).items():
        match_cfg = getattr(intent_cfg, "match", None)
        if not match_cfg:
            continue

        matched = False

        exact = {_normalize(v) for v in (getattr(match_cfg, "normalized_exact", []) or [])}
        if normalized in exact:
            matched = True

        if not matched:
            prefixes = [_normalize(v) for v in (getattr(match_cfg, "normalized_starts_with", []) or [])]
            for prefix in prefixes:
                if not prefix:
                    continue
                if normalized == prefix or normalized.startswith(prefix + " "):
                    matched = True
                    break

        if not matched:
            contains_list = getattr(match_cfg, "normalized_contains_any", []) or []
            for phrase in contains_list:
                if _normalize(phrase) in normalized:
                    matched = True
                    break

        if matched:
            if getattr(intent_cfg, "defer_to_adk", False):
                return None
            return intent_name

async def _generate_reply(intent_name: str, user_message: str) -> str:
    """Probabilistic reply generation via fast LLM, driven by YAML role."""
    config = getattr(cfg, "pre_router", None)
    intents = getattr(config, "intents", None)
    intent_cfg = getattr(intents, intent_name, None)
    role = getattr(intent_cfg, "role", "") if intent_cfg else ""

    gen = getattr(config, "generator", None)
    model = os.getenv(getattr(gen, "model_env_key", "PRE_ROUTER_FAST_MODEL"),
                      getattr(gen, "model_default", "groq/llama-3.1-8b-instant"))
    temperature = float(getattr(gen, "temperature", 0.7))
    max_tokens = int(getattr(gen, "max_tokens", 80))
    timeout = float(getattr(gen, "timeout_seconds", 4))

    def _call() -> str:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": role.strip()},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        msg = getattr(choice, "message", None) if choice else None
        content = getattr(msg, "content", "") if msg else ""
        return (content or "").strip()

    try:
        return await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout)
    except Exception as exc:
        logger.warning("[pre_router] fast LLM failed for intent=%s: %s", intent_name, exc)
        return getattr(config, "emergency_fallback", "") or ""


async def route_pre_adk(
    *,
    message: str,
    user_id: str,
    session_id: str,
) -> Optional[dict[str, Any]]:
    """Two-stage: deterministic detect → probabilistic generate. Returns None to defer to ADK."""
    intent = _detect_intent(message)
    if intent is None:
        return None

    reply = await _generate_reply(intent, message)
    if not reply:
        return None

    return {"intent": intent, "reply": reply, "source": "pre_router"}