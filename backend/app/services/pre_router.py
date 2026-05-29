from __future__ import annotations
import asyncio
import logging
import re
from typing import Any, Optional
import litellm
from app.config.agent_config_loader import cfg
from app.config.model_config_loader import load_model_config

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_intent(message: str) -> Optional[str]:
    """
    Deterministically classify a user message into a configured pre-router intent.
    
    If the pre-router is disabled or no intents are configured, no intent is selected. If the normalized message is empty and an `empty_or_unclear` intent exists, that intent name is returned. Matching uses intent-configured exact, starts-with, or contains-any phrases; if a matched intent has `defer_to_adk` set to True, no intent is selected.
    
    Returns:
        intent_name (str): The matching intent's name, or `None` when no intent is selected.
    """
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
            contain_list = getattr(match_cfg, "normalized_contains_any", []) or []
            for phrase in contain_list:
                if _normalize(phrase) in normalized:
                    matched = True
                    break

        if matched:
            if getattr(intent_cfg, "defer_to_adk", False):
                return None
            return intent_name

    return None

async def _generate_reply(intent_name: str, user_message: str) -> str:
    """
    Generate a reply for the given intent using a fast LLM and the intent's configured role.
    
    Reads generator settings (temperature, max tokens, timeout) and the pre-router fast model from configuration, invokes the model with the intent's role as the system prompt and the user's message as the user prompt, and returns the model's trimmed output. If the model call fails or produces no content, returns the pre_router emergency fallback or an empty string.
    
    Returns:
        reply (str): The trimmed model-generated reply, or the configured emergency fallback/empty string on failure.
    """
    config = getattr(cfg, "pre_router", None)
    intents = getattr(config, "intents", None)
    intent_cfg = getattr(intents, intent_name, None)
    role = getattr(intent_cfg, "role", "") if intent_cfg else ""

    gen = getattr(config, "generator", None)
    model = load_model_config().pre_router_fast_model
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
    """
    Attempt to pre-route a user message by detecting a deterministic intent and, if found, generating an intent-specific reply.
    
    Parameters:
    	message (str): The user's message to classify and respond to.
    	user_id (str): Identifier for the user (accepted for interface compatibility; not used).
    	session_id (str): Identifier for the session (accepted for interface compatibility; not used).
    
    Returns:
    	dict[str, Any]: A dictionary with keys `intent` (the matched intent name), `reply` (the generated reply text), and `source` set to `"pre_router"` when a reply was produced.
    	None: If no intent was detected or no reply could be generated, indicating downstream handling should proceed.
    """
    intent = _detect_intent(message)
    if intent is None:
        return None

    reply = await _generate_reply(intent, message)
    if not reply:
        return None

    return {"intent": intent, "reply": reply, "source": "pre_router"}
