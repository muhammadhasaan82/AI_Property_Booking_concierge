# services/adk_runner.py
"""
ADK 2.0 Runner — Execution bridge between Chainlit and the ADK SequentialAgent.

Manages per-user sessions via InMemorySessionService and provides
`run_adk_turn()` as the single async entry point for the Chainlit UI.

Feature flag: ADK_ENABLED (env var, default "1"). Set to "0" to fall back
to the V1 LangGraph pipeline (run_chat_graph).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from .guardrails import sanitize_input, sanitize_output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------
ADK_ENABLED = os.getenv("ADK_ENABLED", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# Lazy-init globals (created on first call to avoid import-time side effects)
# ---------------------------------------------------------------------------
_session_service: Optional[InMemorySessionService] = None
_runner: Optional[Runner] = None

APP_NAME = "ai_concierge"


def _get_runner() -> Runner:
    """Lazily initialize the ADK Runner and session service."""
    global _session_service, _runner
    if _runner is None:
        from .adk_agents import root_agent

        _session_service = InMemorySessionService()
        _runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=_session_service,
        )
        logger.info("[ADK] Runner initialized with agent '%s'", root_agent.name)
    return _runner


def _get_session_service() -> InMemorySessionService:
    """Return the session service, initializing the runner if needed."""
    _get_runner()
    return _session_service


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_adk_turn(
    user_id: str,
    session_id: str,
    message: str,
) -> str:
    """Process a single conversation turn through the ADK pipeline.

    Args:
        user_id: Unique user identifier (from Chainlit session).
        session_id: Conversation thread ID.
        message: The user's message text.

    Returns:
        The agent's response string.
    """
    # --- Feature flag: V1 fallback ---
    if not ADK_ENABLED:
        return await _v1_fallback(message)

    # --- Sanitize input ---
    cleaned_message, is_safe = sanitize_input(message)
    if not is_safe:
        return "I'm sorry, I can't process that request. Could you rephrase?"

    if not cleaned_message.strip():
        return "I didn't catch that. Could you repeat your question?"

    runner = _get_runner()
    session_service = _get_session_service()

    # --- Ensure session exists ---
    try:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        session = None

    if session is None:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        logger.info("[ADK] Created new session %s for user %s", session_id, user_id)

    # --- Execute the pipeline ---
    user_content = Content(parts=[Part(text=cleaned_message)])

    final_reply = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_reply = "".join(
                        part.text for part in event.content.parts if hasattr(part, "text") and part.text
                    )
                break
    except Exception as e:
        logger.error("[ADK] Pipeline execution error: %s", e, exc_info=True)
        # Fall back to V1 on error
        logger.warning("[ADK] Falling back to V1 pipeline due to error")
        return await _v1_fallback(message)

    # --- Try to get final_reply from session state if event didn't have it ---
    if not final_reply:
        try:
            updated_session = await session_service.get_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
            if updated_session and updated_session.state:
                final_reply = updated_session.state.get("final_reply", "")
                if not final_reply:
                    final_reply = updated_session.state.get("router_output", "")
        except Exception:
            pass

    if not final_reply:
        final_reply = "I'm sorry, I couldn't process your request. Could you try again?"

    # --- Sanitize output ---
    final_reply = sanitize_output(final_reply)

    # --- Best-effort chat logging ---
    try:
        from .db_logging import log_chat
        import asyncio
        asyncio.create_task(log_chat(cleaned_message, final_reply))
    except Exception:
        pass

    return final_reply


async def _v1_fallback(message: str) -> str:
    """Fall back to the V1 LangGraph pipeline."""
    from .graph import run_chat_graph

    result = await run_chat_graph(message=message)
    return result.get("reply", "Sorry, I didn't understand that.")
