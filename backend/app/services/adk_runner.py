# services/adk_runner.py
"""
ADK 2.0 Runner — Execution bridge between Chainlit and the ADK SequentialAgent.

Manages per-user sessions via InMemorySessionService and provides
`run_adk_turn()` as the single async entry point for the Chainlit UI.

Phase 2: Core ADK pipeline.
Phase 3: DPO telemetry capture + tool-loop anomaly detection.

Feature flag: ADK_ENABLED (env var, default "1"). Set to "0" to fall back
to the V1 LangGraph pipeline (run_chat_graph).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from .guardrails import sanitize_input, sanitize_output
from . import anomaly
from . import telemetry

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
    t0 = time.monotonic()

    final_reply = ""
    tool_calls_log: List[Dict[str, Any]] = []
    anomaly_triggered = False

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        ):
            # --- Phase 3: Capture tool call events for telemetry ---
            tool_name, tool_params = _extract_tool_call(event)
            if tool_name:
                # Anomaly check BEFORE recording
                if anomaly.check_tool_loop(session_id, tool_name, tool_params):
                    anomaly_triggered = True
                    tool_calls_log.append({
                        "tool": tool_name,
                        "params_hash": hashlib.md5(
                            json.dumps(tool_params, sort_keys=True, default=str).encode()
                        ).hexdigest()[:12],
                        "result_status": "anomaly_blocked",
                    })
                    break
                anomaly.record_tool_call(session_id, tool_name, tool_params)
                tool_calls_log.append({
                    "tool": tool_name,
                    "params_hash": hashlib.md5(
                        json.dumps(tool_params, sort_keys=True, default=str).encode()
                    ).hexdigest()[:12],
                    "result_status": "ok",
                })

            if event.is_final_response():
                if event.content and event.content.parts:
                    final_reply = "".join(
                        part.text for part in event.content.parts if hasattr(part, "text") and part.text
                    )
                break
    except Exception as e:
        logger.error("[ADK] Pipeline execution error: %s", e, exc_info=True)
        logger.warning("[ADK] Falling back to V1 pipeline due to error")
        return await _v1_fallback(message)

    latency_ms = (time.monotonic() - t0) * 1000.0

    # --- Phase 3: Anomaly — return graceful fallback ---
    if anomaly_triggered:
        final_reply = anomaly.GRACEFUL_FALLBACK_REPLY

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

    # --- Phase 3: Fire-and-forget telemetry + chat logging ---
    try:
        asyncio.create_task(
            telemetry.log_trajectory(
                session_id=session_id,
                user_id=user_id,
                user_message=cleaned_message,
                tool_calls=tool_calls_log,
                final_reply=final_reply,
                latency_ms=latency_ms,
            )
        )
    except Exception:
        pass

    try:
        from .db_logging import log_chat
        asyncio.create_task(log_chat(cleaned_message, final_reply))
    except Exception:
        pass

    return final_reply


def _extract_tool_call(event: Any) -> tuple:
    """Extract tool name and params from an ADK event, if it's a tool call.

    Returns (tool_name, tool_params) or (None, None).
    """
    try:
        # ADK events with function calls have content.parts with function_call
        if event.content and event.content.parts:
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    name = getattr(fc, "name", None)
                    args = getattr(fc, "args", None)
                    if name:
                        return (name, args or {})
    except Exception:
        pass
    return (None, None)


async def _v1_fallback(message: str) -> str:
    """Fall back to the V1 LangGraph pipeline."""
    from .graph import run_chat_graph

    result = await run_chat_graph(message=message)
    return result.get("reply", "Sorry, I didn't understand that.")
