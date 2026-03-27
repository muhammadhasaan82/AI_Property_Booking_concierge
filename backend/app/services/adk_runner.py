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
from typing import Any, AsyncGenerator, Dict, List, Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from ..security.guardrails import sanitize_input, sanitize_output
from ..security import anomaly
from ..observability import telemetry

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
        from ..agents.adk_agents import root_agent

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
) -> AsyncGenerator[str, None]:
    """Process a single conversation turn, yielding text chunks as an async generator.

    Two-Speed Rule:
      - triage_router events (tool calls, routing) are silently consumed.
      - concierge_voice text deltas are yielded immediately to the caller.

    Args:
        user_id: Unique user identifier (from Chainlit session).
        session_id: Conversation thread ID.
        message: The user's message text.

    Yields:
        Text chunks from the concierge_voice agent as they arrive.
    """
    # --- Feature flag: V1 fallback ---
    if not ADK_ENABLED:
        async for chunk in _v1_fallback_stream(message):
            yield chunk
        return

    # --- Sanitize input ---
    cleaned_message, is_safe = sanitize_input(message)
    if not is_safe:
        yield "I'm sorry, I can't process that request. Could you rephrase?"
        return

    if not cleaned_message.strip():
        yield "I didn't catch that. Could you repeat your question?"
        return

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

    streamed_parts: List[str] = []
    tool_calls_log: List[Dict[str, Any]] = []
    anomaly_triggered = False

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        ):
            # --- Tool call events: capture silently, never stream to user ---
            tool_name, tool_params = _extract_tool_call(event)
            if tool_name:
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
                continue  # SILENT: never stream tool calls

            # --- INSTANT VOICE: stream text deltas from concierge_voice only ---
            author = getattr(event, "author", None)
            if author == "concierge_voice" and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        streamed_parts.append(part.text)
                        yield part.text

            # --- Final response: fallback yield if voice produced nothing yet ---
            if event.is_final_response():
                if not streamed_parts and event.content and event.content.parts:
                    fallback_text = "".join(
                        p.text for p in event.content.parts
                        if hasattr(p, "text") and p.text
                    )
                    if fallback_text:
                        streamed_parts.append(fallback_text)
                        yield fallback_text
                break

    except Exception as e:
        logger.error("[ADK] Pipeline execution error: %s", e, exc_info=True)
        logger.warning("[ADK] Falling back to V1 pipeline due to error")
        async for chunk in _v1_fallback_stream(message):
            streamed_parts.append(chunk)
            yield chunk

    latency_ms = (time.monotonic() - t0) * 1000.0
    final_reply = "".join(streamed_parts)

    # --- Anomaly: yield graceful fallback message ---
    if anomaly_triggered:
        final_reply = anomaly.GRACEFUL_FALLBACK_REPLY
        yield final_reply

    # --- Session state fallback if nothing was streamed ---
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
        yield final_reply

    # --- Sanitize for logging only (chunks already delivered) ---
    logged_reply = sanitize_output(final_reply)

    # --- Phase 3: Fire-and-forget telemetry + chat logging ---
    try:
        asyncio.create_task(
            telemetry.log_trajectory(
                session_id=session_id,
                user_id=user_id,
                user_message=cleaned_message,
                tool_calls=tool_calls_log,
                final_reply=logged_reply,
                latency_ms=latency_ms,
            )
        )
    except Exception:
        pass

    try:
        from ..observability.db_logging import log_chat
        asyncio.create_task(log_chat(cleaned_message, logged_reply))
    except Exception:
        pass


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


async def _v1_fallback_stream(message: str) -> AsyncGenerator[str, None]:
    """Fall back to the V1 LangGraph pipeline, yielding reply as a single chunk."""
    from .graph import run_chat_graph

    result = await run_chat_graph(message=message)
    reply = result.get("reply", "Sorry, I didn't understand that.")
    yield reply
