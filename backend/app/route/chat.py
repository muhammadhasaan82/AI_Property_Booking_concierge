# services/adk_runner.py
"""
ADK 2.0 Runner — Execution bridge between Chainlit and the ADK SequentialAgent.

Manages per-user sessions via InMemorySessionService and provides
[run_adk_turn()](cci:1://file:///c:/Users/ASUS/Desktop/Hotel%20booking/backend/app/services/adk_runner.py:233:0-388:12) as the single async entry point for the Chainlit UI.

Phase 2: Core ADK pipeline.
Phase 3: DPO telemetry capture + tool-loop anomaly detection.
Phase 4 (V2): Removed V1 LangGraph fallback — pure ADK pipeline.
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
from .redis_store import get_session_snapshot, save_session_snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag (kept for backward compat — V2 is always enabled)
# ---------------------------------------------------------------------------
ADK_ENABLED = True

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


def _snapshot_has_context(snapshot: Dict[str, Any]) -> bool:
    history = snapshot.get("history", [])
    state = snapshot.get("state", {})
    meta = snapshot.get("meta", {})
    return bool(history) or bool(state) or bool(meta.get("saved_at"))


def _filter_persistent_state(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    return {
        str(key): value
        for key, value in state.items()
        if not str(key).startswith("temp:")
    }


def _deserialize_event(event_payload: Any) -> Optional[Any]:
    if event_payload is None:
        return None

    try:
        from google.adk.events import Event as AdkEvent
    except Exception:
        try:
            from google.adk.events.event import Event as AdkEvent
        except Exception:
            return None

    if isinstance(event_payload, AdkEvent):
        return event_payload

    if hasattr(AdkEvent, "model_validate"):
        try:
            return AdkEvent.model_validate(event_payload)
        except Exception:
            pass

    if isinstance(event_payload, dict):
        try:
            return AdkEvent(**event_payload)
        except Exception:
            pass

    return None


async def _load_or_create_session(
    session_service: InMemorySessionService,
    user_id: str,
    session_id: str,
) -> Any:
    snapshot = await get_session_snapshot(session_id)
    snapshot_meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
    snapshot_user_id = snapshot_meta.get("user_id")
    snapshot_app_name = snapshot_meta.get("app_name")

    if (snapshot_user_id and snapshot_user_id != user_id) or (
        snapshot_app_name and snapshot_app_name != APP_NAME
    ):
        logger.warning(
            "[ADK] Ignoring session snapshot for %s due to metadata mismatch (saved_user=%s current_user=%s saved_app=%s current_app=%s)",
            session_id,
            snapshot_user_id,
            user_id,
            snapshot_app_name,
            APP_NAME,
        )
        snapshot = {"history": [], "state": {}, "meta": {}, "session_id": session_id}

    try:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        session = None

    if session is not None and not _snapshot_has_context(snapshot):
        return session

    if session is not None:
        try:
            await session_service.delete_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception as e:
            logger.warning("[ADK] Could not replace stale in-memory session %s: %s", session_id, e)

    restored_state = _filter_persistent_state(snapshot.get("state", {}))
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        state=restored_state,
        session_id=session_id,
    )

    restored_history = snapshot.get("history", [])
    if isinstance(restored_history, list):
        for restored_event in restored_history:
            event_obj = _deserialize_event(restored_event)
            if event_obj is None:
                continue
            try:
                await session_service.append_event(session, event_obj)
            except Exception as e:
                logger.warning("[ADK] Failed to restore an event for session %s: %s", session_id, e)

    try:
        hydrated_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if hydrated_session is not None:
            return hydrated_session
    except Exception:
        pass

    logger.info("[ADK] Restored session %s for user %s", session_id, user_id)
    return session


async def _persist_session_snapshot(
    session_service: InMemorySessionService,
    user_id: str,
    session_id: str,
) -> Optional[Any]:
    try:
        updated_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as e:
        logger.error("[ADK] Could not retrieve updated session %s: %s", session_id, e, exc_info=True)
        return None

    if updated_session is None:
        return None

    try:
        await save_session_snapshot(
            session_id=session_id,
            history=getattr(updated_session, "events", []) or [],
            state=_filter_persistent_state(getattr(updated_session, "state", {}) or {}),
            metadata={
                "app_name": APP_NAME,
                "user_id": user_id,
                "last_update_time": getattr(updated_session, "last_update_time", None),
            },
        )
    except Exception as e:
        logger.error("[ADK] Could not persist session %s to Redis: %s", session_id, e, exc_info=True)

    return updated_session


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

    session = await _load_or_create_session(
        session_service=session_service,
        user_id=user_id,
        session_id=session_id,
    )
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
        yield "I'm sorry, something went wrong. Please try again."

    latency_ms = (time.monotonic() - t0) * 1000.0
    final_reply = "".join(streamed_parts)
    updated_session = await _persist_session_snapshot(
        session_service=session_service,
        user_id=user_id,
        session_id=session_id,
    )

    # --- Anomaly: yield graceful fallback message ---
    if anomaly_triggered:
        final_reply = anomaly.GRACEFUL_FALLBACK_REPLY
        yield final_reply

    # --- Session state fallback if nothing was streamed ---
    if not final_reply:
        try:
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