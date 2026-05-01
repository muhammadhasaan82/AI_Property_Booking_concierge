"""
ADK 2.0 Runner - Execution bridge between Chainlit and the ADK SequentialAgent.

Uses a Redis-backed ADK session service so the FastAPI container stays
stateless while the agent state lives in Redis snapshots.

Phase 2: Core ADK pipeline.
Phase 3: DPO telemetry capture + tool-loop anomaly detection.
Phase 4 (V2): Removed V1 LangGraph fallback - pure ADK pipeline.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.genai.types import Content, Part
from ..observability import telemetry
from ..security import anomaly
from ..security.guardrails import sanitize_input, sanitize_output
from ..security import policy_router
from .config import (
    ADK_MAX_COGNITIVE_CONTEXT_CHARS,
    ADK_SESSION_MAX_CONTEXT_CHARS,
    ADK_SESSION_MAX_EVENTS,
)
from .redis_store import (
    clear_session_snapshot,
    get_redis_client,
    get_session_snapshot,
    save_session_snapshot,
)
from app.agents.schemas.understanding_frame import UnderstandingFrame
from app.config.response_policies_loader import render_policy_snippet
from app.config.agent_config_loader import cfg as _cfg

ADK_TURN_TIMEOUT = float(os.getenv("ADK_TURN_TIMEOUT", "45"))
logger = logging.getLogger(__name__)
MEM0_ENABLED = os.getenv("MEM0_ENABLED", "1").strip().lower() in ("1", "true", "yes")
ADK_ENABLED = True

_session_service: Optional["RedisSessionService"] = None
_runner: Optional[Runner] = None

APP_NAME = "ai_concierge"


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


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]

    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _jsonable(value.model_dump(mode="json", by_alias=False))
        except Exception:
            try:
                return _jsonable(value.model_dump())
            except Exception:
                pass

    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            public = {key: val for key, val in vars(value).items() if not key.startswith("_")}
            return _jsonable(public)
        except Exception:
            pass

    return str(value)


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


def _extract_text_parts(event: Any) -> str:
    """Extract concatenated text parts from an ADK event content payload."""
    try:
        if event.content and event.content.parts:
            return "".join(
                part.text for part in event.content.parts
                if hasattr(part, "text") and part.text
            )
    except Exception:
        pass
    return ""


def _event_timestamp(event: Any) -> float:
    try:
        return float(getattr(event, "timestamp", 0.0) or 0.0)
    except Exception:
        return 0.0


def _normalize_cognitive_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return ""
    return str(value).strip()


def _truncate_text_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _estimate_event_chars(event: Any) -> int:
    try:
        return len(json.dumps(_jsonable(event), ensure_ascii=False, sort_keys=True))
    except Exception:
        return len(str(event))


def _trim_events_for_context(
    events: List[Any],
    *,
    max_events: int = ADK_SESSION_MAX_EVENTS,
    max_chars: int = ADK_SESSION_MAX_CONTEXT_CHARS,
) -> List[Any]:
    """Keep only the most recent events within the configured context budget."""
    if not events:
        return []

    bounded_max_events = max(int(max_events or 0), 1)
    bounded_max_chars = max(int(max_chars or 0), 1)

    recent_events = list(events[-bounded_max_events:])
    kept_reversed: List[Any] = []
    running_chars = 0

    for event in reversed(recent_events):
        event_chars = max(_estimate_event_chars(event), 1)
        if kept_reversed and running_chars + event_chars > bounded_max_chars:
            break
        kept_reversed.append(event)
        running_chars += event_chars

    trimmed = list(reversed(kept_reversed))
    return trimmed or [recent_events[-1]]


def _already_exists_error(message: str) -> Exception:
    try:
        from google.adk.errors.already_exists_error import AlreadyExistsError

        return AlreadyExistsError(message)
    except Exception:
        return ValueError(message)


class RedisSessionService(BaseSessionService):
    """ADK session service backed by Redis session snapshots."""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def _matches_scope(self, snapshot: Dict[str, Any], app_name: str, user_id: str) -> bool:
        meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
        saved_app = meta.get("app_name")
        saved_user = meta.get("user_id")
        return (not saved_app or saved_app == app_name) and (not saved_user or saved_user == user_id)

    def _build_session(
        self,
        *,
        snapshot: Dict[str, Any],
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        if not _snapshot_has_context(snapshot):
            return None

        if not self._matches_scope(snapshot, app_name, user_id):
            return None

        raw_history = snapshot.get("history", [])
        events = []
        if isinstance(raw_history, list):
            for payload in raw_history:
                event = _deserialize_event(payload)
                if event is not None:
                    events.append(event)

        if config and config.after_timestamp is not None:
            events = [
                event
                for event in events
                if _event_timestamp(event) > float(config.after_timestamp)
            ]

        if config and config.num_recent_events:
            events = events[-config.num_recent_events :]

        trimmed_events = _trim_events_for_context(events)
        if len(trimmed_events) != len(events):
            logger.info(
                "[ADK] Trimmed session %s context from %s events to %s events before model invocation",
                session_id,
                len(events),
                len(trimmed_events),
            )
            events = trimmed_events

        meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
        last_update_time = meta.get("last_update_time")
        if last_update_time is None and events:
            last_update_time = _event_timestamp(events[-1])

        return Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=_filter_persistent_state(snapshot.get("state", {})),
            events=events,
            last_update_time=float(last_update_time or 0.0),
        )

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        resolved_session_id = session_id.strip() if session_id and session_id.strip() else str(uuid4())

        async with self._lock_for(resolved_session_id):
            snapshot = await get_session_snapshot(resolved_session_id)
            existing_session = self._build_session(
                snapshot=snapshot,
                app_name=app_name,
                user_id=user_id,
                session_id=resolved_session_id,
            )
            if existing_session is not None:
                raise _already_exists_error(f"Session with id {resolved_session_id} already exists.")

            if _snapshot_has_context(snapshot) and not self._matches_scope(snapshot, app_name, user_id):
                logger.warning(
                    "[ADK] Overwriting Redis session snapshot for %s due to scope mismatch (saved_user=%s current_user=%s saved_app=%s current_app=%s)",
                    resolved_session_id,
                    snapshot.get("meta", {}).get("user_id"),
                    user_id,
                    snapshot.get("meta", {}).get("app_name"),
                    app_name,
                )

            initial_state = _filter_persistent_state(state)
            created_at = time.time()
            await save_session_snapshot(
                session_id=resolved_session_id,
                history=[],
                state=initial_state,
                metadata={
                    "app_name": app_name,
                    "user_id": user_id,
                    "last_update_time": created_at,
                },
            )

        return Session(
            id=resolved_session_id,
            app_name=app_name,
            user_id=user_id,
            state=initial_state,
            events=[],
            last_update_time=created_at,
        )

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        snapshot = await get_session_snapshot(session_id)
        session = self._build_session(
            snapshot=snapshot,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            config=config,
        )

        if session is None and _snapshot_has_context(snapshot) and not self._matches_scope(snapshot, app_name, user_id):
            logger.warning(
                "[ADK] Ignoring Redis session snapshot for %s due to scope mismatch (saved_user=%s current_user=%s saved_app=%s current_app=%s)",
                session_id,
                snapshot.get("meta", {}).get("user_id"),
                user_id,
                snapshot.get("meta", {}).get("app_name"),
                app_name,
            )

        return session

    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: Optional[str] = None,
    ) -> ListSessionsResponse:
        client = await get_redis_client()
        if client is None:
            return ListSessionsResponse()

        sessions: list[Session] = []
        try:
            async for key in client.scan_iter(match="adk:session:*"):
                payload = await client.get(key)
                if not payload:
                    continue

                try:
                    snapshot = json.loads(payload)
                except (TypeError, ValueError):
                    continue

                meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
                if meta.get("app_name") != app_name:
                    continue
                if user_id is not None and meta.get("user_id") != user_id:
                    continue

                session_id = snapshot.get("session_id") or str(key).rsplit(":", 1)[-1]
                sessions.append(
                    Session(
                        id=session_id,
                        app_name=app_name,
                        user_id=meta.get("user_id", user_id or ""),
                        state={},
                        events=[],
                        last_update_time=float(meta.get("last_update_time") or 0.0),
                    )
                )
        except Exception as exc:
            logger.warning("[ADK] Failed to list Redis sessions: %s", exc)
            return ListSessionsResponse()

        return ListSessionsResponse(sessions=sessions)

    async def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        async with self._lock_for(session_id):
            snapshot = await get_session_snapshot(session_id)
            if _snapshot_has_context(snapshot) and not self._matches_scope(snapshot, app_name, user_id):
                return
            await clear_session_snapshot(session_id)

    async def append_event(self, session: Session, event: Any) -> Any:
        if getattr(event, "partial", None):
            return event

        await super().append_event(session=session, event=event)
        session.last_update_time = _event_timestamp(event) or time.time()

        async with self._lock_for(session.id):
            storage_session = await self.get_session(
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.id,
            )
            if storage_session is None:
                storage_session = Session(
                    id=session.id,
                    app_name=session.app_name,
                    user_id=session.user_id,
                    state={},
                    events=[],
                    last_update_time=session.last_update_time,
                )

            storage_session.events.append(event)
            storage_session.events = _trim_events_for_context(storage_session.events)
            storage_session.last_update_time = session.last_update_time

            if isinstance(session.state, dict):
                storage_session.state.update(_filter_persistent_state(session.state))

            state_delta = getattr(getattr(event, "actions", None), "state_delta", None)
            if isinstance(state_delta, dict):
                storage_session.state.update(_filter_persistent_state(state_delta))

            await save_session_snapshot(
                session_id=storage_session.id,
                history=storage_session.events,
                state=_filter_persistent_state(storage_session.state),
                metadata={
                    "app_name": storage_session.app_name,
                    "user_id": storage_session.user_id,
                    "last_update_time": storage_session.last_update_time,
                },
            )

        return event


def _get_runner() -> Runner:
    """Lazily initialize the ADK Runner and Redis-backed session service."""
    global _session_service, _runner
    if _runner is None:
        from ..agents.adk_agents import root_agent

        _session_service = RedisSessionService()
        _runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=_session_service,
            auto_create_session=True,
        )
        logger.info("[ADK] Runner initialized with agent '%s'", root_agent.name)
    return _runner


def _get_session_service() -> RedisSessionService:
    """Return the session service, initializing the runner if needed."""
    _get_runner()
    return _session_service


async def _build_invocation_state_delta(user_id: str, current_query: str, session_id: str) -> dict[str, Any]:
    user_cognitive_context = ""

    if len(current_query.strip().split()) > 2:
        if MEM0_ENABLED:
            try:
                from .memory_engine import fetch_user_context
                mem0_context = await fetch_user_context(
                    user_id=user_id,
                    current_query=current_query,
                    session_id=session_id,
                )
                user_cognitive_context = _normalize_cognitive_context(mem0_context)
                user_cognitive_context = _truncate_text_chars(
                    user_cognitive_context,
                    ADK_MAX_COGNITIVE_CONTEXT_CHARS,
                )
            except Exception as exc:
                logger.debug("[ADK] Could not fetch cognitive context: %s", exc)
    soft_state = {}
    try:
        snapshot = await get_session_snapshot(session_id)
        soft_state = snapshot.get("state", {}).get("soft_state", {})
    except Exception:
        pass
    return {
        "user_cognitive_context": user_cognitive_context,
        "soft_state": soft_state,
    }

async def _render_voice_from_router_output(
    router_output: str,
    user_cognitive_context: str,
    understanding_frame_json: str = "",
) -> str:
    """Force Node-2 voice synthesis when only router JSON is available."""
    if not router_output or not router_output.strip():
        return ""

    try:
        import litellm
        from ..agents.adk_agents import VOICE_CONFIG, VOICE_INSTRUCTION, VOICE_MODEL
        status = None
        if isinstance(router_output, dict):
            status = router_output.get("status")
        response_policy_snippet = ""
        if _cfg.feature_response_policies and status:
            response_policy_snippet = render_policy_snippet(status)

        system_prompt = (
            VOICE_INSTRUCTION
            .replace("{router_output}", router_output)
            .replace("{user_cognitive_context}", _normalize_cognitive_context(user_cognitive_context))
            .replace("{understanding}", understanding_frame_json or "{}")
        )
        temperature = getattr(VOICE_CONFIG, "temperature", 0.6)

        def _generate() -> str:
            response = litellm.completion(
                model=VOICE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "Generate the final concierge response for this turn.",
                    },
                ],
                temperature=temperature,
            )
            if response_policy_snippet:
                system_prompt = system_prompt + "\n\n" + response_policy_snippet
                
            if getattr(response, "choices", None):
                choice0 = response.choices[0]
                message = getattr(choice0, "message", None)
                content = getattr(message, "content", "") if message else ""
                if isinstance(content, str):
                    return content.strip()

            if isinstance(response, dict):
                return (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )

            return ""

        return await asyncio.to_thread(_generate)
    except Exception as exc:
        logger.warning("[ADK] Voice handoff fallback failed: %s", exc)
        return ""


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
    cleaned_message, is_safe = sanitize_input(message)
    if not is_safe:
        yield "I'm sorry, I can't process that request. Could you rephrase?"
        return

    if not cleaned_message.strip():
        yield "I didn't catch that. Could you repeat your question?"
        return

    runner = _get_runner()
    session_service = _get_session_service()
    state_delta = await _build_invocation_state_delta(user_id=user_id, current_query=cleaned_message, session_id=session_id)
    user_cognitive_context = _normalize_cognitive_context(state_delta.get("user_cognitive_context"))

    user_content = Content(parts=[Part(text=cleaned_message)])
    t0 = time.monotonic()

    streamed_parts: List[str] = []
    tool_calls_log: List[Dict[str, Any]] = []
    anomaly_triggered = False
    router_output = ""
    pipeline_failed_reply = ""

    try:
        async with asyncio.timeout(ADK_TURN_TIMEOUT):
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content,
                state_delta=state_delta,
            ):
                tool_name, tool_params = _extract_tool_call(event)
                if tool_name:
                    if await anomaly.check_tool_loop(session_id, tool_name, tool_params):
                        anomaly_triggered = True
                        tool_calls_log.append({
                            "tool": tool_name,
                            "params_hash": hashlib.md5(
                                json.dumps(tool_params, sort_keys=True, default=str).encode()
                            ).hexdigest()[:12],
                            "result_status": "anomaly_blocked",
                        })
                        break
                    await anomaly.record_tool_call(session_id, tool_name, tool_params)
                    tool_calls_log.append({
                        "tool": tool_name,
                        "params_hash": hashlib.md5(
                            json.dumps(tool_params, sort_keys=True, default=str).encode()
                        ).hexdigest()[:12],
                        "result_status": "ok",
                    })
                    continue

                author = getattr(event, "author", None)
                event_text = _extract_text_parts(event)

                if author == "triage_router" and event_text:
                    router_output = event_text

                if author == "concierge_voice":
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                streamed_parts.append(part.text)
                                yield part.text

                if event.is_final_response():
                    if author == "triage_router":
                        continue
                    if author == "concierge_voice":
                        if not streamed_parts and event_text:
                            streamed_parts.append(event_text)
                            yield event_text
                        break
                    if streamed_parts:
                        break

    except asyncio.TimeoutError:
        logger.error("[ADK] Turn timed out after %ss - check providers keys/networks", ADK_TURN_TIMEOUT)

    except Exception as exc:
        logger.error("[ADK] Pipeline execution error: %s", exc, exc_info=True)
        pipeline_failed_reply = "I'm sorry, something went wrong. Please try again."

    latency_ms = (time.monotonic() - t0) * 1000.0
    final_reply = "".join(streamed_parts)

    try:
        updated_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as exc:
        logger.error("[ADK] Could not retrieve updated session %s: %s", session_id, exc, exc_info=True)
        updated_session = None

    try:
        if updated_session and updated_session.state:
            fresh_soft_state = updated_session.state.get("soft_state")
            if isinstance(fresh_soft_state, dict) and fresh_soft_state:
                current_snapshot = await get_session_snapshot(session_id)
                merged_state = current_snapshot.get("state", {})
                merged_state["soft_state"] = fresh_soft_state
                await save_session_snapshot(session_id=session_id, state=merged_state)
                logger.debug("[ADK] soft state presisted explicity for session %s", session_id)
    except Exception as exc:
        logger.warning("[ADK] Could not persist soft_state to Redis: %s", exc)

    try:
        if updated_session and updated_session.state:
            router_output = router_output or str(updated_session.state.get("router_output", "") or "")
            user_cognitive_context = _normalize_cognitive_context(
                updated_session.state.get("user_cognitive_context", user_cognitive_context)
            )
            understanding_frame_json = ""
            if updated_session and updated_session.state and _cfg.features_understanding_frame:
                raw_frame = upadated_session.state.get("understanding")
                if raw_frame is not None:
                    try:
                        if isintance(raw_frame, dict):
                            frame_obj = UnderstandingFrame(**raw_frame)
                        elif isinstance(raw_frame, UnderstandingFrame):
                            frame_obj = raw_frame
                        elif isinstance(raw_frame, str):
                            import json as _json
                            frame_obj = UnderstandingFrame(**_json.loads(raw_frame))
                        else:
                            frame_obj = None

                        if frame_obj is not None:
                            understanding_frame_json = frame_obj.to_compact_json()
                            logger.debug(
                                "[ADK] Understanding frame captured: intent=%s confidence=%.2f mood=%s",
                                frame_obj.primary_intent, frame_obj.confidence, frame_obj.user_mood,
                            )
                    except Exception as _frame_exc:
                        logger.warning("[ADK] Failed to parso UnderstandingFrame: %s", _frame_exc)

    except Exception:
        pass

    policy_override_json = None
    policy_override_applied = False
    _mode = (_cfg.feature_policy_router_mode or "off").lower()

    if _mode in ("shodow", "enforce") and frame_obj is not None:
        try:
            soft_state_for_policy = (
                updated_session.state.get("soft_state", {})
                if updated_session and updated_session.state else {}
            )
            decision = policy_route.decide(frame_obj, soft_state_for_policy)
            actual_tool = tool_calls_log[-1]["tool"] if tool_calls_log else None
            override = policy_router.compute_override(decision, actual_tool)

            if override:
                override_record = {
                    "mode": _mode,
                    "applied": False,
                    "decision":{
                        "action": decision.get("action"),
                        "tool_name": decision.get("tool_name"),
                        "effective_intent": decision.get("effective_intent"),
                        "matched_priority_id": decision.get("matched_priority_id"),
                        "reasoning": decision.get("reasoning"),
                    },
                    "actual_tool": override.get("actual_tool"),
                }
                logger.info(
                    "[POLICY_OVERRIDE] mode=%s actual=%s policy=%s action=%s reason=%s",
                    _mode,
                    override.get("actual_tool"),
                    override.get("tool_name"),
                    override.get("action"),
                    override.get("reasoning"),
                )

                if _mode == "enforce" and decision.get("action") != "execute_tool":
                    synthetic = policy_route.synthetic_router_output(decsion)
                    router_output = json.dumps(synthetic, ensure_ascii=False)
                    final_reply = ""
                    policy_override_applied = True
                    override_record["applied"] = True
                    logger.info(
                        "[POLICY_OVERRIDE] enforced action=%s status=%s",
                        decision.get("action"), synthetic.get("status"),
                    )
                policy_override_json = json.dumps(override_record, ensure_ascii=False)
        except Exception as _policy_exc:
            logger.warning("[ADK] policy_router failed: %s", _policy_exc)
            
    if anomaly_triggered:
        final_reply = anomaly.GRACEFUL_FALLBACK_REPLY
        yield final_reply

    if not final_reply:
        try:
            if updated_session and updated_session.state:
                final_reply = str(updated_session.state.get("final_reply", "") or "")
        except Exception:
            pass

    if not final_reply and router_output:
        if updated_session and updated_session.state:
            final_reply = str(updated_session.state.get("final_reply", "") or "")

        if not final_reply:
            voice_reply = await _render_voice_from_router_output(
                router_output=router_output,
                user_cognitive_context=user_cognitive_context,
                understanding_frame_json=understanding_frame_json,
            )
            if voice_reply:
                final_reply = voice_reply
                if not streamed_parts and not anomaly_triggered:
                    yield final_reply


    if not final_reply and pipeline_failed_reply:
        final_reply = pipeline_failed_reply
        yield final_reply
    if not final_reply:
        final_reply = "I'm sorry, I couldn't process your request. Could you try again?"
        yield final_reply

    logged_reply = sanitize_output(final_reply)

    try:
        asyncio.create_task(
            telemetry.log_trajectory(
                session_id=session_id,
                user_id=user_id,
                user_message=cleaned_message,
                tool_calls=tool_calls_log,
                final_reply=logged_reply,
                latency_ms=latency_ms,
                cognitive_context=user_cognitive_context or None,
                understanding_frame_json=understanding_frame_json or None,
                policy_override_json=policy_override_json,
                turn_count=turn_count,
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
