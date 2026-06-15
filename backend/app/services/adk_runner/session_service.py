from __future__ import annotations
"""ADK runner submodule."""

import asyncio
import hashlib
import json
import logging
import os
import time
from types import SimpleNamespace
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

from app.config.agent_config_loader import cfg as _cfg
from app.services.redis_store import (
    clear_session_snapshot,
    get_redis_client,
    get_session_snapshot,
    save_session_snapshot,
)

logger = logging.getLogger(__name__)
ADK_TURN_TIMEOUT = float(getattr(_cfg, "runtime_turn_timeout_seconds", 45))
MEM0_ENABLED = os.getenv("MEM0_ENABLED", "1").strip().lower() in ("1", "true", "yes")
APP_NAME = "ai_concierge"


from app.services.adk_runner.state_helpers import (
    _already_exists_error,
    _deserialize_event,
    _estimate_event_chars,
    _event_timestamp,
    _extract_text_parts,
    _filter_persistent_state,
    _merge_state,
    _snapshot_has_context,
    _trim_events_for_context,
)

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

            if not isinstance(storage_session.state, dict):
                storage_session.state = {}
            if isinstance(session.state, dict):
                _merge_state(storage_session.state, _filter_persistent_state(session.state))

            state_delta = getattr(getattr(event, "actions", None), "state_delta", None)
            if isinstance(state_delta, dict):
                _merge_state(storage_session.state, _filter_persistent_state(state_delta))

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


_session_service: Optional[RedisSessionService] = None
_runner: Optional[Runner] = None


def _get_runner() -> Runner:
    """Lazily initialize the ADK Runner and Redis-backed session service."""
    global _session_service, _runner
    if _runner is None:
        from app.agents.adk_agents import root_agent

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

