
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import REDIS_SESSION_TTL_SECONDS, REDIS_URL

logger = logging.getLogger(__name__)

try:
    from redis import asyncio as redis_asyncio
    from redis.exceptions import RedisError
except Exception:
    redis_asyncio = None

    class RedisError(Exception):
        pass


_REDIS_CLIENT = None
_LOCAL_FALLBACK: Dict[str, Dict[str, Any]] = {}

_LAST_PING_AT: float = 0.0
_PING_INTERVAL_SECONDS: float = 5.0

def _session_key(session_id: str) -> str:
    return f"adk:session:{session_id}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

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
            public = {
                key: val
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
            return _jsonable(public)
        except Exception:
            pass

    return str(value)


def _hash_payload(payload: Any) -> str:
    try:
        canonical = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False)
    except Exception:
        canonical = str(payload)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _default_snapshot(session_id: str) -> Dict[str, Any]:
    history: List[Any] = []
    state: Dict[str, Any] = {}
    return {
        "session_id": session_id,
        "history": history,
        "state": state,
        "meta": {
            "saved_at": None,
            "ttl_seconds": REDIS_SESSION_TTL_SECONDS,
            "history_length": 0,
            "history_hash": _hash_payload(history),
            "snapshot_hash": _hash_payload({"history": history, "state": state}),
        },
    }


def _build_snapshot(
    session_id: str,
    history: Optional[List[Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_history = _jsonable(history or [])
    if not isinstance(normalized_history, list):
        normalized_history = [normalized_history]

    normalized_state = _jsonable(state or {})
    if not isinstance(normalized_state, dict):
        normalized_state = {"value": normalized_state}

    base_meta = {
        "saved_at": _utc_now(),
        "ttl_seconds": REDIS_SESSION_TTL_SECONDS,
        "history_length": len(normalized_history),
        "history_hash": _hash_payload(normalized_history),
        "snapshot_hash": _hash_payload({"history": normalized_history, "state": normalized_state}),
    }
    if metadata:
        base_meta.update(_jsonable(metadata) or {})

    return {
        "session_id": session_id,
        "history": normalized_history,
        "state": normalized_state,
        "meta": base_meta,
    }


async def _get_redis_client():
    global _REDIS_CLIENT, _LAST_PING_AT

    if redis_asyncio is None:
        return None

    if _REDIS_CLIENT is None:
        try:
            _REDIS_CLIENT = redis_asyncio.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                health_check_interval=30,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        except Exception as exc:
            logger.error("[Redis] Could not initialize Redis client: %s", exc)
            _REDIS_CLIENT = None
            return None
    now = time.monotonic()
    if now - _LAST_PING_AT > _PING_INTERVAL_SECONDS:
        try:
            await _REDIS_CLIENT.ping()
            _LAST_PING_AT = now
            return _REDIS_CLIENT
        except Exception as exc:
            logger.debug("[Redis] Redis unavailable, using local fallback: %s", exc)
            _REDIS_CLIENT = None
            return None
    return _REDIS_CLIENT

async def get_redis_client():
    return await _get_redis_client()


def _cleanup_local_fallback() -> None:
    now = time.time()
    expired = [sid for sid, entry in _LOCAL_FALLBACK.items() if entry.get("expires_at", 0) <= now]
    for sid in expired:
        _LOCAL_FALLBACK.pop(sid, None)


def _save_local_snapshot(session_id: str, snapshot: Dict[str, Any]) -> None:
    _cleanup_local_fallback()
    _LOCAL_FALLBACK[session_id] = {
        "snapshot": copy.deepcopy(snapshot),
        "expires_at": time.time() + REDIS_SESSION_TTL_SECONDS,
    }


def _load_local_snapshot(session_id: str) -> Dict[str, Any]:
    _cleanup_local_fallback()
    entry = _LOCAL_FALLBACK.get(session_id)
    if not entry:
        return _default_snapshot(session_id)
    snapshot = entry.get("snapshot") or _default_snapshot(session_id)
    return copy.deepcopy(snapshot)


async def get_session_snapshot(session_id: str) -> Dict[str, Any]:
    client = await _get_redis_client()
    if client is None:
        return _load_local_snapshot(session_id)

    try:
        payload = await client.get(_session_key(session_id))
        if not payload:
            return _default_snapshot(session_id)
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            return _default_snapshot(session_id)
        parsed.setdefault("session_id", session_id)
        parsed.setdefault("history", [])
        parsed.setdefault("state", {})
        parsed.setdefault("meta", {})
        return parsed
    except (RedisError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("[Redis] Failed to read session snapshot for %s: %s", session_id, exc)
        return _load_local_snapshot(session_id)


async def save_session_snapshot(
    session_id: str,
    history: Optional[List[Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    snapshot = _build_snapshot(session_id=session_id, history=history, state=state, metadata=metadata)
    client = await _get_redis_client()

    if client is None:
        _save_local_snapshot(session_id, snapshot)
        return

    try:
        payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        await client.set(_session_key(session_id), payload, ex=REDIS_SESSION_TTL_SECONDS)
        _LOCAL_FALLBACK.pop(session_id, None)
    except (RedisError, TypeError, ValueError) as exc:
        logger.error("[Redis] Failed to save session snapshot for %s: %s", session_id, exc)
        _save_local_snapshot(session_id, snapshot)


async def get_session_history(session_id: str) -> List[Any]:
    snapshot = await get_session_snapshot(session_id)
    history = snapshot.get("history", [])
    return history if isinstance(history, list) else []


async def save_session_history(session_id: str, history: List[Any]) -> None:
    snapshot = await get_session_snapshot(session_id)
    await save_session_snapshot(
        session_id=session_id,
        history=history,
        state=snapshot.get("state", {}),
        metadata=snapshot.get("meta", {}),
    )


async def get_session_state(session_id: str) -> Dict[str, Any]:
    snapshot = await get_session_snapshot(session_id)
    state = snapshot.get("state", {})
    return state if isinstance(state, dict) else {}


async def save_session_state(session_id: str, state: Dict[str, Any]) -> None:
    snapshot = await get_session_snapshot(session_id)
    await save_session_snapshot(
        session_id=session_id,
        history=snapshot.get("history", []),
        state=state,
        metadata=snapshot.get("meta", {}),
    )


async def clear_session_snapshot(session_id: str) -> None:
    client = await _get_redis_client()
    if client is not None:
        try:
            await client.delete(_session_key(session_id))
        except RedisError as exc:
            logger.error("[Redis] Failed to clear session snapshot for %s: %s", session_id, exc)
    _LOCAL_FALLBACK.pop(session_id, None)

async def clear_session_for_testing(session_id: str) -> None:
    """Wipe all the state for a session. Use in tests to avoid stale data."""
    await clear_session_snapshot(session_id)
    logger.debug("[Redis] Session %s reset for testing. ", session_id)