# services/anomaly.py
"""
Real-Time Anomaly Detection — Phase 3 OODA Loop Protection.

Detects tool-loop hijacking in the ADK SequentialAgent pipeline.
If the GPT-5 Nano router calls the same tool with identical parameters
≥N times in one session, it flags a [ROUTING_ANOMALY] and forces a
graceful fallback response.

All operations are in-memory (dict + hash) — O(1) per check, <1μs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..services.redis_store import get_redis_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOOL_LOOP_THRESHOLD = int(os.getenv("ANOMALY_TOOL_LOOP_THRESHOLD", "3"))
SESSION_TTL_MINUTES = int(os.getenv("ANOMALY_SESSION_TTL_MINUTES", "30"))
_SESSION_TTL_SECONDS = SESSION_TTL_MINUTES * 60

GRACEFUL_FALLBACK_REPLY = (
    "I seem to be having trouble with that request. Let me try a different "
    "approach — could you rephrase what you're looking for?"
)

# ---------------------------------------------------------------------------
# In-memory storage
# ---------------------------------------------------------------------------
# {session_id: [(tool_name, param_hash, timestamp), ...]}
_session_tool_history: Dict[str, List[Tuple[str, str, float]]] = {}
_lock = threading.Lock()
_last_eviction = time.monotonic()


def _session_key(session_id: str) -> str:
    return f"adk:anomaly:{session_id}"


def _param_hash(params: Any) -> str:
    """Deterministic hash of tool parameters for deduplication."""
    try:
        canonical = json.dumps(params, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canonical = str(params)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()[:12]


def _evict_stale_sessions() -> None:
    """Remove session entries older than SESSION_TTL to prevent memory leaks."""
    global _last_eviction
    now = time.monotonic()
    if now - _last_eviction < 60.0:
        return
    _last_eviction = now

    cutoff = now - _SESSION_TTL_SECONDS
    stale_keys = []
    for sid, entries in _session_tool_history.items():
        if entries and entries[-1][2] < cutoff:
            stale_keys.append(sid)
    for sid in stale_keys:
        _session_tool_history.pop(sid, None)

    if stale_keys:
        logger.debug("[Anomaly] Evicted %d stale sessions", len(stale_keys))


def _load_local_history(session_id: str) -> List[Tuple[str, str, float]]:
    with _lock:
        _evict_stale_sessions()
        return list(_session_tool_history.get(session_id, []))


def _record_local_history(session_id: str, tool_name: str, param_hash: str, timestamp: float) -> None:
    with _lock:
        _evict_stale_sessions()
        history = _session_tool_history.setdefault(session_id, [])
        history.append((tool_name, param_hash, timestamp))


def _clear_local_history(session_id: str) -> None:
    with _lock:
        _session_tool_history.pop(session_id, None)


def _parse_history_entry(entry: Any) -> Optional[Tuple[str, str, float]]:
    try:
        if isinstance(entry, (bytes, bytearray)):
            entry = entry.decode("utf-8")
        if isinstance(entry, str):
            entry = json.loads(entry)
        if not isinstance(entry, dict):
            return None
        tool_name = entry.get("tool")
        param_hash = entry.get("param_hash")
        timestamp = float(entry.get("timestamp", time.monotonic()))
        if not tool_name or not param_hash:
            return None
        return (str(tool_name), str(param_hash), timestamp)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


async def _load_history(session_id: str) -> List[Tuple[str, str, float]]:
    client = await get_redis_client()
    if client is None:
        return _load_local_history(session_id)

    try:
        entries = await client.lrange(_session_key(session_id), 0, -1)
        history: List[Tuple[str, str, float]] = []
        for entry in entries:
            parsed = _parse_history_entry(entry)
            if parsed is not None:
                history.append(parsed)
        return history
    except Exception as exc:
        logger.error("[Anomaly] Failed to read Redis history for session %s: %s", session_id, exc)
        return _load_local_history(session_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def record_tool_call(
    session_id: str,
    tool_name: str,
    tool_params: Any = None,
) -> None:
    """Record a tool invocation for the given session.

    Call this for every tool call event emitted by the ADK pipeline.
    """
    ph = _param_hash(tool_params)
    now = time.monotonic()

    client = await get_redis_client()
    if client is None:
        _record_local_history(session_id, tool_name, ph, now)
        return

    try:
        payload = json.dumps(
            {"tool": tool_name, "param_hash": ph, "timestamp": now},
            sort_keys=True,
        )
        await client.rpush(_session_key(session_id), payload)
        await client.expire(_session_key(session_id), _SESSION_TTL_SECONDS)
        _clear_local_history(session_id)
    except Exception as exc:
        logger.error("[Anomaly] Failed to record Redis tool call for session %s: %s", session_id, exc)
        _record_local_history(session_id, tool_name, ph, now)


async def check_tool_loop(
    session_id: str,
    tool_name: str,
    tool_params: Any = None,
) -> bool:
    """Check if invoking this tool would constitute a routing anomaly.

    Returns True if the same (tool_name, param_hash) has been seen
    ≥ TOOL_LOOP_THRESHOLD times in this session. Logs [ROUTING_ANOMALY].
    """
    ph = _param_hash(tool_params)

    history = await _load_history(session_id)
    identical_count = sum(
        1 for (tn, p, _ts) in history
        if tn == tool_name and p == ph
    )

    if identical_count >= TOOL_LOOP_THRESHOLD:
        logger.warning(
            "[ROUTING_ANOMALY] session=%s tool=%s params_hash=%s count=%d (threshold=%d)",
            session_id,
            tool_name,
            ph,
            identical_count,
            TOOL_LOOP_THRESHOLD,
        )
        return True

    return False


async def get_session_stats(session_id: str) -> Dict[str, Any]:
    """Return tool call statistics for a session (for debugging/telemetry)."""
    history = await _load_history(session_id)

    tool_counts: Dict[str, int] = {}
    for tn, _ph, _ts in history:
        tool_counts[tn] = tool_counts.get(tn, 0) + 1

    return {
        "session_id": session_id,
        "total_tool_calls": len(history),
        "tool_counts": tool_counts,
        "anomaly_threshold": TOOL_LOOP_THRESHOLD,
    }


async def clear_session(session_id: str) -> None:
    """Clean up session data when a conversation ends."""
    client = await get_redis_client()
    if client is not None:
        try:
            await client.delete(_session_key(session_id))
        except Exception as exc:
            logger.error("[Anomaly] Failed to clear Redis history for session %s: %s", session_id, exc)
    _clear_local_history(session_id)
