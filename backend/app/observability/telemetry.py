# services/telemetry.py
"""
DPO Telemetry Pipeline — Phase 3 Continuous Learning.

Captures full ADK SequentialAgent trajectories (tool calls, replies, outcomes)
and tags them as SUCCESS_PATH, DROP_OFF_PATH, or IN_PROGRESS for future
Direct Preference Optimization (DPO) fine-tuning of the GPT-5 Nano router.

Storage: SQLite primary (zero-dependency), Supabase mirror (if available).
All writes are fire-and-forget — ZERO impact on user-facing latency.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DPO_TELEMETRY_ENABLED = os.getenv("DPO_TELEMETRY_ENABLED", "1") not in ("0", "false", "False")

_REPO_ROOT = Path(__file__).resolve().parents[3]
DPO_SQLITE_PATH = os.getenv(
    "DPO_SQLITE_PATH",
    str(_REPO_ROOT / "backend" / "dpo_telemetry.db"),
)

# ---------------------------------------------------------------------------
# SQLite setup (thread-safe, lazy-init)
# ---------------------------------------------------------------------------
_DB_LOCK = threading.Lock()
_DB_INITIALIZED = False

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dpo_trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT,
    trajectory_tag TEXT NOT NULL,
    user_message TEXT,
    tool_calls TEXT,
    final_reply TEXT,
    booking_id TEXT,
    turn_count INTEGER DEFAULT 1,
    latency_ms REAL,
    cognitive_context TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dpo_tag ON dpo_trajectories(trajectory_tag);
CREATE INDEX IF NOT EXISTS idx_dpo_session ON dpo_trajectories(session_id);
CREATE INDEX IF NOT EXISTS idx_dpo_created ON dpo_trajectories(created_at);
"""


def _ensure_sqlite_schema() -> None:
    """Create the SQLite DB and schema if they don't exist."""
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    with _DB_LOCK:
        if _DB_INITIALIZED:
            return
        try:
            conn = sqlite3.connect(DPO_SQLITE_PATH)
            conn.executescript(_SCHEMA_SQL)
            conn.close()
            _DB_INITIALIZED = True
            logger.info("[Telemetry] SQLite schema ready at %s", DPO_SQLITE_PATH)
        except Exception as e:
            logger.warning("[Telemetry] SQLite schema init failed: %s", e)


def _write_sqlite(
    session_id: str,
    user_id: Optional[str],
    trajectory_tag: str,
    user_message: str,
    tool_calls_json: str,
    final_reply: str,
    booking_id: Optional[str],
    turn_count: int,
    latency_ms: Optional[float],
    cognitive_context: Optional[str] = None,
) -> bool:
    """Synchronous SQLite insert (runs in thread pool)."""
    try:
        _ensure_sqlite_schema()
        conn = sqlite3.connect(DPO_SQLITE_PATH, timeout=3.0)
        conn.execute(
            """
            INSERT INTO dpo_trajectories
                (session_id, user_id, trajectory_tag, user_message,
                 tool_calls, final_reply, booking_id, turn_count, latency_ms,
                 cognitive_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_id,
                trajectory_tag,
                user_message,
                tool_calls_json,
                final_reply,
                booking_id,
                turn_count,
                latency_ms,
                cognitive_context,
            ),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.warning("[Telemetry] SQLite write failed: %s", e)
        return False


async def _mirror_supabase(
    session_id: str,
    user_id: Optional[str],
    trajectory_tag: str,
    user_message: str,
    tool_calls_json: str,
    final_reply: str,
    booking_id: Optional[str],
    turn_count: int,
    latency_ms: Optional[float],
    cognitive_context: Optional[str] = None,
) -> bool:
    """Best-effort mirror to Supabase via db_client."""
    try:
        from ..services import db_client
        await db_client.execute(
            """
            INSERT INTO public.dpo_trajectories
                (session_id, user_id, trajectory_tag, user_message,
                 tool_calls, final_reply, booking_id, turn_count, latency_ms,
                 cognitive_context)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                user_id,
                trajectory_tag,
                user_message,
                tool_calls_json,
                final_reply,
                booking_id,
                turn_count,
                latency_ms,
                cognitive_context,
            ),
        )
        return True
    except Exception as e:
        logger.debug("[Telemetry] Supabase mirror skipped: %s", e)
        return False


# ---------------------------------------------------------------------------
# Trajectory classification
# ---------------------------------------------------------------------------
_BOOKING_ID_RE = re.compile(
    r"(?:Booking\s*ID[:\s]*)?([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}|[0-9a-fA-F]{6,8})",
    re.IGNORECASE,
)


def classify_trajectory(
    tool_calls: List[Dict[str, Any]],
    booking_id: Optional[str],
    final_reply: str,
) -> str:
    """Classify a trajectory for DPO tagging.

    Returns:
        SUCCESS_PATH  — booking completed (booking_id present or confirmation detected)
        DROP_OFF_PATH — tool errors, empty reply, or no progress
        IN_PROGRESS   — normal mid-conversation turn
    """
    if booking_id:
        return "SUCCESS_PATH"

    reply_lower = (final_reply or "").lower()
    if any(phrase in reply_lower for phrase in (
        "booking confirmed", "booking id", "successfully booked",
        "payment link ready",
    )):
        return "SUCCESS_PATH"

    if not final_reply or not final_reply.strip():
        return "DROP_OFF_PATH"

    error_count = sum(
        1 for tc in tool_calls
        if tc.get("result_status") in ("error", "not_found", "no_results")
    )
    if error_count >= 2:
        return "DROP_OFF_PATH"

    if len(tool_calls) >= 5 and not any(
        tc.get("tool") == "process_v2_booking" for tc in tool_calls
    ):
        return "DROP_OFF_PATH"

    return "IN_PROGRESS"


def extract_booking_id(text: str) -> Optional[str]:
    """Try to extract a booking ID from the final reply."""
    if not text:
        return None
    m = _BOOKING_ID_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Public API — fire-and-forget
# ---------------------------------------------------------------------------
async def log_trajectory(
    session_id: str,
    user_id: Optional[str],
    user_message: str,
    tool_calls: List[Dict[str, Any]],
    final_reply: str,
    booking_id: Optional[str] = None,
    turn_count: int = 1,
    latency_ms: Optional[float] = None,
    cognitive_context: Optional[str] = None,
) -> None:
    """Log a full trajectory turn. Fire-and-forget — never blocks the caller.

    Args:
        session_id: Conversation thread ID.
        user_id: User identifier.
        user_message: The user's input text.
        tool_calls: List of dicts [{tool, params_hash, result_status}, ...].
        final_reply: The agent's final response.
        booking_id: Extracted booking ID if any.
        turn_count: Turn number in the session.
        latency_ms: Pipeline execution time in milliseconds.
        cognitive_context: Mem0 user preference context active during this turn.
    """
    if not DPO_TELEMETRY_ENABLED:
        return

    if not booking_id:
        booking_id = extract_booking_id(final_reply)

    tag = classify_trajectory(tool_calls, booking_id, final_reply)
    tool_calls_json = json.dumps(tool_calls, default=str)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _write_sqlite,
        session_id,
        user_id,
        tag,
        user_message,
        tool_calls_json,
        final_reply,
        booking_id,
        turn_count,
        latency_ms,
        cognitive_context,
    )

    asyncio.create_task(
        _mirror_supabase(
            session_id,
            user_id,
            tag,
            user_message,
            tool_calls_json,
            final_reply,
            booking_id,
            turn_count,
            latency_ms,
            cognitive_context,
        )
    )

    logger.debug(
        "[Telemetry] Logged [%s] session=%s tools=%d booking=%s mem0=%s",
        tag, session_id, len(tool_calls), booking_id or "none",
        "yes" if cognitive_context else "no",
    )
