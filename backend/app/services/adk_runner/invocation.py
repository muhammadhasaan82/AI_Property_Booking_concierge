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


from app.services.adk_runner.state_helpers import _normalize_cognitive_context, _truncate_text_chars
from app.services.config import ADK_MAX_COGNITIVE_CONTEXT_CHARS
from app.services.redis_store import get_session_snapshot

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

