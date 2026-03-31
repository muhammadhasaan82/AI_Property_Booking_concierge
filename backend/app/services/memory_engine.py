# services/memory_engine.py
"""
Cognitive Memory Engine — Mem0 Integration Layer.

Provides long-term user preference storage and retrieval via the Mem0
managed platform. Acts as the "hippocampus" for the ADK 2.0 pipeline,
enabling the Voice Agent to implicitly personalize responses based on
historical user preferences (allergies, travel habits, accessibility
needs, property preferences) across sessions.

Architectural Guardrails:
  - All calls are async and wrapped in try/except — Mem0 failure = empty
    context, never a crash.
  - 3-second timeout on all Mem0 API calls.
  - If MEM0_API_KEY is not set, the entire module is a no-op.
  - extract_and_store is designed for fire-and-forget via asyncio.create_task.
  - fetch_user_context is designed to be awaited before pipeline execution.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MEM0_API_KEY: str = os.getenv("MEM0_API_KEY", "")
MEM0_TIMEOUT: float = float(os.getenv("MEM0_TIMEOUT", "3.0"))
MEM0_SEARCH_LIMIT: int = int(os.getenv("MEM0_SEARCH_LIMIT", "5"))
MEM0_ENABLED: bool = bool(MEM0_API_KEY)

# ---------------------------------------------------------------------------
# Lazy-init singleton
# ---------------------------------------------------------------------------
_client = None
_init_attempted = False


def _get_client():
    """Lazily initialize the Mem0 AsyncMemoryClient.

    Returns None if:
      - MEM0_API_KEY is not set
      - mem0 package is not installed
      - Client initialization fails
    """
    global _client, _init_attempted

    if _init_attempted:
        return _client

    _init_attempted = True

    if not MEM0_ENABLED:
        logger.info("[Memory] MEM0_API_KEY not set — cognitive memory disabled")
        return None

    try:
        from mem0 import AsyncMemoryClient
        _client = AsyncMemoryClient(api_key=MEM0_API_KEY)
        logger.info("[Memory] Mem0 AsyncMemoryClient initialized")
    except ImportError:
        logger.warning("[Memory] mem0ai package not installed — cognitive memory disabled")
        _client = None
    except Exception as e:
        logger.warning("[Memory] Failed to initialize Mem0 client: %s", e)
        _client = None

    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_and_store(user_id: str, message: str) -> None:
    """Extract durable user preferences from a message and store in Mem0.

    This is a background function designed to be called via
    asyncio.create_task() — fire-and-forget. It never blocks the user.

    Mem0's internal LLM analyzes the message for persistent facts like:
      - "I'm allergic to dogs"
      - "I always travel with my partner"
      - "I need wheelchair accessibility"
      - "I prefer properties under $150/night"

    Args:
        user_id: Persistent user identifier (email, phone, or session-stable ID).
        message: The user's raw message text.
    """
    client = _get_client()
    if client is None:
        return

    if not message or len(message.strip()) < 5:
        return  # Skip trivially short messages

    try:
        messages = [{"role": "user", "content": message}]
        await asyncio.wait_for(
            client.add(messages, user_id=user_id),
            timeout=MEM0_TIMEOUT,
        )
        logger.debug("[Memory] Stored context for user %s", user_id)
    except asyncio.TimeoutError:
        logger.warning("[Memory] Mem0 add() timed out for user %s", user_id)
    except Exception as e:
        logger.warning("[Memory] Failed to store context for user %s: %s", user_id, e)


async def fetch_user_context(user_id: str, current_query: str = "") -> str:
    """Retrieve relevant historical user preferences from Mem0.

    Called before the pipeline fires to enrich the Voice Agent's context.
    Returns a concise string of facts, or empty string on failure.

    Args:
        user_id: Persistent user identifier.
        current_query: The current user message — used for semantic search
                       to retrieve the most relevant stored preferences.

    Returns:
        A semicolon-separated string of relevant user facts, e.g.:
        "User is allergic to dogs; Prefers pet-free properties; Usually books for 2 guests"
        Returns "" if no memories exist, Mem0 is unavailable, or an error occurs.
    """
    client = _get_client()
    if client is None:
        return ""

    try:
        query = current_query if current_query.strip() else "user preferences and history"
        results = await asyncio.wait_for(
            client.search(query, user_id=user_id, limit=MEM0_SEARCH_LIMIT),
            timeout=MEM0_TIMEOUT,
        )

        if not results:
            return ""

        # Mem0 search returns a list of dicts with 'memory' key
        facts = []
        for item in results:
            memory_text = None
            if isinstance(item, dict):
                memory_text = item.get("memory") or item.get("text") or item.get("content")
            elif hasattr(item, "memory"):
                memory_text = item.memory
            if memory_text and isinstance(memory_text, str) and memory_text.strip():
                facts.append(memory_text.strip())

        context = "; ".join(facts) if facts else ""
        if context:
            logger.debug("[Memory] Retrieved %d facts for user %s", len(facts), user_id)
        return context

    except asyncio.TimeoutError:
        logger.warning("[Memory] Mem0 search() timed out for user %s", user_id)
        return ""
    except Exception as e:
        logger.warning("[Memory] Failed to fetch context for user %s: %s", user_id, e)
        return ""


async def get_all_memories(user_id: str) -> list:
    """Retrieve all stored memories for a user. Utility for debugging/admin.

    Args:
        user_id: Persistent user identifier.

    Returns:
        List of memory dicts, or empty list on failure.
    """
    client = _get_client()
    if client is None:
        return []

    try:
        result = await asyncio.wait_for(
            client.get_all(user_id=user_id),
            timeout=MEM0_TIMEOUT,
        )
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.warning("[Memory] Failed to get all memories for user %s: %s", user_id, e)
        return []
