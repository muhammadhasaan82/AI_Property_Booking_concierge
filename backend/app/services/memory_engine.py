"""
Local Cognitive Memory Engine powered by open-source Mem0.

This module uses the self-hosted/open-source Mem0 runtime via
`from mem0 import Memory` and never talks to the managed Mem0 Cloud API.

Architecture:
    - LLM extraction uses the project's existing provider keys.
    - Vector storage stays on infrastructure we control.
    - All public functions are async and failure-safe.
    - Sync Mem0 SDK calls are offloaded with `asyncio.to_thread(...)`.
"""
from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import importlib
import logging
from typing import Any, Callable, Optional

import transformers

transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

MEM0_TIMEOUT: float = float(os.getenv("MEM0_TIMEOUT", "3.0"))
MEM0_SEARCH_LIMIT: int = int(os.getenv("MEM0_SEARCH_LIMIT", "5"))

_client = None
_init_attempted = False


def initialize_memory():
    """Initialize the local Mem0 Memory client with a schema-safe config."""
    try:
        Memory = importlib.import_module("mem0").Memory
    except Exception as exc:
        logger.warning("[Memory] Mem0 import unavailable, disabling memory engine: %s", exc)
        return None

    try:
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": os.getenv("MEM0_COLLECTION_NAME", "ai_concierge_memories"),
                    "path": os.getenv("MEM0_STORAGE_DIR", "backend/.mem0"),
                },
            },
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": os.getenv("MEM0_LLM_MODEL", "groq/llama-3.3-70b-versatile"),
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": os.getenv(
                        "MEM0_EMBEDDER_MODEL",
                        "BAAI/bge-large-en-v1.5",
                    ),
                },
            },
        }

        return Memory.from_config(config)
    except Exception as exc:
        logger.warning("[Memory] Local Mem0 initialization skipped: %s", exc)
        return None


def _normalize_items(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("results", "memories", "data"):
            value = result.get(key)
            if isinstance(value, list):
                return value
    return []


def _call_with_fallbacks(attempts: list[Callable[[], Any]]) -> Any:
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    raise RuntimeError("No Mem0 call attempts were provided")


def _get_client():
    """Lazily initialize the local open-source Mem0 Memory engine."""
    global _client, _init_attempted

    if _init_attempted:
        return _client

    _init_attempted = True
    _client = initialize_memory()

    if _client is not None:
        logger.info(
            "[Memory] Local Mem0 initialized (llm=litellm:%s, embedder=huggingface:%s, vector_store=chroma:%s)",
            os.getenv("MEM0_LLM_MODEL", "groq/llama-3.3-70b-versatile"),
            os.getenv("MEM0_EMBEDDER_MODEL", "BAAI/bge-large-en-v1.5"),
            os.getenv("MEM0_COLLECTION_NAME", "ai_concierge_memories"),
        )
    else:
        logger.warning("[Memory] Local Mem0 memory disabled")

    return _client

async def extract_and_store(user_id: str, message: str) -> None:
    """Extract durable user preferences from a message and store them locally."""
    client = _get_client()
    if client is None:
        return

    MIN_MEMORY_MSG_LEN = int(os.getenv("MEM0_MIN_MSG_LEN", "30"))
    MEMORY_SKIP_PATTERN = {
        "hi", "hello", "ok", "okay", "sure", "thanks", "bye",
        "yes", "no", "great", "got it", "alright", "cool"
    }

    stripped = message.strip().lower()
    if not stripped or len(stripped) < MIN_MEMORY_MSG_LEN:
        return
    if stripped in MEMORY_SKIP_PATTERN or len (stripped.split()) <=3:
        return
    def _store() -> None:
        messages = [{"role": "user", "content": message}]
        _call_with_fallbacks(
            [
                lambda: client.add(messages, user_id=user_id),
                lambda: client.add(messages=messages, user_id=user_id),
                lambda: client.add(data=messages, user_id=user_id),
            ],
        )

    try:
        await asyncio.wait_for(asyncio.to_thread(_store), timeout=MEM0_TIMEOUT)
        logger.debug("[Memory] Stored context for user %s", user_id)
    except asyncio.TimeoutError:
        logger.warning("[Memory] Local Mem0 add() timed out for user %s", user_id)
    except Exception as exc:
        logger.warning("[Memory] Failed to store context for user %s: %s", user_id, exc)

asynce def _build_invocation_state_delta(user_id: str, current_query: str) -> dict[str, Any]:
    user_cognitive_context = ""
    query_words = current_query.strip()
    if len(query_words) > 2:
        try:
            from .memory_engine import fetch_user_context
            mem0_context = await fetch_user_context(user_id = user_id, current_query=current_query)
            user_cognitive_context = _normalize_cognitive_context(mem0_context)
            user_cognitive_context = _truncate_text_chars(user_cognitive_context,
            ADK_MAX_COGNITIVE_CONTEXT_CHARS)
        except Exception as exc:
            logger.debug("[ADK] could not fetch cognitive context: %s", exc)
    return {"user_cognitive_context": user_cognitive_context}    

async def fetch_user_context(user_id: str, current_query: str = "") -> str:
    """Retrieve relevant historical user preferences from local Mem0."""
    client = _get_client()
    if client is None:
        return ""

    query = (current_query or "").strip()
    if not query:
        return ""

    if len(query.split()) <= 2:
        return ""

    def _search() -> list[Any]:
        return _normalize_items(
            _call_with_fallbacks(
                [
                    lambda: client.search(query, user_id=user_id, top_k=MEM0_SEARCH_LIMIT),
                    lambda: client.search(query, user_id=user_id, limit=MEM0_SEARCH_LIMIT),
                    lambda: client.search(query=query, user_id=user_id, top_k=MEM0_SEARCH_LIMIT),
                    lambda: client.search(query=query, user_id=user_id, limit=MEM0_SEARCH_LIMIT),
                    lambda: client.search(
                        query,
                        filters={"AND": [{"user_id": user_id}]},
                        top_k=MEM0_SEARCH_LIMIT,
                    ),
                    lambda: client.search(
                        query,
                        filters={"AND": [{"user_id": user_id}]},
                        limit=MEM0_SEARCH_LIMIT,
                    ),
                ],
            )
        )

    try:
        results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=MEM0_TIMEOUT)
        if not results:
            return ""

        facts = []
        for item in results:
            memory_text = None
            if isinstance(item, dict):
                memory_text = item.get("memory") or item.get("text") or item.get("content")
            elif hasattr(item, "memory"):
                memory_text = item.memory
            elif hasattr(item, "text"):
                memory_text = item.text

            if isinstance(memory_text, str) and memory_text.strip():
                facts.append(memory_text.strip())

        context = "; ".join(facts)
        if context:
            logger.debug("[Memory] Retrieved %d facts for user %s", len(facts), user_id)
        return context
    except asyncio.TimeoutError:
        logger.warning("[Memory] Local Mem0 search() timed out for user %s", user_id)
        return ""
    except Exception as exc:
        logger.warning("[Memory] Failed to fetch context for user %s: %s", user_id, exc)
        return ""


async def get_all_memories(user_id: str) -> list:
    """Retrieve all stored memories for a user from the local Mem0 store."""
    client = _get_client()
    if client is None:
        return []

    def _get_all() -> list[Any]:
        return _normalize_items(
            _call_with_fallbacks(
                [
                    lambda: client.get_all(user_id=user_id),
                    lambda: client.get_all(filters={"AND": [{"user_id": user_id}]}),
                    lambda: client.get_all(filters={"AND": [{"user_id": user_id}]}, version="v2"),
                ],
            )
        )

    try:
        results = await asyncio.wait_for(asyncio.to_thread(_get_all), timeout=MEM0_TIMEOUT)
        return results if isinstance(results, list) else []
    except asyncio.TimeoutError:
        logger.warning("[Memory] Local Mem0 get_all() timed out for user %s", user_id)
        return []
    except Exception as exc:
        logger.warning("[Memory] Failed to get all memories for user %s: %s", user_id, exc)
        return []
