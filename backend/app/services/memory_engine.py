# services/memory_engine.py
"""
Local Cognitive Memory Engine powered by open-source Mem0.

This module uses the self-hosted/open-source Mem0 runtime via
`from mem0 import Memory` and never talks to the managed Mem0 Cloud API.

Architecture:
  - LLM extraction uses the project's existing provider keys such as
    `OPENAI_API_KEY` or `GROQ_API_KEY`.
  - Vector storage stays on infrastructure we control:
      - default: PostgreSQL/pgvector via the existing database connection string
      - optional: local disk persistence (for example, Chroma)
  - All public functions are async and failure-safe.
  - Sync Mem0 SDK calls are offloaded with `asyncio.to_thread(...)`.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from .db_client import build_conninfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MEM0_TIMEOUT: float = float(os.getenv("MEM0_TIMEOUT", "3.0"))
MEM0_SEARCH_LIMIT: int = int(os.getenv("MEM0_SEARCH_LIMIT", "5"))

MEM0_LLM_PROVIDER: str = os.getenv("MEM0_LLM_PROVIDER", "groq").strip().lower()
MEM0_LLM_MODEL: str = os.getenv("MEM0_LLM_MODEL", "llama-3.3-70b-versatile").strip()
MEM0_EMBEDDER_PROVIDER: str = os.getenv("MEM0_EMBEDDER_PROVIDER", "huggingface").strip().lower()
MEM0_EMBEDDER_MODEL: str = os.getenv("MEM0_EMBEDDER_MODEL", "BAAI/bge-large-en-v1.5").strip()

MEM0_VECTOR_STORE: str = os.getenv("MEM0_VECTOR_STORE", "pgvector").strip().lower()
MEM0_COLLECTION_NAME: str = os.getenv("MEM0_COLLECTION_NAME", "ai_concierge_memories").strip()

MEM0_STORAGE_DIR: str = os.getenv("MEM0_STORAGE_DIR", "backend/.mem0").strip()
MEM0_STORAGE_PATH = Path(MEM0_STORAGE_DIR).expanduser()
MEM0_HISTORY_DB_PATH: str = os.getenv("MEM0_HISTORY_DB_PATH", "").strip()


# ---------------------------------------------------------------------------
# Lazy-init singleton
# ---------------------------------------------------------------------------
_client = None
_init_attempted = False


def _provider_api_key(provider: str) -> Optional[str]:
    env_name = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHER_API_KEY",
    }.get(provider)
    return os.getenv(env_name, "").strip() if env_name else None


def _build_llm_config() -> dict[str, Any]:
    config: dict[str, Any] = {"model": MEM0_LLM_MODEL}
    api_key = _provider_api_key(MEM0_LLM_PROVIDER)
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()

    if api_key:
        config["api_key"] = api_key
    if MEM0_LLM_PROVIDER == "openai" and openai_base_url:
        config["openai_base_url"] = openai_base_url

    return {
        "provider": MEM0_LLM_PROVIDER,
        "config": config,
    }


def _build_embedder_config() -> dict[str, Any]:
    config: dict[str, Any] = {"model": MEM0_EMBEDDER_MODEL}
    api_key = _provider_api_key(MEM0_EMBEDDER_PROVIDER)
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()

    if api_key:
        config["api_key"] = api_key
    if MEM0_EMBEDDER_PROVIDER == "openai" and openai_base_url:
        config["openai_base_url"] = openai_base_url

    return {
        "provider": MEM0_EMBEDDER_PROVIDER,
        "config": config,
    }


def _build_vector_store_config() -> dict[str, Any]:
    provider = MEM0_VECTOR_STORE

    if provider in {"postgres", "postgresql", "pgvector"}:
        return {
            "provider": "pgvector",
            "config": {
                "connection_string": build_conninfo(),
                "collection_name": MEM0_COLLECTION_NAME,
            },
        }

    if provider == "supabase":
        return {
            "provider": "supabase",
            "config": {
                "connection_string": build_conninfo(),
                "collection_name": MEM0_COLLECTION_NAME,
            },
        }

    MEM0_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    return {
        "provider": provider,
        "config": {
            "collection_name": MEM0_COLLECTION_NAME,
            "path": str(MEM0_STORAGE_PATH),
        },
    }


def _build_mem0_config() -> dict[str, Any]:
    config: dict[str, Any] = {
        "vector_store": _build_vector_store_config(),
        "llm": _build_llm_config(),
        "embedder": _build_embedder_config(),
    }

    if MEM0_HISTORY_DB_PATH:
        history_path = Path(MEM0_HISTORY_DB_PATH).expanduser()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        config["history_db_path"] = str(history_path)

    return config


def _camelize_vector_store(config: dict[str, Any]) -> dict[str, Any]:
    vector_store = dict(config.get("vector_store", {}))
    inner = dict(vector_store.get("config", {}))

    if "collection_name" in inner:
        inner["collectionName"] = inner.pop("collection_name")
    if "connection_string" in inner:
        inner["connectionString"] = inner.pop("connection_string")

    vector_store["config"] = inner

    alt = dict(config)
    alt["vector_store"] = vector_store
    if "history_db_path" in alt:
        alt["historyDbPath"] = alt.pop("history_db_path")
    return alt


def _build_mem0_config_variants() -> list[dict[str, Any]]:
    base = _build_mem0_config()
    return [base, _camelize_vector_store(base)]


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

    try:
        from mem0 import Memory

        last_error: Optional[Exception] = None
        for config in _build_mem0_config_variants():
            try:
                _client = Memory.from_config(config)
                break
            except Exception as exc:
                last_error = exc
                _client = None

        if _client is None and last_error is not None:
            raise last_error

        logger.info(
            "[Memory] Local Mem0 initialized (llm=%s, embedder=%s, vector_store=%s)",
            MEM0_LLM_PROVIDER,
            MEM0_EMBEDDER_PROVIDER,
            MEM0_VECTOR_STORE,
        )
    except ImportError:
        logger.warning("[Memory] mem0ai package not installed; local memory disabled")
        _client = None
    except Exception as e:
        logger.warning("[Memory] Failed to initialize local Mem0 memory engine: %s", e)
        _client = None

    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_and_store(user_id: str, message: str) -> None:
    """Extract durable user preferences from a message and store them locally."""
    client = _get_client()
    if client is None:
        return

    if not message or len(message.strip()) < 5:
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
    except Exception as e:
        logger.warning("[Memory] Failed to store context for user %s: %s", user_id, e)


async def fetch_user_context(user_id: str, current_query: str = "") -> str:
    """Retrieve relevant historical user preferences from local Mem0."""
    client = _get_client()
    if client is None:
        return ""

    query = current_query.strip() or "user preferences and booking history"

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
    except Exception as e:
        logger.warning("[Memory] Failed to fetch context for user %s: %s", user_id, e)
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
    except Exception as e:
        logger.warning("[Memory] Failed to get all memories for user %s: %s", user_id, e)
        return []
