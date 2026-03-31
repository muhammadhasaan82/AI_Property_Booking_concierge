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

import asyncio
import importlib
import inspect
import json
import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MEM0_TIMEOUT: float = float(os.getenv("MEM0_TIMEOUT", "3.0"))
MEM0_SEARCH_LIMIT: int = int(os.getenv("MEM0_SEARCH_LIMIT", "5"))

MEM0_LLM_PROVIDER: str = os.getenv("MEM0_LLM_PROVIDER", "groq")
MEM0_LLM_MODEL: str = os.getenv("MEM0_LLM_MODEL", "llama-3.3-70b-versatile")
MEM0_EMBEDDER_PROVIDER: str = os.getenv("MEM0_EMBEDDER_PROVIDER", "huggingface")
MEM0_EMBEDDER_MODEL: str = os.getenv("MEM0_EMBEDDER_MODEL", "BAAI/bge-large-en-v1.5")

MEM0_VECTOR_STORE: str = os.getenv("MEM0_VECTOR_STORE", "chroma")
MEM0_COLLECTION_NAME: str = os.getenv("MEM0_COLLECTION_NAME", "ai_concierge_memories")
MEM0_STORAGE_DIR: str = os.getenv("MEM0_STORAGE_DIR", "backend/.mem0")


# ---------------------------------------------------------------------------
# Lazy-init singleton
# ---------------------------------------------------------------------------
_client = None
_init_attempted = False


def _provider_module_candidates(kind: str, provider: str) -> list[str]:
    normalized = provider.strip().replace("-", "_").replace(" ", "_").lower()
    module_map = {
        "vector_store": [
            f"mem0.configs.vector_stores.{normalized}",
            "mem0.vector_stores.configs",
        ],
        "embedder": [
            f"mem0.configs.embeddings.{normalized}",
            "mem0.embeddings.configs",
        ],
        "llm": [
            f"mem0.configs.llms.{normalized}",
            "mem0.llms.configs",
        ],
    }
    return list(dict.fromkeys(module_map.get(kind, [])))


def _resolve_provider_config_model(kind: str, provider: str) -> Optional[type]:
    provider_token = provider.strip().replace("-", "").replace("_", "").lower()

    for module_name in _provider_module_candidates(kind, provider):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        candidates = []
        for value in vars(module).values():
            if not inspect.isclass(value):
                continue
            if not hasattr(value, "model_fields"):
                continue
            if not str(getattr(value, "__module__", "")).startswith(module.__name__):
                continue
            candidates.append(value)

        if not candidates:
            continue

        matching = [
            candidate
            for candidate in candidates
            if provider_token in candidate.__name__.replace("_", "").lower()
        ]
        if matching:
            return sorted(matching, key=lambda item: len(item.__name__))[0]

        if len(candidates) == 1:
            return candidates[0]

    return None


def _allowed_fields_for(kind: str, provider: str) -> set[str]:
    model = _resolve_provider_config_model(kind, provider)
    if model is not None:
        return set(getattr(model, "model_fields", {}).keys())

    fallback_fields = {
        "vector_store": {
            "port",
            "host",
            "path",
            "collection_name",
            "api_key",
            "client",
            "tenant",
        },
        "embedder": {"model"},
        "llm": {"model"},
    }
    return set(fallback_fields.get(kind, set()))


def _field_env_names(kind: str, field_name: str) -> list[str]:
    normalized = field_name.upper()
    prefix = {
        "vector_store": "MEM0_VECTOR_STORE",
        "embedder": "MEM0_EMBEDDER",
        "llm": "MEM0_LLM",
    }.get(kind, "MEM0")

    names = [f"{prefix}_{normalized}", f"MEM0_{normalized}"]

    if kind == "vector_store" and field_name == "path":
        names.insert(1, "MEM0_STORAGE_DIR")
    if kind == "vector_store" and field_name == "collection_name":
        names.insert(1, "MEM0_COLLECTION_NAME")

    return list(dict.fromkeys(names))


def _coerce_config_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return ""

    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except Exception:
            return value

    return value


def _resolve_field_value(kind: str, field_name: str, defaults: dict[str, Any]) -> Any:
    for env_name in _field_env_names(kind, field_name):
        env_value = os.getenv(env_name)
        if env_value not in (None, ""):
            return _coerce_config_value(env_value)

    return defaults.get(field_name)


def _build_component_config(kind: str, provider: str, defaults: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = _allowed_fields_for(kind, provider)
    component_config: dict[str, Any] = {}

    for field_name in allowed_fields:
        value = _resolve_field_value(kind, field_name, defaults)
        if value not in (None, ""):
            component_config[field_name] = value

    return {
        "provider": provider,
        "config": component_config,
    }


def initialize_memory():
    """Initialize the local Mem0 Memory client with a schema-safe config."""
    try:
        Memory = importlib.import_module("mem0").Memory
    except Exception as exc:
        logger.warning("[Memory] Mem0 import unavailable, disabling memory engine: %s", exc)
        return None

    try:
        config = {
            "vector_store": _build_component_config(
                kind="vector_store",
                provider=MEM0_VECTOR_STORE,
                defaults={
                    "collection_name": MEM0_COLLECTION_NAME,
                    "path": MEM0_STORAGE_DIR,
                },
            ),
            "llm": _build_component_config(
                kind="llm",
                provider=MEM0_LLM_PROVIDER,
                defaults={"model": MEM0_LLM_MODEL},
            ),
            "embedder": _build_component_config(
                kind="embedder",
                provider=MEM0_EMBEDDER_PROVIDER,
                defaults={"model": MEM0_EMBEDDER_MODEL},
            ),
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
            "[Memory] Local Mem0 initialized (llm=%s, embedder=%s, vector_store=%s)",
            MEM0_LLM_PROVIDER,
            MEM0_EMBEDDER_PROVIDER,
            MEM0_VECTOR_STORE,
        )
    else:
        logger.warning("[Memory] Local Mem0 memory disabled")

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
    except Exception as exc:
        logger.warning("[Memory] Failed to store context for user %s: %s", user_id, exc)


async def fetch_user_context(user_id: str, current_query: str = "") -> str:
    """Retrieve relevant historical user preferences from local Mem0."""
    client = _get_client()
    if client is None:
        return ""

    query = (current_query or "").strip()
    if not query:
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
