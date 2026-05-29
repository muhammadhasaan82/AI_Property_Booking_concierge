"""
Central, secret-safe model resolution for runtime LLM configuration.

Environment variables win over YAML. This module exposes only resolved model
identifiers and provider names; it never reads or returns API keys or URLs.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "agent_config.yaml"
_LIGHTWEIGHT_GROQ_MODEL = "llama-3.1-8b-instant"
_LIGHTWEIGHT_LITELLM_MODEL = f"groq/{_LIGHTWEIGHT_GROQ_MODEL}"


@dataclass(frozen=True)
class ResolvedModelConfig:
    dispatcher_model: str
    voice_model: str
    pre_router_fast_model: str
    mem0_llm_model: str
    mem0_llm_provider: str

    def public_snapshot(self) -> dict[str, str]:
        return asdict(self)


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, "").strip() or str(default).strip()


def _load_raw_config() -> dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _resolve_model_entry(
    entry: Mapping[str, Any],
    *,
    default_env_key: str,
    safety_default: str,
) -> str:
    env_key = str(entry.get("env_key") or default_env_key)
    default = str(entry.get("default") or safety_default)
    return _env_str(env_key, default)


def _normalize_provider_model(model: str, provider: str) -> str:
    resolved = str(model or "").strip() or _LIGHTWEIGHT_GROQ_MODEL
    provider = str(provider or "").strip().lower()
    if "/" in resolved or not provider:
        return resolved
    return f"{provider}/{resolved}"


def load_model_config(raw: Optional[dict[str, Any]] = None) -> ResolvedModelConfig:
    raw = raw if isinstance(raw, dict) else _load_raw_config()
    models = _dict(raw.get("models"))

    dispatcher_model = _resolve_model_entry(
        _dict(models.get("dispatcher")),
        default_env_key="ADK_DISPATCHER_MODEL",
        safety_default="openai/gpt-5-nano",
    )
    voice_model = _resolve_model_entry(
        _dict(models.get("voice")),
        default_env_key="ADK_VOICE_MODEL",
        safety_default=_LIGHTWEIGHT_LITELLM_MODEL,
    )

    pre_router = _dict(raw.get("pre_router"))
    pre_router_generator = _dict(pre_router.get("generator"))
    pre_router_entry = _dict(models.get("pre_router_fast"))
    pre_router_fast_model = _env_str(
        str(
            pre_router_entry.get("env_key")
            or pre_router_generator.get("model_env_key")
            or "PRE_ROUTER_FAST_MODEL"
        ),
        str(
            pre_router_entry.get("default")
            or pre_router_generator.get("model_default")
            or _LIGHTWEIGHT_LITELLM_MODEL
        ),
    )

    mem0_entry = _dict(models.get("mem0"))
    mem0_provider = _env_str(
        str(mem0_entry.get("provider_env_key") or "MEM0_LLM_PROVIDER"),
        str(mem0_entry.get("provider_default") or "groq"),
    ).lower()
    mem0_raw_model = _env_str(
        str(mem0_entry.get("env_key") or "MEM0_LLM_MODEL"),
        str(mem0_entry.get("default") or _LIGHTWEIGHT_GROQ_MODEL),
    )
    mem0_llm_model = _normalize_provider_model(mem0_raw_model, mem0_provider)

    return ResolvedModelConfig(
        dispatcher_model=dispatcher_model,
        voice_model=voice_model,
        pre_router_fast_model=pre_router_fast_model,
        mem0_llm_model=mem0_llm_model,
        mem0_llm_provider=mem0_provider,
    )


def get_model_config_snapshot() -> dict[str, str]:
    return load_model_config().public_snapshot()


resolved_models = load_model_config()
