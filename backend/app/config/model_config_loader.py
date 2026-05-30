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

from app.config.env_loader import load_backend_env

load_backend_env()

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
        """
        Produce a dictionary mapping each dataclass field name to its string value.
        
        Returns:
            dict[str, str]: Mapping of dataclass field names to their string values.
        """
        return asdict(self)


def _env_str(key: str, default: str) -> str:
    """
    Return an environment-backed string for a configuration key, falling back to a provided default.
    
    Parameters:
        key (str): Environment variable name to read.
        default (str): Fallback value used when the environment variable is unset or empty.
    
    Returns:
        str: The environment variable's value with surrounding whitespace removed if it is set and non-empty; otherwise the `default` converted to `str` with surrounding whitespace removed.
    """
    return os.getenv(key, "").strip() or str(default).strip()


def _load_raw_config() -> dict[str, Any]:
    """
    Load and parse the module's agent_config.yaml into a plain dictionary.
    
    Reads the YAML file located at the module's configured _CONFIG_PATH and returns its contents as a dict. If the YAML file is empty or parses to a falsy value, an empty dict is returned.
    
    Returns:
        dict[str, Any]: The parsed YAML content as a dictionary, or an empty dictionary when no content is present.
    """
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dict(value: Any) -> dict[str, Any]:
    """
    Convert a Mapping-like value to a plain dict or return an empty dict.
    
    Parameters:
        value (Any): The object to convert; treated as a mapping if it implements `collections.abc.Mapping`.
    
    Returns:
        dict[str, Any]: A shallow `dict` copy of `value` when it's a `Mapping`, otherwise an empty dict.
    """
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _resolve_model_entry(
    entry: Mapping[str, Any],
    *,
    default_env_key: str,
    safety_default: str,
) -> str:
    """
    Resolve a model entry into the final model string by consulting an environment variable with fallbacks.
    
    Parameters:
        entry (Mapping[str, Any]): Mapping that may contain:
            - "env_key": name of an environment variable to read for the model.
            - "default": default model string to use if the environment variable is not set.
        default_env_key (str): Environment variable name to use when `entry` does not provide "env_key".
        safety_default (str): Fallback default to use when `entry` does not provide "default".
    
    Returns:
        str: The environment value (trimmed) for the chosen env key if it is non-empty; otherwise the chosen default (trimmed).
    """
    env_key = str(entry.get("env_key") or default_env_key)
    default = str(entry.get("default") or safety_default)
    return _env_str(env_key, default)


def _normalize_provider_model(model: str, provider: str) -> str:
    """
    Normalize a model identifier by ensuring it is namespaced with the provider when appropriate.
    
    Parameters:
        model (str): Candidate model identifier; if empty or whitespace, the lightweight Groq default is used.
        provider (str): Provider name to prefix; compared case-insensitively and lowercased before use.
    
    Returns:
        str: A model identifier. If `model` already contains a '/' or `provider` is empty, returns the resolved model as-is (with empty `model` replaced by the lightweight Groq default). Otherwise returns "`{provider}/{model}`" with `provider` lowercased.
    """
    resolved = str(model or "").strip() or _LIGHTWEIGHT_GROQ_MODEL
    provider = str(provider or "").strip().lower()
    if "/" in resolved or not provider:
        return resolved
    return f"{provider}/{resolved}"


def load_model_config(raw: Optional[dict[str, Any]] = None) -> ResolvedModelConfig:
    """
    Resolve model and provider identifiers from environment variables with YAML as a fallback and return a ResolvedModelConfig.
    
    Parameters:
        raw (Optional[dict[str, Any]]): Optional raw configuration dictionary to use instead of loading the local YAML file. When provided, only a mapping is accepted; otherwise the module's YAML loader is used.
    
    Returns:
        ResolvedModelConfig: Immutable container with the resolved fields:
            - dispatcher_model: Resolved dispatcher model identifier.
            - voice_model: Resolved voice/model identifier.
            - pre_router_fast_model: Resolved pre-router fast generator model identifier.
            - mem0_llm_model: Resolved Mem0 model identifier (provider-prefixed when applicable).
            - mem0_llm_provider: Resolved Mem0 provider name (lowercased).
    """
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
    """
    Provide a string-only snapshot of the currently resolved model configuration.
    
    Returns:
        dict[str, str]: Mapping of configuration field names (e.g., "dispatcher_model", "voice_model",
        "pre_router_fast_model", "mem0_llm_model", "mem0_llm_provider") to their resolved string values.
    """
    return load_model_config().public_snapshot()


resolved_models = load_model_config()
