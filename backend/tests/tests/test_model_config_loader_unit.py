"""
Comprehensive unit tests for app/config/model_config_loader.py.

Covers: _env_str, _dict, _resolve_model_entry, _normalize_provider_model,
load_model_config (with custom raw and env overrides), ResolvedModelConfig,
get_model_config_snapshot, and module-level resolved_models.
"""
from __future__ import annotations

import os
from dataclasses import fields
from typing import Any, Dict

import pytest

from app.config.model_config_loader import (
    ResolvedModelConfig,
    _dict,
    _env_str,
    _normalize_provider_model,
    _resolve_model_entry,
    get_model_config_snapshot,
    load_model_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_ENV_KEYS = (
    "ADK_DISPATCHER_MODEL",
    "ADK_VOICE_MODEL",
    "PRE_ROUTER_FAST_MODEL",
    "MEM0_LLM_PROVIDER",
    "MEM0_LLM_MODEL",
)


def _clear_model_env(monkeypatch):
    for key in _MODEL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# Minimal raw config that mirrors agent_config.yaml models section
_MINIMAL_RAW: Dict[str, Any] = {
    "models": {
        "dispatcher": {
            "env_key": "ADK_DISPATCHER_MODEL",
            "default": "openai/gpt-5-nano",
            "temperature": 0.3,
        },
        "voice": {
            "env_key": "ADK_VOICE_MODEL",
            "default": "groq/llama-3.1-8b-instant",
            "temperature": 0.6,
        },
        "pre_router_fast": {
            "env_key": "PRE_ROUTER_FAST_MODEL",
            "default": "groq/llama-3.1-8b-instant",
        },
        "mem0": {
            "provider_env_key": "MEM0_LLM_PROVIDER",
            "provider_default": "groq",
            "env_key": "MEM0_LLM_MODEL",
            "default": "llama-3.1-8b-instant",
        },
    }
}


# ---------------------------------------------------------------------------
# _env_str
# ---------------------------------------------------------------------------


class TestEnvStr:
    def test_returns_env_value_when_set(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_ABC", "my-model")
        assert _env_str("TEST_KEY_ABC", "default") == "my-model"

    def test_returns_default_when_key_absent(self, monkeypatch):
        monkeypatch.delenv("TEST_KEY_ABC", raising=False)
        assert _env_str("TEST_KEY_ABC", "the-default") == "the-default"

    def test_returns_default_when_env_is_empty_string(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_ABC", "")
        assert _env_str("TEST_KEY_ABC", "the-default") == "the-default"

    def test_strips_whitespace_from_env_value(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_ABC", "  my-model  ")
        assert _env_str("TEST_KEY_ABC", "default") == "my-model"

    def test_strips_whitespace_from_default_when_used(self, monkeypatch):
        monkeypatch.delenv("TEST_KEY_ABC", raising=False)
        assert _env_str("TEST_KEY_ABC", "  default-val  ") == "default-val"

    def test_whitespace_only_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_ABC", "   ")
        assert _env_str("TEST_KEY_ABC", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# _dict
# ---------------------------------------------------------------------------


class TestDictHelper:
    def test_plain_dict_is_returned(self):
        d = {"a": 1}
        assert _dict(d) == {"a": 1}

    def test_none_returns_empty_dict(self):
        assert _dict(None) == {}

    def test_mapping_is_converted(self):
        from collections import OrderedDict
        od = OrderedDict([("x", 10)])
        result = _dict(od)
        assert result == {"x": 10}

    def test_non_mapping_returns_empty_dict(self):
        assert _dict("string") == {}
        assert _dict(42) == {}
        assert _dict([1, 2, 3]) == {}

    def test_empty_dict_returns_empty_dict(self):
        assert _dict({}) == {}


# ---------------------------------------------------------------------------
# _resolve_model_entry
# ---------------------------------------------------------------------------


class TestResolveModelEntry:
    def test_uses_env_key_from_entry_when_env_var_set(self, monkeypatch):
        monkeypatch.setenv("MY_MODEL_KEY", "custom/model")
        entry = {"env_key": "MY_MODEL_KEY", "default": "fallback/model"}
        assert _resolve_model_entry(entry, default_env_key="OTHER_KEY", safety_default="safe") == "custom/model"

    def test_uses_default_from_entry_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("MY_MODEL_KEY", raising=False)
        entry = {"env_key": "MY_MODEL_KEY", "default": "yaml-default/model"}
        assert _resolve_model_entry(entry, default_env_key="OTHER_KEY", safety_default="safe") == "yaml-default/model"

    def test_uses_safety_default_when_entry_has_no_default(self, monkeypatch):
        monkeypatch.delenv("ADK_DISPATCHER_MODEL", raising=False)
        monkeypatch.delenv("MY_MODEL_KEY", raising=False)
        entry = {"env_key": "MY_MODEL_KEY"}
        # entry.get("default") is None → safety_default used
        result = _resolve_model_entry(entry, default_env_key="ADK_DISPATCHER_MODEL", safety_default="openai/gpt-5-nano")
        assert result == "openai/gpt-5-nano"

    def test_fallback_to_default_env_key_when_entry_has_no_env_key(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_KEY", "from-default-key")
        entry = {}  # no env_key in entry
        result = _resolve_model_entry(entry, default_env_key="DEFAULT_KEY", safety_default="safe")
        assert result == "from-default-key"

    def test_empty_entry_uses_safety_default(self, monkeypatch):
        monkeypatch.delenv("ADK_DISPATCHER_MODEL", raising=False)
        result = _resolve_model_entry({}, default_env_key="ADK_DISPATCHER_MODEL", safety_default="safe-model")
        assert result == "safe-model"


# ---------------------------------------------------------------------------
# _normalize_provider_model
# ---------------------------------------------------------------------------


class TestNormalizeProviderModel:
    def test_model_without_slash_gets_provider_prefix(self):
        assert _normalize_provider_model("llama-3.1-8b-instant", "groq") == "groq/llama-3.1-8b-instant"

    def test_model_with_slash_is_preserved(self):
        assert _normalize_provider_model("groq/llama-3.1-8b-instant", "groq") == "groq/llama-3.1-8b-instant"

    def test_empty_model_falls_back_to_lightweight(self):
        # empty model → _LIGHTWEIGHT_GROQ_MODEL = "llama-3.1-8b-instant"
        result = _normalize_provider_model("", "groq")
        assert result == "groq/llama-3.1-8b-instant"

    def test_none_model_falls_back_to_lightweight(self):
        result = _normalize_provider_model(None, "groq")
        assert result == "groq/llama-3.1-8b-instant"

    def test_whitespace_model_falls_back_to_lightweight(self):
        result = _normalize_provider_model("   ", "groq")
        assert result == "groq/llama-3.1-8b-instant"

    def test_empty_provider_returns_model_as_is(self):
        assert _normalize_provider_model("llama-model", "") == "llama-model"

    def test_none_provider_returns_model_as_is(self):
        assert _normalize_provider_model("llama-model", None) == "llama-model"

    def test_provider_is_lowercased(self):
        result = _normalize_provider_model("some-model", "OPENAI")
        assert result == "openai/some-model"

    def test_different_provider_openai(self):
        result = _normalize_provider_model("gpt-4o-mini", "openai")
        assert result == "openai/gpt-4o-mini"

    def test_already_prefixed_with_different_provider(self):
        # slash present → preserved even if provider differs
        assert _normalize_provider_model("openai/gpt-4", "groq") == "openai/gpt-4"


# ---------------------------------------------------------------------------
# load_model_config with custom raw dict (no file IO)
# ---------------------------------------------------------------------------


class TestLoadModelConfigWithCustomRaw:
    def test_returns_resolved_model_config_instance(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config(_MINIMAL_RAW)
        assert isinstance(result, ResolvedModelConfig)

    def test_dispatcher_uses_yaml_default_when_no_env(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config(_MINIMAL_RAW)
        assert result.dispatcher_model == "openai/gpt-5-nano"

    def test_voice_uses_yaml_default_when_no_env(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config(_MINIMAL_RAW)
        assert result.voice_model == "groq/llama-3.1-8b-instant"

    def test_pre_router_uses_yaml_default_when_no_env(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config(_MINIMAL_RAW)
        assert result.pre_router_fast_model == "groq/llama-3.1-8b-instant"

    def test_mem0_provider_defaults_to_groq(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config(_MINIMAL_RAW)
        assert result.mem0_llm_provider == "groq"

    def test_mem0_model_gets_provider_prefix(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config(_MINIMAL_RAW)
        # bare "llama-3.1-8b-instant" + provider "groq" → "groq/llama-3.1-8b-instant"
        assert result.mem0_llm_model == "groq/llama-3.1-8b-instant"

    def test_dispatcher_env_overrides_yaml(self, monkeypatch):
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("ADK_DISPATCHER_MODEL", "openai/gpt-4o")
        result = load_model_config(_MINIMAL_RAW)
        assert result.dispatcher_model == "openai/gpt-4o"

    def test_voice_env_overrides_yaml(self, monkeypatch):
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("ADK_VOICE_MODEL", "groq/custom-voice")
        result = load_model_config(_MINIMAL_RAW)
        assert result.voice_model == "groq/custom-voice"

    def test_pre_router_env_overrides_yaml(self, monkeypatch):
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("PRE_ROUTER_FAST_MODEL", "groq/llama-3.3-8b-fast")
        result = load_model_config(_MINIMAL_RAW)
        assert result.pre_router_fast_model == "groq/llama-3.3-8b-fast"

    def test_mem0_env_overrides_yaml_bare_model(self, monkeypatch):
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("MEM0_LLM_PROVIDER", "groq")
        monkeypatch.setenv("MEM0_LLM_MODEL", "llama-3.1-8b-instant")
        result = load_model_config(_MINIMAL_RAW)
        assert result.mem0_llm_model == "groq/llama-3.1-8b-instant"

    def test_mem0_env_prefixed_model_preserved(self, monkeypatch):
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("MEM0_LLM_MODEL", "groq/llama-3.1-8b-instant")
        result = load_model_config(_MINIMAL_RAW)
        assert result.mem0_llm_model == "groq/llama-3.1-8b-instant"

    def test_mem0_provider_is_lowercased(self, monkeypatch):
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("MEM0_LLM_PROVIDER", "GROQ")
        result = load_model_config(_MINIMAL_RAW)
        assert result.mem0_llm_provider == "groq"

    def test_none_raw_loads_from_file(self, monkeypatch):
        """Passing None should load from the real YAML file without error."""
        _clear_model_env(monkeypatch)
        result = load_model_config(None)
        assert isinstance(result, ResolvedModelConfig)
        assert result.dispatcher_model  # non-empty

    def test_empty_raw_dict_uses_safety_defaults(self, monkeypatch):
        """An empty dict means no models section → safety defaults apply."""
        _clear_model_env(monkeypatch)
        result = load_model_config({})
        assert result.dispatcher_model == "openai/gpt-5-nano"
        assert result.voice_model == "groq/llama-3.1-8b-instant"
        assert result.mem0_llm_model == "groq/llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# ResolvedModelConfig properties
# ---------------------------------------------------------------------------


class TestResolvedModelConfig:
    def _make(self, **kwargs):
        defaults = {
            "dispatcher_model": "openai/gpt-5-nano",
            "voice_model": "groq/llama-3.1-8b-instant",
            "pre_router_fast_model": "groq/llama-3.1-8b-instant",
            "mem0_llm_model": "groq/llama-3.1-8b-instant",
            "mem0_llm_provider": "groq",
        }
        defaults.update(kwargs)
        return ResolvedModelConfig(**defaults)

    def test_is_frozen_dataclass(self):
        cfg = self._make()
        with pytest.raises((AttributeError, TypeError)):
            cfg.dispatcher_model = "changed"  # type: ignore[misc]

    def test_public_snapshot_contains_all_fields(self):
        cfg = self._make()
        snap = cfg.public_snapshot()
        assert set(snap.keys()) == {
            "dispatcher_model",
            "voice_model",
            "pre_router_fast_model",
            "mem0_llm_model",
            "mem0_llm_provider",
        }

    def test_public_snapshot_values_match_fields(self):
        cfg = self._make(dispatcher_model="openai/gpt-4o")
        snap = cfg.public_snapshot()
        assert snap["dispatcher_model"] == "openai/gpt-4o"

    def test_public_snapshot_contains_no_secrets(self):
        cfg = self._make()
        snap = cfg.public_snapshot()
        import json
        serialized = json.dumps(snap).lower()
        for forbidden in ("api_key", "secret", "password", "token", "database_url"):
            assert forbidden not in serialized


# ---------------------------------------------------------------------------
# get_model_config_snapshot
# ---------------------------------------------------------------------------


class TestGetModelConfigSnapshot:
    def test_returns_dict(self):
        snap = get_model_config_snapshot()
        assert isinstance(snap, dict)

    def test_has_expected_keys(self):
        snap = get_model_config_snapshot()
        assert "dispatcher_model" in snap
        assert "voice_model" in snap
        assert "pre_router_fast_model" in snap
        assert "mem0_llm_model" in snap
        assert "mem0_llm_provider" in snap

    def test_values_are_non_empty_strings(self):
        snap = get_model_config_snapshot()
        for key, value in snap.items():
            assert isinstance(value, str), f"{key} must be a string"
            assert value.strip(), f"{key} must be non-empty"


# ---------------------------------------------------------------------------
# load_model_config uses real YAML file (integration)
# ---------------------------------------------------------------------------


class TestLoadModelConfigFromFile:
    def test_dispatcher_default_from_yaml(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config()
        assert result.dispatcher_model == "openai/gpt-5-nano"

    def test_voice_default_is_lightweight(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config()
        assert "70b" not in result.voice_model

    def test_mem0_default_is_lightweight(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config()
        assert "70b" not in result.mem0_llm_model

    def test_pre_router_default_is_lightweight(self, monkeypatch):
        _clear_model_env(monkeypatch)
        result = load_model_config()
        assert "70b" not in result.pre_router_fast_model

    def test_banned_70b_model_not_present_anywhere(self, monkeypatch):
        """Regression: old default groq/llama-3.3-70b-versatile must not appear."""
        _clear_model_env(monkeypatch)
        result = load_model_config()
        banned = "llama-3.3-70b-versatile"
        snap = result.public_snapshot()
        for key, value in snap.items():
            assert banned not in value, f"{key} should not contain {banned}"