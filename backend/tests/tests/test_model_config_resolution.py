from __future__ import annotations

import json
import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from app.config.model_config_loader import load_model_config
from app.main import app


def _clear_model_env(monkeypatch):
    for key in (
        "ADK_DISPATCHER_MODEL",
        "ADK_VOICE_MODEL",
        "PRE_ROUTER_FAST_MODEL",
        "MEM0_LLM_PROVIDER",
        "MEM0_LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_missing_model_env_uses_lightweight_defaults(monkeypatch):
    _clear_model_env(monkeypatch)

    resolved = load_model_config()

    assert resolved.dispatcher_model == "openai/gpt-5-nano"
    assert resolved.voice_model == "groq/llama-3.1-8b-instant"
    assert resolved.pre_router_fast_model == "groq/llama-3.1-8b-instant"
    assert resolved.mem0_llm_model == "groq/llama-3.1-8b-instant"


def test_mem0_groq_provider_prefix_is_added_for_bare_model(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("MEM0_LLM_PROVIDER", "groq")
    monkeypatch.setenv("MEM0_LLM_MODEL", "llama-3.1-8b-instant")

    resolved = load_model_config()

    assert resolved.mem0_llm_model == "groq/llama-3.1-8b-instant"


def test_mem0_prefixed_model_is_preserved(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("MEM0_LLM_PROVIDER", "groq")
    monkeypatch.setenv("MEM0_LLM_MODEL", "groq/llama-3.1-8b-instant")

    resolved = load_model_config()

    assert resolved.mem0_llm_model == "groq/llama-3.1-8b-instant"


def test_backend_app_python_and_yaml_do_not_contain_70b_fallback():
    banned = "groq/llama-3.3-70b-versatile"
    scanned_roots = (Path("app"),)

    for root in scanned_roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            assert banned not in path.read_text(encoding="utf-8")


def _llm_public_text(llm: object) -> str:
    parts = [repr(llm), str(llm)]
    for attr in ("model", "model_name", "_model", "name"):
        value = getattr(llm, attr, None)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts)


def test_adk_agent_construction_uses_resolved_models(monkeypatch):
    _clear_model_env(monkeypatch)

    import app.agents.adk_agents as adk_agents

    adk_agents = importlib.reload(adk_agents)
    resolved = load_model_config()
    banned = "groq/llama-3.3-70b-versatile"
    banned_fragment = "70b-versatile"

    assert adk_agents.DISPATCHER_MODEL == resolved.dispatcher_model
    assert adk_agents.VOICE_MODEL == resolved.voice_model
    assert adk_agents.DISPATCHER_MODEL != banned
    assert adk_agents.VOICE_MODEL != banned
    assert banned_fragment not in _llm_public_text(adk_agents.dispatcher_llm)
    assert banned_fragment not in _llm_public_text(adk_agents.voice_llm)


def test_debug_model_config_exposes_models_without_secrets():
    response = TestClient(app).get("/debug/model-config")

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "dispatcher_model",
        "voice_model",
        "pre_router_fast_model",
        "mem0_llm_model",
        "mem0_llm_provider",
    }

    serialized = json.dumps(data).lower()
    for forbidden in ("api_key", "secret", "database_url", "redis_url", "password"):
        assert forbidden not in serialized
