from __future__ import annotations

import json
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


def test_backend_config_and_services_do_not_contain_70b_fallback():
    banned = "groq/llama-3.3-70b-versatile"
    scanned_roots = (Path("app/config"), Path("app/services"))

    for root in scanned_roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            assert banned not in path.read_text(encoding="utf-8")


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
