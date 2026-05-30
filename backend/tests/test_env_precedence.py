from __future__ import annotations

import json
import os
from pathlib import Path

from app.config.env_loader import (
    get_env_debug_snapshot,
    get_loaded_dotenv_paths,
    load_backend_env,
)
from app.config.model_config_loader import get_model_config_snapshot


def _write_env(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _clear_model_env(monkeypatch) -> None:
    for key in (
        "ADK_DISPATCHER_MODEL",
        "ADK_VOICE_MODEL",
        "OPENAI_API_KEY",
        "DATABASE_URL",
        "SUPABASE_DB_URL",
        "POSTGRES_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_backend_dotenv_overrides_repo_root(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    backend_root = repo_root / "backend"
    repo_root.mkdir()
    backend_root.mkdir()

    root_env = repo_root / ".env"
    backend_env = backend_root / ".env"

    _write_env(
        root_env,
        "\n".join(
            (
                "ADK_DISPATCHER_MODEL=groq/root-model",
                "ADK_VOICE_MODEL=groq/root-voice",
            )
        ),
    )
    _write_env(
        backend_env,
        "\n".join(
            (
                "ADK_DISPATCHER_MODEL=openai/backend-model",
                "ADK_VOICE_MODEL=groq/backend-voice",
            )
        ),
    )

    _clear_model_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret-db")

    loaded_paths = load_backend_env(
        force=True,
        repo_root=repo_root,
        backend_root=backend_root,
    )

    assert os.environ["ADK_DISPATCHER_MODEL"] == "openai/backend-model"
    assert os.environ["ADK_VOICE_MODEL"] == "groq/backend-voice"
    assert loaded_paths == [str(root_env.resolve()), str(backend_env.resolve())]
    assert get_loaded_dotenv_paths() == loaded_paths

    model_snapshot = get_model_config_snapshot()
    assert model_snapshot["dispatcher_model"] == "openai/backend-model"
    assert model_snapshot["voice_model"] == "groq/backend-voice"

    debug_snapshot = get_env_debug_snapshot()
    assert debug_snapshot == {
        "loaded_paths": [str(root_env.resolve()), str(backend_env.resolve())],
        "root_env_loaded": True,
        "backend_env_loaded": True,
    }

    serialized = json.dumps(debug_snapshot).lower()
    for forbidden in (
        "sk-test-secret",
        "postgresql://secret-db",
        "openai/backend-model",
        "groq/backend-voice",
        "api_key",
        "database_url",
        "postgres_url",
    ):
        assert forbidden not in serialized
