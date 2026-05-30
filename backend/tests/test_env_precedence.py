"""
Tests for env_loader precedence.

Required precedence (strongest → weakest):
    1. Pre-existing process environment   (never overwritten)
    2. backend/.env
    3. repo-root .env

Two scenarios are proven here:
    A. backend/.env beats repo-root .env            (test_backend_dotenv_overrides_repo_root)
    B. Pre-existing os.environ beats both files     (test_process_env_beats_both_dotenv_files)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.config.env_loader import (
    get_env_debug_snapshot,
    get_loaded_dotenv_paths,
    load_backend_env,
)
from app.config.model_config_loader import get_model_config_snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_env(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _clear_model_env(monkeypatch) -> None:
    """Remove model/infra env vars so tests start from a clean slate."""
    for key in (
        "ADK_DISPATCHER_MODEL",
        "ADK_VOICE_MODEL",
        "OPENAI_API_KEY",
        "DATABASE_URL",
        "SUPABASE_DB_URL",
        "POSTGRES_URL",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Scenario A – backend/.env wins over repo-root .env
# ---------------------------------------------------------------------------

def test_backend_dotenv_overrides_repo_root(monkeypatch, tmp_path):
    """backend/.env values must override repo-root .env values."""
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

    # Remove these keys from process env so file values are used
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret-db")

    loaded_paths = load_backend_env(
        force=True,
        repo_root=repo_root,
        backend_root=backend_root,
    )

    # backend/.env must win over repo-root .env
    assert os.environ["ADK_DISPATCHER_MODEL"] == "openai/backend-model", (
        "backend/.env should override repo-root .env"
    )
    assert os.environ["ADK_VOICE_MODEL"] == "groq/backend-voice", (
        "backend/.env should override repo-root .env"
    )

    # Path list: root first, then backend
    assert loaded_paths == [str(root_env.resolve()), str(backend_env.resolve())]
    assert get_loaded_dotenv_paths() == loaded_paths

    # Model config snapshot reflects merged file values
    model_snapshot = get_model_config_snapshot()
    assert model_snapshot["dispatcher_model"] == "openai/backend-model"
    assert model_snapshot["voice_model"] == "groq/backend-voice"

    # Debug snapshot must be secret-safe
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
        assert forbidden not in serialized, (
            f"Secret-safe snapshot must not contain: {forbidden!r}"
        )


# ---------------------------------------------------------------------------
# Scenario B – pre-existing process env beats both .env files
# ---------------------------------------------------------------------------

def test_process_env_beats_both_dotenv_files(monkeypatch, tmp_path):
    """A value already in os.environ must NOT be overwritten by either .env file."""
    repo_root = tmp_path / "repo"
    backend_root = repo_root / "backend"
    repo_root.mkdir()
    backend_root.mkdir()

    root_env = repo_root / ".env"
    backend_env = backend_root / ".env"

    _write_env(
        root_env,
        "ADK_DISPATCHER_MODEL=groq/root-model",
    )
    _write_env(
        backend_env,
        "ADK_DISPATCHER_MODEL=openai/backend-model",
    )

    # Pre-set the key in process env BEFORE calling load_backend_env
    monkeypatch.setenv("ADK_DISPATCHER_MODEL", "process-env-winner")

    loaded_paths = load_backend_env(
        force=True,
        repo_root=repo_root,
        backend_root=backend_root,
    )

    # The process env value must survive – neither file may overwrite it
    assert os.environ["ADK_DISPATCHER_MODEL"] == "process-env-winner", (
        "Pre-existing process env must not be overwritten by any .env file"
    )

    # Both files were found and recorded even though their value was suppressed
    assert loaded_paths == [str(root_env.resolve()), str(backend_env.resolve())]
    assert get_loaded_dotenv_paths() == loaded_paths

    debug_snapshot = get_env_debug_snapshot()
    assert debug_snapshot["root_env_loaded"] is True
    assert debug_snapshot["backend_env_loaded"] is True


# ---------------------------------------------------------------------------
# Scenario C – idempotency guard (no force)
# ---------------------------------------------------------------------------

def test_load_backend_env_is_idempotent(monkeypatch, tmp_path):
    """Without force=True, a second call returns the cached path list unchanged."""
    repo_root = tmp_path / "repo"
    backend_root = repo_root / "backend"
    repo_root.mkdir()
    backend_root.mkdir()

    _write_env(repo_root / ".env", "SOME_KEY=value-from-root")

    _clear_model_env(monkeypatch)

    first = load_backend_env(force=True, repo_root=repo_root, backend_root=backend_root)

    # Modify the file on disk after first load
    _write_env(repo_root / ".env", "SOME_KEY=changed-value")

    # Second call without force must return the cached result unchanged
    second = load_backend_env(repo_root=repo_root, backend_root=backend_root)

    assert first == second, "Idempotent call must return the same path list"
    # Env var still has the original value (second call was a no-op)
    assert os.environ.get("SOME_KEY") == "value-from-root"


# ---------------------------------------------------------------------------
# Scenario D – force=True re-reads files
# ---------------------------------------------------------------------------

def test_force_true_re_reads_files(monkeypatch, tmp_path):
    """force=True bypasses the idempotency guard and re-reads .env files."""
    repo_root = tmp_path / "repo"
    backend_root = repo_root / "backend"
    repo_root.mkdir()
    backend_root.mkdir()

    env_file = repo_root / ".env"
    _write_env(env_file, "FORCE_TEST_KEY=first-value")

    monkeypatch.delenv("FORCE_TEST_KEY", raising=False)

    load_backend_env(force=True, repo_root=repo_root, backend_root=backend_root)
    assert os.environ.get("FORCE_TEST_KEY") == "first-value"

    # Simulate a new value appearing – remove key from env so setdefault picks it up
    monkeypatch.delenv("FORCE_TEST_KEY", raising=False)
    _write_env(env_file, "FORCE_TEST_KEY=second-value")

    load_backend_env(force=True, repo_root=repo_root, backend_root=backend_root)
    assert os.environ.get("FORCE_TEST_KEY") == "second-value"
