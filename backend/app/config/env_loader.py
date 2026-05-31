from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_loaded_dotenv_paths: list[str] = []
_root_env_loaded: bool = False
_backend_env_loaded: bool = False
_env_loaded: bool = False


def load_backend_env(
    *,
    force: bool = False,
    repo_root: Path | None = None,
    backend_root: Path | None = None,
) -> list[str]:
    """Load .env files and merge them into os.environ with correct precedence.

    Precedence (strongest → weakest):
        1. Pre-existing process environment  (never overwritten)
        2. backend/.env
        3. repo-root .env

    Implementation strategy
    -----------------------
    ``dotenv_values()`` reads each file into a plain dict WITHOUT touching
    ``os.environ``.  We then merge the two dicts (backend wins over root) and
    apply the result with ``os.environ.setdefault()``, which is a no-op when
    the key already exists – so the real process env always stays strongest.

    Parameters
    ----------
    force:
        When *True* the module-level idempotency guard is bypassed and the
        files are re-read and re-applied.  Useful in tests.
    repo_root:
        Override for the repository root directory.  Defaults to three levels
        above this file (``parents[3]``).
    backend_root:
        Override for the backend root directory.  Defaults to two levels
        above this file (``parents[2]``).

    Returns
    -------
    list[str]
        Absolute paths of the .env files that were found on disk (in
        load-order: root first, then backend).
    """
    global _loaded_dotenv_paths, _root_env_loaded, _backend_env_loaded, _env_loaded


    if _env_loaded and not force:
        return list(_loaded_dotenv_paths)

    resolved_repo_root = Path(repo_root).resolve() if repo_root else _DEFAULT_REPO_ROOT
    resolved_backend_root = (
        Path(backend_root).resolve() if backend_root else _DEFAULT_BACKEND_ROOT
    )

    root_env_path = resolved_repo_root / ".env"
    backend_env_path = resolved_backend_root / ".env"

    loaded_paths: list[str] = []
    root_env_loaded = False
    backend_env_loaded = False


    root_values: dict[str, str | None] = {}
    if root_env_path.is_file():
        root_values = dict(dotenv_values(root_env_path))
        loaded_paths.append(str(root_env_path))
        root_env_loaded = True

    backend_values: dict[str, str | None] = {}
    if backend_env_path.is_file():
        backend_values = dict(dotenv_values(backend_env_path))
        loaded_paths.append(str(backend_env_path))
        backend_env_loaded = True

    merged: dict[str, str | None] = {**root_values, **backend_values}


    import os

    for key, value in merged.items():
        if value is not None:
            os.environ.setdefault(key, value)


    _loaded_dotenv_paths = loaded_paths
    _root_env_loaded = root_env_loaded
    _backend_env_loaded = backend_env_loaded
    _env_loaded = True

    return list(_loaded_dotenv_paths)


def get_loaded_dotenv_paths() -> list[str]:
    """Return the list of .env file paths that were successfully loaded."""
    return list(_loaded_dotenv_paths)


def get_env_debug_snapshot() -> dict[str, object]:
    """Return a secret-safe diagnostic snapshot of the loader's state.

    Intentionally exposes only path metadata and boolean flags – never
    variable names, values, API keys, or connection strings.
    """
    return {
        "loaded_paths": get_loaded_dotenv_paths(),
        "root_env_loaded": _root_env_loaded,
        "backend_env_loaded": _backend_env_loaded,
    }


__all__ = [
    "get_env_debug_snapshot",
    "get_loaded_dotenv_paths",
    "load_backend_env",
]
