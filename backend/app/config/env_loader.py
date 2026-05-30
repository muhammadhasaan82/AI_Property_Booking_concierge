from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_BACKEND_ROOT = Path(__file__).resolve().parents[2]

_loaded_dotenv_paths: list[str] = []
_root_env_loaded = False
_backend_env_loaded = False
_env_loaded = False


def load_backend_env(
    *,
    force: bool = False,
    repo_root: Path | None = None,
    backend_root: Path | None = None,
) -> list[str]:
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

    if root_env_path.is_file():
        load_dotenv(root_env_path, override=False)
        loaded_paths.append(str(root_env_path))
        root_env_loaded = True

    if backend_env_path.is_file():
        load_dotenv(backend_env_path, override=True)
        loaded_paths.append(str(backend_env_path))
        backend_env_loaded = True

    _loaded_dotenv_paths = loaded_paths
    _root_env_loaded = root_env_loaded
    _backend_env_loaded = backend_env_loaded
    _env_loaded = True

    return list(_loaded_dotenv_paths)


def get_loaded_dotenv_paths() -> list[str]:
    return list(_loaded_dotenv_paths)


def get_env_debug_snapshot() -> dict[str, object]:
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
