"""Runtime configuration for backend services.

This module exposes operational settings only.
Lexical intent aliases, regex patterns, property-type aliases, and dynamic
business rules must live in spec/config YAML files, not here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Set

from app.config.env_loader import load_backend_env


# ---------------------------------------------------------------------------
# Legacy dotenv path compatibility
# ---------------------------------------------------------------------------
# Older tests/callers introspect these paths directly. Keep them module-level.
# Loading order remains broadest-to-narrowest:
#   repo .env -> backend .env -> app/services .env

_backend_root = Path(__file__).resolve().parents[1]
_repo_root = Path(__file__).resolve().parents[2]

_env_root = _repo_root / ".env"
_env_backend = _backend_root / ".env"
_env_services = Path(__file__).resolve().parent / ".env"

_dotenv_paths = [
    str(_env_root),
    str(_env_backend),
    str(_env_services),
]

load_backend_env()


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_set(raw: str) -> Set[str]:
    return {token.strip().lower() for token in raw.split(",") if token.strip()}

DATASET_PATH: str = os.getenv("DATASET_PATH", "data/dataset.csv")
MOCK_MODE: bool = _parse_bool(os.getenv("MOCK_MODE", "false"))
PAYMENT_BASE_URL: str = os.getenv("PAYMENT_BASE_URL", "https://example.com/pay")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SESSION_TTL_SECONDS: int = int(os.getenv("REDIS_SESSION_TTL_SECONDS", "86400"))
ADK_SESSION_MAX_EVENTS: int = int(os.getenv("ADK_SESSION_MAX_EVENTS", "6"))
ADK_SESSION_MAX_CONTEXT_CHARS: int = int(os.getenv("ADK_SESSION_MAX_CONTEXT_CHARS", "3500"))
ADK_MAX_COGNITIVE_CONTEXT_CHARS: int = int(os.getenv("ADK_MAX_COGNITIVE_CONTEXT_CHARS", "600"))


def _load_seed_property_types() -> Set[str]:
    """Load canonical property types from taxonomy YAML.

    The taxonomy spec is the source of truth. Do not maintain a duplicate
    hardcoded property-type list in Python.
    """
    try:
        from app.services.property_type_normalizer import get_all_canonical_types
    except Exception as exc:
        raise RuntimeError(
            "Could not import property type normalizer. "
            "Check app.services.property_type_normalizer."
        ) from exc

    types = get_all_canonical_types()

    if not types:
        raise RuntimeError(
            "No property types loaded from specs/property_type_taxonomy.yaml. "
            "Fix the taxonomy file instead of using hardcoded defaults."
        )

    return types


SEED_PROPERTY_TYPES: Set[str] = (
    _parse_csv_set(os.getenv("SEED_PROPERTY_TYPES", ""))
    or _load_seed_property_types()
)


__all__ = [
    "_backend_root",
    "_repo_root",
    "_env_root",
    "_env_backend",
    "_env_services",
    "_dotenv_paths",
    "DATASET_PATH",
    "MOCK_MODE",
    "PAYMENT_BASE_URL",
    "REDIS_URL",
    "REDIS_SESSION_TTL_SECONDS",
    "ADK_SESSION_MAX_EVENTS",
    "ADK_SESSION_MAX_CONTEXT_CHARS",
    "ADK_MAX_COGNITIVE_CONTEXT_CHARS",
    "SEED_PROPERTY_TYPES",
]
