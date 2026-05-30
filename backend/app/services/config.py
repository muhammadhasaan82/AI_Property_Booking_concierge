"""Runtime configuration for backend services.

This module exposes operational settings only.
Lexical intent aliases, regex patterns, property-type aliases, and dynamic
business rules must live in spec/config YAML files, not here.
"""

from __future__ import annotations

import os
from typing import Set

from app.config.env_loader import load_backend_env

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
