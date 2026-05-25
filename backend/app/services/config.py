"""Runtime configuration for backend services.

This module intentionally exposes operational settings only and does not
contain lexical intent-routing lists or dynamic fallback phrase tables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Set

from dotenv import load_dotenv


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_set(raw: str) -> Set[str]:
    return {token.strip().lower() for token in raw.split(",") if token.strip()}

_repo_root = Path(__file__).resolve().parents[3]
_env_root = _repo_root / ".env"
_env_services = Path(__file__).parent / ".env"
if _env_root.exists():
    load_dotenv(_env_root)
if _env_services.exists():
    load_dotenv(_env_services, override=True)

DATASET_PATH: str = os.getenv("DATASET_PATH", "data/dataset.csv")
MOCK_MODE: bool = _parse_bool(os.getenv("MOCK_MODE", "false"))
PAYMENT_BASE_URL: str = os.getenv("PAYMENT_BASE_URL", "https://example.com/pay")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SESSION_TTL_SECONDS: int = int(os.getenv("REDIS_SESSION_TTL_SECONDS", "86400"))
ADK_SESSION_MAX_EVENTS: int = int(os.getenv("ADK_SESSION_MAX_EVENTS", "6"))
ADK_SESSION_MAX_CONTEXT_CHARS: int = int(os.getenv("ADK_SESSION_MAX_CONTEXT_CHARS", "3500"))
ADK_MAX_COGNITIVE_CONTEXT_CHARS: int = int(os.getenv("ADK_MAX_COGNITIVE_CONTEXT_CHARS", "600"))

def _load_seed_property_types() -> Set[str]:
    """Load canonical property types from taxonomy YAML. Falls back to a
    hardcoded minimal set only if the YAML is unavailable at startup."""
    try:
        from app.services.property_type_normalizer import get_all_canonical_types
        types = get_all_canonical_types()
        if types:
            return types
    except Exception:
        pass
    return {
        "apartment", "house", "duplex", "townhouse", "loft", "villa",
        "studio", "condo", "cottage", "bungalow", "penthouse", "guesthouse",
    }
SEED_PROPERTY_TYPES: Set[str] = _parse_csv_set(os.getenv("SEED_PROPERTY_TYPES", "")) or _load_seed_property_types()

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

