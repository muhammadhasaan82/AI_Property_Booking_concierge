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


# --- Environment Loading ---
_repo_root = Path(__file__).resolve().parents[3]
_env_root = _repo_root / ".env"
_env_services = Path(__file__).parent / ".env"
if _env_root.exists():
    load_dotenv(_env_root)
elif _env_services.exists():
    load_dotenv(_env_services)


# ---------------------------------------------------------------------------
# Environment-driven runtime settings
# ---------------------------------------------------------------------------
DATASET_PATH: str = os.getenv("DATASET_PATH", "data/dataset.csv")
MOCK_MODE: bool = _parse_bool(os.getenv("MOCK_MODE", "false"))
PAYMENT_BASE_URL: str = os.getenv("PAYMENT_BASE_URL", "https://example.com/pay")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SESSION_TTL_SECONDS: int = int(os.getenv("REDIS_SESSION_TTL_SECONDS", "86400"))


# Non-routing compatibility constant used by search component.
_DEFAULT_SEED_PROPERTY_TYPES: Final[Set[str]] = {
    "condo",
    "loft",
    "apartment",
    "house",
    "studio",
    "villa",
    "townhouse",
    "flat",
    "cottage",
    "bungalow",
    "penthouse",
    "duplex",
}
SEED_PROPERTY_TYPES: Set[str] = _parse_csv_set(os.getenv("SEED_PROPERTY_TYPES", "")) or set(
    _DEFAULT_SEED_PROPERTY_TYPES
)


__all__ = [
    "DATASET_PATH",
    "MOCK_MODE",
    "PAYMENT_BASE_URL",
    "REDIS_URL",
    "REDIS_SESSION_TTL_SECONDS",
    "SEED_PROPERTY_TYPES",
]

