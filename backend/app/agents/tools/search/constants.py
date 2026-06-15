from __future__ import annotations
"""Module-level search configuration constants."""

from pathlib import Path

from app.config.agent_config_loader import cfg

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = _BACKEND_ROOT / cfg.dataset_relative_path
CITY_COLUMN_CANDIDATES = cfg.city_column_candidates
PROPERTY_RERANK_LIMIT: int = cfg.rerank_limit
PROPERTY_RERANK_TIMEOUT_SECONDS: float = cfg.rerank_timeout
PROPERTY_RESULT_LIMIT_DEFAULT: int = cfg.search_result_limit
PROPERTY_RESULT_LIMIT_MAX: int = cfg.search_result_limit_max
PROPERTY_SUMMARY_THRESHOLD: int = cfg.search_summary_mode_threshold
