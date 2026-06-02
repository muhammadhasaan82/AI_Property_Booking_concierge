from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from app.config.agent_config_loader import cfg

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "property_search_filters.yaml"


class FilterOperatorConfig(BaseModel):
    patterns: List[str] = Field(default_factory=list)


class FilterConfig(BaseModel):
    column: str
    type: str
    match: Optional[str] = None
    aliases: Any = None
    operators: Dict[str, FilterOperatorConfig] = Field(default_factory=dict)
    taxonomy_ref: Optional[str] = None
    taxonomy_aliases: Dict[str, str] = Field(default_factory=dict)
    dataset_column_exists: Optional[bool] = None


class PropertySearchFiltersConfig(BaseModel):
    version: str
    filters: Dict[str, FilterConfig]
    dataset_columns: List[str] = Field(default_factory=list)


def _dataset_path() -> Path:
    return Path(__file__).resolve().parents[2] / cfg.dataset_relative_path


def _dataset_columns() -> List[str]:
    path = _dataset_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [str(col).strip() for col in (reader.fieldnames or []) if str(col).strip()]
    except Exception as exc:
        logger.warning("property_search_filters_loader: could not read dataset columns: %s", exc)
        return []


def _load_taxonomy_aliases(taxonomy_ref: str) -> Dict[str, str]:
    path = (_CONFIG_PATH.parent / taxonomy_ref).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    aliases: Dict[str, str] = {}
    for canonical, spec in (raw.get("property_types") or {}).items():
        canonical_key = str(canonical).strip().lower()
        if not canonical_key:
            continue
        aliases[canonical_key] = canonical_key
        for alias in spec.get("aliases") or []:
            alias_key = str(alias).strip().lower()
            if alias_key:
                aliases[alias_key] = canonical_key
    return aliases


@lru_cache(maxsize=1)
def load_property_search_filters() -> PropertySearchFiltersConfig:
    with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset_columns = _dataset_columns()
    filters: Dict[str, FilterConfig] = {}

    for field_name, spec in (raw.get("filters") or {}).items():
        item = dict(spec or {})
        filter_cfg = FilterConfig.model_validate(item)
        filter_cfg.dataset_column_exists = filter_cfg.column in dataset_columns if dataset_columns else None
        if filter_cfg.taxonomy_ref:
            try:
                filter_cfg.taxonomy_aliases = _load_taxonomy_aliases(filter_cfg.taxonomy_ref)
            except Exception as exc:
                logger.warning(
                    "property_search_filters_loader: failed taxonomy load for %s: %s",
                    field_name,
                    exc,
                )
        if dataset_columns and filter_cfg.dataset_column_exists is False:
            logger.warning(
                "property_search_filters_loader: configured column %r for field %r not found in dataset",
                filter_cfg.column,
                field_name,
            )
        filters[str(field_name)] = filter_cfg

    return PropertySearchFiltersConfig(
        version=str(raw.get("version") or "1.0"),
        filters=filters,
        dataset_columns=dataset_columns,
    )


def get_filter_config(field: str) -> Optional[FilterConfig]:
    return load_property_search_filters().filters.get(field)
