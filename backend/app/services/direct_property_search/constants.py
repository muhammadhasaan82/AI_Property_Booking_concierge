from __future__ import annotations
"""Shared constants for direct property search."""

from pathlib import Path

_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "property_type_taxonomy.yaml"
)
