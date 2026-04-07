"""
app/agents/prompts/loader.py
-----------------------------
Lightweight prompt loader with safe fallback behavior.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(filename: str, *, fallback: Optional[str] = "") -> str:
    path = _PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Prompt missing or unreadable (%s): %s", path, exc)
        return fallback or ""
