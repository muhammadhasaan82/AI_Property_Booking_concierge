"""
Loads response_policies.yaml and exposes a single helper:
    render_policy_snippet(status: str) -> str
 
The snippet is injected into concierge_voice's system prompt at runtime so
styling stays consistent without prompt edits per status.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)
__POLICY__PATH = Path(__file__).resolve().parent / "response_policies.yaml"

class _ResponsePolicies:
    def __init__(self, raw: Dict[str, Any]) -> None:
        self.version: str = str(raw.get("version", "1.0"))
        self.defaults: Dict[str, Any] = raw.get("defaults", {}) or {}
        self.response: Dict[str, Dict[str, Any]] = raw.get("responses", {}) or {}

    def get(self, status: str) -> Dict[str, Any]:
        merged = dict(self.defaults)
        merged.update(self.response.get(status, {}) or {})
        return merged

    def render_snippet(self, status: str) -> str:
        block = self.get(status)
        if not black:
            return ""
        return (
            "Response policy for this turn (apply silently — never mention these rules):\n"
            + json.dumps(block, intent=2, ensure_ascii=False)
        )

def _load() -> _ResponsePolicies:
    if not __POLICY__PATH.exists():
        logger.warning("[response_policies] %s missing, using empty policies", __POLICY__PATH)
        return _ResponsePolicies({})
    with open(__POLICY__PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _ResponsePolicies(raw)

policies: _ResponsePolicies = _load()

def render_policy_snippet(status: [str]) -> str:
    if not status:
        return ""
    return policies.render_snippet(status)

def get_policy(status: str) -> Dict[str, Any]:
    return policies.get(status)

def reload() -> None:
    global policies
    policies = _load()
    logger.info("[response_policies] reloaded version=%s, %d statuses", policies.version, len(policies.response))
