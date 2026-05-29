from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SHORTCUTS_PATH = Path(__file__).resolve().parent / "conversation_shortcuts.yaml"


class ShortcutMatch(BaseModel):
    intent: str
    action: str
    direction: Optional[str] = None
    selection_number: Optional[int] = None
    requires_state: List[str] = Field(default_factory=list)
    requires_any_state: List[str] = Field(default_factory=list)


class ShortcutSpec(BaseModel):
    intent: str
    action: str
    direction: Optional[str] = None
    requires_state: List[str] = Field(default_factory=list)
    requires_any_state: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    entity_schema: Dict[str, Any] = Field(default_factory=dict)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _compile_pattern(template: str) -> re.Pattern:
    escaped = re.escape(template.strip().lower())
    escaped = escaped.replace(re.escape("{number}"), r"(?P<number>\d+)")
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"\b{escaped}\b")


def _spec_body(body: Any) -> Dict[str, Any]:
    data = dict(body or {}) if isinstance(body, dict) else {}
    for key in ("requires_state", "requires_any_state", "examples", "patterns"):
        data[key] = _as_str_list(data.get(key))
    return data


class _ShortcutRouter:
    def __init__(self, raw: Dict[str, Any]) -> None:
        raw = raw if isinstance(raw, dict) else {}
        self.version = str(raw.get("version", "1.0"))
        self.specs: List[ShortcutSpec] = []
        self._compiled: Dict[str, List[re.Pattern]] = {}

        shortcuts = raw.get("shortcuts") or {}
        if not isinstance(shortcuts, dict):
            return

        for intent, body in shortcuts.items():
            try:
                spec = ShortcutSpec(intent=str(intent), **_spec_body(body))
                self.specs.append(spec)
                self._compiled[spec.intent] = [
                    _compile_pattern(pattern) for pattern in spec.patterns
                ]
            except Exception as exc:
                logger.warning("[shortcuts] invalid spec for %r: %s", intent, exc)

    def _state_ok(self, spec: ShortcutSpec, soft_state: Dict[str, Any]) -> bool:
        if spec.requires_state and not all(soft_state.get(key) for key in spec.requires_state):
            return False
        if spec.requires_any_state and not any(
            soft_state.get(key) for key in spec.requires_any_state
        ):
            return False
        return True

    def match(self, message: str, soft_state: Optional[Dict[str, Any]]) -> Optional[ShortcutMatch]:
        norm = _normalize(message)
        if not norm:
            return None

        state = soft_state if isinstance(soft_state, dict) else {}

        for spec in self.specs:
            if not self._state_ok(spec, state):
                continue

            for pattern in self._compiled.get(spec.intent, []):
                match = pattern.search(norm)
                if not match:
                    continue
                number = match.groupdict().get("number")
                return ShortcutMatch(
                    intent=spec.intent,
                    action=spec.action,
                    direction=spec.direction,
                    selection_number=int(number) if number is not None else None,
                    requires_state=spec.requires_state,
                    requires_any_state=spec.requires_any_state,
                )

            for example in spec.examples:
                if norm == _normalize(example):
                    return ShortcutMatch(
                        intent=spec.intent,
                        action=spec.action,
                        direction=spec.direction,
                        requires_state=spec.requires_state,
                        requires_any_state=spec.requires_any_state,
                    )

        return None


def _load() -> _ShortcutRouter:
    if not _SHORTCUTS_PATH.exists():
        logger.warning("[shortcuts] %s missing", _SHORTCUTS_PATH)
        return _ShortcutRouter({})
    with open(_SHORTCUTS_PATH, "r", encoding="utf-8") as f:
        return _ShortcutRouter(yaml.safe_load(f) or {})


shortcut_router = _load()


def match_shortcut(message: str, soft_state: Optional[Dict[str, Any]]) -> Optional[ShortcutMatch]:
    return shortcut_router.match(message, soft_state)


def reload() -> None:
    global shortcut_router
    shortcut_router = _load()