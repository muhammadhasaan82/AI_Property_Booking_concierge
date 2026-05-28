"""
Loads conversation_shortcuts.yaml and resolves deterministic conversational
shortcuts (pagination + numbered option selection) from configured examples.
 
Phrase lists and patterns live ONLY in YAML. This module is a generic matcher
engine: it compiles YAML examples/patterns at load time and returns a typed
ShortcutMatch, or None when nothing matches (so ADK/LLM handles the turn).
"""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_SHORTCUTS_PATH = Path(__file__).resolve().parent / "conversation_shortcuts.yaml"
_NUMBER_TOKEN = "{number}"

class ShortcutMatch(BaseMatch):
    intent: str
    action: str
    direction: Optional[str] = None
    selection_number: Optional[int] = None
    requires_state: List[str] = Field(default_factory=list)

class _ShortcutSpec(BaseModel):
    intent: str
    action: str
    direction: Optional[str] = None
    requires_state: Optional[str] = None
    examples: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)

def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())

def _compile_pattern(template: str) -> re.Pattern:
    """Generic engine: turn a YAML template like 'option {number}' into a regex.
 
    Literal text is escaped; the {number} token becomes a digit capture group.
    The business content stays in YAML — only the compiler lives here.
    """
    parts = template.strip().lower().split(_NUMBER_TOKEN)
    escaped = [re.escape(p.strip()) for p in parts]
    if len(escaped) > 1:
        body = r"\s+(\d+)\b".join(escaped)
    else:
        body = escaped[0]
    return re.compile(body)

class _ShortcutRouter:
    def __init__(self, raw: Dict[str, Any]) -> None:
        self.version: str = str(raw.get("version", "1.0"))
        self.specs: List[_ShortcutSpec] = []
        self._compiled: Dict[str, List[re.Pattern]] = {}
        for intent, body in (raw.get("shortcuts", {}) or {}).items():
            try:
                spec = _ShortcutSpec(intent=intent, **(body or {}))
                self.specs.append(spec)
                self._compiled[intent] = [_compile_pattern(p) for p in spec.patterns]
            except Exception as e:
                logger.warning("[shortcuts] invalid spec for %r: %s", intent, exc)
                
    def _state_ok(self, spec: _ShortcutSpec, soft_state: Dict[str, Any]) -> bool:
        if not spec.requires_state:
            return True
        return all(soft_state.get(key) for key in spec.requires_state)
        
    def match(
        self, message: str, soft_state: Dict[str, Any]
    ) -> Optional[ShortcutMatch]:
        norm = _normalize(message)
        if not norm:
            return None
        soft_state = soft_state if isinstance(soft_state, dict) else {}
        for spec in self.specs:
            if not self._compiled.get(spec.intent) or not self._state_ok(spec, soft_state):
                continue
            for pattern in self._compiled[spec.intent]:
                m = pattern.search(norm)
                if m and m.groups():
                    return ShortcutMatch(
                        intent=spec.intent,
                        action=spec.action,
                        direction=spec.direction,
                        selection_number=int(m.group(1)),
                        requires_state=spec.requires_state,
                    )

            for spec in self.specs:
                if not spec.examples or not self._state_ok(spec, soft_state):
                    continue
                padded = f" {norm} "
                for example in spec.examples:
                    ex = _normalize(example)
                    if ex and (norm == ex or f" {ex} " in padded):
                        return ShortcutMatch(
                            intent=spec.intent,
                            action=spec.action,
                            direction=spec.direction,
                            requires_state=spec.requires_state,
                        )
        return None

    def _load() -> _ShortcutRouter:
        if not _SHORTCUTS_PATH.exists():
            logger.warning("[shortcuts] %s missing, no deterministic shortcuts", _SHORTCUTS_PATH)
            return _ShortcutRouter({})
        with open(_SHORTCUTS_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return _ShortcutRouter(raw)

    shortcut_router: _ShortcutRouter = _load()

    def match_shortcut(
        message: str, soft_state: Optional[Dict[str, Any]]
    ) -> Optional[ShortcutMatch]:
        return shortcut_router.match(message, soft_state)

    def reload() -> None:
        global shortcut_router
        shortcut_router = _load()
        logger.info(
            "[shortcuts] reloaded version=%s, %d shortcuts",
            shortcut_router.version,
            len(shortcut_router.specs),
        )