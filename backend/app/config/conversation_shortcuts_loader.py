"""
YAML-driven deterministic shortcut matcher.

Phrase lists and pattern templates live in conversation_shortcuts.yaml. This
module only normalizes text, compiles generic templates, checks required state,
and returns a typed match for the tool executor.
"""
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
    """
    Convert a YAML value into a cleaned list of strings suitable for shortcut fields.
    
    The function accepts None, a list, or any single value and returns a list of non-empty strings:
    - If `value` is None, returns an empty list.
    - If `value` is a list, converts each element to `str`, strips surrounding whitespace, and keeps only non-empty entries.
    - Otherwise, converts `value` to `str`, strips it, and returns a single-element list when the result is non-empty, or an empty list when it is empty.
    
    Parameters:
        value (Any): The input value from YAML that should be normalized into a list of strings.
    
    Returns:
        List[str]: A list of trimmed, non-empty strings derived from `value`.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize(text: str) -> str:
    """
    Produce a lowercase, whitespace-collapsed form of the input string.
    
    Parameters:
        text (str | None): Input text; `None` is treated as an empty string.
    
    Returns:
        str: The input converted to lowercase with leading/trailing spaces removed and internal whitespace collapsed to single spaces.
    """
    return " ".join((text or "").strip().lower().split())


def _compile_pattern(template: str) -> re.Pattern:
    """
    Compile a template string into a regular expression that matches the whole template, allowing flexible whitespace and an optional numeric capture.
    
    Parameters:
        template (str): Template text where the literal token `{number}` denotes a named capture group `number` for one or more digits. Literal spaces in the template match one or more whitespace characters in input.
    
    Returns:
        re.Pattern: Compiled regular expression that matches the template as a whole and, when `{number}` is used, exposes a named group `number` containing the matched digits.
    """
    escaped = re.escape(template.strip().lower())
    escaped = escaped.replace(re.escape("{number}"), r"(?P<number>\d+)")
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"\b{escaped}\b")


def _spec_body(body: Any) -> Dict[str, Any]:
    """
    Normalize a shortcut YAML "body" into a dict that guarantees canonical list fields.
    
    Parameters:
        body (Any): The raw "body" value from a YAML shortcut entry; may be a dict or any other type.
    
    Returns:
        Dict[str, Any]: A dict (copied from `body` when it is a dict, otherwise empty) with keys
        "requires_state", "requires_any_state", "examples", and "patterns" present and each mapped
        to a list of cleaned strings.
    """
    data = dict(body or {}) if isinstance(body, dict) else {}
    for key in ("requires_state", "requires_any_state", "examples", "patterns"):
        data[key] = _as_str_list(data.get(key))
    return data


class _ShortcutRouter:
    def __init__(self, raw: Dict[str, Any]) -> None:
        """
        Initialize the router by loading shortcut specifications and compiling their patterns.
        
        Given `raw` (typically the dict produced by parsing the YAML file), this constructor:
        - treats a non-dict `raw` as empty,
        - sets `self.version` from `raw["version"]` or `"1.0"` if missing,
        - populates `self.specs` with `ShortcutSpec` instances for each entry in `raw["shortcuts"]` when that value is a dict,
        - compiles each spec's `patterns` into regular expressions and stores them in `self._compiled` keyed by intent,
        - skips any spec that raises an exception during construction or compilation and logs a warning.
        
        Parameters:
            raw (Dict[str, Any]): Parsed configuration mapping (e.g., output of `yaml.safe_load`). Non-dict values are treated as an empty configuration.
        """
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
        """
        Check whether the provided soft_state satisfies the spec's state requirements.
        
        Evaluates the spec's `requires_state` and `requires_any_state` lists against `soft_state` using Python truthiness: all keys listed in `requires_state` must be present and truthy in `soft_state`, and at least one key listed in `requires_any_state` must be present and truthy.
        
        Parameters:
            spec (ShortcutSpec): Shortcut specification containing `requires_state` and `requires_any_state`.
            soft_state (Dict[str, Any]): Mapping of state keys to values; values are checked for truthiness.
        
        Returns:
            bool: `true` if both requirement checks pass, `false` otherwise.
        """
        if spec.requires_state and not all(soft_state.get(key) for key in spec.requires_state):
            return False
        if spec.requires_any_state and not any(
            soft_state.get(key) for key in spec.requires_any_state
        ):
            return False
        return True

    def match(
        self,
        message: str,
        soft_state: Optional[Dict[str, Any]],
    ) -> Optional[ShortcutMatch]:
        """
        Finds the first shortcut that matches a normalized message and current conversational soft state.
        
        Parameters:
            message (str): Incoming user message to normalize and match against compiled shortcut patterns and examples.
            soft_state (Optional[Dict[str, Any]]): Current conversational state used to evaluate spec state requirements; treated as empty dict if not a mapping.
        
        Returns:
            Optional[ShortcutMatch]: A ShortcutMatch populated with `intent`, `action`, optional `direction`, optional integer `selection_number` (if a `{number}` capture was present), and the spec's `requires_state` / `requires_any_state` lists if a matching spec is found; `None` if no spec matches.
        """
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
    """
    Load the conversation shortcuts YAML file and return a router built from its contents.
    
    Returns:
        _ShortcutRouter: Router initialized with parsed YAML data; if the shortcuts file is missing, returns an empty router.
    """
    if not _SHORTCUTS_PATH.exists():
        logger.warning("[shortcuts] %s missing", _SHORTCUTS_PATH)
        return _ShortcutRouter({})
    with open(_SHORTCUTS_PATH, "r", encoding="utf-8") as f:
        return _ShortcutRouter(yaml.safe_load(f) or {})


shortcut_router = _load()


def match_shortcut(
    message: str,
    soft_state: Optional[Dict[str, Any]],
) -> Optional[ShortcutMatch]:
    """
    Finds a configured conversation shortcut that matches the given message and soft conversational state.
    
    Parameters:
        message (str): The incoming user message to match.
        soft_state (Optional[Dict[str, Any]]): Optional mapping of conversational state values used to evaluate a shortcut's `requires_state` and `requires_any_state` constraints.
    
    Returns:
        Optional[ShortcutMatch]: A `ShortcutMatch` containing the matched shortcut's `intent`, `action`, optional `direction`, optional integer `selection_number`, and the spec's `requires_state` / `requires_any_state` lists; `None` if no shortcut matches.
    """
    return shortcut_router.match(message, soft_state)


def reload() -> None:
    """
    Reloads conversation shortcuts from disk into the module-level router.
    
    Replaces the module-level `shortcut_router` with a newly loaded instance constructed
    from the current contents of the conversation_shortcuts.yaml file.
    """
    global shortcut_router
    shortcut_router = _load()
