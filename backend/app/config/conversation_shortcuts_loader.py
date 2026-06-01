"""
YAML-driven deterministic shortcut matcher.

Phrase lists, pattern templates, and generic semantic-cue groups live in
conversation_shortcuts.yaml. This module only normalizes text, compiles
generic templates, checks required state, and returns a typed match for the
tool executor. No shortcut-specific logic is encoded in Python: any new
intent, phrase, pattern, or semantic cue can be added or removed by editing
the YAML alone.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SHORTCUTS_PATH = Path(__file__).resolve().parent / "conversation_shortcuts.yaml"

_CONTRACTIONS: Dict[str, str] = {
    "don't": "dont",
    "dont": "dont",
    "doesn't": "doesnt",
    "doesnt": "doesnt",
    "didn't": "didnt",
    "didnt": "didnt",
    "won't": "wont",
    "wont": "wont",
    "can't": "cant",
    "cant": "cant",
    "couldn't": "couldnt",
    "couldnt": "couldnt",
    "shouldn't": "shouldnt",
    "shouldnt": "shouldnt",
    "wouldn't": "wouldnt",
    "wouldnt": "wouldnt",
    "isn't": "isnt",
    "isnt": "isnt",
    "aren't": "arent",
    "arent": "arent",
    "i'm": "im",
    "im": "im",
    "i'll": "ill",
    "ill": "ill",
    "i've": "ive",
    "ive": "ive",
    "we're": "were",
    "we're": "were",
    "let's": "lets",
    "lets": "lets",
    "that's": "thats",
    "thats": "thats",
    "it's": "its",
    "its": "its",
}

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


class ShortcutMatch(BaseModel):
    intent: str
    action: str
    direction: Optional[str] = None
    selection_number: Optional[int] = None
    requires_state: List[str] = Field(default_factory=list)
    requires_any_state: List[str] = Field(default_factory=list)
    requires_context: Dict[str, Any] = Field(default_factory=dict)
    semantic_cues: Dict[str, List[List[str]]] = Field(default_factory=dict)


class ShortcutSpec(BaseModel):
    intent: str
    action: str
    direction: Optional[str] = None
    requires_state: List[str] = Field(default_factory=list)
    requires_any_state: List[str] = Field(default_factory=list)
    requires_context: Dict[str, Any] = Field(default_factory=dict)
    examples: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    entity_schema: Dict[str, Any] = Field(default_factory=dict)
    semantic_cues: Dict[str, List[List[str]]] = Field(default_factory=dict)


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
    Produce a lowercase, whitespace-collapsed, contraction-canonical form of the input string.

    Normalization rules (applied in order):
      1. Lowercase.
      2. Canonicalise common apostrophe forms (curly quotes, backticks) to `'`.
      3. Expand common English contractions so both "don't" and "dont" map
         to the same token ("don't want" vs "dont want").  This step must
         run *before* punctuation stripping, otherwise the apostrophe is
         gone before we can match the contraction.
      4. Replace every remaining punctuation character with a single space.
      5. Collapse runs of whitespace to a single space and trim.

    Parameters:
        text (str | None): Input text; `None` is treated as an empty string.

    Returns:
        str: The normalised, tokenisation-friendly form of `text`.
    """
    if not text:
        return ""
    lowered = str(text).lower()
    
    for src, dst in (
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u02bc", "'"),
        ("`", "'"),
    ):
        lowered = lowered.replace(src, "'")
   
    for src, dst in _CONTRACTIONS.items():
        lowered = re.sub(rf"\b{re.escape(src)}\b", dst, lowered)
    
    cleaned = _PUNCT_RE.sub(" ", lowered)
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    return _WS_RE.sub(" ", cleaned).strip()


def _tokenize(norm_text: str) -> List[str]:
    """
    Split a normalized text into a deterministic token list.

    Parameters:
        norm_text (str): Text produced by :func:`_normalize`.

    Returns:
        List[str]: Whitespace-separated tokens in their original order.
    """
    return [token for token in norm_text.split(" ") if token]


def _group_matches(
    group: Sequence[str],
    token_set: Iterable[str],
) -> bool:
    """
    Return True when every cue token in `group` is present in `token_set`.

    Empty groups never match (degenerate cues are ignored).

    Parameters:
        group (Sequence[str]): Tokens that must all be present.
        token_set (Iterable[str]): Token set derived from the message.

    Returns:
        bool: True iff `group` is non-empty and a subset of `token_set`.
    """
    if not group:
        return False
    tokens = set(token_set)
    return all(str(token) in tokens for token in group)


def _matches_semantic_cues(
    norm_text: str,
    cues: Optional[Dict[str, Any]],
) -> bool:
    """
    Generic semantic-cue matcher.

    The matcher itself is content-free: it knows nothing about property
    details, booking flows, or any other domain concept.  The YAML is the
    sole source of truth for which cues correspond to which shortcut.

    Matching contract (deterministic, generic):

      * `cues` empty / None  → no match.
      * No non-empty `any` groups → no match (a positive cue is required).
      * Otherwise, the cue matches when:
          - no `none` group has all tokens present in the normalised
            message, AND
          - at least one `any` group has all tokens present in the
            normalised message.

    Parameters:
        norm_text (str): Output of :func:`_normalize` for the user message.
        cues (Optional[Dict[str, Any]]): Cue spec from the YAML shortcut.

    Returns:
        bool: True when the normalized message satisfies the cue spec.
    """
    if not isinstance(cues, dict) or not cues:
        return False

    tokens = set(_tokenize(norm_text))

    any_groups = cues.get("any") or []
    any_groups = [
        group for group in any_groups
        if isinstance(group, (list, tuple)) and group
    ]
    if not any_groups:
        return False

    none_groups = cues.get("none") or []
    for group in none_groups:
        if not isinstance(group, (list, tuple)) or not group:
            continue
        if _group_matches(group, tokens):
            return False

    tokens = set(_tokenize(norm_text))

    none_groups = cues.get("none") or []
    for group in none_groups:
        if not isinstance(group, (list, tuple)):
            continue
        if _group_matches(group, tokens):
            return False

    any_groups = cues.get("any") or []
    if not any_groups:
        return True

    for group in any_groups:
        if not isinstance(group, (list, tuple)):
            continue
        if _group_matches(group, tokens):
            return True
    return False

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
        to a list of cleaned strings, and a normalised "semantic_cues" mapping.
    """
    data = dict(body or {}) if isinstance(body, dict) else {}
    for key in ("requires_state", "requires_any_state", "examples", "patterns"):
        data[key] = _as_str_list(data.get(key))
    requires_context = data.get("requires_context")
    data["requires_context"] = (
        dict(requires_context) if isinstance(requires_context, dict) else {}
    )
    data["semantic_cues"] = _normalise_semantic_cues(data.get("semantic_cues"))
    return data


def _normalise_semantic_cues(raw: Any) -> Dict[str, List[List[str]]]:
    """
    Normalise a YAML `semantic_cues` block into the canonical dict shape.

    The accepted YAML shape is::

        semantic_cues:
          any:  - [token, token, ...]   
          none: - [token, token, ...] 

    Each group is normalised through :func:`_normalize` so YAML authors can
    write phrases like ["don't", "want"] and have them match user messages
    like "I don't want this property" without re-typing the apostrophe.

    Parameters:
        raw (Any): Raw YAML value for the `semantic_cues` key.

    Returns:
        Dict[str, List[List[str]]]: Normalised cue mapping (always has the
        "any" and "none" keys, each an empty list when absent).
    """
    cues: Dict[str, List[List[str]]] = {"any": [], "none": []}
    if not isinstance(raw, dict):
        return cues
    for key in ("any", "none"):
        groups = raw.get(key)
        if not isinstance(groups, list):
            continue
        normalised_groups: List[List[str]] = []
        for group in groups:
            if not isinstance(group, (list, tuple)):
                text = str(group or "").strip()
                if not text:
                    continue
                tokens = [tok for tok in _normalize(text).split(" ") if tok]
                if tokens:
                    normalised_groups.append(tokens)
                continue
            tokens = []
            for token in group:
                token_text = str(token or "").strip()
                if not token_text:
                    continue
                
                norm = _normalize(token_text)
                if not norm:
                    continue

                tokens.extend(part for part in norm.split(" ") if part)
            if tokens:
                normalised_groups.append(tokens)
        cues[key] = normalised_groups
    return cues


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
        for key, expected in spec.requires_context.items():
            if soft_state.get(key) != expected:
                return False
        return True

    def match(
        self,
        message: str,
        soft_state: Optional[Dict[str, Any]],
    ) -> Optional[ShortcutMatch]:
        """
        Finds the first shortcut that matches a normalized message and current conversational soft state.

        Matching is performed in the following order, all gated by the spec's
        state/context requirements (so e.g. property-rejection cues only fire
        when `last_presented_view == "property_details"`):

          1. Compiled `patterns` (regex templates with `{number}` capture).
          2. Exact `examples` (full normalised equality).
          3. Generic `semantic_cues` (token-group matching, content-free in
             Python — driven entirely by the YAML).

        The first match wins, so order in the YAML file acts as priority.

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
                    requires_context=spec.requires_context,
                    semantic_cues=spec.semantic_cues,
                )

            for example in spec.examples:
                if norm == _normalize(example):
                    return ShortcutMatch(
                        intent=spec.intent,
                        action=spec.action,
                        direction=spec.direction,
                        requires_state=spec.requires_state,
                        requires_any_state=spec.requires_any_state,
                        requires_context=spec.requires_context,
                        semantic_cues=spec.semantic_cues,
                    )

            if spec.semantic_cues and _matches_semantic_cues(norm, spec.semantic_cues):
                return ShortcutMatch(
                    intent=spec.intent,
                    action=spec.action,
                    direction=spec.direction,
                    requires_state=spec.requires_state,
                    requires_any_state=spec.requires_any_state,
                    requires_context=spec.requires_context,
                    semantic_cues=spec.semantic_cues,
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
