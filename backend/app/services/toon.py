# services/toon.py
# -*- coding: utf-8 -*-
"""
TOON (Token-Optimized Object Notation) serializer/deserializer.

TOON is a compact, LLM-friendly alternative to JSON that uses
indentation-based structure and header-based arrays.  This module
provides lossless round-trip conversion between Python dicts and
TOON text.

Format rules
─────────────
  key: value            # simple key/value on one line
  key:                  # nested object starts on next indented lines
    child_key: value
  key: []               # uniform array header
    - item1
    - item2

Edge-case encoding:
  • Strings containing colons   → prefixed with `"` and suffixed with `"`
  • Strings containing newlines → newlines replaced with `\\n` literal
  • null / bool / numbers       → inline as-is
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

# ────────────────────── Encoder ──────────────────────

_NEEDS_QUOTING = re.compile(r'[:\n\r]')


def _quote_value(v: str) -> str:
    """Wrap the string in double-quotes if it contains colons, newlines, or
    is empty / looks like a keyword (true/false/null) or a number."""
    if not v:
        return '""'
    if _NEEDS_QUOTING.search(v):
        escaped = v.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
        return f'"{escaped}"'
    # Avoid ambiguity with bool/null/number literals
    lower = v.strip().lower()
    if lower in ('true', 'false', 'null', 'none'):
        return f'"{v}"'
    # If it looks purely numeric, quote to preserve string type
    try:
        float(v)
        return f'"{v}"'
    except ValueError:
        pass
    return v


def _encode_value(val: Any, indent: int = 0) -> str:
    """Recursively encode a Python value to TOON text."""
    prefix = '  ' * indent

    if val is None:
        return 'null'
    if isinstance(val, bool):
        return 'true' if val else 'false'
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        # Finite check
        if val != val:  # NaN
            return 'null'
        return f'{val:g}'
    if isinstance(val, str):
        return _quote_value(val)

    if isinstance(val, list):
        if not val:
            return '[]'
        lines = ['[]']
        for item in val:
            encoded = _encode_value(item, indent + 1)
            if isinstance(item, dict):
                # Multi-line dict item: first line on `- ` prefix, rest indented
                dict_lines = _encode_dict(item, indent + 2)
                lines.append(f'{prefix}  -')
                lines.append(dict_lines)
            else:
                lines.append(f'{prefix}  - {encoded}')
        return '\n'.join(lines)

    if isinstance(val, dict):
        return _encode_dict(val, indent)

    # Fallback: treat as string
    return _quote_value(str(val))


def _encode_dict(d: Dict[str, Any], indent: int = 0) -> str:
    """Encode a dict into TOON lines at the given indent level."""
    prefix = '  ' * indent
    lines: list[str] = []

    for key, val in d.items():
        safe_key = key.replace(':', '\\:') if ':' in key else key

        if isinstance(val, dict):
            if not val:
                lines.append(f'{prefix}{safe_key}: {{}}')
            else:
                lines.append(f'{prefix}{safe_key}:')
                lines.append(_encode_dict(val, indent + 1))
        elif isinstance(val, list):
            encoded = _encode_value(val, indent)
            if '\n' in encoded:
                # Multi-line array
                first, *rest = encoded.split('\n', 1)
                lines.append(f'{prefix}{safe_key}: {first}')
                if rest:
                    lines.append(rest[0])
            else:
                lines.append(f'{prefix}{safe_key}: {encoded}')
        else:
            encoded = _encode_value(val, indent)
            lines.append(f'{prefix}{safe_key}: {encoded}')

    return '\n'.join(lines)


def toon_encode(obj: Any) -> str:
    """Convert a Python object (dict, list, or scalar) to TOON text."""
    if isinstance(obj, dict):
        return _encode_dict(obj, 0)
    return _encode_value(obj, 0)


# ────────────────────── Decoder ──────────────────────

class _ToonDecoder:
    """Line-by-line TOON parser with indent tracking."""

    def __init__(self, text: str):
        self.lines = text.split('\n')
        self.pos = 0

    def _current_indent(self, line: str) -> int:
        stripped = line.lstrip(' ')
        return (len(line) - len(stripped)) // 2

    def _skip_blank(self):
        while self.pos < len(self.lines) and not self.lines[self.pos].strip():
            self.pos += 1

    def parse(self) -> Any:
        self._skip_blank()
        if self.pos >= len(self.lines):
            return None
        line = self.lines[self.pos].strip()
        if line.startswith('- ') or line == '-':
            return self._parse_array_items(self._current_indent(self.lines[self.pos]))
        return self._parse_object(0)

    def _parse_value_str(self, raw: str) -> Any:
        """Parse a raw inline value string."""
        raw = raw.strip()
        if not raw:
            return ''
        if raw == 'null':
            return None
        if raw == 'true':
            return True
        if raw == 'false':
            return False
        if raw == '[]':
            return []
        if raw == '{}':
            return {}

        # Quoted string
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            inner = raw[1:-1]
            # Unescape
            inner = inner.replace('\\n', '\n').replace('\\r', '\r').replace('\\"', '"').replace('\\\\', '\\')
            return inner

        # Try number
        try:
            if '.' in raw or 'e' in raw.lower():
                return float(raw)
            return int(raw)
        except ValueError:
            pass

        # Bare string
        return raw

    def _parse_object(self, expected_indent: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if not line.strip():
                self.pos += 1
                continue
            indent = self._current_indent(line)
            if indent < expected_indent:
                break
            if indent > expected_indent:
                break
            stripped = line.strip()
            if stripped.startswith('- ') or stripped == '-':
                break

            # Parse key: value
            # Handle escaped colons in keys
            colon_idx = self._find_key_colon(stripped)
            if colon_idx == -1:
                self.pos += 1
                continue

            key = stripped[:colon_idx].replace('\\:', ':')
            rest = stripped[colon_idx + 1:].strip()
            self.pos += 1

            if not rest:
                # Nested object on next lines
                result[key] = self._parse_object(expected_indent + 1)
            elif rest == '[]':
                # Array header — items follow
                result[key] = self._parse_array_items(expected_indent + 1)
            elif rest.startswith('[]'):
                # Inline empty array indicator followed by nothing meaningful
                result[key] = self._parse_array_items(expected_indent + 1)
            else:
                result[key] = self._parse_value_str(rest)

        return result

    def _find_key_colon(self, line: str) -> int:
        """Find the first unescaped colon that separates key from value."""
        i = 0
        while i < len(line):
            if line[i] == '\\' and i + 1 < len(line) and line[i + 1] == ':':
                i += 2
                continue
            if line[i] == ':':
                return i
            i += 1
        return -1

    def _parse_array_items(self, expected_indent: int) -> List[Any]:
        items: List[Any] = []
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if not line.strip():
                self.pos += 1
                continue
            indent = self._current_indent(line)
            if indent < expected_indent:
                break
            stripped = line.strip()
            if not stripped.startswith('-'):
                break

            rest = stripped[1:].strip() if len(stripped) > 1 else ''
            self.pos += 1

            if not rest:
                # Multi-line dict item follows
                items.append(self._parse_object(expected_indent + 1))
            else:
                items.append(self._parse_value_str(rest))
        return items


def toon_decode(text: str) -> Any:
    """Parse TOON text back to a Python object."""
    if not text or not text.strip():
        return {}
    decoder = _ToonDecoder(text)
    return decoder.parse()
