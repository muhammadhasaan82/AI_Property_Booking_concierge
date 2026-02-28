"""
Input / Output Guardrails
- Sanitize user input (prompt injection, script injection, excessive length)
- Sanitize bot output (strip leaked internals)
"""

from __future__ import annotations

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Prompt injection patterns (compiled once at import time)
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I),
    re.compile(r"(disregard|forget)\s+(everything|all|your)\s+(above|previous|instructions?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\b", re.I),
    re.compile(r"^(system|assistant)\s*:", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\b", re.I),
    re.compile(r"(reveal|show|print|output)\s+(your|the)\s+(system\s+)?prompt", re.I),
    re.compile(r"jailbreak|DAN\s+mode|developer\s+mode", re.I),
    re.compile(r"\bact\s+as\s+(if\s+you\s+are|a)\b", re.I),
]

_SCRIPT_PATTERNS = [
    re.compile(r"<\s*script\b", re.I),
    re.compile(r"javascript\s*:", re.I),
    re.compile(r"on(load|error|click|mouseover)\s*=", re.I),
]

MAX_INPUT_LENGTH = 2000


# ---------------------------------------------------------------------------
# Input Sanitization
# ---------------------------------------------------------------------------
def sanitize_input(text: str) -> Tuple[str, bool]:
    """Validate and clean user input.

    Returns:
        (cleaned_text, is_safe)
        - cleaned_text: the input with minor cleanup applied
        - is_safe: False only for clearly malicious input
    """
    if not text or not text.strip():
        return text, True

    cleaned = text.strip()

    # Truncate excessively long input
    if len(cleaned) > MAX_INPUT_LENGTH:
        cleaned = cleaned[:MAX_INPUT_LENGTH]

    # Check for script injection
    for pat in _SCRIPT_PATTERNS:
        if pat.search(cleaned):
            print(f"[GUARD] Script injection blocked")
            return "", False

    # Check for prompt injection
    for pat in _INJECTION_PATTERNS:
        if pat.search(cleaned):
            print(f"[GUARD] Prompt injection blocked")
            return "", False

    return cleaned, True


# ---------------------------------------------------------------------------
# Output Sanitization
# ---------------------------------------------------------------------------
_LEAK_PATTERNS = [
    re.compile(r"(system\s+prompt|internal\s+error|traceback|File\s+\")", re.I),
    re.compile(r"sk-[a-zA-Z0-9]{20,}", re.I),           # OpenAI API key
    re.compile(r"eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}", re.I),  # JWT tokens
]


def sanitize_output(reply: str) -> str:
    """Strip potentially leaked internal data from bot replies."""
    if not reply:
        return reply

    for pat in _LEAK_PATTERNS:
        if pat.search(reply):
            # Remove the offending line rather than the whole message
            lines = reply.split("\n")
            lines = [ln for ln in lines if not pat.search(ln)]
            reply = "\n".join(lines)

    return reply.strip()
