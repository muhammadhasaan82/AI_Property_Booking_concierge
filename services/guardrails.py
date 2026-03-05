"""
Input / Output Guardrails
- Sanitize user input (prompt injection, script injection, excessive length)
- Sanitize bot output (strip leaked internals)

Patterns are loaded from config/guardrails.yaml at runtime.
"""

from __future__ import annotations

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

from services.dynamic_config import (
    get_compiled_injection_patterns,
    get_compiled_script_patterns,
    get_compiled_leak_patterns,
    get_guardrails,
)


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
    max_len = get_guardrails().max_input_length
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]

    # Check for script injection
    for pat in get_compiled_script_patterns():
        if pat.search(cleaned):
            logger.warning("Script injection blocked")
            return "", False

    # Check for prompt injection
    for pat in get_compiled_injection_patterns():
        if pat.search(cleaned):
            logger.warning("Prompt injection blocked")
            return "", False

    return cleaned, True


# ---------------------------------------------------------------------------
# Output Sanitization
# ---------------------------------------------------------------------------
def sanitize_output(reply: str) -> str:
    """Strip potentially leaked internal data from bot replies."""
    if not reply:
        return reply

    for pat in get_compiled_leak_patterns():
        if pat.search(reply):
            # Remove the offending line rather than the whole message
            lines = reply.split("\n")
            lines = [ln for ln in lines if not pat.search(ln)]
            reply = "\n".join(lines)

    return reply.strip()
