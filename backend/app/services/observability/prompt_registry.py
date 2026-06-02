"""
Langfuse Prompt Registry

Provides optional fetching of prompts from Langfuse. Local prompts remain the
default source of truth. Never makes app boot depend on Langfuse availability.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from app.prompts import templates
from .langfuse_observer import get_observer, LANGFUSE_PROMPTS_ENABLED

logger = logging.getLogger(__name__)

_LOCAL_PROMPT_MAP = {
    "triage_router_instruction": templates.TRIAGE_ROUTER_INSTRUCTION,
    "concierge_voice_instruction": templates.CONCIERGE_VOICE_INSTRUCTION,
}


def get_local_prompt(name: str) -> str:
    """Retrieve a prompt from local storage (fallback)."""
    return _LOCAL_PROMPT_MAP.get(name, f"Local fallback prompt for: {name}")


def fetch_prompt_from_langfuse(name: str, version: Optional[str] = None, label: Optional[str] = None) -> Optional[Tuple[str, str, str, str]]:
    """
    Attempt to fetch a prompt from Langfuse.
    
    Returns:
        Tuple of (prompt_content, source, version, label) if successful, else None.
    """
    observer = get_observer()
    if not hasattr(observer, '_langfuse') or observer._langfuse is None:
        return None
        
    try:
        prompt = observer._langfuse.get_prompt(
            name=name,
            version=version,
            label=label,
            type="text"
        )
        if prompt and prompt.prompt:
            return str(prompt.prompt), "langfuse", str(prompt.version or "latest"), str(prompt.label or "production")
    except Exception as exc:
        logger.warning("[PromptRegistry] Failed to fetch prompt '%s' from Langfuse: %s", name, exc)
        
    return None


def get_prompt(
    name: str, 
    version: Optional[str] = None, 
    label: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Get a prompt, preferring Langfuse if enabled and available, otherwise falling back to local.
    
    Returns:
        Tuple of (prompt_content, metadata_dict)
    """
    metadata = {
        "prompt_name": name,
        "source": "local",
        "version": "local",
        "label": "local",
    }
    
    if LANGFUSE_PROMPTS_ENABLED:
        langfuse_result = fetch_prompt_from_langfuse(name, version, label)
        if langfuse_result:
            content, source, ver, lbl = langfuse_result
            metadata["source"] = source
            metadata["version"] = ver
            metadata["label"] = lbl
            logger.info("[PromptRegistry] Successfully loaded prompt '%s' from Langfuse (v%s, %s)", name, ver, lbl)
            return content, metadata
        else:
            logger.info("[PromptRegistry] Falling back to local prompt for '%s'", name)
            
    content = get_local_prompt(name)
    return content, metadata
