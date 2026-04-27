"""
app/agents/resolvers/property_resolver.py
------------------------------------------
Resolves fuzzy property references ("the second one", "the $400 place", etc.)
against the active options list using the DISPATCHER_MODEL.

The prompt lives in app/prompts/resolution_prompt.md.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import litellm

from ..status_codes import ENGAGEMENT_STATES, INTENT_CLASSES
from ..tools.helpers import (
    _coerce_bool,
    _extract_json_dict,
    _normalize_extracted_parameters,
    _sanitize_soft_state_for_model,
)
from app.config.agent_config_loader import cfg
from app.agents.prompts.loader import load_prompt

logger = logging.getLogger(__name__)

_RESOLUTION_PROMPT_TEMPLATE: str = load_prompt("resolution_prompt.md")

_FALLBACK_AGENT_RESPONSE_DEFAULT: str = cfg.msg_resolution_default
_FALLBACK_AGENT_RESPONSE_FRUSTRATED: str = cfg.msg_resolution_frustrated


def _build_resolution_prompt(
    user_input: str,
    active_options: List[Dict[str, Any]],
    user_engagement_state: str,
    unresolved_turns: int,
    soft_state: Optional[Dict[str, Any]],
    backend_tool_payload: Optional[Dict[str, Any]],
) -> str:
    """Render the resolution prompt template with runtime context."""
    return _RESOLUTION_PROMPT_TEMPLATE.format(
        user_engagement_state=user_engagement_state,
        unresolved_turns=max(unresolved_turns, 0),
        soft_state=json.dumps(_sanitize_soft_state_for_model(soft_state), ensure_ascii=False),
        active_options=json.dumps(active_options, ensure_ascii=False),
        backend_tool_payload=json.dumps(backend_tool_payload or {}, ensure_ascii=False),
        user_input=user_input,
    )


def _parse_resolution_response(
    raw: Any,
    fallback: Dict[str, Any],
    backend_tool_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Parse and validate the model's resolution JSON response."""
    parsed = _extract_json_dict(raw) or {}

    intent = str(parsed.get("user_intent_classification") or cfg.resolution_fallback_intent).strip().lower()
    if intent not in cfg.resolution_valid_intents:
        intent = cfg.resolution_fallback_intent

    resolved_property_id = parsed.get("resolved_property_id")
    if resolved_property_id in {"null", "", None}:
        resolved_property_id = None
    elif not isinstance(resolved_property_id, str):
        resolved_property_id = str(resolved_property_id)

    internal_reasoning_log = str(
        parsed.get("internal_reasoning_log") or fallback["internal_reasoning_log"]
    ).strip()
    agent_response = str(parsed.get("agent_response") or fallback["agent_response"]).strip()
    requires_human_handoff = _coerce_bool(parsed.get("requires_human_handoff"))

    extracted_parameters = _normalize_extracted_parameters(
        {
            **_normalize_extracted_parameters((backend_tool_payload or {}).get("query_context")),
            **_normalize_extracted_parameters(parsed.get("extracted_parameters")),
        }
    )

    return {
        "internal_reasoning_log": internal_reasoning_log or fallback["internal_reasoning_log"],
        "user_intent_classification": intent,
        "resolved_property_id": resolved_property_id,
        "user_engagement_state": fallback["user_engagement_state"],
        "unresolved_turns": fallback["unresolved_turns"],
        "extracted_parameters": extracted_parameters,
        "agent_response": agent_response or fallback["agent_response"],
        "requires_human_handoff": requires_human_handoff,
    }


def resolve_property_reference(
    user_input: str,
    active_options: List[Dict[str, Any]],
    user_engagement_state: str,
    dispatcher_model: str,
    unresolved_turns: int = 0,
    soft_state: Optional[Dict[str, Any]] = None,
    backend_tool_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve a fuzzy property reference against the active options using the dispatcher LLM.

    Returns a structured resolution dict with:
        - resolved_property_id (str | None)
        - user_intent_classification
        - agent_response (fallback reply if unresolved)
        - extracted_parameters
        - requires_human_handoff
    """
    fallback_response = (
        _FALLBACK_AGENT_RESPONSE_FRUSTRATED
        if user_engagement_state == "exhausted_or_frustrated"
        else _FALLBACK_AGENT_RESPONSE_DEFAULT
    )

    fallback: Dict[str, Any] = {
        "internal_reasoning_log": (
            cfg.msg_resolution_not_matched_log
        ),
        "user_intent_classification": "select_property",
        "resolved_property_id": None,
        "user_engagement_state": user_engagement_state,
        "unresolved_turns": max(unresolved_turns, 0),
        "extracted_parameters": _normalize_extracted_parameters(
            (backend_tool_payload or {}).get("query_context")
        ),
        "agent_response": fallback_response,
        "requires_human_handoff": False,
    }

    try:
        prompt = _build_resolution_prompt(
            user_input=user_input,
            active_options=active_options,
            user_engagement_state=user_engagement_state,
            unresolved_turns=unresolved_turns,
            soft_state=soft_state,
            backend_tool_payload=backend_tool_payload,
        )
        raw = litellm.completion(
            model=dispatcher_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content

        return _parse_resolution_response(raw, fallback, backend_tool_payload)
    except Exception as exc:
        logger.warning("Property resolution LLM call failed: %s", exc)
        return fallback
