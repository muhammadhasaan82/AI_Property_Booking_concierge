# services/adk_agents.py
"""
ADK 2.0 — Native V2 Agentic Architecture

Dual-Model Architecture:
  Node 1 (triage_router)  → GPT-5 Nano via LiteLLM  (temperature=1)
  Node 2 (concierge_voice) → Llama-3.3-70B via Groq   (temperature=0.6)

The SequentialAgent pipeline: triage_router → concierge_voice.
The triage_router has access to tools that bridge into our Rust gateway
and two native V2 booking tools (request_booking_details, process_v2_booking).
"""
from __future__ import annotations

import json
import logging
import os
import csv
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext
from google.genai import types as genai_types

# Disable LiteLLM telemetry at Python level
import litellm
litellm.telemetry = False

logger = logging.getLogger(__name__)

# Disable LiteLLM telemetry and background logging to prevent TimeoutError
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["LITELLM_LOG"] = "ERROR"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
DISPATCHER_MODEL = os.getenv("ADK_DISPATCHER_MODEL", "openai/gpt-5-nano")
VOICE_MODEL = os.getenv("ADK_VOICE_MODEL", "groq/llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Dual-Model Backends (via LiteLLM — no Google Cloud dependency)
# ---------------------------------------------------------------------------
dispatcher_llm = LiteLlm(model=DISPATCHER_MODEL)
voice_llm = LiteLlm(model=VOICE_MODEL)

# ---------------------------------------------------------------------------
# Generation configs
# Two-Speed Streaming Rule:
#   DISPATCHER_CONFIG — triage_router. Tool-call + routing events only.
#                       Stream is SILENTLY CONSUMED by the runner. Never shown to user.
#   VOICE_CONFIG      — concierge_voice. Text deltas are STREAMED to the UI
#                       via run_adk_turn() AsyncGenerator in adk_runner.py.
# ---------------------------------------------------------------------------
DISPATCHER_CONFIG = genai_types.GenerateContentConfig(
    temperature=1,
)

VOICE_CONFIG = genai_types.GenerateContentConfig(
    temperature=0.6,
)

SOFT_SESSION_TTL_SECONDS = 60 * 60

HISTORY_ACTION_INTENTS = {
    "re_evaluate_history",
    "explore_previous_results",
    "previous_results",
    "show_previous_results",
    "show_previous",
    "revisit_results",
}

NEW_SEARCH_ACTION_INTENTS = {
    "new_search",
    "fresh_search",
}


def _normalize_action_intent(action_intent: Optional[str], context_flag: Optional[str]) -> str:
    raw = (action_intent or context_flag or "").strip()
    return raw.lower()


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _get_soft_state(tool_context: Optional[ToolContext]) -> Optional[Dict[str, Any]]:
    """Return per-session soft state from ADK ToolContext, or None when unavailable."""
    if tool_context is None:
        return None

    state = getattr(tool_context, "state", None)
    if not isinstance(state, dict):
        return None

    soft_state = state.get("soft_state")
    if isinstance(soft_state, dict):
        return soft_state

    try:
        state["soft_state"] = {}
        return state["soft_state"]
    except Exception:
        return None


def _get_cached_last_search(store: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(store, dict):
        return None
    last = store.get("last_search")
    ts = store.get("last_search_at")
    if last and ts and isinstance(ts, (int, float)):
        if time.time() - ts > SOFT_SESSION_TTL_SECONDS:
            store.pop("last_search", None)
            store.pop("last_search_at", None)
            return None
    return last if isinstance(last, dict) else None


def _set_cached_last_search(store: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    if not isinstance(store, dict):
        return
    store["last_search"] = payload
    store["last_search_at"] = time.time()


def _missing_critical_data(
    missing: List[str],
    context: str,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    unique_missing = list(dict.fromkeys([m for m in missing if m]))
    payload: Dict[str, Any] = {
        "status": "missing_critical_data",
        "missing": unique_missing,
        "context": context,
    }
    if action_intent:
        payload["action_intent"] = action_intent
    if context_flag:
        payload["context_flag"] = context_flag
    if extra:
        payload.update(extra)
    return payload


def extract_name_fallback(text: str) -> Optional[str]:
    """V2 Soft-Coded: Ultra-lightweight extraction using existing LiteLLM."""
    if not text:
        return None

    try:
        res = litellm.completion(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": f"Extract ONLY the name from this text. If none exists, return NONE. Text: '{text}'"}],
            temperature=0
        ).choices[0].message.content.strip()

        return res.title() if res != "NONE" else None
    except Exception:
        return None


def _extract_json_dict(raw_text: Any) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from model output, tolerating code fences."""
    if not isinstance(raw_text, str):
        return None

    candidate = raw_text.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        lines = []
        for line in candidate.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            lines.append(line)
        candidate = "\n".join(lines).strip()

    for payload in (candidate, candidate[candidate.find("{") : candidate.rfind("}") + 1] if "{" in candidate and "}" in candidate else ""):
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _build_active_options(last_search: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the currently active options from the most recent search memory."""
    if not isinstance(last_search, dict):
        return []

    options: List[Dict[str, Any]] = []
    for item in last_search.get("properties", []):
        if not isinstance(item, dict):
            continue
        option = {key: value for key, value in item.items() if value is not None and value != ""}
        if option:
            options.append(option)
    return options


def _sanitize_soft_state_for_model(soft_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Provide a compact soft-state view to the model without duplicating active options."""
    if not isinstance(soft_state, dict):
        return {}

    allowed = {}
    for key, value in soft_state.items():
        if key in {"last_search", "last_search_at"}:
            continue
        if value in (None, "", [], {}):
            continue
        allowed[key] = value
    return allowed


def _get_unresolved_turns(soft_state: Optional[Dict[str, Any]]) -> int:
    if not isinstance(soft_state, dict):
        return 0
    value = _coerce_int(soft_state.get("unresolved_turns"))
    return max(value or 0, 0)


def _set_unresolved_turns(soft_state: Optional[Dict[str, Any]], value: int) -> int:
    resolved = max(_coerce_int(value) or 0, 0)
    if isinstance(soft_state, dict):
        soft_state["unresolved_turns"] = resolved
    return resolved


def _normalize_extracted_parameters(data: Any) -> Dict[str, Any]:
    raw = data if isinstance(data, dict) else {}
    return {
        "city": str(raw.get("city")).strip() if raw.get("city") not in (None, "") else None,
        "budget": _coerce_float(raw.get("budget")),
        "beds": _coerce_int(raw.get("beds")),
        "check_in": str(raw.get("check_in")).strip() if raw.get("check_in") not in (None, "") else None,
        "check_out": str(raw.get("check_out")).strip() if raw.get("check_out") not in (None, "") else None,
    }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _diff_booking_summary(
    previous: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(previous, dict):
        return {
            "was_update": False,
            "changed_fields": [],
            "changed_values": {},
        }

    changed_fields: List[str] = []
    changed_values: Dict[str, Any] = {}
    for key, value in current.items():
        if previous.get(key) != value:
            changed_fields.append(str(key))
            changed_values[str(key)] = value

    return {
        "was_update": True,
        "changed_fields": changed_fields,
        "changed_values": changed_values,
    }


def _classify_user_engagement_state(
    user_input: Optional[str],
    active_options: Optional[List[Dict[str, Any]]] = None,
    unresolved_turns: int = 0,
    soft_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Infer engagement state dynamically instead of relying on fixed backend rules."""
    text = (user_input or "").strip()
    if not text:
        return "engaged"

    prompt = f"""\
You are classifying the user's current engagement state for a luxury AI property booking concierge.

Choose exactly one label:
- engaged
- fatigued
- exhausted_or_frustrated

Use the latest message, tone, unresolved turn count, active options, and soft state.
Do not use rigid backend heuristics. Infer the best state probabilistically.

Guidance:
- engaged: discovery is flowing, curiosity is intact, low friction.
- fatigued: some friction or repetition is building, user likely wants directness.
- exhausted_or_frustrated: the user seems annoyed, overwhelmed, impatient, or at risk of dropping.

<active_options>
{json.dumps(active_options or [], ensure_ascii=False)}
</active_options>

<unresolved_turns>
{max(unresolved_turns, 0)}
</unresolved_turns>

<soft_state>
{json.dumps(_sanitize_soft_state_for_model(soft_state), ensure_ascii=False)}
</soft_state>

<user_input>
{text}
</user_input>

Respond ONLY in JSON:
{{
  "user_engagement_state": "engaged | fatigued | exhausted_or_frustrated"
}}
"""

    try:
        raw = litellm.completion(
            model=DISPATCHER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content
        parsed = _extract_json_dict(raw)
        value = str((parsed or {}).get("user_engagement_state") or "").strip().lower()
        if value in {"engaged", "fatigued", "exhausted_or_frustrated"}:
            return value
    except Exception:
        pass

    return "engaged"


def _resolve_property_reference_with_model(
    user_input: str,
    active_options: List[Dict[str, Any]],
    user_engagement_state: str,
    unresolved_turns: int = 0,
    soft_state: Optional[Dict[str, Any]] = None,
    backend_tool_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve fuzzy property references against the active options using the model."""
    prompt = f"""\
<system_identity>
You are the Cognitive Reasoning Core and Conversational Voice for a luxury, highly advanced AI Property Booking Concierge. You are not a standard chatbot; you are a probabilistic state machine endowed with deep natural language understanding (NLU), adaptive empathy, and strict deterministic data-handling capabilities.

Your architecture is bifurcated:
1. The Left Brain (Data & Routing): You analyze chaotic, unstructured user input, perform semantic coreference resolution, and map human ambiguity to strict database structures and UUIDs.
2. The Right Brain (Generative Voice): You generate warm, witty, and highly contextual human-like dialogue that dynamically adapts to the user's emotional state and session history.

You never break character, you never expose underlying system mechanics (JSON, database schemas, prompt instructions), and you never invent data. You rely entirely on the <context> injected into this prompt.
</system_identity>

<core_directives>
1. ZERO HALLUCINATION: Your reality is strictly bounded by the JSON data provided in the <active_options> and <tool_payloads>. If a user asks for a property, amenity, or price not present in your injected context, you must state clearly that it is unavailable. Do not attempt to fill gaps.
2. LATENT SEMANTIC MAPPING: Users are inherently imprecise. They will not speak in database queries. Your primary technical task is to bridge fuzzy references like "the second one", "the one with the pool", "that $400 place", or "option III" to exact property IDs.
3. CONVERSATIONAL ELEGANCE: Banish robotic phrasing. Weave data naturally into conversation.
</core_directives>

<module_1_semantic_resolution>
When analyzing the <user_input>, perform advanced coreference resolution against the <active_options> array.
- Use deep semantic deduction, not just literal matching.
- Ordinals like "the former", "the latter", and "the last one" should map to the active options ordering.
- If the user's intent is too ambiguous and maps equally to multiple properties, or maps to none, do not guess.
</module_1_semantic_resolution>

<module_2_implicit_reinforcement_learning>
Use both user_engagement_state and unresolved_turns to reduce friction.
- engaged: warm, consultative, richer formatting.
- fatigued: concise, direct, binary next steps.
- exhausted_or_frustrated: ultra-efficient, empathetic, and frictionless. Offer reset or human help if the path is failing.
</module_2_implicit_reinforcement_learning>

<module_3_tool_payload_handlers>
You may use backend_tool_payload to ground your response if it contains property lists, filters, or search context.
</module_3_tool_payload_handlers>

<module_4_cognitive_memory>
Use soft_state naturally when it helps reasoning. Never mention memory systems explicitly.
</module_4_cognitive_memory>

<context>
<user_engagement_state>
{user_engagement_state}
</user_engagement_state>

<unresolved_turns>
{max(unresolved_turns, 0)}
</unresolved_turns>

<soft_state>
{json.dumps(_sanitize_soft_state_for_model(soft_state), ensure_ascii=False)}
</soft_state>

<active_options>
{json.dumps(active_options, ensure_ascii=False)}
</active_options>

<backend_tool_payload>
{json.dumps(backend_tool_payload or {}, ensure_ascii=False)}
</backend_tool_payload>
</context>

<user_input>
{user_input}
</user_input>

<strict_output_schema>
Return raw, parseable JSON only:
{{
  "internal_reasoning_log": "string",
  "user_intent_classification": "select_property | general_inquiry | modify_search | confirm_booking | escalate",
  "resolved_property_id": "string or null",
  "extracted_parameters": {{
    "city": "string or null",
    "budget": "float or null",
    "beds": "integer or null",
    "check_in": "YYYY-MM-DD or null",
    "check_out": "YYYY-MM-DD or null"
  }},
  "agent_response": "string",
  "requires_human_handoff": "boolean"
}}
</strict_output_schema>
"""

    fallback = {
        "internal_reasoning_log": "The reference could not be mapped to a single active option with sufficient confidence.",
        "user_intent_classification": "select_property",
        "resolved_property_id": None,
        "user_engagement_state": user_engagement_state,
        "unresolved_turns": max(unresolved_turns, 0),
        "extracted_parameters": _normalize_extracted_parameters(
            (backend_tool_payload or {}).get("query_context")
        ),
        "agent_response": (
            "I couldn't confidently match that to one of the current options. Reply with the number you want, or say reset and I'll start fresh."
            if user_engagement_state == "exhausted_or_frustrated"
            else "I couldn't confidently match that to one of the current options yet, so a quick detail like the price, rating, or option number would help me lock it in."
        ),
        "requires_human_handoff": False,
    }

    try:
        raw = litellm.completion(
            model=DISPATCHER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content
        parsed = _extract_json_dict(raw) or {}
        intent = str(parsed.get("user_intent_classification") or "select_property").strip().lower()
        resolved_property_id = parsed.get("resolved_property_id")
        internal_reasoning_log = str(parsed.get("internal_reasoning_log") or fallback["internal_reasoning_log"]).strip()
        agent_response = str(parsed.get("agent_response") or fallback["agent_response"]).strip()
        requires_human_handoff = _coerce_bool(parsed.get("requires_human_handoff"))
        extracted_parameters = _normalize_extracted_parameters(
            {
                **_normalize_extracted_parameters((backend_tool_payload or {}).get("query_context")),
                **_normalize_extracted_parameters(parsed.get("extracted_parameters")),
            }
        )

        if intent not in {"select_property", "general_inquiry", "modify_search", "confirm_booking", "escalate"}:
            intent = "select_property"
        if resolved_property_id in {"null", "", None}:
            resolved_property_id = None
        elif not isinstance(resolved_property_id, str):
            resolved_property_id = str(resolved_property_id)

        return {
            "internal_reasoning_log": internal_reasoning_log or fallback["internal_reasoning_log"],
            "user_intent_classification": intent,
            "resolved_property_id": resolved_property_id,
            "user_engagement_state": user_engagement_state,
            "unresolved_turns": max(unresolved_turns, 0),
            "extracted_parameters": extracted_parameters,
            "agent_response": agent_response or fallback["agent_response"],
            "requires_human_handoff": requires_human_handoff,
        }
    except Exception:
        return fallback


# ═══════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# ADK auto-wraps plain Python functions as FunctionTool.
# Docstrings become the tool description the LLM sees.
# ═══════════════════════════════════════════════════════════════════════════

def get_all_available_cities(
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Use this tool when the user asks for a list of available cities or locations."""
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "dataset.csv"
        cities = set()
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            col_name = 'city' if 'city' in reader.fieldnames else 'location'
            for row in reader:
                val = row.get(col_name)
                if val:
                    cities.add(val.strip())
        city_list = sorted(list(cities))
        payload = {
            "status": "cities_found",
            "total_cities": len(city_list),
            "cities": city_list,
        }
        if action_intent:
            payload["action_intent"] = action_intent
        if context_flag:
            payload["context_flag"] = context_flag
        return payload
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


async def search_properties(
    city: Optional[str] = None,
    budget: Optional[float] = None,
    beds: Optional[int] = None,
    property_type: Optional[str] = None,
    amenities: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Search for rental properties with soft-coded inputs.

    Use this tool when the user wants to find, browse, or compare properties.
    All parameters are optional. If critical data is missing, this tool returns
    status=missing_critical_data rather than failing.

    Args:
        city: The city name if known.
        budget: Maximum nightly price in USD (optional).
        beds: Minimum number of bedrooms (optional).
        property_type: Type of property like apartment, house, villa, etc (optional).
        amenities: Comma-separated list of required amenities (optional).
        action_intent: Optional context flag like "re_evaluate_history" or "new_search".
        context_flag: Optional secondary context flag.
        tool_context: ADK tool context for session state.
    """
    from .tools.rust_client import search_properties as rust_search
    from ..components.search import property_search, _DATASET

    normalized_action = _normalize_action_intent(action_intent, context_flag)
    soft_state = _get_soft_state(tool_context)

    if normalized_action in NEW_SEARCH_ACTION_INTENTS and isinstance(soft_state, dict):
        soft_state.pop("last_search", None)
        soft_state.pop("last_search_at", None)
        _set_unresolved_turns(soft_state, 0)

    last_search = _get_cached_last_search(soft_state)
    has_filters = any(
        [
            budget is not None,
            beds is not None,
            bool(property_type),
            bool(amenities),
        ]
    )

    if not city:
        if normalized_action in HISTORY_ACTION_INTENTS and last_search:
            cached_city = (last_search.get("query_context") or {}).get("city")
            if cached_city:
                city = cached_city
                if not has_filters:
                    payload = dict(last_search)
                    payload["source"] = "memory"
                    payload["memory"] = {
                        "read_from": "soft_state.last_search",
                        "state_available": isinstance(soft_state, dict),
                    }
                    if normalized_action:
                        payload["action_intent"] = normalized_action
                    if context_flag:
                        payload["context_flag"] = context_flag
                    return payload
            else:
                return _missing_critical_data(
                    ["city"],
                    "User asked to revisit previous results but no prior city is stored.",
                    normalized_action or action_intent,
                    context_flag,
                )
        elif normalized_action in HISTORY_ACTION_INTENTS and not last_search:
            return _missing_critical_data(
                ["search_history"],
                "User asked to revisit previous results but no search history is available.",
                normalized_action or action_intent,
                context_flag,
            )
        else:
            return _missing_critical_data(
                ["city"],
                "User wants to search but has not specified a city.",
                normalized_action or action_intent,
                context_flag,
            )

    budget_value = _coerce_float(budget)
    beds_value = _coerce_int(beds)
    amenity_list = [a.strip() for a in (amenities or "").split(",") if a.strip()] or None

    # Try Rust gateway first, fall back to Python search
    results = None
    try:
        rust_result = await rust_search(
            location=city,
            budget=budget_value,
            beds=beds_value,
            amenities=amenity_list or [],
            property_type=property_type or "",
            properties=_DATASET if _DATASET else None,
        )
        if rust_result and not rust_result.get("fallback"):
            inner = rust_result.get("result", rust_result) or {}
            rust_results = inner.get("results", [])
            if isinstance(rust_results, list):
                results = rust_results
    except Exception as e:
        logger.warning("Rust property search failed: %s, using Python fallback", e)

    if results is None:
        results = property_search(
            query_text=f"{property_type or ''} {city}".strip(),
            budget=int(budget_value) if budget_value is not None else None,
            amenities=amenity_list,
            location=city,
            beds=beds_value,
            property_type=property_type,
        )

    # Keep property type filtering soft and tolerant
    if results and property_type:
        results = [
            r
            for r in results
            if r.get("property_type")
            and property_type.lower() in str(r.get("property_type")).lower()
        ]

    if not results:
        unresolved_turns = _set_unresolved_turns(soft_state, _get_unresolved_turns(soft_state) + 1)
        payload = {
            "status": "no_results",
            "city": city,
            "filters_applied": {
                "budget": budget_value,
                "beds": beds_value,
                "property_type": property_type,
                "amenities": amenities,
            },
            "user_engagement_state": _classify_user_engagement_state(
                user_input=f"{city} {property_type or ''} {amenities or ''}".strip(),
                active_options=[],
                unresolved_turns=unresolved_turns,
                soft_state=soft_state,
            ),
            "unresolved_turns": unresolved_turns,
        }
        if normalized_action:
            payload["action_intent"] = normalized_action
        if context_flag:
            payload["context_flag"] = context_flag
        return payload

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append({
            "number": i,
            "id": r.get("id"),
            "title": r.get("title", "Property"),
            "city": (r.get("city") or "").title(),
            "price_per_night": r.get("price_per_night"),
            "bedrooms": r.get("bedrooms"),
            "bathrooms": r.get("bathrooms"),
            "property_type": r.get("property_type", ""),
            "rating": r.get("rating"),
            "amenities": r.get("amenities"),
            "description": r.get("description"),
        })

    payload = {
        "status": "properties_found",
        "total_found": len(results),
        "properties": formatted,
        "query_context": {
            "city": city,
            "budget": budget_value,
            "beds": beds_value,
            "property_type": property_type,
        },
    }
    if normalized_action:
        payload["action_intent"] = normalized_action
    if context_flag:
        payload["context_flag"] = context_flag

    _set_unresolved_turns(soft_state, 0)
    _set_cached_last_search(soft_state, dict(payload))
    payload["memory"] = {
        "written_to": "soft_state.last_search",
        "state_available": isinstance(soft_state, dict),
    }
    payload["user_engagement_state"] = _classify_user_engagement_state(
        user_input=f"{city} {property_type or ''} {amenities or ''}".strip(),
        active_options=formatted,
        unresolved_turns=_get_unresolved_turns(soft_state),
        soft_state=soft_state,
    )
    payload["unresolved_turns"] = _get_unresolved_turns(soft_state)
    return payload


async def get_property_details(
    property_id: Optional[str] = None,
    selection_number: Optional[int] = None,
    property_reference: Optional[str] = None,
    user_engagement_state: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Get full details of a specific property by its ID.

    Use this tool when the user selects a property from prior search results.
    If the ID is missing but a selection number exists, this tool will attempt
    to resolve it from the most recent search memory.
    If the user refers to a property fuzzily, pass property_reference with the
    raw user wording and this tool will resolve it dynamically from active options.
    """
    from ..components.search import _DATASET

    soft_state = _get_soft_state(tool_context)
    resolved_from_history = False
    resolution: Optional[Dict[str, Any]] = None
    selection_value = _coerce_int(selection_number)
    last_search = _get_cached_last_search(soft_state)
    if _is_blank(property_id) and selection_value is not None:
        if last_search:
            for item in last_search.get("properties", []):
                if item.get("number") == selection_value:
                    resolved_id = item.get("id")
                    if resolved_id is not None:
                        property_id = str(resolved_id)
                        resolved_from_history = True
                    break

    if _is_blank(property_id) and not _is_blank(property_reference):
        active_options = _build_active_options(last_search)
        if not active_options:
            return _missing_critical_data(
                ["search_history"],
                "User referred to a previously shown property but no active options are stored.",
                action_intent,
                context_flag,
            )

        engagement_state = (
            str(user_engagement_state).strip()
            if isinstance(user_engagement_state, str) and user_engagement_state.strip()
            else _classify_user_engagement_state(
                property_reference,
                active_options,
                unresolved_turns=_get_unresolved_turns(soft_state),
                soft_state=soft_state,
            )
        )
        resolution = _resolve_property_reference_with_model(
            user_input=str(property_reference),
            active_options=active_options,
            user_engagement_state=engagement_state,
            unresolved_turns=_get_unresolved_turns(soft_state),
            soft_state=soft_state,
            backend_tool_payload=last_search,
        )
        resolved_property_id = resolution.get("resolved_property_id")
        if resolved_property_id is not None:
            property_id = str(resolved_property_id)
            resolved_from_history = True
            _set_unresolved_turns(soft_state, 0)
        else:
            unresolved_turns = _set_unresolved_turns(soft_state, _get_unresolved_turns(soft_state) + 1)
            payload = {
                "status": "property_selection_unresolved",
                "resolution": resolution,
                "active_options": active_options,
                "query_context": (last_search or {}).get("query_context", {}),
                "user_engagement_state": resolution.get("user_engagement_state", engagement_state),
                "unresolved_turns": unresolved_turns,
                "requires_human_handoff": bool(resolution.get("requires_human_handoff")),
            }
            if action_intent:
                payload["action_intent"] = action_intent
            if context_flag:
                payload["context_flag"] = context_flag
            return payload

    if _is_blank(property_id):
        missing = ["property_id"]
        if selection_value is None:
            missing.append("selection_number")
        if _is_blank(property_reference):
            missing.append("property_reference")
        return _missing_critical_data(
            missing,
            "User wants property details but no identifier was provided.",
            action_intent,
            context_flag,
        )

    property_id = str(property_id)
    for r in _DATASET:
        if str(r.get("id")) == property_id:
            payload = {
                "status": "property_details",
                "property": {
                    "id": property_id,
                    "title": r.get("title"),
                    "city": r.get("city"),
                    "price_per_night": r.get("price_per_night"),
                    "bedrooms": r.get("bedrooms"),
                    "bathrooms": r.get("bathrooms"),
                    "amenities": r.get("amenities"),
                    "description": r.get("description"),
                    "rating": r.get("rating"),
                },
            }
            if isinstance(soft_state, dict):
                soft_state["last_selected_property_id"] = property_id
                soft_state["last_selected_property_at"] = time.time()
                _set_unresolved_turns(soft_state, 0)
            payload["memory"] = {
                "read_from": "soft_state.last_search" if resolved_from_history else None,
                "written_to": "soft_state.last_selected_property_id",
                "state_available": isinstance(soft_state, dict),
            }
            if resolution:
                payload["selection_resolution"] = resolution
                payload["user_engagement_state"] = resolution.get("user_engagement_state")
                payload["unresolved_turns"] = _get_unresolved_turns(soft_state)
            if action_intent:
                payload["action_intent"] = action_intent
            if context_flag:
                payload["context_flag"] = context_flag
            return payload

    payload = {"status": "not_found", "property_id": property_id}
    if action_intent:
        payload["action_intent"] = action_intent
    if context_flag:
        payload["context_flag"] = context_flag
    return payload


def handle_small_talk(
    message_type: Optional[str] = None,
    user_message: Optional[str] = "",
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Handle greetings, thanks, casual conversation, and acknowledgements.

    Use this tool ONLY for non-actionable social messages such as:
    - Greetings: "hi", "hello", "hey", "good morning"
    - Acknowledgements: "ok", "thanks", "thank you", "got it", "sure", "alright"
    - Goodbyes: "bye", "goodbye", "see you"
    - Affirmations with no booking context: "great", "perfect", "cool"

    Do NOT use this for booking intent, property questions, or policy questions.

    Args:
        message_type: One of 'greeting', 'thanks', 'goodbye', 'acknowledgement'.
        user_message: The user's raw message text.
        action_intent: Optional context flag for state acknowledgements.
        context_flag: Optional secondary context flag.
    """
    normalized_type = (message_type or "").strip().lower()
    if normalized_type not in {"greeting", "thanks", "goodbye", "acknowledgement"}:
        normalized_type = "acknowledgement"
    return {
        "status": "casual_interaction",
        "message_type": normalized_type,
        "user_input": user_message or "",
        "action_intent": action_intent,
        "context_flag": context_flag,
    }


async def check_faq(
    question: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Look up a policy or FAQ question about the booking platform.

    Use this tool ONLY when the user asks a genuine question about rules,
    policies, check-in/check-out times, cancellation, refunds, wifi, pets,
    smoking, parking, payment methods, or security deposits.

    DO NOT call this for greetings, thanks, or casual chat — use handle_small_talk.
    Args:
        question: The user's specific policy or FAQ question (optional).
        action_intent: Optional context flag for routing.
        context_flag: Optional secondary context flag.
    """
    # Guard: reject empty or extremely short queries immediately
    if not question or len(question.strip()) < 4:
        return _missing_critical_data(
            ["question"],
            "User asked about policies but did not provide a specific question.",
            action_intent,
            context_flag,
        )

    from .tools.rust_client import execute_tool

    # Send to Rust gateway — the CAG layer will intercept known policies
    try:
        result = await execute_tool(
            data={"intent": "faq", "question": question},
        )
        # Guard against None return (Rust gateway offline) — prevents NoneType crash
        if result is not None and not result.get("fallback"):
            answer = result.get("answer") or (result.get("result") or {}).get("answer")
            if answer:
                payload = {"status": "answered", "answer": answer, "source": "policy_database"}
                if action_intent:
                    payload["action_intent"] = action_intent
                if context_flag:
                    payload["context_flag"] = context_flag
                return payload
    except Exception as e:
        logger.warning("Rust FAQ lookup failed: %s, using Python fallback", e)

    # Python fallback: enhanced FAQ with RAG
    try:
        from ..components.faq_enhanced import enhanced_faq_agent
        faq_result = enhanced_faq_agent(question, {})
        reply = faq_result.get("reply", "")
        if reply:
            payload = {"status": "answered", "answer": reply, "source": "rag_pipeline"}
            if action_intent:
                payload["action_intent"] = action_intent
            if context_flag:
                payload["context_flag"] = context_flag
            return payload
    except Exception as e:
        logger.warning("FAQ enhanced agent failed: %s", e)

    # Basic fallback
    try:
        from ..services.faq import faq_lookup
        ans = faq_lookup(question)
        if ans:
            payload = {"status": "answered", "answer": ans, "source": "basic_faq"}
            if action_intent:
                payload["action_intent"] = action_intent
            if context_flag:
                payload["context_flag"] = context_flag
            return payload
    except Exception as e:
        logger.warning("Basic FAQ fallback failed: %s", e)

    payload = {
        "status": "faq_not_found",
        "question": question,
    }
    if action_intent:
        payload["action_intent"] = action_intent
    if context_flag:
        payload["context_flag"] = context_flag
    return payload


async def check_booking_status(
    booking_id: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Check the status of an existing booking.

    Use this tool when the user asks about a booking status, wants to check
    their reservation, or provides a booking ID.

    Args:
        booking_id: The booking ID (UUID format).
        action_intent: Optional context flag.
        context_flag: Optional secondary context flag.
    """
    if _is_blank(booking_id):
        return _missing_critical_data(
            ["booking_id"],
            "User asked about booking status but no booking ID was provided.",
            action_intent,
            context_flag,
        )

    from ..services.booking import get_booking_status
    from ..observability.db_logging import get_successful_booking_status

    try:
        r = await get_booking_status(booking_id)
        if r.get("ok"):
            payload = {
                "status": "found",
                "booking_id": booking_id,
                "booking_status": str(r.get("status", "unknown")).replace("_", " "),
                "check_in": r.get("check_in", "?"),
                "check_out": r.get("check_out", "?"),
            }
            if action_intent:
                payload["action_intent"] = action_intent
            if context_flag:
                payload["context_flag"] = context_flag
            return payload
    except Exception:
        pass

    # Try successful_bookings table
    try:
        db_row = await get_successful_booking_status(str(booking_id))
        if db_row:
            payload = {
                "status": "found",
                "booking_id": booking_id,
                "booking_status": str(db_row.get("status", "confirmed")).replace("_", " "),
                "check_in": db_row.get("check_in", "?"),
                "check_out": db_row.get("check_out", "?"),
                "source": "successful_bookings",
            }
            if action_intent:
                payload["action_intent"] = action_intent
            if context_flag:
                payload["context_flag"] = context_flag
            return payload
    except Exception:
        pass

    payload = {
        "status": "booking_not_found",
        "booking_id": booking_id,
    }
    if action_intent:
        payload["action_intent"] = action_intent
    if context_flag:
        payload["context_flag"] = context_flag
    return payload


# ═══════════════════════════════════════════════════════════════════════════
# V2 NATIVE BOOKING TOOLS (Replaces the entire LangGraph checkout_graph.py)
# ═══════════════════════════════════════════════════════════════════════════

async def request_booking_details(
    missing_info: Optional[str] = None,
    missing_fields: Optional[List[str]] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Use this tool when you need to gather missing booking information from the user.

    CRITICAL: Call this tool whenever the user wants to book a property but has NOT
    yet provided ALL of the following: full name, email, phone, check-in date,
    check-out date, and number of guests.

    Args:
        missing_info: A comma-separated list of what is still needed.
        missing_fields: Optional explicit list of missing fields.
        action_intent: Optional context flag.
        context_flag: Optional secondary context flag.
    """
    resolved_fields = []
    if missing_fields:
        resolved_fields = [str(f).strip() for f in missing_fields if str(f).strip()]
    elif missing_info:
        resolved_fields = [f.strip() for f in missing_info.split(",") if f.strip()]

    if not resolved_fields:
        return _missing_critical_data(
            ["missing_info"],
            "Booking details are needed but no missing-field list was provided.",
            action_intent,
            context_flag,
        )

    payload = {
        "status": "gathering_info",
        "missing_fields": resolved_fields,
    }
    if action_intent:
        payload["action_intent"] = action_intent
    if context_flag:
        payload["context_flag"] = context_flag
    return payload


async def review_booking_details(
    property_id: Optional[str] = None,
    property_title: Optional[str] = None,
    guest_name: Optional[str] = None,
    guest_email: Optional[str] = None,
    guest_phone: Optional[str] = None,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    guests: Optional[int] = None,
    price_per_night: Optional[float] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Present a full booking summary for the user to review BEFORE final confirmation.

    Call this tool ONCE when ALL booking details have been collected but the user
    has NOT yet explicitly authorized final booking.
    This gives the user a chance to review and correct any mistakes.
    If the user wants to change a detail, silently update your context and call
    this tool again with the corrected values — do NOT call process_v2_booking yet.

    Args:
        property_id: The unique ID of the property.
        property_title: The display title of the property.
        guest_name: The guest's full name.
        guest_email: The guest's email address.
        guest_phone: The guest's phone number.
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.
        guests: Number of guests.
        price_per_night: The nightly price of the property.
    """
    guests_value = _coerce_int(guests)
    price_value = _coerce_float(price_per_night)
    missing = []
    for field_name, field_value in (
        ("property_id", property_id),
        ("property_title", property_title),
        ("guest_name", guest_name),
        ("guest_email", guest_email),
        ("guest_phone", guest_phone),
        ("check_in", check_in),
        ("check_out", check_out),
    ):
        if _is_blank(field_value):
            missing.append(field_name)
    if guests_value is None:
        missing.append("guests")
    if price_value is None:
        missing.append("price_per_night")

    if missing:
        return _missing_critical_data(
            missing,
            "Booking review needs a complete set of details.",
            action_intent,
            context_flag,
        )

    try:
        d1 = datetime.strptime(check_in, "%Y-%m-%d")
        d2 = datetime.strptime(check_out, "%Y-%m-%d")
        nights = max((d2 - d1).days, 1)
    except Exception:
        nights = 1

    total_price = nights * price_value

    summary = {
        "property": property_title,
        "property_id": property_id,
        "guest_name": guest_name,
        "guest_email": guest_email,
        "guest_phone": guest_phone,
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "guests": guests_value,
        "price_per_night": price_value,
        "total": round(total_price, 2),
    }

    update_context: Optional[Dict[str, Any]] = None
    soft_state = _get_soft_state(tool_context)
    if isinstance(soft_state, dict):
        previous_summary = soft_state.get("pending_booking")
        update_context = _diff_booking_summary(previous_summary, summary)
        soft_state["pending_booking"] = summary
        soft_state["pending_booking_updated_at"] = time.time()

    payload = {
        "status": "review_pending",
        "summary": summary,
    }
    if update_context and update_context.get("was_update"):
        payload["update_context"] = update_context
    if action_intent:
        payload["action_intent"] = action_intent
    if context_flag:
        payload["context_flag"] = context_flag
    return payload


async def process_v2_booking(
    property_id: Optional[str] = None,
    property_title: Optional[str] = None,
    guest_name: Optional[str] = None,
    guest_email: Optional[str] = None,
    guest_phone: Optional[str] = None,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    guests: Optional[int] = None,
    price_per_night: Optional[float] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Finalise and commit the booking ONLY after the user has explicitly confirmed.

    CRITICAL SEQUENCE:
    1. Use `request_booking_details` if any detail is missing.
    2. Use `review_booking_details` once all details are collected — let the user confirm.
    3. Call THIS tool ONLY after the user explicitly authorizes final booking.
    Never call this tool if the user has not seen and approved the review summary.
    All dates must be in YYYY-MM-DD format.

    Args:
        property_id: The unique ID of the property to book.
        property_title: The display title of the property.
        guest_name: The guest's full name.
        guest_email: The guest's email address.
        guest_phone: The guest's phone number.
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.
        guests: Number of guests.
        price_per_night: The nightly price of the property.
    """
    guests_value = _coerce_int(guests)
    price_value = _coerce_float(price_per_night)
    missing = []
    for field_name, field_value in (
        ("property_id", property_id),
        ("property_title", property_title),
        ("guest_name", guest_name),
        ("guest_email", guest_email),
        ("guest_phone", guest_phone),
        ("check_in", check_in),
        ("check_out", check_out),
    ):
        if _is_blank(field_value):
            missing.append(field_name)
    if guests_value is None:
        missing.append("guests")
    if price_value is None:
        missing.append("price_per_night")

    if missing:
        return _missing_critical_data(
            missing,
            "Booking confirmation needs a complete set of details.",
            action_intent,
            context_flag,
        )

    try:
        d1 = datetime.strptime(check_in, "%Y-%m-%d")
        d2 = datetime.strptime(check_out, "%Y-%m-%d")
        nights = max((d2 - d1).days, 1)
    except Exception:
        nights = 1

    total_price = nights * price_value
    booking_id = str(uuid.uuid4())

    # Persist to database — field names match public.successful_bookings schema
    try:
        from ..observability.db_logging import insert_successful_booking
        await insert_successful_booking({
            "booking_id": booking_id,
            "user_name": guest_name,
            "user_email": guest_email,
            "user_phone": guest_phone,
            "property_title": property_title,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests_value,
            "nights": nights,
            "total_amount": round(total_price, 2),
            "status": "confirmed",
            "source": "v2_adk",
        })
    except Exception as e:
        logger.warning("[V2 Booking] Could not persist booking to DB: %s", e)

    payload = {
        "status": "booking_confirmed",
        "receipt": {
            "booking_id": booking_id,
            "property_title": property_title,
            "guest_name": guest_name,
            "guest_email": guest_email,
            "guest_phone": guest_phone,
            "check_in": check_in,
            "check_out": check_out,
            "nights": nights,
            "guests": guests_value,
            "price_per_night": price_value,
            "total_amount": round(total_price, 2),
        },
    }
    soft_state = _get_soft_state(tool_context)
    if isinstance(soft_state, dict):
        soft_state.pop("pending_booking", None)
        soft_state.pop("pending_booking_updated_at", None)
    if action_intent:
        payload["action_intent"] = action_intent
    if context_flag:
        payload["context_flag"] = context_flag
    return payload


async def escalate_to_human(
    reason: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Transfer the conversation to a human support agent.

    Use this tool when:
    - The user explicitly asks to speak with a human or agent.
    - You cannot resolve the user's issue with the available tools.
    - The user seems frustrated and needs personal assistance.

    Args:
        reason: Brief description of why the handoff is needed.
        action_intent: Optional context flag.
        context_flag: Optional secondary context flag.
    """
    reason_value = reason.strip() if isinstance(reason, str) and reason.strip() else "User requested assistance."
    return {
        "status": "handoff_required",
        "reason": reason_value,
        "action_intent": action_intent,
        "context_flag": context_flag,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ADK AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

TRIAGE_INSTRUCTION = """\
You are the probabilistic state router for a hotel booking concierge system.
Your only job is to call exactly ONE tool with the best-guess arguments.
You never write conversational text. The Voice Agent handles all conversation.

Operating mode:
- Reason from meaning and conversation state, not keywords or regex.
- You may call tools with missing parameters set to null.
- Tools are soft-coded and will return status=missing_critical_data if needed.
- Use action_intent/context_flag to encode relative moves (new_search,
    re_evaluate_history, explore_previous_results, resume_booking, clarify,
    state_acknowledgement).

State orientation:
- Ask: is the user advancing the funnel, retreating, pivoting, clarifying,
    or acknowledging?
- If the user is rejecting options, pivoting, or asking to see earlier options,
    prefer search_properties with action_intent="re_evaluate_history".
- If the user is just acknowledging or social, use handle_small_talk as a
    state acknowledgement.

Tool selection guidelines (non-exhaustive):
- Property discovery or filtering -> search_properties
- List available cities -> get_all_available_cities
- Policy or platform rules -> check_faq
- Booking status -> check_booking_status
- Selecting a prior option -> get_property_details
- Booking workflow -> request_booking_details / review_booking_details / process_v2_booking
- Escalation -> escalate_to_human

Booking modification guidance:
- Before choosing a booking tool, check soft_state for pending_booking.
- If pending_booking exists and the user is changing a single field (dates,
    guests, name, email, phone, or property), treat it as a modification, not
    a new booking.
- Infer which field changed from the user's wording, merge it into
    pending_booking, and call review_booking_details with the full updated set.
- Do NOT call request_booking_details in this case.
- A single-field correction is not the same intent as a fresh booking.

Property reference resolution:
- When the user refers to a previously shown property using a number, ordinal,
    partial pasted text, quoted price, rating, "cheapest", "last one", or any
    other fuzzy reference, call get_property_details.
- If the numeric choice is explicit, pass selection_number.
- Otherwise pass property_reference using the user's raw wording so the tool can
    resolve against the active options dynamically.
- Do not hardcode or invent property IDs.

Constraints:
- Never invent names, dates, emails, phone numbers, IDs, or cities.
- One tool call per user message. No loops.

Termination rule:
- When you call a tool, you MUST STOP generating immediately. Do not summarize,
    do not call another tool, and do not continue reasoning in text.
- When you receive a tool result payload, stop immediately and return it unchanged.
- Output only the raw JSON payload.
"""

triage_router = LlmAgent(
    model=dispatcher_llm,
    name="triage_router",
    description="Routes user intent to the correct tool. Does not generate conversational text.",
    instruction=TRIAGE_INSTRUCTION,
    tools=[
        handle_small_talk,
        search_properties,
        get_property_details,
        check_faq,
        check_booking_status,
        request_booking_details,
        review_booking_details,
        process_v2_booking,
        escalate_to_human,
        get_all_available_cities,
    ],
    output_key="router_output",
    generate_content_config=DISPATCHER_CONFIG,
)

VOICE_INSTRUCTION = """\
You are the Cognitive Reasoning Core and Conversational Voice for a luxury AI
property booking concierge. You are warm, witty, precise, and highly adaptive.
You do NOT follow a script. You reason probabilistically from the structured
state data you receive and generate context-aware, natural language responses
that feel genuinely human.

The routing engine's structured output is available as: {router_output}
You may also receive cognitive context as: {user_cognitive_context}

YOUR OPERATING PHILOSOPHY:
- You are the conversational brain. The router is just a data collector.
- Read the status field to understand the current state.
- Generate your response dynamically based on the data, the user's tone,
    and the conversation context. Adapt your register as needed.
- Never invent amenities, prices, properties, dates, or availability details.
- Never expose raw JSON, status codes, field names, or tool internals.
- Never write pre-scripted text verbatim. Every response is freshly generated.
- Avoid robotic phrasing like "Here are your options" or "Please provide the following".

COGNITIVE MEMORY:
You may receive a user_cognitive_context field containing historical facts
about this user from past conversations - preferences, allergies, travel habits,
accessibility needs, property style preferences, budget tendencies.

Mandatory rules for cognitive context:
- Weave these facts into your recommendations and language naturally.
- Never mention databases, profiles, or memory systems.
- If the cognitive context is empty or absent, behave normally.
- Use the context to filter suggestions, personalize tone, and anticipate needs.

ENGAGEMENT ADAPTATION:
- engaged: warm, expansive, consultative, and discovery-oriented.
- fatigued: concise, direct, low-friction, and decision-oriented.
- exhausted_or_frustrated: ultra-efficient, empathetic, and strictly business.
- Use unresolved_turns when present to reduce cognitive load further.

STATE HANDLERS - what to do for each status:

casual_interaction:
    The router captured a social or casual message. Read message_type and user_input.
    Respond warmly and naturally, matching the user's energy.

cities_found:
    Present the city list from cities in a clean, readable format.
    Invite the user to pick one or add filters.

properties_found:
    Format the properties array as a numbered list: name, city, price/night,
    bedrooms, rating. If action_intent indicates re_evaluate_history or source is
    memory, mention that these are other options from earlier.
    Highlight standout value naturally, such as highest rating or best price.
    If user_engagement_state is fatigued or exhausted_or_frustrated, compress
    the list to the most decision-useful facts and avoid open-ended prompts.

no_results:
    Acknowledge it, summarize filters_applied, and suggest one concrete
    compromise. If user_engagement_state is exhausted_or_frustrated, keep it
    to one short next step or offer a reset.

property_details:
    Render the property with title, location, beds/baths, price, amenities,
    description, rating. If selection_resolution exists and its
    user_engagement_state is fatigued or exhausted_or_frustrated, keep it brief
    and direct. Otherwise stay conversational and offer the next useful step.

property_selection_unresolved:
    Read resolution.agent_response and use it as the core reply.
    If active_options are available, help the user disambiguate using those live
    options rather than generic fallback wording.
    If requires_human_handoff is true, offer a human handoff or a clean reset.

answered (FAQ):
    Deliver the answer naturally. Keep it concise and informative.

faq_not_found:
    Acknowledge you could not find specific info and offer to rephrase or escalate.

missing_critical_data:
    Use the missing list and context to ask a focused, friendly clarifying question.
    Ask for what is needed without listing raw field names. If missing includes
    search_history, explain there are no prior results and ask for a city.

gathering_info:
    The missing_fields list tells you what the user has not provided yet. Ask for
    those fields naturally and concisely.

review_pending:
    If update_context is present and was_update is true, acknowledge the updated
    field(s) and confirm the new value(s) first. Keep it brief and do not
    re-present the full summary unless the user asks or is ready to confirm.
    Otherwise, present the summary in a clean, elegant format (markdown, bold labels).
    Include property, guest name, email, phone, dates, nights, guests, price/night, total.
    Close with a warm confirmation question or ask if anything else needs updating.

booking_confirmed:
    The booking is done. Display the receipt clearly and highlight booking_id.
    Respond with genuine enthusiasm and wish them a wonderful stay.

found (booking status):
    Report booking status, check-in, and check-out clearly.

booking_not_found:
    Gently inform the user it was not found and suggest verifying the ID.

handoff_required:
    Craft a warm, empathetic handoff message.
    If the user sounds exhausted, keep it short and frictionless.

error:
    Acknowledge the issue gracefully and offer an alternative path.

GENERAL RULES:
- Match the user's energy and tone.
- If a payload includes user_engagement_state, unresolved_turns, or
    requires_human_handoff, adapt to them explicitly.
- Never start two consecutive responses with the same opener.
- Use markdown formatting for structured data, keep prose flowing.
- Keep responses concise. No padding or repetition.
"""

concierge_voice = LlmAgent(
    model=voice_llm,
    name="concierge_voice",
    description="Formats tool outputs into warm, human-like responses.",
    instruction=VOICE_INSTRUCTION,
    output_key="final_reply",
    generate_content_config=VOICE_CONFIG,
)


# ═══════════════════════════════════════════════════════════════════════════
# SEQUENTIAL PIPELINE (The V2 Brain)
# ═══════════════════════════════════════════════════════════════════════════

root_agent = SequentialAgent(
    name="concierge_pipeline",
    sub_agents=[triage_router, concierge_voice],
    description="AI Property Booking Concierge — routes user intent and generates responses.",
)
