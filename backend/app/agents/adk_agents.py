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
    """Return per-session soft state from ADK ToolContext, or None when unavailable.

    Liquid State guarantee: this function NEVER raises. It returns None when
    the context is absent, the state attribute is missing, or the state cannot
    be mutated (e.g. read-only proxy during a local test run).
    The caller must always treat a None return as the ephemeral-only path.
    """
    if tool_context is None:
        logger.debug("[SoftState] tool_context is None — running in ephemeral-only mode")
        return None

    state = getattr(tool_context, "state", None)
    if not isinstance(state, dict):
        logger.debug("[SoftState] tool_context.state is not a dict — ephemeral-only mode")
        return None

    soft_state = state.get("soft_state")
    if isinstance(soft_state, dict):
        return soft_state

    try:
        state["soft_state"] = {}
        return state["soft_state"]
    except Exception as exc:
        logger.debug("[SoftState] Could not initialise soft_state bucket: %s — ephemeral-only mode", exc)
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
        if normalized_action in HISTORY_ACTION_INTENTS:
            # Determine WHY history is unavailable so the LLM can be precise.
            if soft_state is None:
                # Redis/state layer is completely unavailable — ephemeral mode.
                return _missing_critical_data(
                    ["search_history"],
                    "No previous search history found in memory.",
                    normalized_action or action_intent,
                    context_flag,
                    extra={"memory_status": "ephemeral_only_redis_unavailable"},
                )
            elif last_search:
                cached_city = (last_search.get("query_context") or {}).get("city")
                if cached_city:
                    city = cached_city
                    if not has_filters:
                        payload = dict(last_search)
                        payload["source"] = "memory"
                        payload["memory"] = {
                            "read_from": "soft_state.last_search",
                            "state_available": True,
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
            else:
                # State is alive but no prior search was cached.
                return _missing_critical_data(
                    ["search_history"],
                    "User asked to revisit previous results but no search history is available in this session.",
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
        payload = {
            "status": "no_results",
            "city": city,
            "filters_applied": {
                "budget": budget_value,
                "beds": beds_value,
                "property_type": property_type,
                "amenities": amenities,
            },
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
            "property_type": r.get("property_type", ""),
            "rating": r.get("rating"),
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

    # Graceful Ephemeral Fallback: write to state if available, otherwise flag it.
    # The LLM sees memory_status and knows it cannot rely on cross-turn history.
    if isinstance(soft_state, dict):
        _set_cached_last_search(soft_state, dict(payload))
        payload["memory"] = {
            "written_to": "soft_state.last_search",
            "state_available": True,
        }
    else:
        logger.warning(
            "[SoftState] Session state unavailable during search_properties — "
            "result NOT cached. Next-turn re-evaluation will not work."
        )
        payload["memory_status"] = "ephemeral_only_redis_unavailable"
        payload["warning"] = (
            "session_state_unavailable_memory_is_ephemeral"
        )
        payload["memory"] = {
            "written_to": None,
            "state_available": False,
        }
    return payload


async def get_property_details(
    property_id: Optional[str] = None,
    selection_number: Optional[int] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Get full details of a specific property by its ID.

    Use this tool when the user selects a property from prior search results.
    If the ID is missing but a selection number exists, this tool will attempt
    to resolve it from the most recent search memory.
    """
    from ..components.search import _DATASET

    soft_state = _get_soft_state(tool_context)
    resolved_from_history = False
    state_was_ephemeral = soft_state is None  # Track before any mutations
    selection_value = _coerce_int(selection_number)

    if _is_blank(property_id) and selection_value is not None:
        if soft_state is None:
            # Graceful Ephemeral Fallback: state layer is down.
            # Cannot resolve selection number from history — tell the LLM explicitly.
            logger.warning(
                "[SoftState] Session state unavailable in get_property_details — "
                "cannot resolve selection_number=%s from history.",
                selection_value,
            )
            return _missing_critical_data(
                ["property_id"],
                "No previous search history found in memory.",
                action_intent,
                context_flag,
                extra={"memory_status": "ephemeral_only_redis_unavailable",
                       "warning": "session_state_unavailable_memory_is_ephemeral"},
            )
        last_search = _get_cached_last_search(soft_state)
        if last_search:
            for item in last_search.get("properties", []):
                if item.get("number") == selection_value:
                    resolved_id = item.get("id")
                    if resolved_id is not None:
                        property_id = str(resolved_id)
                        resolved_from_history = True
                    break

    if _is_blank(property_id):
        missing = ["property_id"]
        if selection_value is None:
            missing.append("selection_number")
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
            # Graceful Ephemeral Fallback: write selection to state if possible,
            # otherwise flag it so the LLM knows history is transient.
            if isinstance(soft_state, dict):
                soft_state["last_selected_property_id"] = property_id
                soft_state["last_selected_property_at"] = time.time()
                payload["memory"] = {
                    "read_from": "soft_state.last_search" if resolved_from_history else None,
                    "written_to": "soft_state.last_selected_property_id",
                    "state_available": True,
                }
            else:
                logger.warning(
                    "[SoftState] Session state unavailable in get_property_details — "
                    "property selection NOT cached. Booking flow will require explicit ID."
                )
                payload["memory_status"] = "ephemeral_only_redis_unavailable"
                payload["warning"] = "session_state_unavailable_memory_is_ephemeral"
                payload["memory"] = {
                    "read_from": None,
                    "written_to": None,
                    "state_available": False,
                }
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

    payload = {
        "status": "review_pending",
        "summary": {
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
        },
    }
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
- Selecting a prior option -> get_property_details (use selection_number when possible)
- Booking workflow -> request_booking_details / review_booking_details / process_v2_booking
- Escalation -> escalate_to_human

Constraints:
- Never invent names, dates, emails, phone numbers, IDs, or cities.
- One tool call per user message. No loops.

Termination rule:
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
You are a dynamic, generative AI hotel booking concierge - warm, witty, and
professionally charming. You do NOT follow a script. You reason probabilistically
from the structured state data you receive and generate context-aware, natural
language responses that feel genuinely human.

The routing engine's structured output is available as: {router_output}
You may also receive cognitive context as: {user_cognitive_context}

YOUR OPERATING PHILOSOPHY:
- You are the conversational brain. The router is just a data collector.
- Read the status field to understand the current state.
- Generate your response dynamically based on the data, the user's tone,
    and the conversation context. Adapt your register as needed.
- Never expose raw JSON, status codes, field names, or tool internals.
- Never write pre-scripted text verbatim. Every response is freshly generated.

COGNITIVE MEMORY:
You may receive a user_cognitive_context field containing historical facts
about this user from past conversations - preferences, allergies, travel habits,
accessibility needs, property style preferences, budget tendencies.

Mandatory rules for cognitive context:
- Weave these facts into your recommendations and language naturally.
- Never mention databases, profiles, or memory systems.
- If the cognitive context is empty or absent, behave normally.
- Use the context to filter suggestions, personalize tone, and anticipate needs.

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
    memory, mention that these are other options from earlier. Close with a
    gentle prompt to pick one.

no_results:
    Acknowledge it, summarize filters_applied, and suggest broadening criteria.

property_details:
    Render the property with title, location, beds/baths, price, amenities,
    description, rating. Close by asking if they want to proceed with booking.

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
    Present the summary in a clean, elegant format (markdown, bold labels).
    Include property, guest name, email, phone, dates, nights, guests, price/night, total.
    Close with a warm confirmation question.

booking_confirmed:
    The booking is done. Display the receipt clearly and highlight booking_id.
    Respond with genuine enthusiasm and wish them a wonderful stay.

found (booking status):
    Report booking status, check-in, and check-out clearly.

booking_not_found:
    Gently inform the user it was not found and suggest verifying the ID.

handoff_required:
    Craft a warm, empathetic handoff message and ask for preferred contact method
    and a convenient time.

error:
    Acknowledge the issue gracefully and offer an alternative path.

MEMORY STATUS (ephemeral_only_redis_unavailable):
    If the payload contains memory_status="ephemeral_only_redis_unavailable",
    this is a soft infrastructure signal. You may handle it as follows:
    - If the user tried to revisit their last search and history was unavailable,
      apologise briefly and ask them to repeat their search criteria.
    - If the current action succeeded (e.g., a search just ran) but memory_status
      is present, you may proceed normally. Only mention it if relevant — for example,
      "Just to note, I won't be able to recall this list in our next message, so
      feel free to refer back to it."
    - NEVER use technical language like "Redis", "state layer", or "ephemeral".
    - Keep it light and human. The session is not broken — only cross-turn recall
      is limited for this message.

GENERAL RULES:
- Match the user's energy and tone.
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
