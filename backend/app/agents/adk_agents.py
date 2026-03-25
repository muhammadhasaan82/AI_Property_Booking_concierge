# services/adk_agents.py
"""
ADK 2.0 — Nested Hybrid Graph (Phase 2: V2 Brain)

Dual-Model Architecture:
  Node 1 (triage_router)  → GPT-5 Nano via LiteLLM  (temperature=1, top_p=0.95, top_k=50)
  Node 2 (concierge_voice) → Llama-3.3-70B via Groq   (temperature=0.6)

The SequentialAgent pipeline: triage_router → concierge_voice.
The triage_router has access to 5 tools that bridge into our Rust gateway
and the isolated LangGraph checkout flow (the Deterministic Vault).
"""
from __future__ import annotations

import json
import logging
import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

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
# ---------------------------------------------------------------------------
DISPATCHER_CONFIG = genai_types.GenerateContentConfig(
    temperature=1,
)

VOICE_CONFIG = genai_types.GenerateContentConfig(
    temperature=0.6,
)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# ADK auto-wraps plain Python functions as FunctionTool.
# Docstrings become the tool description the LLM sees.
# ═══════════════════════════════════════════════════════════════════════════

def get_all_available_cities() -> str:
    """Use this tool ONLY when the user asks for a list of available cities or locations."""
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
        return f"We have properties in these {len(city_list)} cities: " + ", ".join(city_list)
    except Exception as e:
        return f"Could not retrieve cities. Error: {str(e)}"


async def search_properties(
    city: str,
    budget: Optional[float] = None,
    beds: Optional[int] = None,
    property_type: Optional[str] = None,
    amenities: Optional[str] = None,
) -> dict:
    """Search for rental properties in a specific city.

    Use this tool when the user wants to find, browse, or look for properties,
    apartments, houses, villas, or any accommodation.

    Args:
        city: The city or location to search in (required).
        budget: Maximum nightly price in USD (optional).
        beds: Minimum number of bedrooms (optional).
        property_type: Type of property — apartment, house, villa, condo, loft, studio, townhouse (optional).
        amenities: Comma-separated list of required amenities like wifi, pool, parking (optional).
    """
    from .tools.rust_client import search_properties as rust_search
    from ..components.search import property_search, _DATASET

    amenity_list = [a.strip() for a in (amenities or "").split(",") if a.strip()] or None

    # Try Rust gateway first, fall back to Python search
    results = None
    try:
        rust_result = await rust_search(
            location=city,
            budget=budget,
            beds=beds,
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
            budget=int(budget) if budget else None,
            amenities=amenity_list,
            location=city,
            beds=beds,
            property_type=property_type,
        )

    if not results:
        return {
            "status": "no_results",
            "message": f"No properties found in {city} matching your criteria.",
            "suggestion": "Try a different city, adjust budget, or broaden filters.",
        }

    # Return top 10 for the LLM to present
    top = results[:10]
    formatted = []
    for i, r in enumerate(top, 1):
        entry = {
            "number": i,
            "title": r.get("title", "Property"),
            "city": (r.get("city") or "").title(),
            "price_per_night": r.get("price_per_night"),
            "bedrooms": r.get("bedrooms"),
            "property_type": r.get("property_type", ""),
            "rating": r.get("rating"),
            "id": r.get("id"),
        }
        formatted.append(entry)

    return {
        "status": "success",
        "total_found": len(results),
        "showing": len(formatted),
        "properties": formatted,
        "instruction": "Present these as a numbered list. Ask the user to pick one by number.",
    }


async def check_faq(question: str) -> dict:
    """Look up a policy or FAQ question about the booking platform.

    Use this tool when the user asks about rules, policies, check-in times,
    cancellation, refunds, wifi, pets, smoking, parking, payment methods,
    security deposits, or any other platform/property question.

    Args:
        question: The user's policy or FAQ question.
    """
    from .tools.rust_client import execute_tool

    # Send to Rust gateway — the CAG layer will intercept known policies
    try:
        result = await execute_tool(
            data={"intent": "faq", "question": question},
        )
        if result and not result.get("fallback"):
            # CAG hit or database result
            answer = result.get("answer") or result.get("result", {}).get("answer")
            if answer:
                return {"status": "answered", "answer": answer, "source": "policy_database"}
    except Exception as e:
        logger.warning("Rust FAQ lookup failed: %s, using Python fallback", e)

    # Python fallback: enhanced FAQ with RAG
    try:
        from ..components.faq_enhanced import enhanced_faq_agent
        faq_result = enhanced_faq_agent(question, {})
        reply = faq_result.get("reply", "")
        if reply:
            return {"status": "answered", "answer": reply, "source": "rag_pipeline"}
    except Exception:
        pass

    # Basic fallback
    from ..services.faq import faq_lookup
    ans = faq_lookup(question)
    if ans:
        return {"status": "answered", "answer": ans, "source": "basic_faq"}

    return {
        "status": "not_found",
        "message": "I couldn't find a specific answer to that question.",
        "suggestion": "Would you like me to connect you with a human agent?",
    }


async def check_booking_status(booking_id: str) -> dict:
    """Check the status of an existing booking.

    Use this tool when the user asks about a booking status, wants to check
    their reservation, or provides a booking ID.

    Args:
        booking_id: The booking ID (UUID format).
    """
    from ..services.booking import get_booking_status
    from ..observability.db_logging import get_successful_booking_status

    try:
        r = await get_booking_status(booking_id)
        if r.get("ok"):
            return {
                "status": "found",
                "booking_id": booking_id,
                "booking_status": str(r.get("status", "unknown")).replace("_", " "),
                "check_in": r.get("check_in", "?"),
                "check_out": r.get("check_out", "?"),
            }
    except Exception:
        pass

    # Try successful_bookings table
    try:
        db_row = await get_successful_booking_status(str(booking_id))
        if db_row:
            return {
                "status": "found",
                "booking_id": booking_id,
                "booking_status": str(db_row.get("status", "confirmed")).replace("_", " "),
                "check_in": db_row.get("check_in", "?"),
                "check_out": db_row.get("check_out", "?"),
                "source": "successful_bookings",
            }
    except Exception:
        pass

    return {
        "status": "not_found",
        "message": f"Could not find booking {booking_id}.",
        "suggestion": "Please double-check the booking ID and try again.",
    }


async def trigger_checkout_flow(
    user_text: str,
    session_state_json: str = "{}",
) -> dict:
    """Start or continue the property booking checkout process.

    Use this tool when:
    - The user has selected a property and wants to book it.
    - The user is providing booking details (dates, name, email, phone, guests).
    - The user confirms or modifies a booking receipt.
    - The user says "yes" to confirm a booking.

    This tool manages the full checkout pipeline: collecting details,
    showing a receipt, and processing payment.

    Args:
        user_text: The user's latest message.
        session_state_json: JSON string of the current booking session state (filters, booking_args, etc.).
    """
    from ..services.checkout_graph import run_checkout_flow

    try:
        session_state = json.loads(session_state_json) if session_state_json else {}
    except (json.JSONDecodeError, TypeError):
        session_state = {}

    result = await run_checkout_flow(
        user_text=user_text,
        filters=session_state.get("filters", {}),
        booking_args=session_state.get("booking_args", {}),
        status_args=session_state.get("status_args", {}),
        payment_args=session_state.get("payment_args", {}),
    )

    return {
        "status": "checkout_response",
        "reply": result.get("reply", ""),
        "updated_state": {
            "filters": result.get("filters", {}),
            "booking_args": result.get("booking_args", {}),
            "status_args": result.get("status_args", {}),
            "payment_args": result.get("payment_args", {}),
        },
        "tool_result": result.get("tool_result", {}),
    }


async def escalate_to_human(reason: str) -> dict:
    """Transfer the conversation to a human support agent.

    Use this tool when:
    - The user explicitly asks to speak with a human or agent.
    - You cannot resolve the user's issue with the available tools.
    - The user seems frustrated and needs personal assistance.

    Args:
        reason: Brief description of why the handoff is needed.
    """
    return {
        "status": "handoff",
        "message": (
            f"I'll connect you with a human specialist right away. "
            f"Reason: {reason}\n\n"
            "Please share your email or phone number and a preferred contact time."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ADK AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

TRIAGE_INSTRUCTION = """\
You are a hotel booking concierge routing engine. Your ONLY job is to analyze
the user's message and call the correct tool. You do NOT generate conversational
responses. You ONLY output tool calls.

ROUTING RULES:
1. If the user asks about properties, cities, apartments, houses, or accommodation
   → call `search_properties`
2. If the user asks about policies, rules, check-in times, cancellation, refunds,
   wifi, pets, smoking, parking, payment methods, or any FAQ
   → call `check_faq`
3. If the user provides a booking ID or asks about booking status
   → call `check_booking_status`
4. If the user is in a booking flow (selecting property, providing dates/name/email/phone,
   confirming receipt, saying "yes" to book)
   → call `trigger_checkout_flow`
5. If the user asks to speak to a human or you cannot help
   → call `escalate_to_human`
6. If the user says hello or greets you, respond with a brief greeting and ask how
   you can help. This is the ONLY case where you generate text instead of a tool call.

NEVER generate long conversational responses. NEVER add pleasantries to tool calls.
ALWAYS prefer calling a tool over generating text.
"""

triage_router = LlmAgent(
    model=dispatcher_llm,
    name="triage_router",
    description="Routes user intent to the correct tool. Does not generate conversational text.",
    instruction=TRIAGE_INSTRUCTION,
    tools=[
        search_properties,
        check_faq,
        check_booking_status,
        trigger_checkout_flow,
        escalate_to_human,
        get_all_available_cities,
    ],
    output_key="router_output",
    generate_content_config=DISPATCHER_CONFIG,
)

VOICE_INSTRUCTION = """\
You are a warm, professional hotel booking concierge. Your job is to take the
raw tool output from the routing engine and transform it into a friendly,
human-like response for the user.

The routing engine's output is available as: {router_output}

RULES:
- If the output contains property search results, present them as a clean numbered
  list with title, city, price, and bedrooms. End with "Reply with the number of
  the property you'd like to book."
- If the output contains an FAQ answer, present it conversationally.
- If the output contains booking status, present it clearly with dates and status.
- If the output contains a checkout response, pass the reply through as-is (it
  comes from the deterministic booking system and must not be modified).
- If the output is a handoff, present the handoff message warmly.
- If the output is a greeting or simple text, respond naturally.
- Keep responses concise. Do not repeat raw JSON to the user.
- Do not invent information not present in the router output.
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
