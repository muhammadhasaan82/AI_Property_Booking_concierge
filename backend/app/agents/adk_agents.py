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

def get_all_available_cities() -> dict:
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
        
        return {
            "status": "success",
            "total_cities": len(city_list),
            "cities_list": ", ".join(city_list),
            "instruction": "Stop calling tools. Pass this exact list of cities to the voice agent to present to the user."
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Could not retrieve cities. Error: {str(e)}"
        }


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
        city: The exact city name (required). CRITICAL: Preserve multi-word cities like "New York" completely. Do not treat "new" as an adjective.
        budget: Maximum nightly price in USD (optional).
        beds: Minimum number of bedrooms (optional).
        property_type: Type of property — apartment, house, villa, condo, loft, studio, townhouse. CRITICAL: You MUST extract this if mentioned by the user. Do not leave blank if the user specifies a type.
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

    # ---> STRICT PROPERTY TYPE FILTER <---
    if results and property_type:
        results = [
            r for r in results
            if r.get("property_type") and property_type.lower() in str(r.get("property_type")).lower()
        ]
    # --------------------------------------

    if not results:
        return {
            "status": "no_results",
            "message": f"No properties found in {city} matching your criteria.",
            "suggestion": "Try a different city, adjust budget, or broaden filters.",
        }

    # V2: Uncapped results. Show exactly what the database found.
    top = results
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
        "instruction": "Present these as a numbered list. Ask the user to pick one by number. Remember the property details (id, title, price_per_night) for when they want to book.",
    }


async def get_property_details(property_id: str) -> dict:
    """Get full details of a specific property by its ID.

    Call this when the user selects a property from search results (for example, 'option 7').
    """
    from ..components.search import _DATASET

    for r in _DATASET:
        if str(r.get("id")) == property_id:
            return {
                "status": "property_details",
                "property": {
                    "title": r.get("title"),
                    "city": r.get("city"),
                    "price_per_night": r.get("price_per_night"),
                    "bedrooms": r.get("bedrooms"),
                    "bathrooms": r.get("bathrooms"),
                    "amenities": r.get("amenities"),
                    "description": r.get("description"),
                    "rating": r.get("rating")
                },
                "instruction": "Present the property details beautifully using markdown (Title, City, Beds/Baths, Price, Amenities, Description). CRITICAL: Do NOT write the words 'Property Card' at the top. Just show the details naturally. End by asking: 'Would you like to proceed with booking this property?'"
            }

    return {"status": "error", "message": "Property not found."}


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


# ═══════════════════════════════════════════════════════════════════════════
# V2 NATIVE BOOKING TOOLS (Replaces the entire LangGraph checkout_graph.py)
# ═══════════════════════════════════════════════════════════════════════════

async def request_booking_details(missing_info: str) -> dict:
    """Use this tool when you need to gather missing booking information from the user.

    CRITICAL: Call this tool whenever the user wants to book a property but has NOT
    yet provided ALL of the following: full name, email, phone, check-in date,
    check-out date, and number of guests.

    Args:
        missing_info: A comma-separated list of what is still needed (e.g., "full name, email, phone, check-in date, check-out date, number of guests").
    """
    return {
        "status": "gathering_info",
        "instruction": f"Stop calling tools. Ask the user to provide the following missing information to continue their booking: {missing_info}"
    }


async def review_booking_details(
    property_id: str,
    property_title: str,
    guest_name: str,
    guest_email: str,
    guest_phone: str,
    check_in: str,
    check_out: str,
    guests: int,
    price_per_night: float,
) -> dict:
    """Present a full booking summary for the user to review BEFORE final confirmation.

    Call this tool ONCE when ALL booking details have been collected but the user
    has NOT yet explicitly said 'yes', 'confirm', 'go ahead', or similar.
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
    try:
        d1 = datetime.strptime(check_in, "%Y-%m-%d")
        d2 = datetime.strptime(check_out, "%Y-%m-%d")
        nights = max((d2 - d1).days, 1)
    except Exception:
        nights = 1

    total_price = nights * price_per_night

    return {
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
            "guests": guests,
            "price_per_night": f"${price_per_night:.2f}",
            "total": f"${total_price:.2f}",
        },
        "instruction": "Stop calling tools. Present this booking summary beautifully and ask: 'Everything looks great! Shall I go ahead and confirm this booking, or would you like to change anything?'"
    }


async def process_v2_booking(
    property_id: str,
    property_title: str,
    guest_name: str,
    guest_email: str,
    guest_phone: str,
    check_in: str,
    check_out: str,
    guests: int,
    price_per_night: float,
) -> dict:
    """Finalise and commit the booking ONLY after the user has explicitly confirmed.

    CRITICAL SEQUENCE:
    1. Use `request_booking_details` if any detail is missing.
    2. Use `review_booking_details` once all details are collected — let the user confirm.
    3. Call THIS tool ONLY after the user explicitly says 'yes', 'confirm', 'go ahead', or similar.
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
    try:
        d1 = datetime.strptime(check_in, "%Y-%m-%d")
        d2 = datetime.strptime(check_out, "%Y-%m-%d")
        nights = max((d2 - d1).days, 1)
    except Exception:
        nights = 1

    total_price = nights * price_per_night
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
            "guests": guests,
            "nights": nights,
            "total_amount": round(total_price, 2),
            "status": "confirmed",
            "source": "v2_adk",
        })
    except Exception as e:
        logger.warning("[V2 Booking] Could not persist booking to DB: %s", e)

    return {
        "status": "booking_confirmed",
        "receipt": {
            "booking_id": booking_id,
            "property": property_title,
            "guest": guest_name,
            "email": guest_email,
            "phone": guest_phone,
            "dates": f"{check_in} to {check_out}",
            "nights": nights,
            "guests": guests,
            "price_per_night": f"${price_per_night:.2f}",
            "total": f"${total_price:.2f}",
        },
        "instruction": "Stop calling tools. Congratulate the user and display the full receipt beautifully."
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
the user's message and call the correct tool.

ROUTING RULES:
1. Property search / browsing
   → call `search_properties` or `get_all_available_cities`
2. Policies, FAQ, rules, cancellation, refunds, amenities
   → call `check_faq`
3. User gives a booking ID or asks about an existing booking
   → call `check_booking_status`
4. User selects a numbered property from search results (e.g. "option 3", "the second one")
   → call `get_property_details` with that property's ID. Do NOT request booking details yet.
5. User asks to book — strict 3-step sequence:
   STEP A — Gather: If ANY of (name, email, phone, check-in, check-out, guests) are missing
            → call `request_booking_details` listing exactly what is still needed.
   STEP B — Review: Once ALL details are present but the user has NOT yet confirmed
            → call `review_booking_details` with all collected values.
            If the user wants to correct a detail (e.g. "change the dates"),
            silently update your context and call `review_booking_details` again
            with the corrected values. Do NOT call `process_v2_booking` yet.
   STEP C — Confirm: ONLY after the user explicitly replies "yes", "confirm",
            "go ahead", "book it", or any clear affirmation after seeing the review
            → call `process_v2_booking` with the final agreed values.
6. User asks for a human or cannot be helped
   → call `escalate_to_human`
7. Greeting only
   → respond with a brief greeting via the voice agent.

ANTI-HALLUCINATION RULES (strictly enforced):
- NEVER invent, guess, or assume dates, guest count, name, email, or phone.
- NEVER call `process_v2_booking` without the user having seen and verbally confirmed
  the review summary. The review step is mandatory — there are NO shortcuts.
- Accumulate details provided across multiple messages. Do NOT re-ask for details
  the user has already given.
- If the user tries to correct a mistake during review, update and re-show the review
  (call `review_booking_details` again) — never escalate to human for corrections.

CRITICAL: Once a tool returns a result, output it as plain text for the voice agent.
Do NOT call the same tool twice in a row. NEVER generate conversational text yourself.
"""

triage_router = LlmAgent(
    model=dispatcher_llm,
    name="triage_router",
    description="Routes user intent to the correct tool. Does not generate conversational text.",
    instruction=TRIAGE_INSTRUCTION,
    tools=[
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
You are a warm, professional hotel booking concierge. Your job is to take the
raw tool output from the routing engine and transform it into a friendly,
human-like response for the user.

The routing engine's output is available as: {router_output}

RULES:
- search results (status: success): Present as a clean numbered list — title, city,
  price/night, bedrooms. End with: "Which one catches your eye?"

- property details (status: property_details): Format beautifully with markdown.
  DO NOT write the words "Property Card". Show title, city, beds/baths, price,
  amenities, description. End with: "Would you like to book this property?"

- gathering info (status: gathering_info): Sound like a warm concierge.
  Open with "Great choice! I just need a few details to secure this for you:"
  or "Almost there! Could you share...". Ask naturally. Never mention YYYY-MM-DD
  format — just say 'check-in date' or 'arrival date'.

- review pending (status: review_pending): Present the booking summary as a
  clean, elegant card. Use this structure:
    ✨ **Booking Review**
    🏠 **Property:** <property>
    👤 **Guest:** <guest_name>
    📧 **Email:** <guest_email>
    📞 **Phone:** <guest_phone>
    📅 **Check-in:** <check_in>  →  **Check-out:** <check_out>
    🌙 **Nights:** <nights>  |  👥 **Guests:** <guests>
    💰 **Price/night:** <price_per_night>  |  **Total:** <total>
  Then ask warmly: "Everything looks perfect! Shall I go ahead and confirm this
  booking, or would you like to adjust anything?"

- booking confirmed (status: booking_confirmed): Congratulate enthusiastically!
  Show the receipt as a clean summary with all fields. End with the booking ID
  prominently so they can reference it later.

- handoff (status: handoff): Deliver the message warmly.

- Keep all responses warm, concise, and human. Never expose raw JSON or tool names.
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
