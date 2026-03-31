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
            "status": "cities_found",
            "total_cities": len(city_list),
            "cities": city_list,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
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
            "city": city,
            "filters_applied": {
                "budget": budget,
                "beds": beds,
                "property_type": property_type,
                "amenities": amenities,
            },
        }

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

    return {
        "status": "properties_found",
        "total_found": len(results),
        "properties": formatted,
        "query_context": {
            "city": city,
            "budget": budget,
            "beds": beds,
            "property_type": property_type,
        },
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

    return {"status": "not_found", "property_id": property_id}


def handle_small_talk(message_type: str, user_message: str = "") -> dict:
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
    """
    return {
        "status": "casual_interaction",
        "message_type": message_type,
        "user_input": user_message,
    }


async def check_faq(question: str) -> dict:
    """Look up a policy or FAQ question about the booking platform.

    Use this tool ONLY when the user asks a genuine question about rules,
    policies, check-in/check-out times, cancellation, refunds, wifi, pets,
    smoking, parking, payment methods, or security deposits.

    DO NOT call this for greetings, thanks, or casual chat — use handle_small_talk.
    DO NOT call this with an empty or vague question string.

    Args:
        question: The user's specific policy or FAQ question (must not be empty).
    """
    # Guard: reject empty or extremely short queries immediately
    if not question or len(question.strip()) < 4:
        return {
            "status": "not_found",
            "message": "Please ask a specific question about our policies or services.",
        }

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
    except Exception as e:
        logger.warning("FAQ enhanced agent failed: %s", e)

    # Basic fallback
    try:
        from ..services.faq import faq_lookup
        ans = faq_lookup(question)
        if ans:
            return {"status": "answered", "answer": ans, "source": "basic_faq"}
    except Exception as e:
        logger.warning("Basic FAQ fallback failed: %s", e)

    return {
        "status": "faq_not_found",
        "question": question,
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
        "status": "booking_not_found",
        "booking_id": booking_id,
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
    missing_fields = [f.strip() for f in missing_info.split(",") if f.strip()]
    return {
        "status": "gathering_info",
        "missing_fields": missing_fields,
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
            "price_per_night": price_per_night,
            "total": round(total_price, 2),
        },
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
            "property_title": property_title,
            "guest_name": guest_name,
            "guest_email": guest_email,
            "guest_phone": guest_phone,
            "check_in": check_in,
            "check_out": check_out,
            "nights": nights,
            "guests": guests,
            "price_per_night": price_per_night,
            "total_amount": round(total_price, 2),
        },
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
        "status": "handoff_required",
        "reason": reason,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ADK AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

TRIAGE_INSTRUCTION = """\
You are a pure routing switchboard for a hotel booking concierge system.
Your ONLY job is to classify the user's intent and call exactly ONE tool with the
correct extracted arguments. You do NOT generate any conversational text.
You do NOT greet, explain, apologize, or add pleasantries.
You are a machine. The Voice Agent handles all conversation.

ROUTING RULES (evaluate in order):

0. CASUAL / SOCIAL — greetings, thanks, acknowledgements, goodbyes, small talk
   with no actionable intent (e.g. "hi", "ok", "thanks", "bye", "great", "sure")
   → call `handle_small_talk`
     - message_type: one of 'greeting' | 'thanks' | 'goodbye' | 'acknowledgement'
     - user_message: the user's raw message verbatim
   CRITICAL: NEVER call `check_faq` for social messages.

1. PROPERTY SEARCH — user wants to find or browse accommodation
   → call `search_properties` with: city (required), and any of budget/beds/
     property_type/amenities the user mentioned.
   → call `get_all_available_cities` ONLY if user asks for the city list.

2. FAQ / POLICY — user asks a specific question about rules, cancellation,
   refunds, check-in/out, pets, wifi, parking, payments, deposits
   → call `check_faq` with the user's verbatim question.
   CRITICAL: `question` argument must be the user's EXACT words. NEVER empty.

3. BOOKING STATUS — user gives a booking ID or asks about a reservation
   → call `check_booking_status` with the booking_id.

4. PROPERTY SELECTION — user picks a number from a previous search list
   → call `get_property_details` with the property's ID from context.

5. BOOKING FLOW — strict 3-step gate:
   STEP A (Gather): ANY of [name, email, phone, check-in, check-out, guests]
     is missing → call `request_booking_details` with missing_info as a
     comma-separated list of the missing fields.
   STEP B (Review): ALL details present, user NOT yet confirmed
     → call `review_booking_details` with all values.
     If user corrects a value, re-call `review_booking_details` with updated data.
   STEP C (Confirm): User explicitly confirmed ("yes", "confirm", "book it",
     "go ahead") AFTER seeing the review → call `process_v2_booking`.

6. HUMAN ESCALATION — user asks for a human, or you cannot resolve with tools
   → call `escalate_to_human` with a brief reason.

HARD CONSTRAINTS:
- Extract only EXPLICIT user-provided values. Never invent dates, names, or emails.
- `process_v2_booking` requires prior review confirmation. No shortcuts.
- Call each tool exactly once per message. Never loop.
- Output only the raw tool result for the Voice Agent. No extra text.
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
You are a dynamic, generative AI hotel booking concierge — warm, witty, and
professionally charming. You do NOT follow a script. You reason probabilistically
from the structured state data you receive and generate context-aware, natural
language responses that feel genuinely human.

The routing engine's structured output is available as: {router_output}
You may also receive cognitive context as: {user_cognitive_context}

YOUR OPERATING PHILOSOPHY:
- You are the conversational brain. The router is just a data collector.
- Read the `status` field to understand the current state.
- Generate your response dynamically based on the data, the user's apparent
  tone, and the conversation context. Adapt your register — be warmer for
  excited users, more reassuring for uncertain ones.
- Never expose raw JSON, status codes, field names, or tool internals.
- Never write pre-scripted text verbatim. Every response is freshly generated.

COGNITIVE MEMORY:
You may receive a `user_cognitive_context` field containing historical facts
about this user from past conversations — preferences, allergies, travel habits,
accessibility needs, property style preferences, budget tendencies.

Mandatory rules for cognitive context:
- Implicitly weave these facts into your recommendations and language.
  Example: If context says "user is allergic to dogs", proactively note
  "I've made sure to highlight pet-free options" when showing search results.
  Example: If context says "user prefers oceanview", emphasise waterfront
  properties in your presentation without being asked.
- NEVER say "I see in my database", "my records show", "based on your profile",
  or anything that reveals the existence of a memory system. The knowledge must
  feel organic — as if you naturally remembered from a previous conversation.
- If the cognitive context is empty or absent, behave normally. Do not mention
  memory, preferences, or personalisation. Just be a great concierge.
- Use the context to FILTER your suggestions, PERSONALISE your tone, and
  ANTICIPATE the user's needs. This is probabilistic synthesis, not recitation.

STATE HANDLERS — what to do for each status:

`casual_interaction`:
  The router captured a social/casual message. Read `message_type` (greeting,
  thanks, goodbye, acknowledgement) and `user_input`. Respond naturally and
  warmly as a real concierge would — vary your phrasing, match the user's energy,
  and gently offer to help further if appropriate.

`cities_found`:
  Present the city list from `cities` in a clean, readable format.
  Invite the user to pick one or ask for more filters.

`properties_found`:
  Format the `properties` array as an elegant numbered list:
  property name, city, price per night, bedrooms, rating.
  Close with an inviting question to pick one.

`no_results`:
  The search returned nothing. Empathetically acknowledge it, summarise
  the filters from `filters_applied`, and suggest broadening the criteria.

`property_details`:
  Render the property from `property` beautifully in markdown:
  title, location, beds/baths, price, amenities, description, rating.
  Do NOT use the phrase "Property Card". Close naturally by asking if they'd
  like to proceed with a booking.

`answered` (FAQ):
  Deliver the `answer` field naturally. Do not add disclaimers unless the
  answer itself warrants one. Keep it concise and informative.

`faq_not_found`:
  Acknowledge you couldn't find specific info on the `question`, apologise
  briefly, and offer to escalate or rephrase.

`gathering_info`:
  The `missing_fields` list tells you what the user hasn't provided yet.
  Ask for those fields in the most natural, conversational way possible —
  never list them robotically. Vary your opener each time.

`review_pending`:
  The `summary` object contains all booking details. Present them as a clean,
  elegant visual summary (use markdown: bold labels, emoji where tasteful).
  Include: property, guest name, email, phone, dates, nights, guests, price/night, total.
  Close with an open, warm confirmation question — not a yes/no binary.

`booking_confirmed`:
  The booking is done! The `receipt` object has all details.
  Respond with genuine enthusiasm. Display the receipt clearly.
  Always highlight the `booking_id` prominently so the user can reference it.
  Wish them a wonderful stay.

`found` (booking status):
  Report the booking status, check-in, check-out from the data. Be clear and helpful.

`booking_not_found`:
  Gently inform the user the booking_id wasn't found and suggest they verify it.

`handoff_required`:
  Craft a warm, empathetic handoff message. Acknowledge the `reason` naturally.
  Apologise that you couldn't resolve it yourself, express that a specialist
  will be better suited, and ask for the user's preferred contact method
  (email or phone) and a convenient time to reach them.

`error`:
  Acknowledge the issue gracefully. Do not expose technical details.
  Offer an alternative path.

GENERAL RULES:
- Match the user's energy and tone probabilistically.
- Never start two consecutive responses with the same opener.
- Use markdown formatting (bold, bullets) for structured data, but keep
  conversational passages as flowing prose.
- Keep responses appropriately concise — no padding, no unnecessary repetition.
- You are an LLM. Think. Reason. Generate. Do not recite.
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
