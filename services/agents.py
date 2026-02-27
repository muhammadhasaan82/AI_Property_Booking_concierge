# services/agents.py
from __future__ import annotations
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import httpx
from dotenv import load_dotenv
from .tracing import span

from .search import property_search
from .booking import (
    create_booking,
    update_booking_status,
    get_or_create_user,
    get_booking_status,
)
from .faq import faq_lookup
from .faq_enhanced import enhanced_faq_agent, initialize_faq_system
from .whatsapp import send_payment_link_async
from .nlp_extractor import extract_filters, extract_property_type, KNOWN_CITIES, CITY_ALIASES
from . import nlp_engine
from .db_logging import insert_booking_details

# -----------------------
# Load env
# -----------------------
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_TOP_P = os.getenv("OPENAI_TOP_P", "0.95")
OPENAI_TOP_K = os.getenv("OPENAI_TOP_K", "50")
OPENAI_FREQUENCY_PENALTY = os.getenv("OPENAI_FREQUENCY_PENALTY")
OPENAI_PRESENCE_PENALTY = os.getenv("OPENAI_PRESENCE_PENALTY")
OPENAI_MAX_TOKENS = os.getenv("OPENAI_MAX_TOKENS")
LLM_STRUCTURED = os.getenv("LLM_STRUCTURED", "1") not in ("0", "false", "False")

# -----------------------
# Intent helpers — NLP-powered (delegates to nlp_engine)
# -----------------------
# Structural regex (not semantic, kept minimal)
_DATE_PAT = re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2})\b")
_EMAIL_PAT = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _is_greeting(t: str) -> bool:
    return nlp_engine.is_greeting(t or "")

def _is_ack(t: str) -> bool:
    return nlp_engine.is_acknowledgment(t or "")

def _is_yes(t: str) -> bool:
    return nlp_engine.classify_affirmation(t or "") == "yes"

def _is_no(t: str) -> bool:
    return nlp_engine.classify_affirmation(t or "") == "no"

def _is_handoff_request(t: str) -> bool:
    return nlp_engine.is_handoff_request(t or "")

def _is_availability_query(t: str) -> bool:
    return nlp_engine.is_availability_query(t or "")

def _is_status_query(t: str) -> bool:
    return nlp_engine.is_status_query(t or "")

def _is_end(t: str) -> bool:
    return nlp_engine.is_end_request(t or "")

# -----------------------
# Selection & slot helpers — NLP-powered
# -----------------------

def _parse_selection_index(t: str) -> int | None:
    return nlp_engine.extract_cardinal(t or "")

def _parse_email(t: str) -> str | None:
    return nlp_engine.extract_email(t or "")

def _parse_dates(t: str) -> list[str]:
    return nlp_engine.extract_dates(t or "")

def _parse_guests(t: str) -> int | None:
    return nlp_engine.extract_guests(t or "")

def _parse_name(t: str) -> str | None:
    return nlp_engine.extract_person_name(t or "")

def _parse_phone(t: str) -> str | None:
    return nlp_engine.extract_phone(t or "")

def _parse_booking_id(t: str) -> str | None:
    return nlp_engine.extract_booking_id(t or "")

def _looks_like_property_search(t: str) -> bool:
    return nlp_engine.is_property_search(t or "")

def get_available_cities() -> List[str]:
    """Get list of unique cities from the dataset for display."""
    from .nlp_extractor import KNOWN_CITIES
    cities = sorted([c.title() for c in KNOWN_CITIES if c and len(c) > 2])
    seen = set()
    unique_cities = []
    for city in cities:
        if city.lower() not in seen:
            seen.add(city.lower())
            unique_cities.append(city)
    return unique_cities[:20]

def _wants_property_search_request(t: str) -> bool:
    return nlp_engine.wants_property_search_request(t or "")

def _wants_modification(t: str) -> bool:
    return nlp_engine.wants_modification(t or "")

def _detect_requested_fields(t: str) -> List[str]:
    return nlp_engine.detect_requested_fields(t or "")

def triage_intent(user_text: str) -> str:
    t = user_text or ""

    # IMPORTANT: Check greetings FIRST, before any other intent
    if _is_greeting(t): return "greeting"

    # Check for FAQ/Policy questions EARLY in the flow
    try:
        if nlp_engine.detect_faq_intent(t):
            return "faq"
    except Exception:
        tl = t.lower()
        if any(w in tl for w in ["wifi", "faq", "policy", "check-in time",
                                  "rules", "password", "refund", "cancel", "terms"]):
            return "faq"

    # Check for booking status queries BEFORE property search
    if _is_status_query(t): return "status_update"

    # Check for explicit end requests BEFORE other intents
    if _is_end(t): return "end"

    # Check for property search BEFORE confirmation intents
    if _looks_like_property_search(t): return "property_search"

    # Then check for other specific intents
    if _is_handoff_request(t): return "handoff"
    if _is_availability_query(t): return "availability"

    # If user mentions specific fields to change, treat as confirmation
    try:
        if _detect_requested_fields(t):
            return "confirmation"
    except Exception:
        pass

    # If user wants to modify, keep them in confirmation flow
    if _wants_modification(t): return "confirmation"

    # If user is giving selection/booking info, go to confirmation
    if (_EMAIL_PAT.search(t) or _DATE_PAT.search(t) or _parse_phone(t) is not None or
        _parse_name(t) is not None or _is_yes(t) or _is_no(t) or
        _parse_guests(t) is not None or _parse_selection_index(t) is not None):
        return "confirmation"

    if _is_ack(t): return "confirmation"

    # Booking triggers — semantic agent routing
    _BOOKING_TRIGGERS = [" book ", "reserve", "hold", "lock it", "go ahead", "confirm it"]
    if any(w in t.lower() for w in _BOOKING_TRIGGERS): return "booking"
    if any(w in t.lower() for w in ["check in", "check-in", "check out", "check-out", "status"]):
        return "status_update"
    if any(w in t.lower() for w in ["pay", "payment", "link", "invoice"]):
        return "payment_link"
    return "property_search"

# -----------------------
# LLM reply w/ optional streaming
# -----------------------
# Update the llm_reply_from_results function in agents.py:

async def llm_reply_from_results(
    user_text: str,
    results: List[Dict[str, Any]],
    locale: str = "en",
    known_filters: Dict[str, Any] | None = None,
    stream: bool = False,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    # Take up to 15 results for display
    safe_results = (results or [])[:15]
    numbered = []
    
    # Create numbered list for ALL results we have
    for i, r in enumerate(safe_results, start=1):
        city = (r.get("city") or "").title()
        title = r.get("title") or "Option"
        price = r.get("price_per_night")
        price_txt = f" — about ${price}/night" if price is not None else ""
        numbered.append(f"{i}. {title} — {city}{price_txt}")

    if not OPENAI_API_KEY:
        if results:
            # Use actual count of results shown
            return f"Yes—found {len(numbered)} options.\n\n" + "\n".join(numbered) + "\n\nReply with a number (e.g., 1 or 2) to choose."
        kf = known_filters or {}
        if not (kf.get("location") or kf.get("city")): 
            return "No match yet—what city should I search in (and a nightly budget)?"
        if not kf.get("budget"): 
            return "No match yet—what's your target nightly budget (approx)?"
        return "No match yet—any preferred dates or must-have amenities?"

    # Rest of the OpenAI logic...
    system = ("You are a warm, concise vacation-rental concierge. Use ONLY provided JSON results and the provided numbered list. "
            f"Keep replies very short. Language: {locale}")
    style = ("If results:\n- Start: 'Yes—found X options.' where X is the EXACT number of items in the numbered list.\n"
           "- Show ONLY the provided numbered list.\n- End: 'Reply with a number (e.g., 1 or 2) to choose.'\n"
           "If no results: ask ONE follow-up from {city,budget,dates,beds,amenities}.")
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User asked: {user_text}"},
        {"role": "user", "content": "Here are the ONLY properties you may use (JSON):"},
        {"role": "user", "content": json.dumps(safe_results, ensure_ascii=False)},
        {"role": "user", "content": "Numbered list:"},
        {"role": "user", "content": "\n".join(numbered) if numbered else "(no items)"},
        {"role": "user", "content": f"IMPORTANT: You found exactly {len(numbered)} options. Use this exact number."},
        {"role": "user", "content": "Known filters:"},
        {"role": "user", "content": json.dumps(known_filters or {}, ensure_ascii=False)},
        {"role": "user", "content": style},
    ]
    
    # Build OpenAI Chat Completions payload
    payload: Dict[str, Any] = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": OPENAI_TEMPERATURE,
    }
    # Optional knobs if provided
    try:
        if OPENAI_MAX_TOKENS is not None and str(OPENAI_MAX_TOKENS).strip() != "":
            payload["max_tokens"] = int(float(OPENAI_MAX_TOKENS))
    except Exception:
        pass
    try:
        if OPENAI_TOP_P is not None and str(OPENAI_TOP_P).strip() != "":
            payload["top_p"] = float(OPENAI_TOP_P)
    except Exception:
        pass
    if OPENAI_FREQUENCY_PENALTY not in (None, "", "None"):
        try:
            payload["frequency_penalty"] = float(OPENAI_FREQUENCY_PENALTY)  # type: ignore[arg-type]
        except Exception:
            pass
    if OPENAI_PRESENCE_PENALTY not in (None, "", "None"):
        try:
            payload["presence_penalty"] = float(OPENAI_PRESENCE_PENALTY)  # type: ignore[arg-type]
        except Exception:
            pass

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # If streaming requested, stream token deltas and invoke callback
    if stream:
        try:
            payload_stream = {**payload, "stream": True}
            out_chunks: List[str] = []
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload_stream,
                ) as resp:
                    if resp.status_code != 200:
                        # Fall back to non-streaming or template below
                        text = await resp.aread()
                        print(f"[LLM] stream HTTP {resp.status_code}: {text[:200]!r}")
                    else:
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            # OpenAI streams as Server-Sent Events: lines starting with 'data: '
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                evt = json.loads(data_str)
                                delta = (((evt or {}).get("choices") or [{}])[0] or {}).get("delta") or {}
                                chunk = delta.get("content") or ""
                                if chunk:
                                    out_chunks.append(chunk)
                                    if stream_callback:
                                        try:
                                            stream_callback(chunk)
                                        except Exception:
                                            # Never let callback issues break the stream
                                            pass
                            except Exception:
                                # Ignore malformed lines
                                continue
            combined = "".join(out_chunks).strip()
            if combined:
                return combined
        except Exception as e:
            print(f"[LLM] streaming error: {e}")

    # Non-streaming fallback (or if streaming failed)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        if r.status_code == 200:
            body = r.json()
            content = (((body or {}).get("choices") or [{}])[0] or {}).get("message", {}).get("content", "")
            if content:
                return content.strip()
        else:
            print(f"[LLM] HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[LLM] non-streaming error: {e}")
    
    # At the end of the function, update the fallback:
    if results:
        return f"Yes—found {len(numbered)} options.\n\n" + "\n".join(numbered) + "\n\nReply with a number (e.g., 1 or 2) to choose."
    
    kf = known_filters or {}
    if not (kf.get("location") or kf.get("city")): 
        return "Which city should I search in?"
    if not kf.get("budget"): 
        return "What's your target nightly budget?"
    return "Any dates or must-have amenities?"

# -----------------------
# Message formatters
# -----------------------
def _fmt_if_present(label: str, value: Any) -> str:
    return f"- {label}: {value}\n" if value not in (None, "", [], {}) else ""

def _format_property_full(p: Dict[str, Any]) -> str:
    """
    Render a rich, but concise property card using only fields that exist.
    Falls back gracefully when a field is missing.
    """
    title = p.get("title", "Property")
    city = (p.get("city") or "").title()
    prop_type = p.get("property_type") or ""
    bedrooms = p.get("bedrooms")
    bathrooms = p.get("bathrooms") or p.get("baths")
    beds = p.get("beds")
    price = p.get("price_per_night")
    rating = p.get("rating") or p.get("review_score")
    amenities = p.get("amenities") or []
    desc = p.get("description") or p.get("summary") or ""
    one_line_desc = (desc.strip().split("\n")[0])[:160] if desc else ""

    s = f"**{title}** — {city}\n"
    if prop_type:
        s += f"- Type: {prop_type}\n"
    if bedrooms is not None:
        s += f"- Bedrooms: {bedrooms}\n"
    if bathrooms is not None:
        s += f"- Bathrooms: {bathrooms}\n"
    if beds is not None:
        s += f"- Beds: {beds}\n"
    if price is not None:
        s += f"- Price: ${price}/night\n"
    if rating is not None:
        s += f"- Rating: {rating}\n"
    if amenities:
        s += f"- Amenities: {', '.join(map(str, amenities[:12]))}\n"
    if one_line_desc:
        s += f"- About: {one_line_desc}\n"

    return s.strip()

def _format_property_brief(p: Dict[str, Any]) -> str:
    title = p.get("title", "Property")
    city = (p.get("city") or "").title()
    price = p.get("price_per_night", "N/A")
    bedrooms = p.get("bedrooms", "N/A")
    return f"**{title}** — {city}\n- Bedrooms: {bedrooms}\n- Price: ${price}/night"

# -----------------------
# Agents
# -----------------------
def greeting_agent(filters: Dict[str, Any], user_text: str = "") -> Dict[str, Any]:
    # Clear any stale booking context when greeting
    clean_filters = {k: v for k, v in filters.items() 
                    if k not in ["awaiting_field", "awaiting_selection_confirm", "receipt_shown", 
                                 "recent_property_id", "recent_selection_index", "selected_property",
                                 "name", "phone", "email", "check_in", "check_out", "guests"]}
    
    city=clean_filters.get("location") or clean_filters.get("city")
    beds=clean_filters.get("beds"); budget=clean_filters.get("budget")
    parts=[]
    if city: parts.append(city)
    if beds: parts.append(f"{beds} beds")
    if budget: parts.append(f"budget ${budget}")
    hint=f" (noted: {', '.join(parts)})" if parts else ""

    name = _parse_name(user_text) or clean_filters.get("name")
    if name:
        clean_filters["name"] = name
        return {"reply": f"Hi {name.title()}! 👋 I'm your property assistant{hint}. How can I help you today?", "filters": clean_filters}

    return {"reply": f"Hi there! 👋 I'm your property assistant{hint}. How can I help you today?", "filters": clean_filters}

async def confirmation_agent(user_text: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Selection + slot fill → receipt → final yes/no
    - shows a full property card on selection
    - robust single-date handling using `awaiting_field`
    """
    persisted = {**(filters or {})}

    # PRIORITY 1: Handle final confirmation responses (after receipt shown) - CHECK THIS FIRST
    if persisted.get("receipt_shown"):
        if _is_yes(user_text):
            # Clear any lingering post-mod/cancel prompts and proceed to booking
            persisted.pop("awaiting_post_mod_choice", None)
            persisted.pop("awaiting_post_cancel_choice", None)
            persisted.pop("receipt_shown", None)
            # Also clear any booking-related flags to prevent re-entry
            persisted.pop("awaiting_field", None)
            persisted.pop("modifying_dates", None)
            return {
                "reply": "🎯 Perfect! Creating your booking now...",
                "tool_result": {"ok": True, "ready_for_booking": True},
                "filters": persisted,
                "booking_args": {
                    "property_id": persisted.get("recent_property_id"),
                    "check_in": persisted.get("check_in"),
                    "check_out": persisted.get("check_out"),
                    "guests": persisted.get("guests"),
                    "name": persisted.get("name"),
                    "email": persisted.get("email"),
                    "phone": persisted.get("phone"),
                    "selected_property": persisted.get("selected_property"),
                },
            }

        if _is_no(user_text):
            # Clear the receipt flag but KEEP all user data, then immediately ask what to modify
            persisted.pop("receipt_shown", None)
            persisted.pop("awaiting_post_mod_choice", None)
            # Allow user to still say "search different properties" if they want, but default to modification list
            persisted["awaiting_field"] = "modification_choice"
            return {
                "reply": "No problem, the booking has been cancelled. What would you like to modify — dates, guests, name, phone, email, or property?",
                "filters": persisted,
                "tool_result": {"ok": False, "cancelled": True, "need": ["modification"]}
            }
        
        # If user asks for total bill or receipt again
        tl = (user_text or "").lower().strip()
        if any(phrase in tl for phrase in ["total bill", "total", "bill", "receipt", "show total", "my total", "please show me my total", "Yes please show me my total", "Yes please proceed me to my total"]):
            # Re-render the receipt without changing any data
            selected_property = persisted.get("selected_property") or {}
            title = selected_property.get("title","Property")
            city = (selected_property.get("city") or "").title()
            price_per_night = float(selected_property.get("price_per_night") or 0)
            from datetime import datetime
            try:
                ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                nights=max(1, (co-ci).days)
            except Exception:
                nights=1
            total=int(price_per_night*nights)
            receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}

    # If we're in any modification sub-flow, ensure post-cancel choice doesn't interfere
    if persisted.get("awaiting_field") in {"modification","modification_choice","check_in","check_out","guests","name","phone","email"} or persisted.get("modifying_dates"):
        persisted.pop("awaiting_post_cancel_choice", None)

    # High-priority: After a modification, the user may say 'no' to proceed to receipt or 'yes' to modify more
    if persisted.get("awaiting_post_mod_choice"):
        tl = (user_text or "").lower().strip()
        proceed_phrases = [
            "proceed","continue","receipt","total","total bill","bill","show","final","summary",
            "confirm","no","see","see receipt","see total","show total","show receipt","go ahead","next","done","updated total",
            "yes i want to proceed","yes proceed","proceed to total","proceed to payment","show me total bill","please show me my total bill",
            "i want to proceed","i want total bill","total bill for payment","proceed to total bill"
        ]
        modify_phrases = [
            "modify","change","edit","update","adjust","more","another","again","i want to make more changes",
            "i want make more changes","make more changes","want to change","want to modify","i want to change","i want to modify"
        ]
        # Check for simple "yes" first - this should proceed to receipt
        if tl.strip() == "yes":
            persisted.pop("awaiting_post_mod_choice", None)
            persisted.pop("awaiting_post_cancel_choice", None)
            # Render the updated receipt
            selected_property = persisted.get("selected_property") or {}
            title = selected_property.get("title","Property")
            city = (selected_property.get("city") or "").title()
            price_per_night = float(selected_property.get("price_per_night") or 0)
            from datetime import datetime
            try:
                ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                nights=max(1, (co-ci).days)
            except Exception:
                nights=1
            total=int(price_per_night*nights)
            receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
            persisted["receipt_shown"] = True
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}
        
        if any(p in tl for p in proceed_phrases):
            persisted.pop("awaiting_post_mod_choice", None)
            persisted.pop("awaiting_post_cancel_choice", None)
            # Ensure all required fields are present before rendering receipt; otherwise ask for the next missing field
            required=["name","phone","email","check_in","check_out","guests","selected_property"]
            prompts={
                "name":"Please share your full name.",
                "phone":"Please share your phone number.",
                "email":"Please share your email address.",
                "check_in":"What is your check-in date (YYYY-MM-DD)?",
                "check_out":"What is your check-out date (YYYY-MM-DD)?",
                "guests":"How many guests?",
            }
            for rk in ["name","phone","email","check_in","check_out","guests"]:
                if not persisted.get(rk):
                    persisted["awaiting_field"]=rk
                    return {"reply":prompts[rk], "filters": persisted, "tool_result": {"ok": False, "need": [rk]}}
            # Render the updated receipt
            selected_property = persisted.get("selected_property") or {}
            title = selected_property.get("title","Property")
            city = (selected_property.get("city") or "").title()
            price_per_night = float(selected_property.get("price_per_night") or 0)
            from datetime import datetime
            try:
                ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                nights=max(1, (co-ci).days)
            except Exception:
                nights=1
            total=int(price_per_night*nights)
            receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
            persisted["receipt_shown"] = True
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}

        if any(p in tl for p in modify_phrases):
            persisted.pop("awaiting_post_mod_choice", None)
            persisted.pop("awaiting_post_cancel_choice", None)
            persisted["awaiting_field"] = "modification_choice"
            return {
                "reply": "What would you like to modify — dates, guests, name, phone, email, or property?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["modification"]}
            }

        # If user didn't answer clearly, and we're not in modification mode, proceed to receipt automatically
        if not persisted.get("modifying_dates"):
            # Ensure required fields, else ask for next missing
            for rk in ["name","phone","email","check_in","check_out","guests"]:
                if not persisted.get(rk):
                    prompts={
                        "name":"Please share your full name.",
                        "phone":"Please share your phone number.",
                        "email":"Please share your email address.",
                        "check_in":"What is your check-in date (YYYY-MM-DD)?",
                        "check_out":"What is your check-out date (YYYY-MM-DD)?",
                        "guests":"How many guests?",
                    }
                    persisted["awaiting_field"]=rk
                    return {"reply":prompts[rk], "filters": persisted, "tool_result": {"ok": False, "need": [rk]}}
            # Render receipt
            selected_property = persisted.get("selected_property") or {}
            title = selected_property.get("title","Property")
            city = (selected_property.get("city") or "").title()
            price_per_night = float(selected_property.get("price_per_night") or 0)
            from datetime import datetime
            try:
                ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                nights=max(1, (co-ci).days)
            except Exception:
                nights=1
            total=int(price_per_night*nights)
            receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
            persisted["receipt_shown"] = True
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}

        return {
            "reply": "Would you like to proceed to the updated receipt, or make another change?",
            "filters": persisted,
            "tool_result": {"ok": False, "need": ["post_mod_choice"]}
        }

    # Global branch: user explicitly asks to search different properties
    if _wants_property_search_request(user_text):
        keep_keys = [
            "location","city","budget","amenities","beds",
            "results_index_map","last_results","results",
            # Keep user-provided details to avoid losing context
            "name","phone","email","check_in","check_out","guests"
        ]
        reset = {k:v for k,v in persisted.items() if k in keep_keys}
        # Clear any selection-specific flags
        for k in ["recent_selection_index","recent_property_id","selected_property","awaiting_selection_confirm","awaiting_field","receipt_shown"]:
            reset.pop(k, None)
        return {
            "reply": "Sure — let's explore more options. Tell me what you're looking for (city, budget, dates, beds, amenities).",
            "filters": reset,
            "tool_result": {"ok": False, "need": ["restart"]}
        }

    # Handle numeric selection (but not if we're collecting guest numbers)
    sel = _parse_selection_index(user_text)
    awaiting_guests = persisted.get("awaiting_field") == "guests"
    
    # If we're waiting for guests and got a number, treat it as guest count, not selection
    if awaiting_guests and re.match(r"^\s*\d+\s*$", user_text.strip()):
        sel = None
    
    if sel is not None and not awaiting_guests:
        idx_map = persisted.get("results_index_map") or {}
        prop_id = idx_map.get(sel)
        results = persisted.get("last_results") or persisted.get("results") or []
        chosen = next((p for p in results if p.get("id")==prop_id), None)
        if not chosen:
            # Dynamic message based on available options
            max_option = max(idx_map.keys()) if idx_map else 4
            options_str = ", ".join(str(i) for i in range(1, max_option + 1))
            return {"reply":f"Sorry, I couldn't find that option. Please choose from: {options_str}.",
                    "tool_result":{"ok":False,"need":["property_selection"]},"filters":persisted}
        card = _format_property_full(chosen)
        persisted.update({
            "recent_selection_index": sel,
            "recent_property_id": prop_id,
            "selected_property": chosen,
            "awaiting_selection_confirm": True,
            "awaiting_field": None,
        })
        # If all required booking fields are already present, show the final receipt immediately
        required = ["name","phone","email","check_in","check_out","guests"]
        if all(persisted.get(k) for k in required):
            # Build and return receipt
            title = chosen.get("title","Property")
            city = (chosen.get("city") or "").title()
            price_per_night = float(chosen.get("price_per_night") or 0)
            from datetime import datetime
            try:
                ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                nights=max(1, (co-ci).days)
            except Exception:
                nights=1
            total=int(price_per_night*nights)
            receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
            persisted["receipt_shown"]=True
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}
        # Otherwise, ask for a quick confirmation on the selected property
        return {
            "reply": f"🏠 Selected:\n\n{card}\n\nWould you like to book this one? (yes/no)",
            "tool_result":{"ok":False,"need":["booking_confirmation"],"property_id":prop_id},
            "filters": persisted
        }

    # If user answered yes/no to selection
    if persisted.get("awaiting_selection_confirm"):
        # While awaiting selection confirm, do NOT parse name/phone/email/dates/guests
        # Treat common affirmations like "yes please" as yes
        tl=(user_text or "").strip().lower()
        if tl in {"yes please","yes pls","sure please","pls yes","yup please","yeah please"}:
            user_text = "yes"
        if _is_no(user_text):
            for k in ["recent_property_id","recent_selection_index","selected_property","awaiting_selection_confirm","awaiting_field"]:
                persisted.pop(k, None)
            # End the chat gracefully when user declines to book the selected property
            return {"reply": "No worries — thanks for visiting! Have a lovely day ✨", "tool_result": {"ok": False, "end": True}, "filters": persisted}
        if _is_yes(user_text) or (tl in {"yes sure","sure yes","yes please","yes pls","sure"}):
            persisted["awaiting_selection_confirm"] = False
            # If we already have required fields, show receipt directly; otherwise ask for next missing
            required=["name","phone","email","check_in","check_out","guests"]
            missing=[k for k in required if not persisted.get(k)]
            if not missing:
                selected_property = persisted.get("selected_property") or {}
                title = selected_property.get("title","Property")
                city = (selected_property.get("city") or "").title()
                price_per_night = float(selected_property.get("price_per_night") or 0)
                from datetime import datetime
                try:
                    ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                    co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                    nights=max(1, (co-ci).days)
                except Exception:
                    nights=1
                total=int(price_per_night*nights)
                receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
                persisted["receipt_shown"]=True
                return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}
            # Ask for the next missing field
            prompts={
                "name":"Please share your full name.",
                "phone":"Please share your phone number.",
                "email":"Please share your email address.",
                "check_in":"What is your check-in date (YYYY-MM-DD)?",
                "check_out":"What is your check-out date (YYYY-MM-DD)?",
                "guests":"How many guests?",
            }
            nxt=missing[0]
            persisted["awaiting_field"]=nxt
            return {"reply":prompts[nxt], "tool_result": {"ok": False, "need": [nxt]}, "filters": persisted}
        # If neither explicit yes nor no, keep asking for confirmation
        return {"reply": "Please reply with yes or no to continue.",
                "tool_result": {"ok": False, "need": ["booking_confirmation"]},
                "filters": persisted}

    # Handle user's choice after cancelling a receipt (search again vs modify)
    if persisted.get("awaiting_post_cancel_choice") and not persisted.get("awaiting_field"):
        tl = (user_text or "").lower().strip()

        # User wants to search for different properties
        if any(phrase in tl for phrase in [
            "search", "different properties", "different property", "other properties", "other property", "browse",
            "show", "more options", "yes", "find", "look", "search different properties", "find different property",
            "i want to search for different properties", "i would like to search for different properties"
        ]):
            persisted.pop("awaiting_post_cancel_choice", None)
            persisted.pop("awaiting_post_mod_choice", None)
            # Keep user info but clear property selection
            keep_keys = [
                "name", "phone", "email", "check_in", "check_out", "guests",
                "location", "city", "budget", "amenities", "beds"
            ]
            reset = {k: v for k, v in persisted.items() if k in keep_keys}
            return {
                "reply": "Sure! What type of property are you looking for? (city, budget, amenities, property type)",
                "filters": reset,
                "tool_result": {"ok": False, "need": ["restart"]}
            }

        # User wants to modify requirements (and may have already specified which one)
        if any(phrase in tl for phrase in [
            "modify", "modification", "change", "edit", "update", "correct", "fix", "adjust", "tweak"
        ]):
            requested = _detect_requested_fields(user_text)
            persisted.pop("awaiting_post_cancel_choice", None)
            persisted.pop("awaiting_post_mod_choice", None)

            # If a specific field is mentioned, jump straight to that flow
            if requested:
                # Property → restart search preserving user details
                if "property" in requested:
                    keep_keys = [
                        "name", "phone", "email", "check_in", "check_out", "guests",
                        "location", "city", "budget", "amenities", "beds"
                    ]
                    reset = {k: v for k, v in persisted.items() if k in keep_keys}
                    return {
                        "reply": "Let's find a different property. What are you looking for?",
                        "filters": reset,
                        "tool_result": {"ok": False, "need": ["restart"]}
                    }
                # Dates (both)
                if "dates" in requested or ("check_in" in requested and "check_out" in requested):
                    persisted["check_in"] = None
                    persisted["check_out"] = None
                    persisted["modifying_dates"] = True
                    persisted["awaiting_field"] = "check_in"
                    return {
                        "reply": "What's the new check-in date (YYYY-MM-DD)?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["check_in"]}
                    }
                # Single date targets
                if "check_in" in requested:
                    persisted["modifying_dates"] = True
                    persisted["awaiting_field"] = "check_in"
                    return {
                        "reply": "What's the new check-in date (YYYY-MM-DD)?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["check_in"]}
                    }
                if "check_out" in requested:
                    persisted["modifying_dates"] = True
                    persisted["awaiting_field"] = "check_out"
                    return {
                        "reply": "What's the new check-out date (YYYY-MM-DD)?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["check_out"]}
                    }
                # Guests
                if "guests" in requested:
                    persisted["awaiting_field"] = "guests"
                    return {
                        "reply": "How many guests will be staying?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["guests"]}
                    }
                # Identity/contact
                if "name" in requested:
                    persisted["awaiting_field"] = "name"
                    return {
                        "reply": "What's the correct full name?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["name"]}
                    }
                if "phone" in requested:
                    persisted["awaiting_field"] = "phone"
                    return {
                        "reply": "What's the correct phone number?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["phone"]}
                    }
                if "email" in requested:
                    persisted["awaiting_field"] = "email"
                    return {
                        "reply": "What's the correct email address?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["email"]}
                    }

            # Otherwise ask a focused clarification
            persisted["awaiting_field"] = "modification_choice"
            return {
                "reply": "What would you like to modify — dates, guests, name, phone, email, or property?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["modification"]}
            }

        # Neither clear yes nor modify - ask again
        return {
            "reply": ("Please specify: would you like to:\n"
                     "1. Search for different properties, or\n"
                     "2. Modify your current requirements?\n\n"
                     "Just tell me what you'd prefer!"),
            "filters": persisted,
            "tool_result": {"ok": False, "need": ["clarification"]}
        }

        # After a modification was applied, ask whether to proceed or modify more
        if persisted.get("awaiting_post_mod_choice"):
            tl = (user_text or "").lower().strip()
            proceed_phrases = [
                "proceed","continue","receipt","total","bill","show","final","summary","confirm","see","go ahead","next","done","payment","pay"
            ]
            modify_phrases = [
                "modify","change","edit","update","adjust","more","another","again","yes","tweak","fix"
            ]
            if any(p in tl for p in proceed_phrases):
                # Skip rendering confirmation receipt; proceed directly to booking/payment
                persisted.pop("awaiting_post_mod_choice", None)
                selected_property = persisted.get("selected_property") or {}
                price_per_night = float(selected_property.get("price_per_night") or 0)
                from datetime import datetime
                try:
                    ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                    co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                    nights=max(1, (co-ci).days)
                except Exception:
                    nights=1
                total=int(price_per_night*nights)

                return {
                    "reply": f"Proceeding to payment. Total amount: ${total}.",
                    "filters": persisted,
                    "tool_result": {"ok": True, "ready_for_booking": True},
                    "booking_args": {
                        "property_id": persisted.get("recent_property_id"),
                        "check_in": persisted.get("check_in"),
                        "check_out": persisted.get("check_out"),
                        "guests": persisted.get("guests"),
                        "name": persisted.get("name"),
                        "email": persisted.get("email"),
                        "phone": persisted.get("phone"),
                        "selected_property": persisted.get("selected_property"),
                    },
                }

            if any(p in tl for p in modify_phrases):
                persisted.pop("awaiting_post_mod_choice", None)
                persisted["awaiting_field"] = "modification_choice"
                return {
                    "reply": "What would you like to modify — dates, guests, name, phone, email, or property?",
                    "filters": persisted,
                    "tool_result": {"ok": False, "need": ["modification"]}
                }

            return {
                "reply": "Would you like to proceed to the updated receipt, or make another change?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["post_mod_choice"]}
            }
    
    # Add handling for modification_choice
    if persisted.get("awaiting_field") == "modification_choice":
        requested = _detect_requested_fields(user_text)
        
        if not requested:
            # Try to parse direct field mentions
            tl = user_text.lower()
            if "date" in tl:
                requested = ["dates"]
            elif "guest" in tl:
                requested = ["guests"]
            elif "name" in tl:
                requested = ["name"]
            elif "phone" in tl:
                requested = ["phone"]
            elif "email" in tl:
                requested = ["email"]
            elif "property" in tl:
                requested = ["property"]
            else:
                return {
                    "reply": "Please specify what you'd like to change: dates, guests, name, phone, email, or property?",
                    "filters": persisted,
                    "tool_result": {"ok": False, "need": ["modification"]}
                }
        
        persisted.pop("awaiting_field", None)
        
        # Handle each modification type
        if "dates" in requested or "check_in" in requested or "check_out" in requested:
            # User wants to change dates - clear existing dates and ask for both check-in and check-out
            persisted["check_in"] = None
            persisted["check_out"] = None
            persisted["awaiting_field"] = "check_in"
            persisted["modifying_dates"] = True
            return {
                "reply": "What is your new check-in date (YYYY-MM-DD)?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["check_in"]}
            }
        elif "guests" in requested:
            persisted["awaiting_field"] = "guests"
            return {
                "reply": "How many guests?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["guests"]}
            }
        elif "name" in requested:
            persisted["awaiting_field"] = "name"
            return {
                "reply": "What's your full name?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["name"]}
            }
        elif "phone" in requested:
            persisted["awaiting_field"] = "phone"
            return {
                "reply": "What's your phone number?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["phone"]}
            }
        elif "email" in requested:
            persisted["awaiting_field"] = "email"
            return {
                "reply": "What's your email address?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["email"]}
            }
        elif "property" in requested:
            keep_keys = [
                "name", "phone", "email", "check_in", "check_out", "guests",
                "location", "city", "budget", "amenities", "beds"
            ]
            reset = {k: v for k, v in persisted.items() if k in keep_keys}
            return {
                "reply": "Let's find a different property. What are you looking for?",
                "filters": reset,
                "tool_result": {"ok": False, "need": ["restart"]}
            }
        
        if "dates" in requested:
            # Clear existing dates when modifying
            persisted["check_in"] = None
            persisted["check_out"] = None
            persisted["awaiting_field"] = "check_in"
            persisted["modifying_dates"] = True
            return {
                "reply": "What's the new check-in date (YYYY-MM-DD)?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["check_in"]}
            }
        
        # Treat single date mentions as a full dates update (ask both sequentially)
        if "check_in" in requested and "dates" not in requested:
            persisted["awaiting_field"] = "check_in"
            return {
                "reply": "What's the new check-in date (YYYY-MM-DD)?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["check_in"]}
            }
        
        if "check_out" in requested and "dates" not in requested:
            persisted["awaiting_field"] = "check_out"
            return {
                "reply": "What's the new check-out date (YYYY-MM-DD)?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["check_out"]}
            }
        
        if "guests" in requested:
            persisted["awaiting_field"] = "guests"
            return {
                "reply": "How many guests will be staying?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["guests"]}
            }
        
        if "name" in requested:
            persisted["awaiting_field"] = "name"
            return {
                "reply": "What's the correct full name?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["name"]}
            }
        
        if "phone" in requested:
            persisted["awaiting_field"] = "phone"
            return {
                "reply": "What's the correct phone number?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["phone"]}
            }
        
        if "email" in requested:
            persisted["awaiting_field"] = "email"
            return {
                "reply": "What's the correct email address?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["email"]}
            }

    # Parse incremental fields
    # If still awaiting selection confirmation, skip parsing to avoid capturing phrases like "yes please" as name
    if persisted.get("awaiting_selection_confirm"):
        parsed_name = None
        parsed_phone = None
        parsed_email = None
        parsed_dates = None
        parsed_guests = None
    else:
        parsed_name = _parse_name(user_text)
    parsed_phone = _parse_phone(user_text)
    parsed_email = _parse_email(user_text)
    parsed_dates = _parse_dates(user_text)
    parsed_guests = _parse_guests(user_text)

    # Fast-path: if user explicitly asked to change dates, immediately route to date prompts
    if _wants_modification(user_text):
        try:
            requested_now = _detect_requested_fields(user_text)
        except Exception:
            requested_now = []
        if ("dates" in requested_now) or ("check_in" in requested_now) or ("check_out" in requested_now):
            persisted["modifying_dates"] = True
            if not persisted.get("check_in"):
                persisted["awaiting_field"] = "check_in"
                return {"reply": "What's the new check-in date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
            if not persisted.get("check_out"):
                persisted["awaiting_field"] = "check_out"
                return {"reply": "What's the new check-out date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}

    # If the bot explicitly asked to modify a specific field, overwrite it even if it already exists
    just_applied_field_update = False
    awaited = (persisted.get("awaiting_field") or "").strip()
    if awaited == "email" and parsed_email:
        persisted["email"] = parsed_email
        persisted["awaiting_field"] = None
        just_applied_field_update = True
    elif awaited == "phone" and parsed_phone:
        persisted["phone"] = parsed_phone
        persisted["awaiting_field"] = None
        just_applied_field_update = True
    elif awaited == "name" and (parsed_name or user_text.strip()):
        # Be lenient for names when explicitly awaiting
        cand = parsed_name or user_text.strip()
        if cand and len(cand) >= 2 and not any(ch in cand for ch in ['@','#','$','%','^','&','*']):
            persisted["name"] = cand
            persisted["awaiting_field"] = None
            just_applied_field_update = True
    elif awaited == "guests" and parsed_guests is not None:
        try:
            persisted["guests"] = int(parsed_guests)
        except Exception:
            persisted["guests"] = parsed_guests
        persisted["awaiting_field"] = None
        just_applied_field_update = True

    # Global branch: user explicitly wants to modify requirements (even if receipt not visible)
    if _wants_modification(user_text):
        requested = _detect_requested_fields(user_text)
        if not requested:
            persisted["awaiting_field"] = "modification"
            return {
                "reply": "What would you like to modify — dates, guests, name, phone, email, or property?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["modification"]}
            }
        # If location change requested, route to search but preserve details
        if "location" in requested:
            keep_keys = [
                "name","phone","email","check_in","check_out","guests","budget","amenities","beds"
            ]
            reset = {k:v for k,v in persisted.items() if k in keep_keys}
            return {
                "reply": "Sure — which city should I search in?",
                "filters": reset,
                "tool_result": {"ok": False, "need": ["restart"]}
            }
        # If property change requested, route back to search
        if "property" in requested:
            keep_keys = [
                "name","phone","email","check_in","check_out","guests","budget","amenities","beds","location","city"
            ]
            reset = {k:v for k,v in persisted.items() if k in keep_keys}
            return {
                "reply": "Okay, let's search for a different property. What should I look for (city, budget, amenities, beds)?",
                "filters": reset,
                "tool_result": {"ok": False, "need": ["restart"]}
            }
        # For specific fields, set awaiting_field and prompt accordingly
        if "dates" in requested or ("check_in" in requested and "check_out" in requested):
            persisted["check_in"] = None
            persisted["check_out"] = None
            persisted["awaiting_field"] = "check_in"
            return {"reply": "Sure — what's the new check-in date (YYYY-MM-DD)? Then I'll ask for check-out.", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
        if "check_in" in requested:
            persisted["check_in"] = None
            persisted["awaiting_field"] = "check_in"
            return {"reply": "Got it — what's the new check-in date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
        if "check_out" in requested:
            persisted["check_out"] = None
            persisted["awaiting_field"] = "check_out"
            return {"reply": "Got it — what's the new check-out date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
        if "guests" in requested:
            persisted["guests"] = None
            persisted["awaiting_field"] = "guests"
            return {"reply": "How many guests now?", "filters": persisted, "tool_result": {"ok": False, "need": ["guests"]}}
        if "name" in requested:
            persisted["name"] = None
            persisted["awaiting_field"] = "name"
            return {"reply": "What's the correct full name?", "filters": persisted, "tool_result": {"ok": False, "need": ["name"]}}
        if "phone" in requested:
            persisted["phone"] = None
            persisted["awaiting_field"] = "phone"
            return {"reply": "What's the correct phone number?", "filters": persisted, "tool_result": {"ok": False, "need": ["phone"]}}
        if "email" in requested:
            persisted["email"] = None
            persisted["awaiting_field"] = "email"
            return {"reply": "What's the correct email address?", "filters": persisted, "tool_result": {"ok": False, "need": ["email"]}}

    # If we're explicitly awaiting a name and got text, be more lenient
    if persisted.get("awaiting_field") == "name" and not parsed_name:
        # Accept any reasonable text as a name when explicitly asked
        t_clean = user_text.strip()
        if t_clean and len(t_clean) >= 2 and not any(char in t_clean for char in ['@', '#', '$', '%', '^', '&', '*']):
            # Check it's not a question or command
            t_lower = t_clean.lower()
            if not any(word in t_lower for word in ["show", "available", "dates", "what", "when", "how", "?"]):
                parsed_name = t_clean

    if parsed_name and not persisted.get("name"): persisted["name"]=parsed_name
    if parsed_phone and not persisted.get("phone"): persisted["phone"]=parsed_phone
    if parsed_email and not persisted.get("email"): persisted["email"]=parsed_email
    
    # Handle guest parsing - only set if we're awaiting guests or if it's clearly a guest number
    if parsed_guests and not persisted.get("guests"):
        # Only set guests if we're explicitly awaiting it or if the text clearly mentions guests
        if persisted.get("awaiting_field") == "guests" or "guest" in user_text.lower() or "people" in user_text.lower():
            try: persisted["guests"]=int(parsed_guests)
            except: persisted["guests"]=parsed_guests

    # Date handling with awaiting_field
    if parsed_dates:
        if persisted.get("awaiting_field") == "check_in":
            y,m,d = parsed_dates[0].split("-"); persisted["check_in"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            # If modifying dates, ask for check-out next explicitly; otherwise clear awaiting_field
            if persisted.get("modifying_dates"):
                persisted["awaiting_field"] = "check_out"
                return {"reply": "Thanks. Please share the new check-out date (YYYY-MM-DD).", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
            else:
                persisted["awaiting_field"] = None
        elif persisted.get("awaiting_field") == "check_out":
            y,m,d = parsed_dates[0].split("-"); persisted["check_out"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            persisted["awaiting_field"]=None
            # Clear the modifying_dates flag when both dates are set
            was_modifying = bool(persisted.pop("modifying_dates", None))
            # Only in modification mode: ask whether to modify anything else
            if was_modifying:
                persisted["awaiting_post_mod_choice"] = True
                return {
                    "reply": "Dates updated. Do you want to proceed to the total bill for payment or make more changes?",
                    "filters": persisted,
                    "tool_result": {"ok": False, "need": ["post_mod_choice"]}
                }
        else:
            applied_check_in = False
            applied_check_out = False
            if len(parsed_dates)>=1 and not persisted.get("check_in"):
                y,m,d = parsed_dates[0].split("-"); persisted["check_in"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                applied_check_in = True
            if len(parsed_dates)>=2 and not persisted.get("check_out"):
                y,m,d = parsed_dates[1].split("-"); persisted["check_out"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                applied_check_out = True
            # If only check-in was provided during modification, prompt for check-out next
            if applied_check_in and not persisted.get("check_out") and persisted.get("modifying_dates"):
                persisted["awaiting_field"] = "check_out"
                return {"reply": "Thanks. Please share the new check-out date (YYYY-MM-DD).", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
            # If both dates are now set, clear awaiting markers for a clean re-render
            if persisted.get("check_in") and persisted.get("check_out"):
                persisted["awaiting_field"] = None
                was_modifying = bool(persisted.pop("modifying_dates", None))
                if was_modifying:
                    # After dates update, ask whether to modify anything else or proceed to the updated receipt
                    persisted["awaiting_post_mod_choice"] = True
                    return {
                        "reply": "Dates updated. Do you want to proceed to the total bill for payment or make more changes?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["post_mod_choice"]}
                    }
                # If not in modification mode, just continue with normal flow
                # Don't return here, let the normal booking flow continue

    # If we already showed a receipt
    if persisted.get("receipt_shown"):
        # Inline direct modifications when no explicit prompt is pending
        if not persisted.get("awaiting_field"):
            direct_update_applied = False
            # Phone
            if parsed_phone and parsed_phone != persisted.get("phone"):
                persisted["phone"] = parsed_phone
                direct_update_applied = True
            # Email
            if parsed_email and parsed_email != persisted.get("email"):
                persisted["email"] = parsed_email
                direct_update_applied = True
            # Guests
            if parsed_guests is not None and parsed_guests != persisted.get("guests"):
                try:
                    persisted["guests"] = int(parsed_guests)
                except Exception:
                    persisted["guests"] = parsed_guests
                direct_update_applied = True
            # Dates: if two dates provided together, update both
            if parsed_dates and len(parsed_dates) >= 2:
                try:
                    y,m,d = parsed_dates[0].split("-"); persisted["check_in"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                    y,m,d = parsed_dates[1].split("-"); persisted["check_out"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                    direct_update_applied = True
                except Exception:
                    pass
            elif parsed_dates and len(parsed_dates) == 1:
                # If a single date is provided while viewing the receipt:
                # - If a date is missing, ask specifically for the missing one
                # - If both dates already exist, treat as an update and re-render
                if not persisted.get("check_in"):
                    persisted["awaiting_field"] = "check_in"
                    return {"reply": "Thanks. Please share the new check-in date (YYYY-MM-DD).", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
                if not persisted.get("check_out"):
                    persisted["awaiting_field"] = "check_out"
                    return {"reply": "Thanks. Please share the new check-out date (YYYY-MM-DD).", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
                # If both dates are already present, re-render with current values
                direct_update_applied = True

            if direct_update_applied:
                # Re-render receipt with updated values
                selected_property = persisted.get("selected_property") or {}
                title = selected_property.get("title","Property")
                city = (selected_property.get("city") or "").title()
                price_per_night = float(selected_property.get("price_per_night") or 0)
                from datetime import datetime
                try:
                    ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                    co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                    nights=max(1, (co-ci).days)
                except Exception:
                    nights=1
                total=int(price_per_night*nights)
                receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
                # Ensure no lingering modification prompt flags remain when rendering receipt
                persisted["receipt_shown"] = True
                persisted["awaiting_field"] = None
                persisted.pop("awaiting_post_mod_choice", None)
                persisted.pop("awaiting_post_cancel_choice", None)
                return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}
        # If a targeted field was just updated, immediately re-render the receipt with the new values
        if just_applied_field_update:
            selected_property = persisted.get("selected_property") or {}
            title = selected_property.get("title","Property")
            city = (selected_property.get("city") or "").title()
            price_per_night = float(selected_property.get("price_per_night") or 0)
            from datetime import datetime
            try:
                ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                nights=max(1, (co-ci).days)
            except Exception:
                nights=1
            total=int(price_per_night*nights)
            receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
            # Ensure no lingering modification prompt flags remain when rendering receipt
            persisted["receipt_shown"] = True
            persisted["awaiting_field"] = None
            persisted.pop("awaiting_post_mod_choice", None)
            persisted.pop("awaiting_post_cancel_choice", None)
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}
        # If we asked what to modify, process the user's specification
        if persisted.get("awaiting_field") == "modification":
            requested = _detect_requested_fields(user_text)
            if not requested:
                return {
                    "reply": "Please specify what you'd like to modify — dates, guests, name, phone, email, or property?",
                    "filters": persisted,
                    "tool_result": {"ok": False, "need": ["modification"]}
                }
            # Handle property change directly by routing back to search
            if "property" in requested:
                keep_keys = [
                    "location","city","budget","amenities","beds",
                    "results_index_map","last_results","results",
                    "name","phone","email","check_in","check_out","guests"
                ]
                reset = {k:v for k,v in persisted.items() if k in keep_keys}
                for k in ["recent_selection_index","recent_property_id","selected_property","awaiting_selection_confirm","awaiting_field","receipt_shown"]:
                    reset.pop(k, None)
                return {
                    "reply": "Okay, let's search for a different property. What should I look for (city, budget, amenities, beds)?",
                    "filters": reset,
                    "tool_result": {"ok": False, "need": ["restart"]}
                }

            # Apply multiple modifications in one go if possible
            updates_applied = []
            need_next: str | None = None

            # Dates handling (if both dates provided or specific date target specified)
            if "dates" in requested:
                if parsed_dates and len(parsed_dates) >= 2:
                    try:
                        y,m,d = parsed_dates[0].split("-"); persisted["check_in"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                        y,m,d = parsed_dates[1].split("-"); persisted["check_out"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                        updates_applied.append("dates")
                    except Exception:
                        need_next = need_next or "check_in"
                else:
                    # Ask both dates explicitly when user says 'dates'
                    persisted["check_in"] = None
                    persisted["check_out"] = None
                    persisted["awaiting_field"] = "check_in"
                    return {"reply": "Sure — what's the new check-in date (YYYY-MM-DD)? Then I'll ask for check-out.", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
            if "check_in" in requested and "dates" not in requested:
                if parsed_dates:
                    try:
                        y,m,d = parsed_dates[0].split("-"); persisted["check_in"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                        updates_applied.append("check_in")
                    except Exception:
                        need_next = need_next or "check_in"
                else:
                    need_next = need_next or "check_in"
            if "check_out" in requested and "dates" not in requested:
                # Prefer the last date in message if two provided
                if parsed_dates:
                    try:
                        tgt = parsed_dates[-1]
                        y,m,d = tgt.split("-"); persisted["check_out"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                        updates_applied.append("check_out")
                    except Exception:
                        need_next = need_next or "check_out"
                else:
                    need_next = need_next or "check_out"

            # Guests
            if "guests" in requested:
                if parsed_guests:
                    try:
                        persisted["guests"] = int(parsed_guests)
                    except Exception:
                        persisted["guests"] = parsed_guests
                    updates_applied.append("guests")
                else:
                    need_next = need_next or "guests"

            # Identity/contact fields
            if "name" in requested:
                if parsed_name:
                    persisted["name"] = parsed_name
                    updates_applied.append("name")
                else:
                    need_next = need_next or "name"
            if "phone" in requested:
                if parsed_phone:
                    persisted["phone"] = parsed_phone
                    updates_applied.append("phone")
                else:
                    need_next = need_next or "phone"
            if "email" in requested:
                if parsed_email:
                    persisted["email"] = parsed_email
                    updates_applied.append("email")
                else:
                    need_next = need_next or "email"

            # If something still needed, set the next required field and prompt
            if need_next:
                persisted["awaiting_field"] = need_next
                prompts={
                    "name":"What's the correct full name?",
                    "phone":"What's the correct phone number?",
                    "email":"What's the correct email address?",
                    "check_in":"What's the new check-in date (YYYY-MM-DD)?",
                    "check_out":"Please share the new check-out date (YYYY-MM-DD).",
                    "guests":"How many guests now?",
                }
                return {"reply":prompts.get(need_next, "Please provide the updated information."), "filters": persisted, "tool_result": {"ok": False, "need": [need_next]}}

            # All requested updates applied; if both dates present, proceed to ask next action
            persisted["awaiting_field"] = None
            if persisted.get("check_in") and persisted.get("check_out"):
                persisted["awaiting_post_mod_choice"] = True
                return {
                    "reply": "All set. Would you like to proceed to the updated receipt, or make another change?",
                    "filters": persisted,
                    "tool_result": {"ok": False, "need": ["post_mod_choice"]}
                }
            # If something is still missing (edge case), ask explicitly for the missing one
            missing = []
            if not persisted.get("check_in"): missing.append("check_in")
            if not persisted.get("check_out"): missing.append("check_out")
            if missing:
                need = missing[0]
                persisted["awaiting_field"] = need
                prompts={
                    "check_in":"What's the new check-in date (YYYY-MM-DD)?",
                    "check_out":"Please share the new check-out date (YYYY-MM-DD).",
                }
                return {"reply":prompts.get(need, "Please provide the updated information."), "filters": persisted, "tool_result": {"ok": False, "need": [need]}}
            # Fallback to proceed prompt
            persisted["awaiting_post_mod_choice"] = True
            return {
                "reply": "All set. Would you like to proceed to the updated receipt, or make another change?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["post_mod_choice"]}
            }
        # (Removed unconditional re-render to ensure 'no' branch triggers correctly)

        # Allow direct inline updates even when receipt isn't currently shown, if we just updated a field and all data is present
        # This makes the UX consistent after cancellation + modification
    if not persisted.get("receipt_shown") and just_applied_field_update:
        required=["name","phone","email","check_in","check_out","guests","selected_property"]
        if all(persisted.get(k) for k in required):
            selected_property = persisted.get("selected_property") or {}
            title = selected_property.get("title","Property")
            city = (selected_property.get("city") or "").title()
            price_per_night = float(selected_property.get("price_per_night") or 0)
            from datetime import datetime
            try:
                ci=datetime.strptime(persisted.get("check_in",""), "%Y-%m-%d")
                co=datetime.strptime(persisted.get("check_out",""), "%Y-%m-%d")
                nights=max(1, (co-ci).days)
            except Exception:
                nights=1
            total=int(price_per_night*nights)
            receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
            persisted["receipt_shown"] = True
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}

        # Branch: user wants to search different properties after seeing receipt
        if _wants_property_search_request(user_text):
            # Keep search-related filters and last results; clear booking-only flags
            keep_keys = [
                "location","city","budget","amenities","beds",
                "results_index_map","last_results","results",
                # Preserve user-provided details
                "name","phone","email","check_in","check_out","guests"
            ]
            reset = {k:v for k,v in persisted.items() if k in keep_keys}
            reset.pop("recent_selection_index", None)
            reset.pop("recent_property_id", None)
            reset.pop("selected_property", None)
            reset.pop("awaiting_selection_confirm", None)
            reset.pop("awaiting_field", None)
            reset.pop("receipt_shown", None)
            return {
                "reply": "Sure — let's explore more options. Tell me what you're looking for (city, budget, dates, beds, amenities).",
                "filters": reset,
                "tool_result": {"ok": False, "need": ["restart"]}
            }


    # Branch: user explicitly wants to modify requirements
    if _wants_modification(user_text):
        requested = _detect_requested_fields(user_text)
        # If user didn't specify which field, ask a focused clarification
        if not requested:
            persisted["awaiting_field"] = "modification"
            return {
                "reply": "What would you like to modify — dates, guests, name, phone, email, or property?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["modification"]}
            }
        # If property change requested, route back to search
        if "property" in requested:
            keep_keys = [
                "location","city","budget","amenities","beds",
                "results_index_map","last_results","results"
            ]
            reset = {k:v for k,v in persisted.items() if k in keep_keys}
            return {
                "reply": "Okay, let's pick a different property. What should I search for (city, budget, amenities, beds)?",
                "filters": reset,
                "tool_result": {"ok": False, "need": ["restart"]}
            }
        # Dates path: always ask both dates, starting with check-in
        if "dates" in requested or ("check_in" in requested and "check_out" in requested):
            persisted["check_in"] = None
            persisted["check_out"] = None
            persisted["modifying_dates"] = True
            persisted["awaiting_field"] = "check_in"
            return {"reply": "What's the new check-in date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
        if "check_in" in requested:
            persisted["modifying_dates"] = True
            persisted["awaiting_field"] = "check_in"
            return {"reply": "What's the new check-in date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
        if "check_out" in requested:
            persisted["modifying_dates"] = True
            persisted["awaiting_field"] = "check_out"
            return {"reply": "What's the new check-out date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
        if "guests" in requested:
            persisted["awaiting_field"] = "guests"
            return {"reply": "How many guests now?", "filters": persisted, "tool_result": {"ok": False, "need": ["guests"]}}
        if "name" in requested:
            persisted["awaiting_field"] = "name"
            return {"reply": "What's the correct full name?", "filters": persisted, "tool_result": {"ok": False, "need": ["name"]}}
        if "phone" in requested:
            persisted["awaiting_field"] = "phone"
            return {"reply": "What's the correct phone number?", "filters": persisted, "tool_result": {"ok": False, "need": ["phone"]}}
        if "email" in requested:
            persisted["awaiting_field"] = "email"
            return {"reply": "What's the correct email address?", "filters": persisted, "tool_result": {"ok": False, "need": ["email"]}}



    # Stateless safety: if user says no right after a summary-like context was likely shown, but `receipt_shown` isn't set
    # Honor the cancellation and ask next action, preserving collected info
    if _is_no(user_text) and not persisted.get("awaiting_selection_confirm") and not persisted.get("awaiting_field") and not persisted.get("receipt_shown") and not persisted.get("awaiting_post_mod_choice"):
        return {"reply":"No problem, The booking has been cancelled. Would you like to search for different properties or modify your requirements?",
                "filters": persisted}

    # Ask next missing field (with awaiting_field marker)
    required=["name","phone","email","check_in","check_out","guests"]
    prompts={
        "name":"Please share your full name.",
        "phone":"Please share your phone number.",
        "email":"Please share your email address.",
        "check_in":"What is your check-in date (YYYY-MM-DD)?",
        "check_out":"What is your check-out date (YYYY-MM-DD)?",
        "guests":"How many guests?",
    }
    for k in required:
        if not persisted.get(k):
            persisted["awaiting_field"]=k
            persisted["awaiting_selection_confirm"] = False  # Clear this flag when asking for fields
            return {"reply":prompts[k], "tool_result":{"ok":False,"need":[k]}, "filters":persisted}

    # Build receipt (only if not already shown)
    if not persisted.get("receipt_shown"):
        selected_property = persisted.get("selected_property") or {}
    title = selected_property.get("title","Property")
    city = (selected_property.get("city") or "").title()
    price_per_night = float(selected_property.get("price_per_night") or 0)

    from datetime import datetime
    try:
        ci=datetime.strptime(persisted["check_in"], "%Y-%m-%d")
        co=datetime.strptime(persisted["check_out"], "%Y-%m-%d")
        nights=max(1, (co-ci).days)
    except Exception:
        nights=1
    total=int(price_per_night*nights)

    receipt=f"""📋 **BOOKING SUMMARY**

**Guest Information**
- Name: {persisted.get("name")}
- Phone: {persisted.get("phone")}
- Email: {persisted.get("email")}

**Property Details**
- {title}
- Location: {city}
- Price per night: ${int(price_per_night)}

**Booking Details**
- Check-in: {persisted.get("check_in")}
- Check-out: {persisted.get("check_out")}
- Number of nights: {nights}
- Number of guests: {persisted.get("guests")}

💰 **TOTAL AMOUNT: ${total}**

✅ **Would you like to confirm this booking?**
Reply **yes** to confirm and proceed with payment, or **no** to cancel."""
    persisted["receipt_shown"]=True
    persisted["awaiting_field"]=None
    return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}

    # Default return to ensure we always return a dictionary
    return {"reply": "I'm sorry, I didn't understand that. Could you please clarify?", "filters": persisted}

def faq_agent(user_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced FAQ agent that uses semantic search on policy documents"""
    # Try the enhanced FAQ system first
    try:
        result = enhanced_faq_agent(user_text, context)
        return result
    except Exception as e:
        print(f"Enhanced FAQ failed, falling back to basic: {e}")
        # Fallback to basic FAQ lookup
        ans = faq_lookup(user_text)
        return {"reply": ans if ans else "I couldn't find that in the FAQs. Want me to connect you to support or try a quick search?"}

def _need_booking_fields(args: Dict[str, Any]) -> List[str]:
    missing=[]
    for k in ["property_id","check_in","check_out","name","email","phone"]:
        if not args.get(k): missing.append(k)
    return missing

async def booking_agent(args: Dict[str, Any]) -> Dict[str, Any]:
    missing=_need_booking_fields(args)
    if missing:
        return {"reply": f"To finalize the booking I need: {', '.join(missing).replace('_',' ')}.",
                "tool_result":{"ok":False,"need":missing}}

    # ──────────────────────────────────────────────────────────
    # Pre-validate via Rust BookingValidatorTool (TOON protocol)
    # If the gateway is unreachable, skip validation and proceed
    # ──────────────────────────────────────────────────────────
    try:
        from .rust_client import validate_booking

        rust_validation = await validate_booking(
            property_id=args.get("property_id", ""),
            check_in=args.get("check_in", ""),
            check_out=args.get("check_out", ""),
            guests=int(args.get("guests", 1)),
            email=args.get("email", "")
        )

        if not rust_validation.get("fallback"):
            # Unwrap: the gateway wraps in {ok, result, ...}
            inner = rust_validation.get("result", rust_validation)
            is_valid = inner.get("valid", True)
            errors = inner.get("errors", [])
            warnings = inner.get("warnings", [])

            if not is_valid:
                error_list = "\n".join(f"• {e}" for e in errors)
                warn_text = ""
                if warnings:
                    warn_text = "\n\n⚠️ Warnings:\n" + "\n".join(f"• {w}" for w in warnings)
                return {
                    "reply": f"❌ **Booking validation failed:**\n\n{error_list}{warn_text}\n\nPlease correct the above and try again.",
                    "tool_result": {"ok": False, "need": ["correction"], "validation_errors": errors, "warnings": warnings},
                }
            # Validation passed — log any warnings
            if warnings:
                print(f"[RUST] Booking validation warnings: {warnings}")
            print(f"[RUST] Booking validated OK via gateway")
    except Exception as e:
        print(f"[RUST] Booking validation offload failed: {e}, proceeding with Python logic")
    
    # Try to create user and booking
    try:
        user_id=get_or_create_user(name=args["name"], email=args["email"], phone=args.get("phone"))
    except Exception as e:
        # If database is not configured, use mock mode for testing
        import os
        if not os.getenv("SUPABASE_URL") or "Supabase env not set" in str(e):
            # Mock mode - simulate successful booking
            user_id = "mock_user_123"
            print("[INFO] Running in mock mode - no real booking created")
        else:
            return {"tool_result":{"ok":False,"error":f"user create failed: {e}"},
                    "reply":"Sorry, I couldn't create your profile. Could you recheck your name/email/phone?"}
    payload={
        "user_id": user_id,
        "property_id": args["property_id"],
        "check_in": args["check_in"],
        "check_out": args["check_out"],
        "guests": int(args.get("guests",1)),
        "phone": args.get("phone"),
    }
    
    # Try to create booking
    try:
        r=create_booking(payload)
    except Exception as e:
        # If database is not configured, use mock mode
        import os
        if not os.getenv("SUPABASE_URL") or "Supabase env not set" in str(e):
            # Mock successful booking
            import uuid
            r = {
                "ok": True,
                "booking_id": str(uuid.uuid4())[:8],
                "status": "confirmed",
                "payment_url": "https://example.com/pay/mock"
            }
            print("[INFO] Mock booking created successfully")
        else:
            r = {"ok": False, "error": str(e)}

    prop_title=(args.get("selected_property") or {}).get("title","")
    ptype="apartment"
    for t in ["villa","condo","house","loft","studio","townhouse","apartment","flat","cottage","bungalow","penthouse"]:
        if t in prop_title.lower(): ptype=t; break

    if r.get("ok"):
        msg=f"""🎉 **Booking Confirmed!**

✅ Your **{ptype}** has been successfully booked!

**Booking Details**
- Booking ID: {r.get('booking_id','N/A')}
- Property: {prop_title or 'Property'}
- Check-in: {args.get('check_in')}
- Check-out: {args.get('check_out')}
- Guests: {args.get('guests',1)}

📧 A payment link has been sent to your email/WhatsApp.

**Thank you for booking with us! Have a wonderful stay!** 🌟
"""
        # Insert booking details row (best-effort)
        try:
            # Compute nights + total amount (based on price if available from selected_property)
            nights = 1
            try:
                from datetime import datetime
                ci = datetime.strptime(args.get('check_in',''), "%Y-%m-%d")
                co = datetime.strptime(args.get('check_out',''), "%Y-%m-%d")
                nights = max(1, (co-ci).days)
            except Exception:
                pass
            selected = (args.get('selected_property') or {})
            price = selected.get('price_per_night') or 0
            total = float(price or 0) * float(nights or 1)
            prop_title = selected.get('title') or 'Property'
            city = (selected.get('city') or '').title()
            price_txt = f" — about ${int(price)}/night" if price else ""
            prop_desc = f"{prop_title} — {city}{price_txt}".strip()
            booking_code = str(r.get('booking_id'))
            row = {
                "booking_id": r.get("booking_id"),
                "booking_code": booking_code if booking_code else None,
                "property_type": ptype,
                "property_description": prop_desc,
                "client_name": args.get("name"),
                "client_phone": args.get("phone"),
                "client_email": args.get("email"),
                "check_in": args.get("check_in"),
                "check_out": args.get("check_out"),
                "guests": int(args.get("guests",1)),
                "nights": int(nights),
                "total_amount": total,
                "payment": "TRUE",  # mark paid on success; adjust if you add real payments
            }
            insert_booking_details(row)
        except Exception:
            pass
    else:
        msg=f"Sorry, I couldn't create the booking: {r.get('error')}"
    
    # If booking was successful, add flag to clear booking state
    if r.get("ok"):
        return {"tool_result": {**r, "clear_booking_state": True}, "reply": msg}
    else:
        return {"tool_result": r, "reply": msg}

def availability_agent(filters: Dict[str, Any]) -> Dict[str, Any]:
    pid=None
    idx_map=filters.get("results_index_map") or {}
    recent=filters.get("recent_selection_index")
    if recent and idx_map: pid=idx_map.get(int(recent))
    city=filters.get("location") or filters.get("city")
    city_n=city.title() if city else None
    
    # Check if we have recent search results to show
    last_results = filters.get("last_results") or filters.get("results") or []
    
    if last_results:
        # We have properties, guide user to select one first
        reply = ("I have properties available! Please select a property first by choosing its number, "
                 "then I can help you with specific dates. ")
        if len(last_results) <= 4:
            options = ", ".join(str(i+1) for i in range(len(last_results)))
            reply += f"Choose from: {options}."
    elif pid:
        reply=(f"For checking availability, please provide your preferred dates:\n"
               f"- Check-in date (YYYY-MM-DD)\n"
               f"- Check-out date (YYYY-MM-DD)\n\n"
               f"All our properties have flexible availability. Once you provide dates, I'll confirm the booking details.")
    else:
        reply=("Our properties are generally available year-round!\n\n"
               "To check specific availability and make a booking, please:\n"
               "1. First, search for properties (e.g., 'find apartment in New York')\n"
               "2. Select a property from the results\n"
               "3. Then provide your check-in and check-out dates (YYYY-MM-DD format)")
    return {"reply": reply}

def status_agent(user_text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Answer booking status questions given a booking_id.
    Behaviors:
      - If booking_id missing → ask only for booking ID
      - If user asks status/check-in/check-out → fetch and answer
      - If user asks to update to check-in/checkout → perform update
    """
    action=(args.get("action") or "").lower()
    booking_id=args.get("booking_id") or _parse_booking_id(user_text)

    # If we don't have an id yet, request it clearly
    if not booking_id:
        return {"reply":"Please provide your Booking ID (e.g., 57015107-d414-409c-843e-b6a6b15d9b59)."}

    # Follow-up after showing status
    if action == "followup":
        return {"reply": "You're welcome! Would you like to ask anything else or end this chat session? Say 'end' to close."}

    # Query current status / dates
    if action in ("query","status","check","info","details","date","dates") or not action:
        r=get_booking_status(booking_id)
        if r.get("ok"):
            s=str(r.get("status",""))
            s_human = s.replace("_"," ") if s else "unknown"
            ci = r.get("check_in") or "?"
            co = r.get("check_out") or "?"
            return {"tool_result": r, "reply": f"Booking {booking_id}: **{s_human}**\n- Check-in: {ci}\n- Check-out: {co}\n\nWould you like to ask anything else or end this chat session? (say 'end' to close)"}
        return {"tool_result": r, "reply": f"Sorry—{r.get('error','unable to find that booking')}."}

    # Update flow (explicit request)
    if action in ("check_in","check-in"): new_status="checked_in"
    elif action in ("check_out","check-out"): new_status="checked_out"
    else: new_status=args.get("new_status")
    if not new_status:
        return {"reply":"Do you want to check in or check out? If you only want to know status, say 'status'."}
    current=(args.get("current_status") or "pending")
    r=update_booking_status(booking_id=booking_id, current_status=current, new_status=new_status)
    msg="Status updated successfully." if r.get("ok") else f"Sorry, that didn't work: {r.get('error')}"
    return {"tool_result": r, "reply": msg}

async def payment_agent(args: Dict[str, Any]) -> Dict[str, Any]:
    r=await send_payment_link_async(
        booking_id=args.get("booking_id","demo"),
        phone=args.get("phone","+10000000000"),
        url=args.get("payment_url","https://example/pay/123"),
    )
    msg="I've sent the payment link via WhatsApp." if r.get("ok") else f"Couldn't send the link: {r.get('error')}"
    return {"tool_result": r, "reply": msg}

async def property_agent(user_text: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    clean_filters = {k: v for k, v in filters.items() if not callable(v)}
    
    extracted = extract_filters(user_text, clean_filters)
    prop_type = extract_property_type(user_text)
    enhanced = f"{prop_type} {user_text}" if prop_type else user_text

    # Check if a city was requested but not found in our dataset
    requested_city = extracted.get("location") or extracted.get("city")
    
    # If user text mentions a city-like query but no valid city was extracted
    city_keywords = ["in", "at", "near", "around", "located"]
    might_be_city_request = any(keyword in user_text.lower() for keyword in city_keywords)
    
    # ──────────────────────────────────────────────────────────
    # Try Rust gateway for property search (heavy computation)
    # Falls back to Python property_search if gateway unavailable
    # ──────────────────────────────────────────────────────────
    results = None
    try:
        from .rust_client import search_properties
        from .search import _DATASET

        rust_result = await search_properties(
            location=requested_city or "",
            budget=extracted.get("budget"),
            beds=extracted.get("beds"),
            amenities=extracted.get("amenities") or [],
            property_type=prop_type or "",
            properties=_DATASET if _DATASET else None,
        )

        if not rust_result.get("fallback"):
            # Extract results from the Rust response
            inner = rust_result.get("result", rust_result)
            rust_results = inner.get("results", [])
            if isinstance(rust_results, list):
                results = rust_results
                print(f"[RUST] Property search returned {len(results)} results via gateway")
    except Exception as e:
        print(f"[RUST] Property search offload failed: {e}, using Python fallback")
    
    # Fallback: use local Python search
    if results is None:
        results = property_search(
            query_text=enhanced,
            budget=extracted.get("budget"),
            amenities=extracted.get("amenities"),
            location=requested_city,
            beds=extracted.get("beds"),
            property_type=prop_type,
        )
    
    # If no results and user seems to be asking for a specific location
    if not results and might_be_city_request:
        # Check if this might be an invalid city request
        from .nlp_extractor import KNOWN_CITIES
        
        # If no valid city was found in the text
        if not requested_city:
            available_cities = get_available_cities()
            city_list = ", ".join(available_cities[:10])  # Show first 10 cities
            
            reply = (
                "Unfortunately, I couldn't find properties in that location. "
                "I have properties available in these US cities:\n\n"
                f"**{city_list}**, and more.\n\n"
                "Would you like to search in one of these cities? Just tell me which city you prefer."
            )
            
            extracted["awaiting_city_selection"] = True
            return {"results": [], "reply": reply, "filters": extracted}
    
    # Regular flow with proper list counting - show all results up to a reasonable limit
    max_display = min(len(results), 15)  # Show up to 15 results
    index_map = {i+1: r.get("id") for i, r in enumerate(results[:max_display])}
    extracted.update({
        "results_index_map": index_map, 
        "last_results": results[:max_display]
    })

    stream = bool(filters.get("stream", False))
    stream_cb = filters.get("stream_callback") if callable(filters.get("stream_callback")) else None
    
    reply = await llm_reply_from_results(
        user_text=user_text,
        results=results,
        locale=filters.get("locale", "en"),
        known_filters=extracted,
        stream=stream,
        stream_callback=stream_cb,
    )
    
    return {"results": results, "reply": reply, "filters": extracted}

def handoff_agent(user_text: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    city=filters.get("location") or filters.get("city")
    return {"reply": f"Okay — I'll connect you with a human specialist{f' about {city.title()}' if city else ''}. "
                     "Please share your email or phone number and a preferred time.",
            "tool_result":{"handoff":True}}
