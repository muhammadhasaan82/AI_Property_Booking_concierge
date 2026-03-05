# services/agents.py
from __future__ import annotations
from services.confirmation_helpers import _render_receipt
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import httpx
from dotenv import load_dotenv
from .tracing import span
import logging

logger = logging.getLogger(__name__)

from .search import property_search
from .booking import (
    create_booking,
    update_booking_status,
    get_or_create_user,
    get_booking_status,
)
from .faq import faq_lookup
from .faq_enhanced import enhanced_faq_agent, initialize_faq_system
from services import confirmation_helpers
from .whatsapp import send_payment_link_async
from .nlp_extractor import extract_filters, extract_property_type, KNOWN_CITIES, CITY_ALIASES
from . import nlp_engine
from .db_logging import (
    insert_booking_details,
    insert_successful_booking,
    get_successful_booking_status,
)
import services.config as config
from .dynamic_config import get_vocabulary
from .state_keys import SK, STATE_KEYS

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
OPENAI_FREQUENCY_PENALTY = os.getenv("OPENAI_FREQUENCY_PENALTY")
OPENAI_PRESENCE_PENALTY = os.getenv("OPENAI_PRESENCE_PENALTY")
OPENAI_MAX_TOKENS = os.getenv("OPENAI_MAX_TOKENS")
LLM_STRUCTURED = os.getenv("LLM_STRUCTURED", "1") not in ("0", "false", "False")

SOFT_INTENT_ROUTER = os.getenv("SOFT_INTENT_ROUTER", "1") not in ("0", "false", "False")

_ALLOWED_INTENTS = {
    "greeting", "faq", "confirmation", "property_search", "booking",
    "status_update", "payment_link", "handoff", "availability", "end",
}


def _llm_route_intent(user_text: str, filters: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Use LLM structured output to classify intent with minimal hardcoded rules."""
    if not (OPENAI_API_KEY and LLM_STRUCTURED and SOFT_INTENT_ROUTER):
        return None

    text = (user_text or "").strip()
    if not text:
        return None

    context = {
        "has_selected_property": bool((filters or {}).get(SK.selected_property)),
        "has_booking_progress": any((filters or {}).get(k) for k in config.REQUIRED_FIELDS),
        SK.awaiting_field: (filters or {}).get(SK.awaiting_field),
        SK.receipt_shown: bool((filters or {}).get(SK.receipt_shown)),
    }

    payload: Dict[str, Any] = {
        "model": OPENAI_CHAT_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an intent classifier for a hotel booking chatbot. "
                    "Classify the user's message into ONE intent. Return strict JSON: {intent, confidence, brief_reason}. "
                    "Intent must be one of: greeting, faq, confirmation, property_search, booking, "
                    "status_update, payment_link, handoff, availability, end.\n\n"
                    "CRITICAL RULES:\n"
                    "- 'faq' = user is asking about rules, policy, refund, cancellation, pets, smoking, check-in time, "
                    "wifi password, security deposit, amenities, or any property/platform question. "
                    "Classify as 'faq' EVEN IF the user is mid-booking. A policy question always overrides booking context.\n"
                    "- 'confirmation' = user is selecting a numbered option (e.g. 'option 3', 'go for number 5', "
                    "'I will take the second one'), providing booking details (name, phone, email, dates, guests), "
                    "or affirming/declining a booking step (yes/no in booking context).\n"
                    "- 'property_search' = user is looking for a place or asking about properties. If they say 'hello I need a loft in NYC', it is 'property_search', NOT 'greeting'.\n"
                    "- 'greeting' = ONLY pure greetings like 'hi', 'hello', 'hey', 'good morning' with NO other intent.\n"
                    "- If the message contains ANY reference to selecting an option number, it is 'confirmation', NOT 'greeting'.\n"
                    "- Words like 'sure', 'ok', 'go ahead' combined with option/number references = 'confirmation'.\n"
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"message": text, "context": context}, ensure_ascii=False),
            },
        ],
    }

    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }, json=payload)
        if r.status_code != 200:
            return None
        body = r.json()
        content = (((body or {}).get("choices") or [{}])[0] or {}).get("message", {}).get("content", "")
        if not content:
            return None
        parsed = json.loads(content)
        intent = str(parsed.get("intent", "")).strip()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        if intent in _ALLOWED_INTENTS and confidence >= 0.45:
            return intent
    except Exception:
        return None
    return None


def _slot_prompt(field: str) -> str:
    return config.FIELD_PROMPTS.get(field, "Please share that detail.")

def _vocab():
    return get_vocabulary()

def _nlp_fallback():
    return _vocab().nlp_fallback


def _classify_proceed_or_modify(user_text: str) -> str | None:
    """Return 'proceed', 'modify', or None based on user intent.

    Uses the canonical phrase lists from config so every call-site
    shares the same decision logic.
    """
    tl = (user_text or "").lower().strip()
    if not tl:
        return None

    # Exact "yes" is treated as proceed (existing convention)
    if tl == "yes":
        return "proceed"

    vocab = _vocab()
    if any(p in tl for p in vocab.proceed_phrases):
        return "proceed"
    if any(p in tl for p in vocab.modify_phrases):
        return "modify"
    return None

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

def _llm_extract_booking_fields(text: str) -> dict:
    """Soft-coded fallback to extract all booking fields using the LLM in one pass."""
    if not (OPENAI_API_KEY and LLM_STRUCTURED) or not text.strip() or len(text.strip()) < 3:
        return {}
        
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract booking details from the user's text. Return strict JSON with keys: "
                    "'name', 'email', 'phone', 'guests' (integer), 'dates' (list of YYYY-MM-DD strings), "
                    "'location' (string). If a field is not found, omit it or set to null."
                )
            },
            {"role": "user", "content": text}
        ]
    }
    
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }, json=payload)
        if r.status_code == 200:
            content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                parsed = json.loads(content)
                return parsed
    except Exception:
        pass
    return {}


def _parse_selection_index(t: str) -> int | None:
    return nlp_engine.extract_cardinal(t or "")

def _parse_email(t: str) -> str | None:
    text = t or ""
    res = nlp_engine.extract_email(text)
    if res: return res
    return _llm_extract_booking_fields(text).get("email")

def _parse_dates(t: str) -> list[str]:
    text = t or ""
    res = nlp_engine.extract_dates(text)
    if res: return res
    return _llm_extract_booking_fields(text).get("dates") or []

def _parse_guests(t: str) -> int | None:
    text = t or ""
    res = nlp_engine.extract_guests(text)
    if res is not None: return res
    try:
        val = _llm_extract_booking_fields(text).get("guests")
        return int(val) if val else None
    except Exception:
        return None

def _parse_name(t: str) -> str | None:
    text = t or ""
    # Try fast NLP / regex first
    name = nlp_engine.extract_person_name(text)
    name_rejects = set(_nlp_fallback().parse_name_reject_exact)
    if name:
        n = name.strip().lower()
        # Prevent greetings/ack words from being stored as a person's name.
        if n in name_rejects:
            return None
        return name
    # Fallback to soft-coded LLM extraction
    extracted = _llm_extract_booking_fields(text).get("name")
    if isinstance(extracted, str) and len(extracted) >= 2:
        n = extracted.strip().lower()
        if n in name_rejects:
            return None
        return extracted.title()
    return None

def _parse_phone(t: str) -> str | None:
    text = t or ""
    res = nlp_engine.extract_phone(text)
    if res: return res
    return _llm_extract_booking_fields(text).get("phone")

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
    return unique_cities

def _wants_property_search_request(t: str) -> bool:
    return nlp_engine.wants_property_search_request(t or "")

def _wants_modification(t: str) -> bool:
    return nlp_engine.wants_modification(t or "")

def _detect_requested_fields(t: str) -> List[str]:
    return nlp_engine.detect_requested_fields(t or "")



def triage_intent(user_text: str, filters: Optional[Dict[str, Any]] = None) -> str:
    t = user_text or ""
    active_filters = filters or {}
    tl = t.lower().strip()

    # Keep strict deterministic guards for critical intents.
    if _is_greeting(t):
        return "greeting"
    if _is_end(t):
        return "end"

    # --- STRICT LIST SELECTION GUARD (Must be BEFORE the LLM call!) ---
    if (filters or {}).get("last_results"):
        if _parse_selection_index(t) is not None and "?" not in t and not any(k in t.lower() for k in ["policy", "refund", "cancel", "rules", "faq"]):
            return "confirmation"

    # If user is currently deciding on a selected property, keep follow-up
    # turns in confirmation for yes/no, reselection, or "back to list" requests.
    if active_filters.get(SK.awaiting_selection_confirm):
        if (
            _is_yes(t)
            or _is_no(t)
            or _parse_selection_index(t) is not None
            or nlp_engine.wants_previous_results_sync(tl)
        ):
            return "confirmation"

    if _is_ack(t):
        return "confirmation"


    # Soft-coded LLM intent router (prioritized as requested by user to be super soft-coded).
    llm_intent = _llm_route_intent(t, filters)
    if llm_intent:
        return llm_intent

    # ── FAQ detection BEFORE LLM — policy/rule questions must ALWAYS break out
    # of any flow (confirmation, booking, etc). Run this first, it's fast & free.
    try:
        faq_hit = nlp_engine.detect_faq_intent(t)
    except Exception:
        faq_hit = any(w in t.lower() for w in _vocab().faq_fallback_keywords)

    if faq_hit:
        return "faq"

    # Status checks come after FAQ so policy questions like "refund/check-in policy"
    # are not hijacked by status routing.
    if _is_status_query(t):
        tl = t.lower()
        if not any(p in tl for p in _nlp_fallback().status_resume_phrases):
            return "status_update"

    # NLP-driven and keyword fallback routing (secondary pass — catches LLM misses).
    if _looks_like_property_search(t):
        return "property_search"
    if _is_handoff_request(t):
        return "handoff"
    if _is_availability_query(t):
        return "availability"

    try:
        if _detect_requested_fields(t):
            return "confirmation"
    except Exception:
        pass

    if _wants_modification(t):
        return "confirmation"

    if (_EMAIL_PAT.search(t) or _DATE_PAT.search(t) or _parse_phone(t) is not None or
        _parse_name(t) is not None or _is_yes(t) or _is_no(t) or
        _parse_guests(t) is not None or _parse_selection_index(t) is not None):
        return "confirmation"

    vocab = _vocab()
    if any(w in t.lower() for w in vocab.booking_triggers):
        return "booking"
    if any(w in t.lower() for w in vocab.status_keywords):
        return "status_update"
    if any(w in t.lower() for w in vocab.payment_keywords):
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
            total_count = len(results)
            shown_count = len(numbered)
            header = f"Yes—found {total_count} options"
            if total_count > shown_count:
                header += f", here are the top {shown_count}:"
            else:
                header += ":"
            return f"{header}\n\n" + "\n".join(numbered) + "\n\nReply with a number (e.g., 1 or 2) to choose."
        kf = known_filters or {}
        if not (kf.get("location") or kf.get("city")): 
            return "No match yet—what city should I search in (and a nightly budget)?"
        if not kf.get("budget"): 
            return "No match yet—what's your target nightly budget (approx)?"
        return "No match yet—any preferred dates or must-have amenities?"

    total_count = len(results)
    shown_count = len(numbered)
    
    system = ("You are a warm, concise vacation-rental concierge. Use ONLY provided JSON results and the provided numbered list. "
            "Think step by step: First identify which properties match the user's needs, then present them clearly. "
            f"Keep replies very short. Language: {locale}")
            
    header_instruction = f"Start: 'Yes—found {total_count} options.'"
    if total_count > shown_count:
        header_instruction = f"Start: 'Yes—found {total_count} options, here are the top {shown_count}.'"
            
    style = (f"If results:\n- {header_instruction}\n"
           "- Show ONLY the provided numbered list.\n- End: 'Reply with a number (e.g., 1 or 2) to choose.'\n"
           "If no results: ask ONE follow-up from {city,budget,dates,beds,amenities}.")
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User asked: {user_text}"},
        {"role": "user", "content": "Here are the ONLY properties you may use (JSON):"},
        {"role": "user", "content": json.dumps(safe_results, ensure_ascii=False)},
        {"role": "user", "content": "Numbered list:"},
        {"role": "user", "content": "\n".join(numbered) if numbered else "(no items)"},
        {"role": "user", "content": f"IMPORTANT: You received {total_count} results, but you are only showing the top {shown_count} options in the list above. Acknowledge this."},
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
                        logger.warning("LLM stream HTTP %d: %s", resp.status_code, text[:200])
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
            logger.warning("LLM streaming error: %s", e)

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
            logger.warning("LLM HTTP %d: %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.warning("LLM non-streaming error: %s", e)
    
    # At the end of the function, update the fallback:
    if results:
        total_count = len(results)
        shown_count = len(numbered)
        header = f"Yes—found {total_count} options"
        if total_count > shown_count:
            header += f", here are the top {shown_count}:"
        else:
            header += ":"
        return f"{header}\n\n" + "\n".join(numbered) + "\n\nReply with a number (e.g., 1 or 2) to choose."
    
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
                    if k not in [SK.awaiting_field, SK.awaiting_selection_confirm, SK.receipt_shown, 
                                 SK.recent_property_id, SK.recent_selection_index, SK.selected_property,
                                 "name", "phone", "email", "check_in", "check_out", "guests"]}
    
    city=clean_filters.get("location") or clean_filters.get("city")
    beds=clean_filters.get("beds"); budget=clean_filters.get("budget")
    parts=[]
    if city: parts.append(city)
    if beds: parts.append(f"{beds} beds")
    if budget: parts.append(f"budget ${budget}")
    hint=f" (noted: {', '.join(parts)})" if parts else ""

    name = clean_filters.get("name")
    # Do not parse greetings like "hello" as names.
    if user_text and not _is_greeting(user_text):
        name = _parse_name(user_text) or name
    if name:
        clean_filters["name"] = name
        return {"reply": f"Hi {name.title()}! 👋 I'm your property assistant{hint}. How can I help you today?", "filters": clean_filters}

    return {"reply": f"Hi there! 👋 I'm your property assistant{hint}. How can I help you today?", "filters": clean_filters}

async def _confirmation_agent_impl(user_text: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Selection + slot fill → receipt → final yes/no
    - shows a full property card on selection
    - robust single-date handling using `awaiting_field`
    """
    persisted = {**(filters or {})}

    # PRIORITY 1: Handle final confirmation responses (after receipt shown) - CHECK THIS FIRST
    if persisted.get(SK.receipt_shown):
        if _is_yes(user_text):
            # Clear any lingering post-mod/cancel prompts and proceed to booking
            persisted.pop(SK.awaiting_post_mod_choice, None)
            persisted.pop(SK.awaiting_post_cancel_choice, None)
            persisted.pop(SK.receipt_shown, None)
            # Also clear any booking-related flags to prevent re-entry
            persisted.pop(SK.awaiting_field, None)
            persisted.pop(SK.modifying_dates, None)
            return {
                "reply": "🎯 Perfect! Creating your booking now...",
                "tool_result": {"ok": True, "ready_for_booking": True},
                "filters": persisted,
                "booking_args": {
                    "property_id": persisted.get(SK.recent_property_id),
                    "check_in": persisted.get("check_in"),
                    "check_out": persisted.get("check_out"),
                    "guests": persisted.get("guests"),
                    "name": persisted.get("name"),
                    "email": persisted.get("email"),
                    "phone": persisted.get("phone"),
                    SK.selected_property: persisted.get(SK.selected_property),
                },
            }

        if _is_no(user_text):
            # Clear the receipt flag but KEEP all user data, then immediately ask what to modify
            persisted.pop(SK.receipt_shown, None)
            persisted.pop(SK.awaiting_post_mod_choice, None)
            # Allow user to still say "search different properties" if they want, but default to modification list
            persisted[SK.awaiting_field] = "modification_choice"
            return {
                "reply": "No problem, the booking has been cancelled. What would you like to modify — dates, guests, name, phone, email, or property?",
                "filters": persisted,
                "tool_result": {"ok": False, "cancelled": True, "need": ["modification"]}
            }
        
        # If user asks for total bill or receipt again
        if nlp_engine.is_receipt_request(user_text):
            receipt = _render_receipt(persisted)
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}

    # If we're in any modification sub-flow, ensure post-cancel choice doesn't interfere
    if persisted.get(SK.awaiting_field) in {"modification","modification_choice","check_in","check_out","guests","name","phone","email"} or persisted.get(SK.modifying_dates):
        persisted.pop(SK.awaiting_post_cancel_choice, None)

    # High-priority: After a modification, the user may say 'no' to proceed to receipt or 'yes' to modify more
    if persisted.get(SK.awaiting_post_mod_choice):
        tl = (user_text or "").lower().strip()
        proceed_phrases = _vocab().proceed_phrases
        modify_phrases = _vocab().modify_phrases
        # Check for simple "yes" first - this should proceed to receipt
        if tl.strip() == "yes":
            persisted.pop(SK.awaiting_post_mod_choice, None)
            persisted.pop(SK.awaiting_post_cancel_choice, None)
            receipt = _render_receipt(persisted)
            persisted[SK.receipt_shown] = True
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}
        
        if any(p in tl for p in proceed_phrases):
            persisted.pop(SK.awaiting_post_mod_choice, None)
            persisted.pop(SK.awaiting_post_cancel_choice, None)
            # Ensure all required fields are present before rendering receipt; otherwise ask for the next missing field
            required = config.REQUIRED_FIELDS + [SK.selected_property]
            for rk in config.REQUIRED_FIELDS:
                if not persisted.get(rk):
                    persisted[SK.awaiting_field]=rk
                    return {"reply": _slot_prompt(rk), "filters": persisted, "tool_result": {"ok": False, "need": [rk]}}
            return confirmation_helpers._try_show_receipt(persisted)

        if any(p in tl for p in modify_phrases):
            persisted.pop(SK.awaiting_post_mod_choice, None)
            persisted.pop(SK.awaiting_post_cancel_choice, None)
            persisted[SK.awaiting_field] = "modification_choice"
            return {
                "reply": "What would you like to modify — dates, guests, name, phone, email, or property?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["modification"]}
            }

        # If user didn't answer clearly, and we're not in modification mode, proceed to receipt automatically
        if not persisted.get(SK.modifying_dates):
            # Ensure required fields, else ask for next missing
            for rk in config.REQUIRED_FIELDS:
                if not persisted.get(rk):
                    persisted[SK.awaiting_field]=rk
                    return {"reply": _slot_prompt(rk), "filters": persisted, "tool_result": {"ok": False, "need": [rk]}}
            return confirmation_helpers._try_show_receipt(persisted)

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
            *config.REQUIRED_FIELDS,
        ]
        reset = {k:v for k,v in persisted.items() if k in keep_keys}
        # Clear any selection-specific flags
        for k in [SK.recent_selection_index,SK.recent_property_id,SK.selected_property,SK.awaiting_selection_confirm,SK.awaiting_field,SK.receipt_shown]:
            reset.pop(k, None)
        return {
            "reply": "Sure — let's explore more options. Tell me what you're looking for (city, budget, dates, beds, amenities).",
            "filters": reset,
            "tool_result": {"ok": False, "need": ["restart"]}
        }

    # Handle numeric selection (but not if we're collecting guest numbers)
    sel = _parse_selection_index(user_text)
    awaiting_field = (persisted.get(SK.awaiting_field) or "").strip()
    awaiting_guests = awaiting_field == "guests"
    # If we are actively collecting any booking field, never treat numbers/ordinals
    # as property selection in this turn.
    if awaiting_field and awaiting_field not in {"modification", "modification_choice"}:
        sel = None
    # Note: do NOT suppress sel when awaiting_selection_confirm; user may re-select a different item.
    # The awaiting_selection_confirm block below handles this case explicitly.
    
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
            SK.recent_selection_index: sel,
            SK.recent_property_id: prop_id,
            SK.selected_property: chosen,
            SK.awaiting_selection_confirm: True,
            SK.awaiting_field: None,
        })
        # If all required booking fields are already present, show the final receipt immediately
        required = list(config.REQUIRED_FIELDS)
        if all(persisted.get(k) for k in required):
            return confirmation_helpers._try_show_receipt(persisted)
        # Otherwise, ask for a quick confirmation on the selected property
        return {
            "reply": f"🏠 Selected:\n\n{card}\n\nWould you like to book this one? (yes/no)",
            "tool_result":{"ok":False,"need":["booking_confirmation"],"property_id":prop_id},
            "filters": persisted
        }

    # If user answered yes/no to selection
    if persisted.get(SK.awaiting_selection_confirm):
        # While awaiting selection confirm, do NOT parse name/phone/email/dates/guests
        tl=(user_text or "").strip().lower()

        # FIX B: If user types a new numeric selection (e.g. "6"), handle it here instead of
        # asking "please reply yes or no". Clear stale selection state and fall through
        # to the sel-handling block which will show the proper property card.
        if sel is not None:
            persisted.pop(SK.awaiting_selection_confirm, None)
            persisted.pop(SK.selected_property, None)
            persisted.pop(SK.recent_property_id, None)
            persisted.pop(SK.recent_selection_index, None)
            # Fall through — sel block below will handle showing the property card

        # FIX A: "no, back to list" / "show other properties" — re-show last results, NOT end session
        elif _is_no(user_text) or await nlp_engine.wants_previous_results_async(tl):
            last_results = persisted.get("last_results") or persisted.get("results") or []
            idx_map = persisted.get("results_index_map") or {}
            for k in [SK.recent_property_id, SK.recent_selection_index, SK.selected_property, SK.awaiting_selection_confirm, SK.awaiting_field]:
                persisted.pop(k, None)
            if last_results and idx_map:
                # Re-display the previous search results
                lines = []
                for num in sorted(idx_map.keys()):
                    pid = idx_map[num]
                    prop = next((p for p in last_results if p.get("id") == pid), None)
                    if prop:
                        title = prop.get("title", "Property")
                        city = (prop.get("city") or "").title()
                        price = int(float(prop.get("price_per_night") or 0))
                        lines.append(f"{num}. {title} — {city} — about ${price}/night")
                if lines:
                    listing = "\n".join(lines)
                    return {
                        "reply": f"Sure! Here are your previous results:\n\n{listing}\n\nReply with a number to choose.",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["property_selection"]},
                    }
            # No cached results — prompt a fresh search
            return {
                "reply": "No previous results found. What would you like to search for? (city, property type, budget)",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["restart"]},
            }

        if sel is None and _is_yes(user_text):
            persisted[SK.awaiting_selection_confirm] = False
            # If we already have required fields, show receipt directly; otherwise ask for next missing
            required=list(config.REQUIRED_FIELDS)
            missing=[k for k in required if not persisted.get(k)]
            if not missing:
                return confirmation_helpers._try_show_receipt(persisted)
            # Ask for the next missing field
            nxt=missing[0]
            persisted[SK.awaiting_field]=nxt
            return {"reply":config.FIELD_PROMPTS.get(nxt, f"Please provide your {nxt}."), "tool_result": {"ok": False, "need": [nxt]}, "filters": persisted}
        # If neither explicit yes nor no (and not a new numeric selection), keep asking
        if sel is None:
            return {"reply": "Please reply with yes or no to continue.",
                    "tool_result": {"ok": False, "need": ["booking_confirmation"]},
                    "filters": persisted}

    # Handle user's choice after cancelling a receipt (search again vs modify)
    if persisted.get(SK.awaiting_post_cancel_choice) and not persisted.get(SK.awaiting_field):
        tl = (user_text or "").lower().strip()

        # User wants to search for different properties
        if _wants_property_search_request(user_text):
            persisted.pop(SK.awaiting_post_cancel_choice, None)
            persisted.pop(SK.awaiting_post_mod_choice, None)
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
        if _wants_modification(user_text):
            requested = _detect_requested_fields(user_text)
            persisted.pop(SK.awaiting_post_cancel_choice, None)
            persisted.pop(SK.awaiting_post_mod_choice, None)

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
                    persisted[SK.modifying_dates] = True
                    persisted[SK.awaiting_field] = "check_in"
                    return {
                        "reply": "What's the new check-in date (YYYY-MM-DD)?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["check_in"]}
                    }
                # Single date targets
                if "check_in" in requested:
                    persisted[SK.modifying_dates] = True
                    persisted[SK.awaiting_field] = "check_in"
                    return {
                        "reply": "What's the new check-in date (YYYY-MM-DD)?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["check_in"]}
                    }
                if "check_out" in requested:
                    persisted[SK.modifying_dates] = True
                    persisted[SK.awaiting_field] = "check_out"
                    return {
                        "reply": "What's the new check-out date (YYYY-MM-DD)?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["check_out"]}
                    }
                # Guests
                if "guests" in requested:
                    persisted[SK.awaiting_field] = "guests"
                    return {
                        "reply": "How many guests will be staying?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["guests"]}
                    }
                # Identity/contact
                if "name" in requested:
                    persisted[SK.awaiting_field] = "name"
                    return {
                        "reply": "What's the correct full name?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["name"]}
                    }
                if "phone" in requested:
                    persisted[SK.awaiting_field] = "phone"
                    return {
                        "reply": "What's the correct phone number?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["phone"]}
                    }
                if "email" in requested:
                    persisted[SK.awaiting_field] = "email"
                    return {
                        "reply": "What's the correct email address?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["email"]}
                    }

            # Otherwise ask a focused clarification
            persisted[SK.awaiting_field] = "modification_choice"
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
        if persisted.get(SK.awaiting_post_mod_choice):
            tl = (user_text or "").lower().strip()
            proceed_phrases = _vocab().proceed_phrases
            modify_phrases = _vocab().modify_phrases
            if any(p in tl for p in proceed_phrases):
                # Skip rendering confirmation receipt; proceed directly to booking/payment
                persisted.pop(SK.awaiting_post_mod_choice, None)
                selected_property = persisted.get(SK.selected_property) or {}
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
                        "property_id": persisted.get(SK.recent_property_id),
                        "check_in": persisted.get("check_in"),
                        "check_out": persisted.get("check_out"),
                        "guests": persisted.get("guests"),
                        "name": persisted.get("name"),
                        "email": persisted.get("email"),
                        "phone": persisted.get("phone"),
                        SK.selected_property: persisted.get(SK.selected_property),
                    },
                }

            if any(p in tl for p in modify_phrases):
                persisted.pop(SK.awaiting_post_mod_choice, None)
                persisted[SK.awaiting_field] = "modification_choice"
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
    if persisted.get(SK.awaiting_field) == "modification_choice":
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
        
        persisted.pop(SK.awaiting_field, None)
        
        # Handle each modification type
        if "dates" in requested or "check_in" in requested or "check_out" in requested:
            # User wants to change dates - clear existing dates and ask for both check-in and check-out
            persisted["check_in"] = None
            persisted["check_out"] = None
            persisted[SK.awaiting_field] = "check_in"
            persisted[SK.modifying_dates] = True
            return {
                "reply": "What is your new check-in date (YYYY-MM-DD)?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["check_in"]}
            }
        elif "guests" in requested:
            persisted[SK.awaiting_field] = "guests"
            return {
                "reply": "How many guests?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["guests"]}
            }
        elif "name" in requested:
            persisted[SK.awaiting_field] = "name"
            return {
                "reply": "What's your full name?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["name"]}
            }
        elif "phone" in requested:
            persisted[SK.awaiting_field] = "phone"
            return {
                "reply": "What's your phone number?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["phone"]}
            }
        elif "email" in requested:
            persisted[SK.awaiting_field] = "email"
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
            persisted[SK.awaiting_field] = "check_in"
            persisted[SK.modifying_dates] = True
            return {
                "reply": "What's the new check-in date (YYYY-MM-DD)?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["check_in"]}
            }
        
        # Treat single date mentions as a full dates update (ask both sequentially)
        if "check_in" in requested and "dates" not in requested:
            persisted[SK.awaiting_field] = "check_in"
            return {
                "reply": "What's the new check-in date (YYYY-MM-DD)?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["check_in"]}
            }
        
        if "check_out" in requested and "dates" not in requested:
            persisted[SK.awaiting_field] = "check_out"
            return {
                "reply": "What's the new check-out date (YYYY-MM-DD)?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["check_out"]}
            }
        
        if "guests" in requested:
            persisted[SK.awaiting_field] = "guests"
            return {
                "reply": "How many guests will be staying?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["guests"]}
            }
        
        if "name" in requested:
            persisted[SK.awaiting_field] = "name"
            return {
                "reply": "What's the correct full name?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["name"]}
            }
        
        if "phone" in requested:
            persisted[SK.awaiting_field] = "phone"
            return {
                "reply": "What's the correct phone number?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["phone"]}
            }
        
        if "email" in requested:
            persisted[SK.awaiting_field] = "email"
            return {
                "reply": "What's the correct email address?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["email"]}
            }

    # Parse incremental fields
    # If still awaiting selection confirmation, skip parsing to avoid capturing phrases like "yes please" as name
    if persisted.get(SK.awaiting_selection_confirm):
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
            persisted[SK.modifying_dates] = True
            if not persisted.get("check_in"):
                persisted[SK.awaiting_field] = "check_in"
                return {"reply": "What's the new check-in date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
            if not persisted.get("check_out"):
                persisted[SK.awaiting_field] = "check_out"
                return {"reply": "What's the new check-out date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}

    # If the bot explicitly asked to modify a specific field, overwrite it even if it already exists
    just_applied_field_update = False
    awaited = (persisted.get(SK.awaiting_field) or "").strip()
    if awaited == "email" and parsed_email:
        persisted["email"] = parsed_email
        persisted[SK.awaiting_field] = None
        just_applied_field_update = True
    elif awaited == "phone" and parsed_phone:
        persisted["phone"] = parsed_phone
        persisted[SK.awaiting_field] = None
        just_applied_field_update = True
    elif awaited == "name" and (parsed_name or user_text.strip()):
        # Be lenient for names when explicitly awaiting
        cand = parsed_name or user_text.strip()
        name_invalid_chars = set(_nlp_fallback().parse_name_invalid_chars)
        if cand and len(cand) >= 2 and not any(ch in cand for ch in name_invalid_chars):
            words = re.findall(r"[a-zA-Z][a-zA-Z'-]*", cand)
            disallowed = set(_nlp_fallback().parse_name_disallowed_words)
            if 1 <= len(words) <= 3 and not any(w.lower() in disallowed for w in words):
                persisted["name"] = cand
                persisted[SK.awaiting_field] = None
                just_applied_field_update = True
    elif awaited == "guests" and parsed_guests is not None:
        try:
            persisted["guests"] = int(parsed_guests)
        except Exception:
            persisted["guests"] = parsed_guests
        persisted[SK.awaiting_field] = None
        just_applied_field_update = True

    # Global branch: user explicitly wants to modify requirements (even if receipt not visible)
    if _wants_modification(user_text):
        requested = _detect_requested_fields(user_text)
        if not requested:
            persisted[SK.awaiting_field] = "modification"
            return {
                "reply": "What would you like to modify — dates, guests, name, phone, email, or property?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["modification"]}
            }
        # If location change requested, route to search but preserve details
        if "location" in requested:
            keep_keys = [
                *config.REQUIRED_FIELDS, "budget", "amenities", "beds"
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
                *config.REQUIRED_FIELDS, "budget", "amenities", "beds", "location", "city"
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
            persisted[SK.awaiting_field] = "check_in"
            return {"reply": "Sure — what's the new check-in date (YYYY-MM-DD)? Then I'll ask for check-out.", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
        if "check_in" in requested:
            persisted["check_in"] = None
            persisted[SK.awaiting_field] = "check_in"
            return {"reply": "Got it — what's the new check-in date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
        if "check_out" in requested:
            persisted["check_out"] = None
            persisted[SK.awaiting_field] = "check_out"
            return {"reply": "Got it — what's the new check-out date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
        if "guests" in requested:
            persisted["guests"] = None
            persisted[SK.awaiting_field] = "guests"
            return {"reply": "How many guests now?", "filters": persisted, "tool_result": {"ok": False, "need": ["guests"]}}
        if "name" in requested:
            persisted["name"] = None
            persisted[SK.awaiting_field] = "name"
            return {"reply": "What's the correct full name?", "filters": persisted, "tool_result": {"ok": False, "need": ["name"]}}
        if "phone" in requested:
            persisted["phone"] = None
            persisted[SK.awaiting_field] = "phone"
            return {"reply": "What's the correct phone number?", "filters": persisted, "tool_result": {"ok": False, "need": ["phone"]}}
        if "email" in requested:
            persisted["email"] = None
            persisted[SK.awaiting_field] = "email"
            return {"reply": "What's the correct email address?", "filters": persisted, "tool_result": {"ok": False, "need": ["email"]}}

    # If we're explicitly awaiting a name and got text, be more lenient
    if persisted.get(SK.awaiting_field) == "name" and not parsed_name:
        # Accept any reasonable text as a name when explicitly asked
        t_clean = user_text.strip()
        name_invalid_chars = set(_nlp_fallback().parse_name_invalid_chars)
        if t_clean and len(t_clean) >= 2 and not any(char in t_clean for char in name_invalid_chars):
            disallowed = set(_nlp_fallback().parse_name_disallowed_words)
            words = re.findall(r"[a-zA-Z][a-zA-Z'-]*", t_clean)
            t_lower = t_clean.lower()
            if (
                1 <= len(words) <= 3
                and not any(w.lower() in disallowed for w in words)
                and not any(phrase in t_lower for phrase in _nlp_fallback().parse_name_disallowed_phrases)
            ):
                parsed_name = t_clean

    if parsed_name and not persisted.get("name"): persisted["name"]=parsed_name
    if parsed_phone and not persisted.get("phone"): persisted["phone"]=parsed_phone
    if parsed_email and not persisted.get("email"): persisted["email"]=parsed_email
    
    # Handle guest parsing - only set if we're awaiting guests or if it's clearly a guest number
    if parsed_guests and not persisted.get("guests"):
        # Only set guests if we're explicitly awaiting it or if the text clearly mentions guests
        guest_context_terms = _nlp_fallback().guest_context_terms
        if (
            persisted.get(SK.awaiting_field) == "guests"
            or any(term in user_text.lower() for term in guest_context_terms)
        ):
            try: persisted["guests"]=int(parsed_guests)
            except: persisted["guests"]=parsed_guests

    # Date handling with awaiting_field
    if parsed_dates:
        if persisted.get(SK.awaiting_field) == "check_in":
            y,m,d = parsed_dates[0].split("-"); persisted["check_in"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            # If modifying dates, ask for check-out next explicitly; otherwise clear awaiting_field
            if persisted.get(SK.modifying_dates):
                persisted[SK.awaiting_field] = "check_out"
                return {"reply": "Thanks. Please share the new check-out date (YYYY-MM-DD).", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
            else:
                persisted[SK.awaiting_field] = None
        elif persisted.get(SK.awaiting_field) == "check_out":
            y,m,d = parsed_dates[0].split("-"); persisted["check_out"]=f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            persisted[SK.awaiting_field]=None
            # Clear the modifying_dates flag when both dates are set
            was_modifying = bool(persisted.pop(SK.modifying_dates, None))
            # Only in modification mode: ask whether to modify anything else
            if was_modifying:
                persisted[SK.awaiting_post_mod_choice] = True
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
            if applied_check_in and not persisted.get("check_out") and persisted.get(SK.modifying_dates):
                persisted[SK.awaiting_field] = "check_out"
                return {"reply": "Thanks. Please share the new check-out date (YYYY-MM-DD).", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
            # If both dates are now set, clear awaiting markers for a clean re-render
            if persisted.get("check_in") and persisted.get("check_out"):
                persisted[SK.awaiting_field] = None
                was_modifying = bool(persisted.pop(SK.modifying_dates, None))
                if was_modifying:
                    # After dates update, ask whether to modify anything else or proceed to the updated receipt
                    persisted[SK.awaiting_post_mod_choice] = True
                    return {
                        "reply": "Dates updated. Do you want to proceed to the total bill for payment or make more changes?",
                        "filters": persisted,
                        "tool_result": {"ok": False, "need": ["post_mod_choice"]}
                    }
                # If not in modification mode, just continue with normal flow
                # Don't return here, let the normal booking flow continue

    # If we already showed a receipt
    if persisted.get(SK.receipt_shown):
        # Inline direct modifications when no explicit prompt is pending
        if not persisted.get(SK.awaiting_field):
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
                    persisted[SK.awaiting_field] = "check_in"
                    return {"reply": "Thanks. Please share the new check-in date (YYYY-MM-DD).", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
                if not persisted.get("check_out"):
                    persisted[SK.awaiting_field] = "check_out"
                    return {"reply": "Thanks. Please share the new check-out date (YYYY-MM-DD).", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
                # If both dates are already present, re-render with current values
                direct_update_applied = True

            if direct_update_applied:
                # Re-render receipt with updated values
                receipt = _render_receipt(persisted)
                # Ensure no lingering modification prompt flags remain when rendering receipt
                persisted[SK.receipt_shown] = True
                persisted[SK.awaiting_field] = None
                persisted.pop(SK.awaiting_post_mod_choice, None)
                persisted.pop(SK.awaiting_post_cancel_choice, None)
                return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}
        # If a targeted field was just updated, immediately re-render the receipt with the new values
        if just_applied_field_update:
            receipt = _render_receipt(persisted)
            # Ensure no lingering modification prompt flags remain when rendering receipt
            persisted[SK.receipt_shown] = True
            persisted[SK.awaiting_field] = None
            persisted.pop(SK.awaiting_post_mod_choice, None)
            persisted.pop(SK.awaiting_post_cancel_choice, None)
            return {"reply":receipt, "tool_result":{"ok":False,"need":["final_confirmation"],"show_receipt":True}, "filters":persisted}
        # If we asked what to modify, process the user's specification
        if persisted.get(SK.awaiting_field) == "modification":
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
                    *config.REQUIRED_FIELDS,
                ]
                reset = {k:v for k,v in persisted.items() if k in keep_keys}
                for k in [SK.recent_selection_index,SK.recent_property_id,SK.selected_property,SK.awaiting_selection_confirm,SK.awaiting_field,SK.receipt_shown]:
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
                    persisted[SK.awaiting_field] = "check_in"
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
                persisted[SK.awaiting_field] = need_next
                return {"reply":config.FIELD_MODIFICATION_PROMPTS.get(need_next, "Please provide the updated information."), "filters": persisted, "tool_result": {"ok": False, "need": [need_next]}}

            # All requested updates applied; if both dates present, proceed to ask next action
            persisted[SK.awaiting_field] = None
            if persisted.get("check_in") and persisted.get("check_out"):
                persisted[SK.awaiting_post_mod_choice] = True
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
                persisted[SK.awaiting_field] = need
                return {"reply":config.FIELD_MODIFICATION_PROMPTS.get(need, "Please provide the updated information."), "filters": persisted, "tool_result": {"ok": False, "need": [need]}}
            # Fallback to proceed prompt
            persisted[SK.awaiting_post_mod_choice] = True
            return {
                "reply": "All set. Would you like to proceed to the updated receipt, or make another change?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["post_mod_choice"]}
            }
        # (Removed unconditional re-render to ensure 'no' branch triggers correctly)

        # Allow direct inline updates even when receipt isn't currently shown, if we just updated a field and all data is present
        # This makes the UX consistent after cancellation + modification
    if not persisted.get(SK.receipt_shown) and just_applied_field_update:
        required = config.REQUIRED_FIELDS + [SK.selected_property]
        if all(persisted.get(k) for k in required):
            return confirmation_helpers._try_show_receipt(persisted)

        # Branch: user wants to search different properties after seeing receipt
        if _wants_property_search_request(user_text):
            # Keep search-related filters and last results; clear booking-only flags
            keep_keys = [
                "location","city","budget","amenities","beds",
                "results_index_map","last_results","results",
                # Preserve user-provided details
                *config.REQUIRED_FIELDS,
            ]
            reset = {k:v for k,v in persisted.items() if k in keep_keys}
            reset.pop(SK.recent_selection_index, None)
            reset.pop(SK.recent_property_id, None)
            reset.pop(SK.selected_property, None)
            reset.pop(SK.awaiting_selection_confirm, None)
            reset.pop(SK.awaiting_field, None)
            reset.pop(SK.receipt_shown, None)
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
            persisted[SK.awaiting_field] = "modification"
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
            persisted[SK.modifying_dates] = True
            persisted[SK.awaiting_field] = "check_in"
            return {"reply": "What's the new check-in date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
        if "check_in" in requested:
            persisted[SK.modifying_dates] = True
            persisted[SK.awaiting_field] = "check_in"
            return {"reply": "What's the new check-in date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_in"]}}
        if "check_out" in requested:
            persisted[SK.modifying_dates] = True
            persisted[SK.awaiting_field] = "check_out"
            return {"reply": "What's the new check-out date (YYYY-MM-DD)?", "filters": persisted, "tool_result": {"ok": False, "need": ["check_out"]}}
        if "guests" in requested:
            persisted[SK.awaiting_field] = "guests"
            return {"reply": "How many guests now?", "filters": persisted, "tool_result": {"ok": False, "need": ["guests"]}}
        if "name" in requested:
            persisted[SK.awaiting_field] = "name"
            return {"reply": "What's the correct full name?", "filters": persisted, "tool_result": {"ok": False, "need": ["name"]}}
        if "phone" in requested:
            persisted[SK.awaiting_field] = "phone"
            return {"reply": "What's the correct phone number?", "filters": persisted, "tool_result": {"ok": False, "need": ["phone"]}}
        if "email" in requested:
            persisted[SK.awaiting_field] = "email"
            return {"reply": "What's the correct email address?", "filters": persisted, "tool_result": {"ok": False, "need": ["email"]}}



    # Stateless safety: if user says no right after a summary-like context was likely shown, but `receipt_shown` isn't set
    # Honor the cancellation and ask next action, preserving collected info
    if _is_no(user_text) and not persisted.get(SK.awaiting_selection_confirm) and not persisted.get(SK.awaiting_field) and not persisted.get(SK.receipt_shown) and not persisted.get(SK.awaiting_post_mod_choice):
        return {"reply":"No problem, The booking has been cancelled. Would you like to search for different properties or modify your requirements?",
                "filters": persisted}

    # Ask next missing field (with awaiting_field marker)
    required=list(config.REQUIRED_FIELDS)
    for k in required:
        if not persisted.get(k):
            persisted[SK.awaiting_field]=k
            persisted[SK.awaiting_selection_confirm] = False  # Clear this flag when asking for fields
            return {"reply":config.FIELD_PROMPTS.get(k, f"Please provide your {k}."), "tool_result":{"ok":False,"need":[k]}, "filters":persisted}

    # Build receipt only when it has not already been shown in this state.
    if not persisted.get(SK.receipt_shown):
        return confirmation_helpers._try_show_receipt(persisted)

    # Receipt was already shown but message was not strongly classified.
    return {
        "reply": "I want to make sure I get this right. Please reply with a clear 'yes' to confirm and proceed with payment, or 'no' to cancel and modify.",
        "filters": persisted,
    }


async def confirmation_agent(user_text: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    return await _confirmation_agent_impl(user_text, filters)

def faq_agent(user_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced FAQ agent that uses semantic search on policy documents"""
    # Try the enhanced FAQ system first
    try:
        result = enhanced_faq_agent(user_text, context)
        return result
    except Exception as e:
        logger.warning("Enhanced FAQ failed, falling back to basic: %s", e)
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
                logger.warning("RUST booking validation warnings: %s", warnings)
            logger.info("RUST booking validated OK via gateway")
    except Exception as e:
        logger.warning("RUST booking validation offload failed: %s, proceeding with Python logic", e)
    
    # Try to create user and booking
    try:
        user_id = await get_or_create_user(name=args["name"], email=args["email"], phone=args.get("phone"))
    except Exception as e:
        # If database is not configured, use mock mode for testing
        import os
        if not os.getenv("SUPABASE_URL") or "Supabase env not set" in str(e):
            # Mock mode - simulate successful booking
            user_id = "mock_user_123"
            logger.info("Running in mock mode - no real booking created")
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
        r = await create_booking(payload)
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
                "payment_url": f"{config.PAYMENT_BASE_URL}/mock"
            }
            logger.info("Mock booking created successfully")
        else:
            r = {"ok": False, "error": str(e)}

    prop_title=(args.get(SK.selected_property) or {}).get("title","")
    ptype="apartment"
    for t in sorted(config.SEED_PROPERTY_TYPES):
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
            selected = (args.get(SK.selected_property) or {})
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
            await insert_booking_details(row)
            # Persist successful bookings in a dedicated table for status checks via chat.
            successful_row = {
                "booking_id": booking_code if booking_code else str(r.get("booking_id") or ""),
                "status": str(r.get("status") or "confirmed"),
                "check_in": args.get("check_in"),
                "check_out": args.get("check_out"),
                "user_name": args.get("name"),
                "user_email": args.get("email"),
                "user_phone": args.get("phone"),
                "property_title": prop_title,
                "property_type": ptype,
                "city": city,
                "guests": int(args.get("guests", 1)),
                "nights": int(nights),
                "total_amount": total,
                "payment_url": r.get("payment_url"),
                "source": str(r.get("note") or "db"),
            }
            await insert_successful_booking(successful_row)
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
    recent=filters.get(SK.recent_selection_index)
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

async def status_agent(user_text: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Answer booking status questions given a booking_id.
    Behaviors:
      - If booking_id missing → ask only for booking ID
      - If user asks status/check-in/check-out → fetch and answer
      - If user asks to update to check-in/checkout → perform update
    """
    action=(args.get("action") or "").lower()
    nlp_fb = _nlp_fallback()
    booking_id=args.get("booking_id") or _parse_booking_id(user_text)

    # If we don't have an id yet, request it clearly
    if not booking_id:
        return {"reply":"Please provide your Booking ID (e.g., 57015107-d414-409c-843e-b6a6b15d9b59)."}

    # Follow-up after showing status
    if action == "followup":
        return {"reply": "You're welcome! Would you like to ask anything else or end this chat session? Say 'end' to close."}

    # Query current status / dates
    if action in set(nlp_fb.status_query_actions) or not action:
        r = await get_booking_status(booking_id)
        if r.get("ok"):
            s=str(r.get("status",""))
            s_human = s.replace("_"," ") if s else "unknown"
            ci = r.get("check_in") or "?"
            co = r.get("check_out") or "?"
            return {"tool_result": r, "reply": f"Booking {booking_id}: **{s_human}**\n- Check-in: {ci}\n- Check-out: {co}\n\nWould you like to ask anything else or end this chat session? (say 'end' to close)"}

        db_row = await get_successful_booking_status(str(booking_id))
        if db_row:
            s = str(db_row.get("status") or "confirmed")
            s_human = s.replace("_", " ")
            ci = db_row.get("check_in") or "?"
            co = db_row.get("check_out") or "?"
            return {
                "tool_result": {"ok": True, "source": "successful_bookings", "booking_id": booking_id, **db_row},
                "reply": f"Booking {booking_id}: **{s_human}**\n- Check-in: {ci}\n- Check-out: {co}\n\nWould you like to ask anything else or end this chat session? (say 'end' to close)",
            }

        return {"tool_result": r, "reply": f"Sorry—{r.get('error','unable to find that booking')}."}
    # Update flow (explicit request)
    if action in set(nlp_fb.status_check_in_actions):
        new_status = "checked_in"
    elif action in set(nlp_fb.status_check_out_actions):
        new_status = "checked_out"
    else: new_status=args.get("new_status")
    if not new_status:
        return {"reply":"Do you want to check in or check out? If you only want to know status, say 'status'."}
    current=(args.get("current_status") or "pending")
    r = await update_booking_status(booking_id=booking_id, current_status=current, new_status=new_status)
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
    user_tl = (user_text or "").lower().strip()

    def _city_list_text() -> str:
        cities = get_available_cities()
        if not cities:
            return "San Diego, New York, Miami"
        # wrap into lines of 5 cities for readability
        lines = []
        for i in range(0, len(cities), 5):
            lines.append(", ".join(cities[i:i + 5]))
        return "\n".join(lines)

    def _unavailable_city_reply() -> str:
        return (
            "Unfortunately, there's no availability for that city or location right now. "
            "Would you like to see which cities are available? (yes/no)"
        )

    # Step 1: follow-up after unavailable city prompt
    if clean_filters.get(SK.awaiting_unavailable_city_choice):
        if _is_yes(user_text):
            extracted[SK.awaiting_unavailable_city_choice] = False
            extracted[SK.awaiting_city_selection] = True
            return {
                "results": [],
                "filters": extracted,
                "reply": (
                    "Here are all available cities where properties are listed:\n\n"
                    f"{_city_list_text()}\n\n"
                    "Please pick one city from this list."
                ),
            }
        if _is_no(user_text):
            extracted.pop(SK.awaiting_unavailable_city_choice, None)
            extracted.pop(SK.awaiting_city_selection, None)
            extracted.pop(SK.awaiting_property_type_choice, None)
            return {
                "results": [],
                "filters": extracted,
                "reply": "Bye! Thanks for visiting. Have a nice day.",
                "tool_result": {"ok": True, "end": True},
            }
        return {
            "results": [],
            "filters": extracted,
            "reply": "Please reply with yes or no. Would you like to see available cities?",
        }

    # Step 2: user is selecting a city from shown list
    if clean_filters.get(SK.awaiting_city_selection):
        if _is_no(user_text) or any(
            p in user_tl for p in _nlp_fallback().unavailable_city_decline_phrases
        ):
            extracted.pop(SK.awaiting_city_selection, None)
            extracted.pop(SK.awaiting_property_type_choice, None)
            return {
                "results": [],
                "filters": extracted,
                "reply": "Bye! Thanks for visiting. We will try to reach your city soon.",
                "tool_result": {"ok": True, "end": True},
            }
        city_pick = extracted.get("location") or extracted.get("city")
        if not city_pick:
            return {
                "results": [],
                "filters": extracted,
                "reply": (
                    "I couldn't match that city. Please choose one city from this list:\n\n"
                    f"{_city_list_text()}"
                ),
            }
        extracted["location"] = city_pick
        extracted["city"] = city_pick
        extracted[SK.awaiting_city_selection] = False
        extracted[SK.awaiting_property_type_choice] = True
        return {
            "results": [],
            "filters": extracted,
            "reply": (
                "Great choice. What property type would you like in that city "
                "(loft, apartment, condo, house, studio, villa, townhouse, or any)?"
            ),
        }

    # Step 3: ask property type after city selection, then search
    if clean_filters.get(SK.awaiting_property_type_choice):
        if not prop_type and not any(p in user_tl for p in _nlp_fallback().property_type_any_phrases):
            return {
                "results": [],
                "filters": extracted,
                "reply": (
                    "Please share a property type (loft, apartment, condo, house, studio, villa, townhouse) "
                    "or say 'any'."
                ),
            }
        extracted[SK.awaiting_property_type_choice] = False
        if prop_type:
            extracted["property_type"] = prop_type

    requested_city = extracted.get("location") or extracted.get("city")

    # Detect unknown city mentions like "find apartment in lisbon" when city isn't in dataset.
    if not requested_city:
        from .nlp_extractor import KNOWN_CITIES, CITY_ALIASES
        candidate = None
        nlp_fb = _nlp_fallback()
        m = re.search(nlp_fb.city_candidate_prefix_pattern, user_tl) if nlp_fb.city_candidate_prefix_pattern else None
        if m:
            raw = m.group(1).strip(" .,-")
            if nlp_fb.city_candidate_split_pattern:
                raw = re.split(nlp_fb.city_candidate_split_pattern, raw)[0].strip()
            words = [w for w in raw.split() if w]
            if 1 <= len(words) <= 3:
                bad = set(nlp_fb.city_candidate_block_words)
                if not any(w in bad for w in words):
                    cand = " ".join(words)
                    if cand not in KNOWN_CITIES and cand not in CITY_ALIASES:
                        candidate = cand

        if candidate:
            extracted["requested_unavailable_city"] = candidate
            extracted[SK.awaiting_unavailable_city_choice] = True
            extracted.pop(SK.awaiting_city_selection, None)
            extracted.pop(SK.awaiting_property_type_choice, None)
            return {"results": [], "reply": _unavailable_city_reply(), "filters": extracted}

    enhanced = f"{prop_type} {user_text}" if prop_type else user_text

    # Try Rust gateway for property search and fall back to Python search.
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
            inner = rust_result.get("result", rust_result)
            rust_results = inner.get("results", [])
            if isinstance(rust_results, list):
                results = rust_results
                logger.info("RUST property search returned %d results via gateway", len(results))
    except Exception as e:
        logger.warning("RUST property search offload failed: %s, using Python fallback", e)

    if results is None:
        results = property_search(
            query_text=enhanced,
            budget=extracted.get("budget"),
            amenities=extracted.get("amenities"),
            location=requested_city,
            beds=extracted.get("beds"),
            property_type=prop_type,
        )

    # If city is known/requested but no listings, trigger unavailable-city flow.
    if requested_city and not results:
        extracted[SK.awaiting_unavailable_city_choice] = True
        return {"results": [], "reply": _unavailable_city_reply(), "filters": extracted}

    max_display = min(len(results), 15)
    index_map = {i + 1: r.get("id") for i, r in enumerate(results[:max_display])}
    extracted.update({
        "results_index_map": index_map,
        "last_results": results[:max_display],
        SK.awaiting_unavailable_city_choice: False,
        SK.awaiting_city_selection: False,
        SK.awaiting_property_type_choice: False,
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

