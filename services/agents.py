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
from .dynamic_config import get_routing_policies, get_vocabulary
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


def _build_llm_router_system_prompt(context: Dict[str, Any]) -> str:
    """Build the intent-router system prompt from dynamic state and vocabulary."""
    base_system = (
        "You are an intent classifier for a hotel booking chatbot. "
        "Classify the user's message into ONE intent. Return strict JSON: {intent, confidence, brief_reason}. "
        "Intent must be one of: greeting, faq, confirmation, property_search, booking, "
        "status_update, payment_link, handoff, availability, end.\n\n"
        "CRITICAL RULES:\n"
        "- 'faq' = user is asking about rules, policy, refund, cancellation, pets, smoking, check-in time, "
        "wifi password, security deposit, amenities, payment methods, how to pay, accepted payments, or any property/platform question. "
        "Classify as 'faq' EVEN IF the user is mid-booking. A policy question always overrides booking context.\n"
        "- 'payment_link' = user is actively ready to pay for an existing booking (e.g., 'send me the link', 'I am ready to pay'). Do NOT use for general questions about accepted payment methods.\n"
        "- 'confirmation' = user is selecting a numbered option, providing booking details "
        "(name, phone, email, dates, guests), or affirming/declining a booking step "
        "(yes/no in booking context).\n"
        "- If the message is only a number and there is NO active property list in context, "
        "do NOT classify it as 'confirmation'.\n"
        "- 'property_search' = user is looking for a place or asking about properties.\n"
        "- 'greeting' = ONLY pure greetings like 'hi', 'hello', 'hey', 'good morning' with NO other intent.\n"
        "- If the message contains ANY reference to selecting an option number, it is 'confirmation', NOT 'greeting'.\n"
        "- Words like 'sure', 'ok', 'go ahead' combined with option/number references = 'confirmation'.\n"
    )

    prompt_sections = [base_system]

    state_lines = [
        "[ACTIVE STATE]",
        f"- has_selected_property: {bool(context.get('has_selected_property'))}",
        f"- has_booking_progress: {bool(context.get('has_booking_progress'))}",
        f"- has_last_results: {bool(context.get('has_last_results'))}",
        f"- last_results_count: {int(context.get('last_results_count') or 0)}",
        f"- awaiting_field: {context.get(SK.awaiting_field) or 'none'}",
        f"- receipt_shown: {bool(context.get(SK.receipt_shown))}",
    ]
    prompt_sections.append("\n".join(state_lines))

    property_types = [pt for pt in get_vocabulary().seed_property_types if pt]
    if property_types:
        prompt_sections.append(
            "Valid property types in our database: " + ", ".join(property_types) + "."
        )

    if context.get("has_last_results"):
        count = int(context.get("last_results_count") or 0)
        prompt_sections.append(
            "[CRITICAL STATE OVERRIDE]: You just showed the user a numbered list of "
            f"{count} properties. If their message is a number (for example '1' or '{count}') "
            "or an ordinal, they are making a selection. You MUST classify this intent strictly "
            "as 'confirmation'. Do NOT classify it as 'property_search'."
        )

    awaiting_field = context.get(SK.awaiting_field)
    if awaiting_field:
        prompt_sections.append(
            "[CRITICAL STATE OVERRIDE]: You are currently awaiting the user to provide "
            f"their '{awaiting_field}'. Treat their input as a 'confirmation' of this data."
        )

    return "\n\n".join(prompt_sections)


def _llm_route_intent(user_text: str, filters: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Use LLM structured output to classify intent with minimal hardcoded rules."""
    if not (OPENAI_API_KEY and LLM_STRUCTURED and SOFT_INTENT_ROUTER):
        return None

    text = (user_text or "").strip()
    if not text:
        return None

    active_filters = filters or {}
    context = {
        "has_selected_property": bool(active_filters.get(SK.selected_property)),
        "has_booking_progress": any(active_filters.get(field) for field in config.REQUIRED_FIELDS),
        "has_last_results": bool(active_filters.get("last_results")),
        "last_results_count": len(active_filters.get("last_results") or []),
        SK.awaiting_field: active_filters.get(SK.awaiting_field),
        SK.receipt_shown: bool(active_filters.get(SK.receipt_shown)),
    }
    system_prompt = _build_llm_router_system_prompt(context)

    payload: Dict[str, Any] = {
        "model": OPENAI_CHAT_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
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


def _apply_contextual_triage_policies(
    user_text: str,
    filters: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Evaluate soft-coded routing rules that rely only on current text + state."""
    active_filters = filters or {}
    awaiting_field = active_filters.get(SK.awaiting_field)
    has_cardinal = nlp_engine.has_cardinal_extraction(user_text or "")
    no_selected_property = not active_filters.get(SK.selected_property)
    no_awaiting_field = not active_filters.get(SK.awaiting_field)

    for policy in get_routing_policies().sorted_policies:
        cond = policy.condition

        # Pre-LLM guard only evaluates conditions that do not depend on a prior intent.
        if (
            cond.intent is not None
            or cond.intent_in is not None
            or cond.always
            or cond.is_safe is not None
            or cond.has_booking_context is not None
            or cond.has_field_data is not None
            or cond.lacks_explicit_status_keywords is not None
            or cond.no_active_selection is not None
        ):
            continue

        match = True
        if cond.filter_key is not None and not active_filters.get(cond.filter_key):
            match = False
        if cond.any_filter_key is not None and not any(active_filters.get(k) for k in cond.any_filter_key):
            match = False
        if cond.has_context_key is not None and not active_filters.get(cond.has_context_key):
            match = False
        if cond.lacks_context_key is not None and active_filters.get(cond.lacks_context_key):
            match = False
        if cond.awaiting_field_in is not None and awaiting_field not in cond.awaiting_field_in:
            match = False
        if cond.requires_cardinal_extraction is not None and cond.requires_cardinal_extraction != has_cardinal:
            match = False
        if cond.no_selected_property is not None and cond.no_selected_property != no_selected_property:
            match = False
        if cond.no_awaiting_field is not None and cond.no_awaiting_field != no_awaiting_field:
            match = False

        if match and policy.route not in {"_from_intent", "_faq_return"}:
            return policy.route

    return None


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
# Intent helpers ,  NLP-powered (delegates to nlp_engine)
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
# Selection & slot helpers ,  NLP-powered
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



def _is_contextual_guest_input(user_text: str, filters: Optional[Dict[str, Any]] = None) -> bool:
    tl = (user_text or "").lower()
    if (filters or {}).get(SK.awaiting_field) == "guests":
        return True
    return any(term in tl for term in _nlp_fallback().guest_context_terms)


def triage_intent(user_text: str, filters: Optional[Dict[str, Any]] = None) -> str:
    t = user_text or ""
    active_filters = filters or {}
    tl = t.lower().strip()

    # 1. --- IRONCLAD SELECTION GUARD ---
    # 1. --- IRONCLAD SELECTION GUARD ---
    if active_filters.get("last_results"):
        if nlp_engine.has_cardinal_extraction(t):
            return "confirmation"
        # SHIELD: Catch explicit selection verbs dynamically from vocabulary.yaml
        if any(w in tl for w in _nlp_fallback().selection_explicit_verbs):
            return "confirmation"
        # SHIELD: Catch if the user copy-pasted the property title instead of typing the number!
        for prop in active_filters["last_results"]:
            title = prop.get("title", "").lower()
            if title and len(title) > 4 and title in tl:
                return "confirmation"

    # 2. ðŸ›‘ GLOBAL CONTEXT PRESERVER SHIELD (Soft-Coded via NLP Engine) ðŸ›‘
    # Dynamically prevents short acknowledgments, greetings, or 2-letter gibberish from wiping active flows.
    is_useless_or_greeting = _is_ack(t) or _is_greeting(t) or len(re.sub(r"[^\w]", "", tl)) <= 2
    
    if is_useless_or_greeting:
        if active_filters.get(SK.awaiting_field) == "booking_id":
            return "status_update"
        if active_filters.get(SK.awaiting_field) or active_filters.get(SK.awaiting_selection_confirm) or active_filters.get(SK.receipt_shown) or active_filters.get(SK.awaiting_post_mod_choice):
            return "confirmation"
        if active_filters.get(SK.awaiting_city_selection) or active_filters.get(SK.awaiting_property_type_choice) or active_filters.get(SK.awaiting_unavailable_city_choice):
            return "property_search"
        # If not in an active flow and it's a greeting, process as greeting
        if _is_greeting(t):
            return "greeting"

    # Keep strict deterministic guard for end.
    if _is_end(t):
        return "end"

    # 3. 🛡️ IDENTITY SHIELD 🛡️
    # Soft-coded check for queries about the bot's identity.
    if any(phrase in tl for phrase in _nlp_fallback().identity_phrases):
        return "greeting"

    # If user is currently deciding on a selected property...
    if active_filters.get(SK.awaiting_selection_confirm):
        if (
            _is_yes(t)
            or _is_no(t)
            or nlp_engine.has_cardinal_extraction(t)
            or nlp_engine.wants_previous_results_sync(tl)
            or nlp_engine.wants_property_search_request(tl)
        ):
            return "confirmation"

    # Master Shield for Booking IDs (UUIDs)
    if re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", user_text):
        return "status_update"
        
    # 1. Master Shield for Active Cancellations
    # Deterministic check ensures cancellation actions bypass the FAQ entirely.
    if ("cancel" in tl or "delete" in tl) and "policy" not in tl:
        return "status_update"

    # 2. ðŸ›‘ SHIELD: Generic Location Inquiry (Soft-Coded) ðŸ›‘
    # If the user asks about the 'city' category generally, route to search.
    if re.search(r"\b(cit(y|ies)|location|where)\b", tl):
        return "property_search"

    # 🛑 SUPER SHIELD: FAQ / Policy Breakout 🛑
    # Runs before input fields so users can ask questions mid-booking.
    try:
        faq_hit = nlp_engine.detect_faq_intent(t)
    except Exception:
        faq_hit = any(w in tl for w in _vocab().faq_fallback_keywords)

    if faq_hit:
        return "faq"

    # Master Shield for Input Fields
    _conf_fields = {"name", "phone", "email", "guests", "check_in", "check_out", "modification_choice", "modification"}
    if active_filters.get(SK.awaiting_field) in _conf_fields:
        return "confirmation"

    # Master Shield for Receipt
    if active_filters.get(SK.receipt_shown) and (_is_yes(t) or _is_no(t)):
        return "confirmation"

    policy_route = _apply_contextual_triage_policies(t, active_filters)
    if policy_route:
        return policy_route

    # Soft-coded LLM intent router (prioritized as requested by user to be super soft-coded).
    llm_intent = _llm_route_intent(t, filters)
    if llm_intent:
        return llm_intent

    # Status checks come after FAQ so policy questions like "refund/check-in policy"
    # are not hijacked by status routing.
    if _is_status_query(t):
        tl = t.lower()
        if not any(p in tl for p in _nlp_fallback().status_resume_phrases):
            return "status_update"

    # NLP-driven and keyword fallback routing (secondary pass ,  catches LLM misses).
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
        (_parse_guests(t) is not None and _is_contextual_guest_input(t, active_filters))):
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
        price_txt = f" ,  about ${price}/night" if price is not None else ""
        numbered.append(f"{i}. {title} ,  {city}{price_txt}")

    if not OPENAI_API_KEY:
        if results:
            total_count = len(results)
            shown_count = len(numbered)
            header = f"Yes, found {total_count} options"
            if total_count > shown_count:
                header += f", here are the top {shown_count}:"
            else:
                header += ":"
            return f"{header}\n\n" + "\n".join(numbered) + "\n\nReply with your desired option."
        kf = known_filters or {}
        if not (kf.get("location") or kf.get("city")): 
            return "No match yet, what city should I search in (and a nightly budget)?"
        if not kf.get("budget"): 
            return "No match yet, what's your target nightly budget (approx)?"
        return "No match yet, any preferred dates or must-have amenities?"

    total_count = len(results)
    shown_count = len(numbered)
    
    system = ("You are a warm, concise vacation-rental concierge. Use ONLY provided JSON results and the provided numbered list. "
            "Think step by step: First identify which properties match the user's needs, then present them clearly. "
            f"Keep replies very short. Language: {locale}")
            
    header_instruction = f"Start: 'Yes, found {total_count} options.'"
    if total_count > shown_count:
        header_instruction = f"Start: 'Yes, found {total_count} options, here are the top {shown_count}.'"
            
    style = (f"If results:\n- {header_instruction}\n"
           "- Show ONLY the provided numbered list.\n- End: 'Reply with your desired option.'\n"
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
        header = f"Yes,found {total_count} options"
        if total_count > shown_count:
            header += f", here are the top {shown_count}:"
        else:
            header += ":"
        return f"{header}\n\n" + "\n".join(numbered) + "\n\nReply with your desired option."
    
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

    s = f"**{title}** ,  {city}\n"
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
    return f"**{title}** ,  {city}\n- Bedrooms: {bedrooms}\n- Price: ${price}/night"


def _coerce_name_when_awaited(user_text: str, parsed_name: Optional[str]) -> Optional[str]:
    if parsed_name:
        return parsed_name

    candidate = (user_text or "").strip()
    if not candidate:
        return None

    name_invalid_chars = set(_nlp_fallback().parse_name_invalid_chars)
    if len(candidate) < 2 or any(char in candidate for char in name_invalid_chars):
        return None

    disallowed = set(_nlp_fallback().parse_name_disallowed_words)
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]*", candidate)
    candidate_lower = candidate.lower()
    if (
        1 <= len(words) <= 3
        and not any(word.lower() in disallowed for word in words)
        and not any(
            phrase in candidate_lower
            for phrase in _nlp_fallback().parse_name_disallowed_phrases
        )
    ):
        return candidate
    return None

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
    hint = f" (noted: {', '.join(parts)})" if parts else ""
    reply_msg = f"Hi there! I'm your AI Hotel Concierge{hint}. I can help you search for properties, check availability, book your stay, and answer policy questions. How can I help you today?"

    return {"reply": reply_msg, "filters": clean_filters}

async def _confirmation_agent_impl(user_text: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Selection + slot fill -> receipt -> final yes/no
    - shows a full property card on selection
    - keeps confirmation state transitions delegated to helpers
    """
    original_filters = filters or {}
    persisted = {**original_filters}

    if persisted.get(SK.awaiting_field) in {
        "modification",
        "modification_choice",
        "check_in",
        "check_out",
        "guests",
        "name",
        "phone",
        "email",
    } or persisted.get(SK.modifying_dates):
        persisted.pop(SK.awaiting_post_cancel_choice, None)

    response = confirmation_helpers.handle_final_confirmation(
        user_text,
        persisted,
        _is_yes,
        _is_no,
    )
    if response:
        return response

    if persisted.get(SK.receipt_shown) and nlp_engine.is_receipt_request(user_text):
        return {
            "reply": _render_receipt(persisted),
            "tool_result": {"ok": False, "need": ["final_confirmation"], "show_receipt": True},
            "filters": persisted,
        }

    parsed_dates = _parse_dates(user_text) or []

    response = confirmation_helpers.handle_post_modification_choice(user_text, persisted)
    if response:
        return response

    response = confirmation_helpers.handle_restart_search_request(
        user_text,
        persisted,
        parsed_dates=parsed_dates,
        wants_property_search_request=_wants_property_search_request,
    )
    if response:
        return response

    sel = _parse_selection_index(user_text)
    
    # 🛠️ FIX: Smart Keyword Matching for copy-pasted property names!
    if sel is None and persisted.get("last_results"):
        user_lower = user_text.lower().strip()
        
        for i, prop in enumerate(persisted["last_results"], start=1):
            title = prop.get("title", "").lower()
            
            # Extract main keywords from the property title (e.g., "4br", "apartment")
            key_words = [w for w in re.findall(r'\b\w+\b', title) if len(w) >= 3]
            
            # If the user's text contains at least 2 of these keywords, it's a match!
            matches = sum(1 for w in key_words if w in user_lower)
            if matches >= 2 or (title and title in user_lower):
                sel = i
                break

    awaiting_field = (persisted.get(SK.awaiting_field) or "").strip()
    if awaiting_field and awaiting_field not in {"modification", "modification_choice"}:
        sel = None
    if awaiting_field == "guests" and re.match(r"^\s*\d+\s*$", user_text.strip()):
        sel = None

    response = confirmation_helpers.handle_property_selection(
        user_text,
        persisted,
        sel,
        _format_property_full,
    )
    if response:
        return response

    response = confirmation_helpers.handle_selection_confirm(
        user_text,
        persisted,
        _is_yes,
        _is_no,
        wants_previous_results=nlp_engine.wants_previous_results_sync,
        wants_property_search_request=_wants_property_search_request,
    )
    if response:
        return response

    requested_fields = _detect_requested_fields(user_text)
    response = confirmation_helpers.handle_post_cancel_choice(
        user_text,
        persisted,
        requested_fields=requested_fields,
        wants_property_search_request=_wants_property_search_request,
        wants_modification=_wants_modification,
    )
    if response:
        return response

    if persisted.get(SK.awaiting_selection_confirm):
        parsed_name = None
        parsed_phone = None
        parsed_email = None
        parsed_guests = None
        parsed_dates_for_fields: List[str] = []
    else:
        parsed_name = _parse_name(user_text)
        parsed_phone = _parse_phone(user_text)
        parsed_email = _parse_email(user_text)
        parsed_guests = _parse_guests(user_text)
        parsed_dates_for_fields = parsed_dates

    resolved_name = _coerce_name_when_awaited(user_text, parsed_name)
    original_awaited = (persisted.get(SK.awaiting_field) or "").strip()
    is_answering_prompt = (
        (original_awaited in {"check_in", "check_out"} and bool(parsed_dates_for_fields))
        or (
            original_awaited == "guests"
            and (parsed_guests is not None or user_text.strip().isdigit())
        )
        or (original_awaited == "name" and bool(resolved_name))
        or (original_awaited == "phone" and bool(parsed_phone))
        or (original_awaited == "email" and bool(parsed_email))
    )

    if original_awaited in {"modification", "modification_choice"}:
        return confirmation_helpers.route_requested_modifications(
            persisted,
            requested_fields,
            parsed_name=parsed_name,
            parsed_phone=parsed_phone,
            parsed_email=parsed_email,
            parsed_guests=parsed_guests,
            parsed_dates=parsed_dates_for_fields,
            allow_inline_apply=True,
        )

    capture = confirmation_helpers.capture_awaited_field(
        persisted,
        user_text,
        parsed_name=parsed_name,
        parsed_phone=parsed_phone,
        parsed_email=parsed_email,
        parsed_guests=parsed_guests,
        parsed_dates=parsed_dates_for_fields,
        resolve_name_candidate=lambda text: _coerce_name_when_awaited(text, parsed_name),
    )
    if capture["response"]:
        return capture["response"]

    just_applied_field_update = bool(capture["updated"])

    if parsed_name and not persisted.get("name"):
        persisted["name"] = parsed_name
    if parsed_phone and not persisted.get("phone"):
        persisted["phone"] = parsed_phone
    if parsed_email and not persisted.get("email"):
        persisted["email"] = parsed_email

    if parsed_guests is not None and not persisted.get("guests"):
        guest_context_terms = _nlp_fallback().guest_context_terms
        if original_awaited == "guests" or any(
            term in user_text.lower() for term in guest_context_terms
        ):
            try:
                persisted["guests"] = int(parsed_guests)
            except Exception:
                persisted["guests"] = parsed_guests

    if parsed_dates_for_fields:
        if not persisted.get("check_in"):
            normalized_check_in = confirmation_helpers.normalize_date_value(
                parsed_dates_for_fields[0]
            )
            if normalized_check_in:
                persisted["check_in"] = normalized_check_in
        if len(parsed_dates_for_fields) >= 2 and not persisted.get("check_out"):
            normalized_check_out = confirmation_helpers.normalize_date_value(
                parsed_dates_for_fields[1]
            )
            if normalized_check_out:
                persisted["check_out"] = normalized_check_out

    if _wants_modification(user_text) and not is_answering_prompt:
        return confirmation_helpers.route_requested_modifications(
            persisted,
            requested_fields,
            parsed_name=parsed_name,
            parsed_phone=parsed_phone,
            parsed_email=parsed_email,
            parsed_guests=parsed_guests,
            parsed_dates=parsed_dates_for_fields,
            allow_inline_apply=True,
        )

    response = confirmation_helpers.handle_inline_receipt_updates(
        persisted,
        parsed_name=parsed_name,
        parsed_phone=parsed_phone,
        parsed_email=parsed_email,
        parsed_guests=parsed_guests,
        parsed_dates=parsed_dates_for_fields,
    )
    if response:
        return response

    if persisted.get(SK.receipt_shown) and just_applied_field_update:
        return confirmation_helpers._render_updated_receipt(persisted)

    current_awaited = (persisted.get(SK.awaiting_field) or "").strip()
    if current_awaited and not just_applied_field_update:
        if current_awaited in {"check_in", "check_out", "guests", "name", "phone", "email"}:
            if persisted.get(current_awaited):
                return confirmation_helpers._ask_for_modification_field(
                    current_awaited,
                    persisted,
                )
            return confirmation_helpers._ask_for_field(current_awaited, persisted)

    if not persisted.get(SK.receipt_shown) and just_applied_field_update:
        required = config.REQUIRED_FIELDS + [SK.selected_property]
        if all(persisted.get(key) for key in required):
            return confirmation_helpers._try_show_receipt(persisted)

    if (
        _is_no(user_text)
        and not persisted.get(SK.awaiting_selection_confirm)
        and not persisted.get(SK.awaiting_field)
        and not persisted.get(SK.receipt_shown)
        and not persisted.get(SK.awaiting_post_mod_choice)
    ):
        persisted[SK.awaiting_post_cancel_choice] = True
        return {
            "reply": (
                "No problem, the booking has been cancelled. Would you like to "
                "search for different properties or modify your requirements?"
            ),
            "filters": persisted,
            "tool_result": {"ok": False, "cancelled": True, "need": ["clarification"]},
        }

    missing_field = confirmation_helpers._next_missing_field(persisted)
    if missing_field:
        persisted[SK.awaiting_selection_confirm] = False
        return confirmation_helpers._ask_for_field(missing_field, persisted)

    if not persisted.get(SK.receipt_shown):
        return confirmation_helpers._try_show_receipt(persisted)

    return {
        "reply": (
            "I want to make sure I get this right. Please reply with a clear "
            "'yes' to confirm and proceed with payment, or 'no' to cancel and modify."
        ),
        "filters": persisted,
    }


async def confirmation_agent(user_text: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    return await _confirmation_agent_impl(user_text, filters)

def faq_agent(user_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced FAQ agent that uses semantic search on policy documents"""
    # Try the enhanced FAQ system first
    try:
        from .faq_enhanced import enhanced_faq_agent

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
        if "property_id" in missing:
            return {
                "reply": "It looks like you don't have an active booking in progress right now. Would you like me to help you find a property?",
                "filters": args
            }
        else:
            field_name = missing[0].replace("_", " ")
            args[SK.awaiting_field] = missing[0]
            return {
                "reply": f"We are almost done with your booking! I just need your {field_name}.",
                "filters": args
            }

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Pre-validate via Rust BookingValidatorTool (TOON protocol)
    # If the gateway is unreachable, skip validation and proceed
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        from .rust_client import validate_booking

        rust_validation = await validate_booking(
            property_id=args.get("property_id", ""),
            check_in=args.get("check_in", ""),
            check_out=args.get("check_out", ""),
            guests=int(args.get("guests", 1)),
            email=args.get("email", "")
        )

        if rust_validation and not rust_validation.get("fallback"):
            # Unwrap: the gateway wraps in {ok, result, ...}
            inner = rust_validation.get("result", rust_validation) or {}
            is_valid = inner.get("valid", True)
            errors = inner.get("errors", [])
            warnings = inner.get("warnings", [])

            if not is_valid:
                error_list = "\n".join(f"â€¢ {e}" for e in errors)
                warn_text = ""
                if warnings:
                    warn_text = "\n\nâš ï¸ Warnings:\n" + "\n".join(f"â€¢ {w}" for w in warnings)
                return {
                    "reply": f"âŒ **Booking validation failed:**\n\n{error_list}{warn_text}\n\nPlease correct the above and try again.",
                    "tool_result": {"ok": False, "need": ["correction"], "validation_errors": errors, "warnings": warnings},
                }
            # Validation passed ,  log any warnings
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
        msg=f"""ðŸŽ‰ **Booking Confirmed!**

âœ… Your **{ptype}** has been successfully booked!

**Booking Details**
- Booking ID: {r.get('booking_id','N/A')}
- Property: {prop_title or 'Property'}
- Check-in: {args.get('check_in')}
- Check-out: {args.get('check_out')}
- Guests: {args.get('guests',1)}

ðŸ“§ A payment link has been sent to your email/WhatsApp.

**Thank you for booking with us! Have a wonderful stay!** ðŸŒŸ
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
            price_txt = f" ,  about ${int(price)}/night" if price else ""
            prop_desc = f"{prop_title} ,  {city}{price_txt}".strip()
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
      - If booking_id missing â†’ ask only for booking ID
      - If user asks status/check-in/check-out â†’ fetch and answer
      - If user asks to update to check-in/checkout â†’ perform update
    """
    action=(args.get("action") or "").lower()
    nlp_fb = _nlp_fallback()
    booking_id=args.get("booking_id") or _parse_booking_id(user_text)

    # If we don't have an id yet, request it clearly
    if not booking_id:
        # Inject the awaiting flag into the filters so triage_intent can preserve context
        filters = {**args, SK.awaiting_field: "booking_id"}
        if "cancel" in user_text.lower():
            return {"reply": "I can help you cancel that. Please provide your Booking ID.", "filters": filters}
        return {"reply":"Please provide your Booking ID (e.g., 57015107-d414-409c-843e-b6a6b15d9b59).", "filters": filters}
    else:
        # Clear the flag once we have the ID
        args[SK.awaiting_field] = None

    # ðŸ›‘ MASTER SHIELD: Intercept Cancellation Requests ðŸ›‘
    if "cancel" in user_text.lower() or "delete" in user_text.lower():
        from .booking import delete_booking
        res = await delete_booking(booking_id)
        if res.get("ok"):
            return {"tool_result": {"ok": True, "deleted": True}, "reply": f"âœ… Booking `{booking_id}` has been successfully cancelled and completely removed from our database."}
        else:
            return {"tool_result": {"ok": False}, "reply": f"Sorry, I couldn't delete the booking: {res.get('error')}"}

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

        return {"tool_result": r, "reply": f"Sorry, {r.get('error','unable to find that booking')}."}
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
    requested_city = extracted.get("city") or extracted.get("location")

    # ðŸ›‘ FIX: Define these helpers at the TOP so they are available to the shields below ðŸ›‘
    def _city_list_text() -> str:
        cities = get_available_cities()
        if not cities:
            return "San Diego, New York, Miami"
        lines = []
        for i in range(0, len(cities), 5):
            lines.append(", ".join(cities[i:i + 5]))
        return "\n".join(lines)

    def _unavailable_city_reply() -> str:
        return (
            "Unfortunately, there's no availability for that city or location right now. "
            "Would you like to see which cities are available? (yes/no)"
        )

    # ðŸ›‘ SOFT-CODED: Generic City List Request ðŸ›‘
    # (Now this can safely call _city_list_text!)
    if re.search(r"\b(city|cities|location|where)\b", user_tl) and not requested_city:
        return {
            "results": [],
            "filters": extracted,
            "reply": (
                "We have properties available in many locations! "
                "Here are all the cities where you can currently book:\n\n"
                f"{_city_list_text()}\n\n"
                "Which city would you like to search in?"
            ),
        }

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
        
        # ðŸ›‘ SHIELD: If user already provided the property type, skip the prompt and search! ðŸ›‘
        if prop_type or any(p in user_tl for p in _nlp_fallback().property_type_any_phrases):
            if prop_type:
                extracted["property_type"] = prop_type
            extracted[SK.awaiting_property_type_choice] = False
            # (By not returning here, the code naturally falls through to execute the search!)
        else:
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

        if rust_result and not rust_result.get("fallback"):
            inner = rust_result.get("result", rust_result) or {}
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

    # ðŸ›‘ INTENSELY SOFT-CODED SHIELD: Vector Store Semantic Mismatch ðŸ›‘
    if results and not prop_type:
        user_tl = (user_text or "").lower()
        # Extract the actual property types the vector store retrieved
        retrieved_types = {str(r.get("property_type", "")).lower() for r in results[:5] if r.get("property_type")}
        
        # If the vector store substituted an unavailable word (like 'resort') for available types
        # the retrieved types won't exist in the user's prompt. We catch this dynamically!
        if retrieved_types and not any(rt in user_tl for rt in retrieved_types):
            import services.config as config
            available_list = ", ".join(p.title() for p in sorted(config.SEED_PROPERTY_TYPES))
            extracted[SK.awaiting_property_type_choice] = True
            
            reply = (
                f"To help me find the perfect match, could you specify the type of property you are looking for? "
                f"Our available private rentals include: **{available_list}**.\n\n"
                f"Which of these would you prefer?"
            )
            return {"results": [], "reply": reply, "filters": extracted}

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

    stream_cb = filters.get("stream_callback") if callable(filters.get("stream_callback")) else None
    stream = stream_cb is not None

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
    return {"reply": f"Okay ,  I'll connect you with a human specialist{f' about {city.title()}' if city else ''}. "
                     "Please share your email or phone number and a preferred time.",
            "tool_result":{"handoff":True}}


