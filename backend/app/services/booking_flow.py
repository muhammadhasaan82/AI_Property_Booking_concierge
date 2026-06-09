from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.agents.state.booking_state import (
    clear_awaiting_field,
    compute_missing_booking_fields,
    get_booking_state,
    set_awaiting_field,
)
from app.agents.status_codes import Source, Status
from app.agents.tools.helpers import _coerce_float, _coerce_int
from app.agents.tools.search import get_all_available_cities, return_to_previous_results
from app.agents.tools.support import check_faq
from app.config.agent_config_loader import cfg
from app.services.observability.langfuse_observer import get_observer, summarize_booking_state
from app.config.booking_schema_loader import (
    get_ask_order,
    get_field_aliases,
    get_field_display_name,
    get_field_prompt,
    get_required_fields,
    get_required_numeric_fields,
    next_field_to_ask,
    validate_field,
)

logger = logging.getLogger(__name__)

_ACTIVE_BOOKING_STAGES = {
    "collecting_details",
    "awaiting_confirmation",
    "awaiting_modification_choice",
    "modifying_details",
    "awaiting_property_reselection",
}
_FAQ_KEYWORDS = (
    "pet",
    "pets",
    "parking",
    "cancellation",
    "refund",
    "policy",
    "policies",
    "check-in time",
    "check in time",
    "check-out time",
    "check out time",
    "smoking",
    "wifi",
    "amenities",
    "payment",
    "accessibility",
    "allowed",
)
_CHECK_IN_DATE_PATTERN = r"\b(check[- ]?in(?: date)?|arrival(?: date)?)\b"
_CHECK_OUT_DATE_PATTERN = r"\b(check[- ]?out(?: date)?|checkout(?: date)?|departure(?: date)?)\b"
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _format_money(value: Any) -> str:
    amount = _coerce_float(value) or 0.0
    return f"${amount:,.2f}"


def _format_display_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, cfg.date_format)
    except Exception:
        return value
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"


def _field_alias_map() -> Dict[str, List[str]]:
    aliases = get_field_aliases()
    merged: Dict[str, List[str]] = {}
    for field in set(get_required_fields() + get_required_numeric_fields()):
        values = [field]
        values.extend(aliases.get(field, []))
        merged[field] = [alias for alias in values if alias]
    return merged


def _property_title_from_soft_state(soft_state: Dict[str, Any]) -> str:
    selected = soft_state.get("booking_selected_property")
    if isinstance(selected, dict):
        title = selected.get("title")
        if title:
            return str(title)
    state = soft_state.get("booking_state")
    if isinstance(state, dict):
        title = state.get("property_title")
        if title:
            return str(title)
    return "this property"


def _property_id_matches(item: Any, selected_id: str) -> bool:
    if not isinstance(item, dict):
        return False
    for key in ("id", "property_id"):
        value = item.get(key)
        if value is not None and str(value) == selected_id:
            return True
    return False


def _resolve_selected_property_from_soft_state(
    soft_state: Dict[str, Any],
    selected_id: str,
) -> Dict[str, Any]:
    for key in ("booking_selected_property",):
        candidate = soft_state.get(key)
        if _property_id_matches(candidate, selected_id):
            return dict(candidate)

    for state_key in ("visible_results", "all_search_results"):
        items = soft_state.get(state_key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if _property_id_matches(item, selected_id):
                return dict(item)

    last_search = soft_state.get("last_search")
    if isinstance(last_search, dict):
        for item in last_search.get("properties", []) or []:
            if _property_id_matches(item, selected_id):
                return dict(item)

    option_map = soft_state.get("active_property_options_map")
    if isinstance(option_map, dict):
        for item in option_map.values():
            if _property_id_matches(item, selected_id):
                resolved = dict(item)
                resolved.setdefault("id", resolved.get("property_id"))
                return resolved

    return {}


def _friendly_prompt_for_field(field: str) -> str:
    prompt = get_field_prompt(field)
    if prompt:
        return prompt
    return f"Please provide {get_field_display_name(field).lower()}."


def _details_request_reply(property_title: str, missing_fields: List[str]) -> str:
    labels = [
        f"- {get_field_display_name(field)}"
        for field in missing_fields
    ]
    template = str(getattr(cfg, "booking_details_request_prompt_template", "") or "").strip()
    if not template:
        return ""
    return template.format(
        property_title=property_title,
        missing_fields_bullets="\n".join(labels),
    )


def _review_reply(summary: Dict[str, Any]) -> str:
    template = str(getattr(cfg, "booking_review_prompt_template", "") or "").strip()
    if not template:
        return ""
    return template.format(
        property_title=summary.get("property_title") or summary.get("property") or "Property",
        guest_name=summary.get("guest_name") or "",
        guest_email=summary.get("guest_email") or "",
        guest_phone=summary.get("guest_phone") or "",
        check_in=_format_display_date(str(summary.get("check_in") or "")),
        check_out=_format_display_date(str(summary.get("check_out") or "")),
        guests=summary.get("guests") or "",
        nights=summary.get("nights") or "",
        price_per_night=_format_money(summary.get("price_per_night")),
        total=_format_money(summary.get("total")),
    )


def _receipt_reply(receipt: Dict[str, Any]) -> str:
    template = str(getattr(cfg, "booking_receipt_prompt_template", "") or "").strip()
    if not template:
        return ""
    return template.format(
        registration_id=receipt.get("booking_id") or "",
        property_title=receipt.get("property_title") or "",
        guest_name=receipt.get("guest_name") or "",
        guest_email=receipt.get("guest_email") or "",
        guest_phone=receipt.get("guest_phone") or "",
        check_in=_format_display_date(str(receipt.get("check_in") or "")),
        check_out=_format_display_date(str(receipt.get("check_out") or "")),
        guests=receipt.get("guests") or "",
        nights=receipt.get("nights") or "",
        price_per_night=_format_money(receipt.get("price_per_night")),
        total=_format_money(receipt.get("total_amount")),
        booking_status=receipt.get("status") or cfg.booking_confirmed_status,
    )

BOOKING_ID_PATTERN = re.compile(r"BK-\d{8}-[A-Z0-9]+", re.I)

_BOOKING_STATUS_INTENT_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"\bmy\s+booking\s+id\s+is\b",
        r"\bbooking\s+id\b",
        r"\bcheck\s+my\s+booking\s+status\b",
        r"\bcheck\s+booking\s+status\b",
        r"\bmy\s+booking\s+status\b",
        r"\bregistration\s+id\b",
        r"\bmy\s+registration\s+id\s+is\b",
        r"\bi\s+want\s+to\s+check\s+my\s+booking\s+status\b",
        r"\bstatus\s+of\s+my\s+booking\b",
        r"\bbooking\s+status\b",
    ]
]


def _detect_booking_status_intent(message: str) -> bool:
    """
    Deterministically detect whether the user message expresses a booking-status intent.

    Matches the normalized message against a configured set of regex patterns
    (e.g. "my booking ID is", "check my booking status", "registration ID").
    Also returns True when a booking ID token (BK-YYYYMMDD-XXXXXXXX) is present
    in the message even without an explicit intent phrase.

    Parameters:
        message (str): The raw user message.

    Returns:
        bool: True if the message is recognised as a booking-status request.
    """
    if not message:
        return False
    normalized = _normalize(message)
    if not normalized:
        return False
    for pattern in _BOOKING_STATUS_INTENT_PATTERNS:
        if pattern.search(normalized):
            return True
    if BOOKING_ID_PATTERN.search(message):
        return True
    return False


def _extract_booking_id(message: str) -> Optional[str]:
    """
    Extract the first booking/registration ID token (BK-YYYYMMDD-XXXXXXXX) from a message.

    Parameters:
        message (str): The raw user message to scan.

    Returns:
        Optional[str]: The matched booking ID string, or None if no match is found.
    """
    if not message:
        return None
    match = BOOKING_ID_PATTERN.search(message)
    return match.group(0).upper() if match else None


def _render_booking_status_reply(receipt: Dict[str, Any]) -> str:
    """
    Render a deterministic booking status reply from a receipt dictionary.

    Uses the configured `booking_status_prompt_template` from agent_config.yaml.
    Falls back to the receipt template if the status template is not configured.

    Parameters:
        receipt (Dict[str, Any]): Booking receipt dict with keys: booking_id,
            property_title, guest_name, guest_email, guest_phone, check_in,
            check_out, guests, nights, price_per_night, total_amount, status.

    Returns:
        str: The rendered status reply string.
    """
    template = str(
        getattr(cfg, "booking_status_prompt_template", "") or ""
    ).strip()
    if not template:
        return _receipt_reply(receipt)
    return template.format(
        registration_id=receipt.get("booking_id") or "",
        property_title=receipt.get("property_title") or "",
        guest_name=receipt.get("guest_name") or "",
        guest_email=receipt.get("guest_email") or "",
        guest_phone=receipt.get("guest_phone") or "",
        check_in=_format_display_date(str(receipt.get("check_in") or "")),
        check_out=_format_display_date(str(receipt.get("check_out") or "")),
        guests=receipt.get("guests") or "",
        nights=receipt.get("nights") or "",
        price_per_night=_format_money(receipt.get("price_per_night")),
        total_amount=_format_money(receipt.get("total_amount")),
        booking_status=receipt.get("status") or cfg.booking_confirmed_status,
    )


def _lookup_booking_in_session(
    booking_id: str,
    soft_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Search for a booking receipt matching `booking_id` in the current session soft_state.

    Lookup order (returns the first match):
      A. soft_state.booking_receipt (single receipt dict)
      B. soft_state.booking_receipts (list/history of receipts)
      C. soft_state top-level booking_review / pending_booking (compatibility snapshot)

    Parameters:
        booking_id (str): The booking ID to search for (case-insensitive match).
        soft_state (Dict[str, Any]): Current session soft_state dictionary.

    Returns:
        Optional[Dict[str, Any]]: The matching receipt dict, or None if not found.
    """
    if not booking_id or not isinstance(soft_state, dict):
        return None

    target = booking_id.strip().upper()

    receipt = soft_state.get("booking_receipt")
    if isinstance(receipt, dict):
        rid = str(receipt.get("booking_id") or "").strip().upper()
        if rid == target:
            return dict(receipt)

    receipts_list = soft_state.get("booking_receipts")
    if isinstance(receipts_list, list):
        for entry in receipts_list:
            if not isinstance(entry, dict):
                continue
            rid = str(entry.get("booking_id") or "").strip().upper()
            if rid == target:
                return dict(entry)

    for key in ("booking_review", "pending_booking"):
        candidate = soft_state.get(key)
        if not isinstance(candidate, dict):
            continue
        rid = str(candidate.get("booking_id") or candidate.get("registration_id") or "").strip().upper()
        if rid == target:
            receipt = dict(candidate)
            receipt.setdefault("booking_id", rid)
            receipt.setdefault("status", soft_state.get("booking_status") or cfg.booking_confirmed_status)
            receipt.setdefault("property_title", candidate.get("property_title") or candidate.get("property"))
            receipt.setdefault("total_amount", candidate.get("total"))
            return receipt

    return None


def _latest_session_receipt(soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return the most recent booking receipt available in the session soft_state.

    Checks soft_state.booking_receipt first, then the last entry of
    soft_state.booking_receipts if present as a non-empty list.

    Parameters:
        soft_state (Dict[str, Any]): Current session soft_state.

    Returns:
        Optional[Dict[str, Any]]: The latest receipt dict, or None if none found.
    """
    if not isinstance(soft_state, dict):
        return None

    receipt = soft_state.get("booking_receipt")
    if isinstance(receipt, dict):
        return dict(receipt)

    receipts_list = soft_state.get("booking_receipts")
    if isinstance(receipts_list, list) and receipts_list:
        last = receipts_list[-1]
        if isinstance(last, dict):
            return dict(last)

    return None


async def handle_booking_status_check(
    message: str,
    soft_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Deterministic booking-status handler — checks session soft_state before falling back to DB/ADK.

    Behaviour:
      1. Detects booking-status intent in the message.
      2. Extracts a booking ID (BK-YYYYMMDD-XXXXXXXX) if present.
      3. If ID is present and found in session → returns deterministic status.
      4. If ID is present but NOT found in session → attempts DB lookup;
         if DB also fails → returns not-found message.
      5. If no ID but session has a booking_receipt → returns latest booking status.
      6. If no ID and no receipt → asks user for booking ID.
      7. If booking-status intent is NOT detected → returns None (let pipeline continue).

    Parameters:
        message (str): The user's raw message.
        soft_state (Dict[str, Any]): Current session soft_state (mutable dict from Redis snapshot).

    Returns:
        Optional[Dict[str, Any]]: A payload with `deterministic_reply` when handled,
        or None when booking-status intent is not detected (pass-through).
    """
    if not _detect_booking_status_intent(message):
        return None

    booking_id = _extract_booking_id(message)

    if booking_id:
        receipt = _lookup_booking_in_session(booking_id, soft_state)
        if receipt:
            return {
                "status": Status.FOUND,
                "receipt": receipt,
                "deterministic_reply": _render_booking_status_reply(receipt),
            }

        try:
            from app.services.booking import get_booking_status

            db_result = await get_booking_status(booking_id)
            if db_result.get("ok"):
                merged = {
                    "booking_id": booking_id,
                    "status": db_result.get("status") or cfg.booking_confirmed_status,
                    "check_in": db_result.get("check_in") or "",
                    "check_out": db_result.get("check_out") or "",
                }
                return {
                    "status": Status.FOUND,
                    "receipt": merged,
                    "deterministic_reply": _render_booking_status_reply(merged),
                }
        except Exception as exc:
            logger.debug("[booking_status] DB lookup failed for %s: %s", booking_id, exc)

        not_found_template = str(
            getattr(cfg, "booking_status_not_found_template", "") or ""
        ).strip()
        if not not_found_template:
            not_found_template = "It looks like your booking wasn't found in our system."
        return {
            "status": Status.BOOKING_NOT_FOUND,
            "deterministic_reply": not_found_template,
        }

    latest = _latest_session_receipt(soft_state)
    if latest:
        return {
            "status": Status.FOUND,
            "receipt": latest,
            "deterministic_reply": _render_booking_status_reply(latest),
        }

    ask_template = str(
        getattr(cfg, "booking_status_ask_for_id_template", "") or ""
    ).strip()
    if not ask_template:
        ask_template = (
            "I'd be happy to check your booking status. "
            "Could you share your registration ID? "
            "It looks like BK-YYYYMMDD-XXXXXXXX."
        )
    return {
        "status": Status.GATHERING_INFO,
        "deterministic_reply": ask_template,
    }


def _is_booking_faq(message: str) -> bool:
    normalized = _normalize(message)
    if not normalized:
        return False
    if any(keyword in normalized for keyword in _FAQ_KEYWORDS):
        return True
    if "?" in message and not re.search(r"\b(email|phone|guest|guests|check[- ]?in|check[- ]?out|name)\b", normalized):
        return True
    return False


def _seed_property_fields(soft_state: Dict[str, Any], state: Dict[str, Any]) -> None:
    property_id = soft_state.get("booking_property_id") or soft_state.get("last_selected_property_id")
    selected = soft_state.get("booking_selected_property")
    if property_id:
        state["property_id"] = str(property_id)
    if isinstance(selected, dict):
        if selected.get("title"):
            state["property_title"] = str(selected.get("title"))
        if selected.get("price_per_night") is not None:
            state["price_per_night"] = _coerce_float(selected.get("price_per_night"))


def _parse_natural_date(text: str) -> Optional[str]:
    """
    Parse an ISO or common natural-language date from `text` and return it formatted using `cfg.date_format`.
    
    Searches for an ISO `YYYY-M-D` token or natural forms like "1st of January, 2025" and "January 1, 2025". If a valid date is found it is returned as a string formatted with `cfg.date_format`; otherwise `None` is returned.
    
    Parameters:
        text (str): Input text to scan for a date.
    
    Returns:
        Optional[str]: A date string formatted per `cfg.date_format` if parsing succeeds, `None` otherwise.
    """
    if not text:
        return None
    cleaned = re.sub(r"[\n\r]", " ", text).strip(" ,.;:")
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", cleaned)
    if iso:
        try:
            parsed = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            return parsed.strftime(cfg.date_format)
        except ValueError:
            return None

    patterns = [
        re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(?P<month>[a-z]+)\s*,?\s*(?P<year>\d{4})\b", re.I),
        re.compile(r"\b(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<year>\d{4})\b", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(cleaned)
        if not match:
            continue
        month_raw = str(match.group("month")).lower()
        month = _MONTHS.get(month_raw)
        if month is None:
            continue
        try:
            parsed = date(int(match.group("year")), month, int(match.group("day")))
            return parsed.strftime(cfg.date_format)
        except ValueError:
            return None
    return None


def _find_all_dates(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract all dates mentioned in `text` and produce normalized date strings paired with their character spans.
    
    Parameters:
        text (str): Input text to scan for dates. An empty or falsy value yields an empty list.
    
    Returns:
        List[Tuple[str, int, int]]: A list of tuples `(date_str, start_index, end_index)` where `date_str` is formatted using `cfg.date_format`, and `start_index`/`end_index` are the character span of the matched date in the original text. The results are ordered by earliest start position and filtered so matches do not overlap. Recognizes ISO `YYYY-M-D` and common natural-language forms such as "1st of January, 2025" and "January 1, 2025".
    """
    if not text:
        return []
    results = []
    
    iso_pat = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
    for m in iso_pat.finditer(text):
        try:
            parsed = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            results.append((parsed.strftime(cfg.date_format), m.start(), m.end()))
        except ValueError:
            pass

    pat1 = re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(?P<month>[a-z]+)\s*,?\s*(?P<year>\d{4})\b", re.I)
    for m in pat1.finditer(text):
        month_raw = m.group("month").lower()
        month = _MONTHS.get(month_raw)
        if month is not None:
            try:
                parsed = date(int(m.group("year")), month, int(m.group("day")))
                results.append((parsed.strftime(cfg.date_format), m.start(), m.end()))
            except ValueError:
                pass

    pat2 = re.compile(r"\b(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<year>\d{4})\b", re.I)
    for m in pat2.finditer(text):
        month_raw = m.group("month").lower()
        month = _MONTHS.get(month_raw)
        if month is not None:
            try:
                parsed = date(int(m.group("year")), month, int(m.group("day")))
                results.append((parsed.strftime(cfg.date_format), m.start(), m.end()))
            except ValueError:
                pass

    results.sort(key=lambda x: (x[1], -(x[2] - x[1])))
    filtered = []
    last_end = -1
    for p_date, start, end in results:
        if start >= last_end:
            filtered.append((p_date, start, end))
            last_end = end
    return filtered


def _extract_dates_by_association(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Associate one or two date mentions in `text` with `check_in` and `check_out`.
    
    Attempts to identify date tokens in the input and assign them to check-in and check-out by proximity to explicit check-in/check-out phrases; when two or more dates are present it will prefer the closest matches and fall back to using the first two dates. If no dates are found, both return values are None.
    
    Parameters:
        text (str): Input text containing zero or more date mentions.
    
    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple `(check_in_date, check_out_date)` where each element is a normalized date string (as produced by the module's date-extraction helpers) or `None` when no appropriate date could be associated.
    """
    dates_found = _find_all_dates(text)
    if not dates_found:
        return None, None
        
    check_in_matches = list(re.finditer(_CHECK_IN_DATE_PATTERN, text, re.I))
    check_out_matches = list(re.finditer(_CHECK_OUT_DATE_PATTERN, text, re.I))
    
    check_in_date = None
    check_out_date = None
    
    if len(dates_found) == 1:
        d_val, d_start, d_end = dates_found[0]
        best_type = None
        min_dist = 999999
        for m in check_in_matches:
            if m.end() <= d_start:
                dist = d_start - m.end()
            else:
                dist = (m.start() - d_end) * 5
            if dist < min_dist:
                min_dist = dist
                best_type = "check_in"
        for m in check_out_matches:
            if m.end() <= d_start:
                dist = d_start - m.end()
            else:
                dist = (m.start() - d_end) * 5
            if dist < min_dist:
                min_dist = dist
                best_type = "check_out"
        if best_type == "check_in":
            check_in_date = d_val
        elif best_type == "check_out":
            check_out_date = d_val
        return check_in_date, check_out_date

    for d_val, d_start, d_end in dates_found:
        best_type = None
        min_dist = 999999
        for m in check_in_matches:
            if m.end() <= d_start:
                dist = d_start - m.end()
            else:
                dist = (m.start() - d_end) * 5
            if dist < min_dist:
                min_dist = dist
                best_type = "check_in"
        for m in check_out_matches:
            if m.end() <= d_start:
                dist = d_start - m.end()
            else:
                dist = (m.start() - d_end) * 5
            if dist < min_dist:
                min_dist = dist
                best_type = "check_out"
        
        if best_type == "check_in" and not check_in_date:
            check_in_date = d_val
        elif best_type == "check_out" and not check_out_date:
            check_out_date = d_val

    if len(dates_found) >= 2:
        if not check_in_date and not check_out_date:
            check_in_date = dates_found[0][0]
            check_out_date = dates_found[1][0]
        elif check_in_date and not check_out_date:
            for d_val, _, _ in dates_found:
                if d_val != check_in_date:
                    check_out_date = d_val
                    break
        elif check_out_date and not check_in_date:
            for d_val, _, _ in dates_found:
                if d_val != check_out_date:
                    check_in_date = d_val
                    break
                    
    return check_in_date, check_out_date


def _field_segments(message: str) -> Dict[str, str]:
    """
    Extracts text segments following detected field mentions in the message.
    
    Searches for whole-word, case-insensitive aliases for known booking fields (from _field_alias_map()). For each field found (first match only) returns the substring from the end of that match up to the start of the next detected field (or the end of the message), with surrounding punctuation and whitespace removed.
    
    Parameters:
    	message (str): The user message to scan for field aliases.
    
    Returns:
    	Dict[str, str]: Mapping from canonical field name to the extracted text segment (may be an empty string if no text follows the match).
    """
    normalized_message = message or ""
    alias_map = _field_alias_map()
    matches: List[Tuple[int, int, str]] = []
    for field, aliases in alias_map.items():
        for alias in aliases:
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.I)
            match = pattern.search(normalized_message)
            if match:
                matches.append((match.start(), match.end(), field))
                break
    matches.sort(key=lambda item: item[0])
    segments: Dict[str, str] = {}
    for index, (_start, end, field) in enumerate(matches):
        next_start = matches[index + 1][0] if index + 1 < len(matches) else len(normalized_message)
        segments[field] = normalized_message[end:next_start].strip(" ,.;:-")
    return segments


def _extract_email(text: str) -> Optional[str]:
    match = re.search(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", text, re.I)
    return match.group(1).strip() if match else None


def _extract_phone(text: str) -> Optional[str]:
    keyword_match = re.search(
        r"(?:phone|mobile|contact number|phone number|number)\s*(?:is|to|:)?\s*([+()0-9][0-9+\-\s()]{6,})",
        text,
        re.I,
    )
    candidate = keyword_match.group(1) if keyword_match else None
    if candidate is None:
        fallback = re.findall(r"[+()0-9][0-9+\-\s()]{6,}", text)
        if len(fallback) == 1:
            candidate = fallback[0]
    if candidate is None:
        return None
    stripped = candidate.strip()
    digits = re.sub(r"\D", "", stripped)
    return stripped if len(digits) >= 7 else None


def _extract_guests(text: str) -> Optional[int]:
    patterns = (
        r"\b(?:we are|for|around|approximately|about)?\s*(\d+)\s+(?:guest|guests|people|persons)\b",
        r"\bguests?\s*(?:is|are|:)?\s*(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return _coerce_int(match.group(1))
    return None


def _extract_name(text: str) -> Optional[str]:
    match = re.search(
        r"(?:my full name is|full name is|my name is|name is)\s+([a-z][a-z .'-]+)",
        text,
        re.I,
    )
    if not match:
        return None
    value = match.group(1).strip(" ,.;:")
    return value if len(value) >= 2 else None


def _extract_name_for_field(text: str) -> Optional[str]:
    patterns = (
        r"(?:my full name is|full name is|my name is|name is|change name to|update name to|modify name to|name to)\s+([a-z][a-z .'-]+)",
        r"^(?:it is|it's|its)\s+([a-z][a-z .'-]+)$",
        r"^([a-z][a-z .'-]+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        value = match.group(1).strip(" ,.;:")
        if len(value) >= 2:
            return value
    return None


def _extract_updates_from_message(
    message: str,
    awaiting_field: Optional[str],
) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    """
    Extract booking-related field values from a user's message and identify which fields were mentioned or failed to parse.
    
    Parses the input text for guest_name, guest_email, guest_phone, guests (count), check_in, check_out, and property_id. For check-in/check-out the function recognizes explicit date labels in the text and uses proximity-based association when explicit labels are not present. If an awaiting_field is provided, the function will attempt field-specific extraction from the full message as a fallback. Fields that are mentioned but could not be parsed are reported in the errors mapping with a user-facing prompt.
    
    Parameters:
        message (str): The user's raw message text.
        awaiting_field (Optional[str]): A single field name the system is currently awaiting (e.g., "guest_name", "check_in"); used to drive targeted extraction/fallback parsing.
    
    Returns:
        Tuple[Dict[str, Any], List[str], Dict[str, str]]:
            - updates: Mapping of fields to extracted values for fields successfully parsed.
            - mentioned_fields: Ordered list of unique field names that were detected or targeted (preserves first-seen order).
            - errors: Mapping of field names to user-friendly prompt strings for fields that were referenced but could not be parsed.
    """
    updates: Dict[str, Any] = {}
    mentioned_fields: List[str] = []
    errors: Dict[str, str] = {}
    segments = _field_segments(message)
    normalized = message or ""

    if _extract_name(normalized):
        updates["guest_name"] = _extract_name(normalized)
        mentioned_fields.append("guest_name")
    elif "guest_name" in segments:
        mentioned_fields.append("guest_name")

    email = _extract_email(normalized)
    if email:
        updates["guest_email"] = email
        mentioned_fields.append("guest_email")
    elif "guest_email" in segments:
        mentioned_fields.append("guest_email")

    phone = _extract_phone(normalized)
    if phone:
        updates["guest_phone"] = phone
        mentioned_fields.append("guest_phone")
    elif "guest_phone" in segments:
        mentioned_fields.append("guest_phone")

    guests = _extract_guests(normalized)
    if guests is not None:
        updates["guests"] = guests
        mentioned_fields.append("guests")
    elif "guests" in segments:
        mentioned_fields.append("guests")

    assoc_check_in, assoc_check_out = _extract_dates_by_association(normalized)
    check_in_segment = segments.get("check_in")
    check_out_segment = segments.get("check_out")
    has_check_in_phrase = "check_in" in segments or bool(re.search(_CHECK_IN_DATE_PATTERN, normalized, re.I))
    has_check_out_phrase = "check_out" in segments or bool(re.search(_CHECK_OUT_DATE_PATTERN, normalized, re.I))
    explicit_date_labels_present = has_check_in_phrase or has_check_out_phrase

    if has_check_in_phrase:
        mentioned_fields.append("check_in")
        explicit_check_in = _parse_natural_date(check_in_segment or "")
        if explicit_check_in:
            updates["check_in"] = explicit_check_in
        elif "check_in" not in segments and assoc_check_in:
            updates["check_in"] = assoc_check_in
        else:
            errors["check_in"] = _friendly_prompt_for_field("check_in")

    if has_check_out_phrase:
        mentioned_fields.append("check_out")
        explicit_check_out = _parse_natural_date(check_out_segment or "")
        if explicit_check_out:
            updates["check_out"] = explicit_check_out
        elif "check_out" not in segments and assoc_check_out:
            updates["check_out"] = assoc_check_out
        else:
            errors["check_out"] = _friendly_prompt_for_field("check_out")

    if has_check_in_phrase and has_check_out_phrase:
        if "check_in" not in updates or "check_out" not in updates:
            all_dates = _find_all_dates(normalized)
            if len(all_dates) >= 2:
                updates["check_in"] = all_dates[0][0]
                updates["check_out"] = all_dates[1][0]
                errors.pop("check_in", None)
                errors.pop("check_out", None)

    if awaiting_field and awaiting_field not in updates:
        mentioned_fields.append(awaiting_field)
        if awaiting_field == "guest_name":
            guest_name = _extract_name_for_field(normalized)
            if guest_name:
                updates[awaiting_field] = guest_name
        elif awaiting_field == "guest_email":
            email = _extract_email(normalized)
            if email:
                updates[awaiting_field] = email
        elif awaiting_field == "guest_phone":
            phone = _extract_phone(normalized)
            if phone:
                updates[awaiting_field] = phone
        elif awaiting_field == "guests":
            guest_count = _extract_guests(normalized)
            if guest_count is None:
                bare_number = re.search(r"\b(\d+)\b", normalized)
                guest_count = _coerce_int(bare_number.group(1)) if bare_number else None
            if guest_count is not None:
                updates[awaiting_field] = guest_count
        elif awaiting_field in {"check_in", "check_out"}:
            if not explicit_date_labels_present:
                parsed = _parse_natural_date(normalized)
                if parsed:
                    updates[awaiting_field] = parsed
                else:
                    errors[awaiting_field] = _friendly_prompt_for_field(awaiting_field)
        elif awaiting_field == "property_id":
            updates[awaiting_field] = normalized.strip()
    return updates, list(dict.fromkeys(mentioned_fields)), errors


def _selected_property(soft_state: Dict[str, Any]) -> Dict[str, Any]:
    selected = soft_state.get("booking_selected_property")
    return dict(selected) if isinstance(selected, dict) else {}


def _replace_booking_state(soft_state: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    target = get_booking_state(soft_state)
    target.clear()
    target.update(state)
    soft_state["booking_state_updated_at"] = time.time()
    return target


def _occupancy_validation_message(occupancy_max: Any) -> str:
    template = str(getattr(cfg, "booking_validation_capacity_template", "") or "").strip()
    if not template:
        return ""
    return template.format(occupancy_max=occupancy_max)


def _checkout_validation_message(check_in_value: str) -> str:
    template = str(
        getattr(cfg, "booking_validation_checkout_after_checkin_template", "") or ""
    ).strip()
    if not template:
        return ""
    return template.format(check_in_display=_format_display_date(check_in_value))


def _next_missing_field(state: Dict[str, Any]) -> Optional[str]:
    missing = compute_missing_booking_fields(state)
    filtered = [field for field in missing if field in set(cfg.booking_details_request_fields)]
    if filtered:
        return next_field_to_ask(filtered)
    return next_field_to_ask(missing)


def _review_summary_from_state(soft_state: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    check_in = str(state.get("check_in") or "")
    check_out = str(state.get("check_out") or "")
    price_per_night = _coerce_float(state.get("price_per_night")) or 0.0
    check_in_dt = datetime.strptime(check_in, cfg.date_format)
    check_out_dt = datetime.strptime(check_out, cfg.date_format)
    nights = max((check_out_dt - check_in_dt).days, 1)
    total = round(nights * price_per_night, 2)
    property_title = str(state.get("property_title") or _property_title_from_soft_state(soft_state))
    return {
        "property": property_title,
        "property_title": property_title,
        "property_id": str(state.get("property_id") or ""),
        "guest_name": str(state.get("guest_name") or ""),
        "guest_email": str(state.get("guest_email") or ""),
        "guest_phone": str(state.get("guest_phone") or ""),
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "guests": _coerce_int(state.get("guests")) or 0,
        "price_per_night": price_per_night,
        "total": total,
    }


def _sync_review_state(soft_state: Dict[str, Any], summary: Dict[str, Any]) -> None:
    soft_state["booking_review"] = dict(summary)
    soft_state["pending_booking"] = dict(summary)
    soft_state["pending_booking_updated_at"] = time.time()


def _ensure_property_seeded(soft_state: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(get_booking_state(soft_state))
    _seed_property_fields(soft_state, state)
    return _replace_booking_state(soft_state, state)


def start_booking_for_selected_property(soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selected_id_raw = soft_state.get("last_selected_property_id")
    if selected_id_raw is None or not str(selected_id_raw).strip():
        return None

    selected_id = str(selected_id_raw).strip()
    selected_property = _resolve_selected_property_from_soft_state(soft_state, selected_id)
    soft_state["booking_property_id"] = selected_id
    if selected_property:
        soft_state["booking_selected_property"] = dict(selected_property)
    soft_state["booking_required_fields"] = list(cfg.booking_details_request_fields)

    state = _ensure_property_seeded(soft_state)
    property_title = str(state.get("property_title") or _property_title_from_soft_state(soft_state))
    validated_state, next_field, reply = _validate_and_commit_state(
        soft_state,
        updates={},
        mentioned_fields=[],
        extraction_errors={},
    )
    if next_field is None:
        clear_awaiting_field(soft_state)
        return _review_payload_from_state(soft_state, validated_state)

    observer = get_observer()
    trace = observer.trace(name="booking_flow")
    trace.update(metadata={
        "previous_booking_stage": soft_state.get("booking_stage"),
        "next_booking_stage": "collecting_details",
        "missing_fields": list(cfg.booking_details_request_fields),
    })
    trace.end()
    soft_state["booking_stage"] = "collecting_details"
    soft_state["last_presented_view"] = "booking_details_request"
    set_awaiting_field(soft_state, [next_field])
    collected_detail_fields = [
        field for field in cfg.booking_details_request_fields
        if validated_state.get(field) not in (None, "")
    ]
    return {
        "status": "booking_details_required",
        "property_id": selected_id,
        "property": soft_state.get("booking_selected_property"),
        "required_fields": list(cfg.booking_details_request_fields),
        "deterministic_reply": (
            reply
            if collected_detail_fields
            else _details_request_reply(property_title, list(cfg.booking_details_request_fields))
        ),
    }


def resume_booking_flow(soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    observer = get_observer()
    previous_stage = str(soft_state.get("booking_stage") or "").strip()
    
    with observer.trace(name="booking_flow", metadata={
        "previous_booking_stage": previous_stage,
        "soft_state_summary": summarize_booking_state(soft_state)
    }) as trace:
        stage = str(soft_state.get("booking_stage") or "").strip()
    if stage not in _ACTIVE_BOOKING_STAGES:
        return None
    state = _ensure_property_seeded(soft_state)
    if stage == "awaiting_confirmation":
        summary = soft_state.get("booking_review") or soft_state.get("pending_booking")
        if isinstance(summary, dict):
            return {
                "status": Status.REVIEW_PENDING,
                "summary": summary,
                "deterministic_reply": _review_reply(summary),
            }
    property_title = str(state.get("property_title") or _property_title_from_soft_state(soft_state))
    next_field = soft_state.get("awaiting_field")
    if isinstance(next_field, str) and next_field.strip():
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": [next_field],
            "deterministic_reply": _friendly_prompt_for_field(next_field),
        }
    missing_fields = [
        field
        for field in cfg.booking_details_request_fields
        if state.get(field) in (None, "")
    ]
    if missing_fields:
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": missing_fields,
            "deterministic_reply": _details_request_reply(property_title, missing_fields),
        }
    summary = _review_summary_from_state(soft_state, state)
    soft_state["booking_stage"] = "awaiting_confirmation"
    _sync_review_state(soft_state, summary)
    return {
        "status": Status.REVIEW_PENDING,
        "summary": summary,
        "deterministic_reply": _review_reply(summary),
    }


def _modification_field_from_message(message: str) -> Optional[str]:
    normalized = _normalize(message)
    alias_map = _field_alias_map()
    field_order = [
        "property_id",
        "guest_name",
        "guest_email",
        "guest_phone",
        "check_in",
        "check_out",
        "guests",
    ]
    for field in field_order:
        aliases = alias_map.get(field, [])
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return field
    return None


def handle_review_modification_request(message: str, soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    field = _modification_field_from_message(message)
    if field == "property_id":
        soft_state["booking_stage"] = "awaiting_property_reselection"
        soft_state["last_presented_view"] = "property_list"
        return return_to_previous_results(soft_state)

    if field:
        soft_state["booking_stage"] = "modifying_details"
        set_awaiting_field(soft_state, [field])
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": [field],
            "deterministic_reply": _friendly_prompt_for_field(field),
        }

    soft_state["booking_stage"] = "awaiting_modification_choice"
    clear_awaiting_field(soft_state)
    return {
        "status": Status.GATHERING_INFO,
        "deterministic_reply": str(getattr(cfg, "booking_modification_prompt_template", "") or "").strip(),
    }


async def confirm_booking_review(soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary = soft_state.get("booking_review") or soft_state.get("pending_booking")
    if not isinstance(summary, dict):
        state = _ensure_property_seeded(soft_state)
        summary = _review_summary_from_state(soft_state, state)

    registration_id = (
        f"{cfg.booking_registration_id_prefix}-{datetime.utcnow():%Y%m%d}-{uuid4().hex[:8].upper()}"
    )
    receipt = {
        "booking_id": registration_id,
        "property_title": summary.get("property_title") or summary.get("property"),
        "guest_name": summary.get("guest_name"),
        "guest_email": summary.get("guest_email"),
        "guest_phone": summary.get("guest_phone"),
        "check_in": summary.get("check_in"),
        "check_out": summary.get("check_out"),
        "nights": summary.get("nights"),
        "guests": summary.get("guests"),
        "price_per_night": summary.get("price_per_night"),
        "total_amount": summary.get("total"),
        "status": cfg.booking_confirmed_status,
    }

    try:
        from app.observability.db_logging import insert_successful_booking

        await insert_successful_booking(
            {
                "booking_id": registration_id,
                "user_name": summary.get("guest_name"),
                "user_email": summary.get("guest_email"),
                "user_phone": summary.get("guest_phone"),
                "property_title": summary.get("property_title") or summary.get("property"),
                "check_in": summary.get("check_in"),
                "check_out": summary.get("check_out"),
                "guests": summary.get("guests"),
                "nights": summary.get("nights"),
                "total_amount": summary.get("total"),
                "status": cfg.booking_confirmed_status,
                "source": cfg.booking_source_tag,
            }
        )
    except Exception as exc:
        logger.warning("[booking_flow] Could not persist confirmed booking: %s", exc)

    observer = get_observer()
    trace = observer.trace(name="booking_flow")
    trace.update(metadata={
        "previous_booking_stage": soft_state.get("booking_stage"),
        "next_booking_stage": "confirmed",
        "review_generated": True,
        "receipt_generated": True,
        "registration_id_present": bool(registration_id),
    })
    trace.end()
    soft_state["booking_stage"] = "confirmed"
    soft_state["booking_status"] = cfg.booking_confirmed_status
    soft_state["booking_registration_id"] = registration_id
    soft_state["booking_receipt"] = dict(receipt)
    soft_state["last_presented_view"] = "booking_receipt"
    clear_awaiting_field(soft_state)

    return {
        "status": Status.BOOKING_CONFIRMED,
        "receipt": receipt,
        "deterministic_reply": _receipt_reply(receipt),
    }


def _validate_and_commit_state(
    soft_state: Dict[str, Any],
    updates: Dict[str, Any],
    mentioned_fields: List[str],
    extraction_errors: Dict[str, str],
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    current_state = dict(get_booking_state(soft_state))
    _seed_property_fields(soft_state, current_state)

    candidate_state = dict(current_state)
    errors: Dict[str, str] = dict(extraction_errors)
    for field, value in updates.items():
        temp_state = dict(candidate_state)
        temp_state[field] = value
        ok, err = validate_field(field, value, current_state=temp_state)
        if ok:
            candidate_state[field] = value
        else:
            errors[field] = err or _friendly_prompt_for_field(field)

    guests_value = _coerce_int(candidate_state.get("guests"))
    selected_property = _selected_property(soft_state)
    occupancy_max = _coerce_int(selected_property.get("occupancy_max")) if isinstance(selected_property, dict) else None
    if occupancy_max is not None and guests_value is not None and guests_value > occupancy_max:
        candidate_state.pop("guests", None)
        errors["guests"] = _occupancy_validation_message(occupancy_max)

    if "check_in" in errors:
        candidate_state.pop("check_in", None)
        candidate_state.pop("check_out", None)
        errors.pop("check_out", None)
    else:
        check_in = candidate_state.get("check_in") or updates.get("check_in")
        check_out = candidate_state.get("check_out") or updates.get("check_out")
        if check_in and check_out:
            try:
                check_in_dt = datetime.strptime(str(check_in), cfg.date_format)
                check_out_dt = datetime.strptime(str(check_out), cfg.date_format)
                if check_out_dt <= check_in_dt:
                    candidate_state.pop("check_out", None)
                    errors["check_out"] = (
                        _checkout_validation_message(str(check_in))
                        or "Check-out must be after check-in."
                    )
            except Exception:
                pass

    committed = _replace_booking_state(soft_state, candidate_state)
    missing_field = _next_missing_field(committed)
    error_field = None
    if errors:
        if "check_in" in errors:
            error_field = "check_in"
        elif "check_out" in errors:
            error_field = "check_out"
        else:
            for field in get_ask_order():
                if field in errors:
                    error_field = field
                    break
            if error_field is None:
                error_field = next(iter(errors))

    if error_field:
        set_awaiting_field(soft_state, [error_field])
        return committed, error_field, errors[error_field]

    if missing_field:
        set_awaiting_field(soft_state, [missing_field])
        return committed, missing_field, _friendly_prompt_for_field(missing_field)

    clear_awaiting_field(soft_state)
    return committed, None, None


def _review_payload_from_state(soft_state: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    summary = _review_summary_from_state(soft_state, state)
    soft_state["booking_stage"] = "awaiting_confirmation"
    soft_state["last_presented_view"] = "booking_review"
    _sync_review_state(soft_state, summary)
    return {
        "status": Status.REVIEW_PENDING,
        "summary": summary,
        "deterministic_reply": _review_reply(summary),
    }


async def handle_active_booking_turn(
    message: str,
    soft_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    observer = get_observer()
    previous_stage = str(soft_state.get("booking_stage") or "").strip()
    
    with observer.trace(name="booking_flow", metadata={
        "previous_booking_stage": previous_stage,
        "soft_state_summary": summarize_booking_state(soft_state)
    }) as trace:
        stage = str(soft_state.get("booking_stage") or "").strip()
    if stage not in _ACTIVE_BOOKING_STAGES:
        return None

    if _is_booking_faq(message):
        tool_context = SimpleNamespace(state={"soft_state": soft_state})
        faq_payload = await check_faq(question=message, tool_context=tool_context)
        answer = str(faq_payload.get("answer") or "").strip()
        resume_prompt = str(getattr(cfg, "booking_faq_resume_prompt", "") or "").strip()
        if answer and resume_prompt:
            faq_payload["deterministic_reply"] = f"{answer}\n\n{resume_prompt}"
        elif answer:
            faq_payload["deterministic_reply"] = answer
        return faq_payload

    if stage in {"awaiting_confirmation", "awaiting_modification_choice"}:
        normalized = _normalize(message)
        should_interpret_as_modification = (
            stage == "awaiting_modification_choice"
            or re.search(r"\b(change|update|modify|edit)\b", normalized) is not None
        )
        if should_interpret_as_modification:
            field = _modification_field_from_message(message)
            if field == "property_id":
                return handle_review_modification_request(message, soft_state)
            if field:
                inline_updates, _mentioned_fields, inline_errors = _extract_updates_from_message(
                    message,
                    field,
                )
                soft_state["booking_stage"] = "modifying_details"
                set_awaiting_field(soft_state, [field])
                if field in inline_updates and field not in inline_errors:
                    stage = "modifying_details"
                else:
                    return {
                        "status": Status.GATHERING_INFO,
                        "missing_fields": [field],
                        "deterministic_reply": _friendly_prompt_for_field(field),
                    }
            elif stage == "awaiting_modification_choice":
                return {
                    "status": Status.GATHERING_INFO,
                    "deterministic_reply": str(getattr(cfg, "booking_modification_prompt_template", "") or "").strip(),
                }
            else:
                return handle_review_modification_request(message, soft_state)
        elif stage == "awaiting_confirmation":
            return None

    if stage == "awaiting_property_reselection":
        return {
            "status": Status.GATHERING_INFO,
            "deterministic_reply": "Please choose a different property from the current results.",
        }

    awaiting_field = soft_state.get("awaiting_field")
    awaiting_field_name = str(awaiting_field).strip() if isinstance(awaiting_field, str) else None
    updates, mentioned_fields, extraction_errors = _extract_updates_from_message(
        message,
        awaiting_field_name,
    )
    state, next_field, reply = _validate_and_commit_state(
        soft_state,
        updates=updates,
        mentioned_fields=mentioned_fields,
        extraction_errors=extraction_errors,
    )

    if next_field:
        soft_state["booking_stage"] = "modifying_details" if stage in {"modifying_details", "awaiting_modification_choice"} else "collecting_details"
        soft_state["last_presented_view"] = "booking_details_request"
        return {
            "status": Status.GATHERING_INFO,
            "missing_fields": [next_field],
            "deterministic_reply": reply,
        }

    return _review_payload_from_state(soft_state, state)


def list_available_cities_payload() -> Dict[str, Any]:
    payload = get_all_available_cities()
    cities = payload.get("cities") or []
    if not isinstance(cities, list):
        cities = []
    city_text = ", ".join(str(city) for city in cities)
    payload["deterministic_reply"] = (
        f"Our service is currently available in: {city_text}."
        if city_text
        else "I couldn't load the available cities right now."
    )
    return payload
