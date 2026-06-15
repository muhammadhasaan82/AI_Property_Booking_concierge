from __future__ import annotations
from app.services.booking.constants import BOOKING_ID_PATTERN
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.config.agent_config_loader import cfg
from app.config.booking_schema_loader import get_ask_order
from app.agents.tools.helpers import _coerce_int
from app.services.booking.constants import (
    _AMENDMENT_NAME_VALUE_PATTERNS,
    _CHECK_IN_DATE_PATTERN,
    _CHECK_OUT_DATE_PATTERN,
    _extract_booking_id,
)
from app.services.booking.date_parser import (
    _extract_dates_by_association,
    _find_all_dates,
    _parse_natural_date,
    _parse_single_structured_date,
)
from app.services.booking.formatting import _friendly_prompt_for_field
from app.services.booking.state import _field_alias_map
import re

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
    # Prefer explicit labels first so date years like "2026\nGuests: 5"
    # are not misread as "2026 guests".
    patterns = (
        r"\bguests?\s*(?:is|are|to|:)?\s*(\d+)\b",
        r"\b(?:we are|for|around|approximately|about)\s+(?!\d{4}\b)(\d+)\s+(?:guest|guests|people|persons)\b",
        r"\b(?!\d{4}\b)(\d+)\s+(?:guest|guests|people|persons)\b",
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

def _field_type_from_schema(field: str) -> str:
    from app.config.booking_schema_loader import booking_schema
    spec = booking_schema.booking.validators.get(field)
    if not spec:
        return "text"
    if spec.type == "date":
        return "date"
    if spec.type in ("integer", "float"):
        return "number"
    if spec.type == "regex":
        if "email" in field.lower() or (spec.pattern and "@" in spec.pattern):
            return "email"
        if "phone" in field.lower() or "mobile" in field.lower():
            return "phone"
    return "text"

def _is_valid_candidate_for_field(field: str, val: str) -> bool:
    if not val:
        return False
    val_str = val.strip().strip("'\"")
    if not val_str:
        return False
        
    f_type = _field_type_from_schema(field)
    if f_type == "email":
        return "@" in val_str and "." in val_str
    elif f_type == "phone":
        digits = re.sub(r"\D", "", val_str)
        if len(digits) < 5:
            return False
        return not bool(re.search(r"[a-zA-Z]", val_str))
    elif f_type == "date":
        structured_match = re.search(
            r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
            val_str,
        )
        if structured_match:
            return _parse_single_structured_date(structured_match.group(0)) is not None
        parsed_nat = _parse_natural_date(val_str)
        return parsed_nat is not None
    elif f_type == "number":
        try:
            float(val_str)
            return True
        except ValueError:
            return False
    elif f_type == "text":
        if "@" in val_str:
            return False
        digits = re.sub(r"\D", "", val_str)
        if len(digits) >= 7 and not re.search(r"[a-zA-Z]", val_str):
            return False
        if _parse_single_structured_date(val_str) or _parse_natural_date(val_str):
            return False
        if val_str.isdigit():
            return False
        if val_str.lower() in ("yes", "no", "y", "n", "correct", "sure", "ok", "okay"):
            return False
        return len(val_str) >= 2
    return True

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

    # Accept multiline label-value input, e.g.:
    # Full name: Jonathan Banks
    # Email: jonathan.banks@gmail.com
    # Phone: 03312223366
    # Check-in: June 19, 2026
    # Check-out: June 30, 2026
    # Guests: 5
    for field, raw_segment in segments.items():
        value = (raw_segment or "").strip().strip(" ,.;:-").strip()
        if not value:
            continue

        if field == "guest_name":
            if _is_valid_candidate_for_field(field, value):
                updates.setdefault(field, value)
                mentioned_fields.append(field)

        elif field == "guest_email":
            email_value = _extract_email(value)
            if email_value and _is_valid_candidate_for_field(field, email_value):
                updates.setdefault(field, email_value)
                mentioned_fields.append(field)

        elif field == "guest_phone":
            phone_value = _extract_phone(value) or value
            if _is_valid_candidate_for_field(field, phone_value):
                updates.setdefault(field, phone_value)
                mentioned_fields.append(field)

        elif field == "guests":
            guest_value = _coerce_int(value)
            if guest_value is not None:
                updates.setdefault(field, guest_value)
                mentioned_fields.append(field)


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
        if assoc_check_in and assoc_check_out:
            updates["check_in"] = assoc_check_in
            updates["check_out"] = assoc_check_out
            errors.pop("check_in", None)
            errors.pop("check_out", None)
        elif assoc_check_in and "check_in" not in updates:
            updates["check_in"] = assoc_check_in
            errors.pop("check_in", None)
        elif assoc_check_out and "check_out" not in updates:
            updates["check_out"] = assoc_check_out
            errors.pop("check_out", None)
        if "check_in" not in updates or "check_out" not in updates:
            all_dates = _find_all_dates(normalized)
            if len(all_dates) >= 2:
                if "check_in" not in updates:
                    updates["check_in"] = all_dates[0][0]
                    errors.pop("check_in", None)
                if "check_out" not in updates:
                    for d_val, _, _ in all_dates:
                        if d_val != updates.get("check_in"):
                            updates["check_out"] = d_val
                            errors.pop("check_out", None)
                            break

    parts = [p.strip() for p in re.split(r'[,;|]', normalized) if p.strip()]
    
    is_compact_raw = False
    if len(parts) >= 2:
        is_compact_raw = True
        for part in parts:
            if part.count(" ") > 3:
                is_compact_raw = False
                break
                
    if is_compact_raw:
        ask_order = get_ask_order()
        

        updates.pop("check_in", None)
        updates.pop("check_out", None)
        if "check_in" in mentioned_fields:
            mentioned_fields.remove("check_in")
        if "check_out" in mentioned_fields:
            mentioned_fields.remove("check_out")
            
        labeled_updates = {}
        unlabeled_parts = []
        for part in parts:
            matched_field = None
            matched_val = None
            for field in ask_order:
                aliases = _field_alias_map().get(field, [])
                sorted_aliases = sorted(aliases, key=len, reverse=True)
                for alias in sorted_aliases:
                    pattern = re.compile(rf"^{re.escape(alias)}\b(?:\s*[:=-]|\s+is\b|\s+would\s+be\b)?\s*(.*)$", re.I)
                    m = pattern.match(part)
                    if m:
                        matched_field = field
                        matched_val = m.group(1).strip()
                        break
                if matched_field:
                    break
            if matched_field:
                labeled_updates[matched_field] = matched_val
            else:
                unlabeled_parts.append(part)
                
        email_parts = []
        phone_parts = []
        date_parts = []
        int_parts = []
        text_parts = []
        
        for part in unlabeled_parts:
            is_date_candidate = bool(re.search(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b", part))
            is_date_candidate = is_date_candidate or bool(re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", part, re.I))
            
            if is_date_candidate:
                date_parts.append(part)
            elif "@" in part and "." in part:
                email_parts.append(part)
            elif re.sub(r"\D", "", part) and len(re.sub(r"\D", "", part)) >= 5 and not re.search(r"[a-zA-Z]", part):
                phone_parts.append(part)
            elif part.isdigit():
                int_parts.append(part)
            else:
                text_parts.append(part)
                
        unfilled_fields = [field for field in ask_order if field not in updates and field not in labeled_updates]
        
        email_fields = [f for f in unfilled_fields if _field_type_from_schema(f) == "email"]
        phone_fields = [f for f in unfilled_fields if _field_type_from_schema(f) == "phone"]
        date_fields = [f for f in unfilled_fields if _field_type_from_schema(f) == "date"]
        int_fields = [f for f in unfilled_fields if _field_type_from_schema(f) == "number"]
        text_fields = [f for f in unfilled_fields if _field_type_from_schema(f) == "text"]
        
        for i, field in enumerate(email_fields):
            if i < len(email_parts):
                labeled_updates[field] = email_parts[i]
        for i, field in enumerate(phone_fields):
            if i < len(phone_parts):
                labeled_updates[field] = phone_parts[i]
        for i, field in enumerate(date_fields):
            if i < len(date_parts):
                labeled_updates[field] = date_parts[i]
        for i, field in enumerate(int_fields):
            if i < len(int_parts):
                labeled_updates[field] = int_parts[i]
        for i, field in enumerate(text_fields):
            if i < len(text_parts):
                labeled_updates[field] = text_parts[i]
                
        cleaned_delimited_updates = {}
        for field, val in labeled_updates.items():
            if _is_valid_candidate_for_field(field, val):
                if _field_type_from_schema(field) == "date":
                    parsed = _parse_single_structured_date(val)
                    if parsed:
                        cleaned_delimited_updates[field] = parsed.strftime(cfg.date_format)
                    else:
                        parsed_nat = _parse_natural_date(val)
                        if parsed_nat:
                            cleaned_delimited_updates[field] = parsed_nat
                elif _field_type_from_schema(field) == "number":
                    guest_count = _coerce_int(val)
                    if guest_count is not None:
                        cleaned_delimited_updates[field] = guest_count
                else:
                    cleaned_delimited_updates[field] = val
                    
        for k, v in cleaned_delimited_updates.items():
            if k not in updates:
                updates[k] = v
                if k not in mentioned_fields:
                    mentioned_fields.append(k)

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

def _sanitize_message_for_amendment_extraction(message: str) -> str:
    sanitized = BOOKING_ID_PATTERN.sub("", message or "")
    return re.sub(r"\s+", " ", sanitized).strip()

def _extract_amendment_name_value(text: str) -> Optional[str]:
    if not (text or "").strip():
        return None
    for pattern in _AMENDMENT_NAME_VALUE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip(" ,.;:'\"")
        if _is_valid_candidate_for_field("guest_name", value):
            return value
    return None

def _extract_amendment_phone_value(text: str) -> Optional[str]:
    return _extract_phone(text)

def _extract_amendment_field_value(
    sanitized_message: str,
    field: str,
) -> Tuple[Optional[Any], Optional[str]]:
    if field == "guest_name":
        value = _extract_amendment_name_value(sanitized_message)
        return value, None

    if field == "guest_phone":
        return _extract_amendment_phone_value(sanitized_message), None

    if field == "guest_email":
        return _extract_email(sanitized_message), None

    field_updates, _, field_errors = _extract_updates_from_message(sanitized_message, field)
    if field in field_updates:
        return field_updates[field], None
    if field in field_errors:
        return None, field_errors[field]
    return None, None

def _extract_amendment_updates(
    message: str,
    fields: List[str],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    sanitized = _sanitize_message_for_amendment_extraction(message)
    updates: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for field in fields:
        value, error = _extract_amendment_field_value(sanitized, field)
        if value is not None:
            updates[field] = value
        elif error:
            errors[field] = error
    return updates, errors

