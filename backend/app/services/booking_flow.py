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


def _field_segments(message: str) -> Dict[str, str]:
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

    for field in ("check_in", "check_out"):
        segment = segments.get(field)
        if segment is None:
            continue
        mentioned_fields.append(field)
        parsed = _parse_natural_date(segment)
        if parsed:
            updates[field] = parsed
        else:
            errors[field] = _friendly_prompt_for_field(field)

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

    # Cross-field checkout rule: use values from updates too when validate_field
    # rejected check_out and it was not committed to candidate_state.
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
        if "check_out" in errors:
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
