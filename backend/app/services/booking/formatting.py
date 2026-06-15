from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.config.agent_config_loader import cfg
from app.agents.tools.helpers import _coerce_float
from app.config.booking_schema_loader import (
    get_amendable_fields,
    get_amendment_display_name,
    get_field_display_name,
    get_field_prompt,
)

def _format_money(value: Any) -> str:
    amount = _coerce_float(value) or 0.0
    return f"${amount:,.2f}"

def _format_display_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, cfg.date_format)
    except Exception:
        return value
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"

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

def _amendment_choice_reply() -> str:
    labels = [get_amendment_display_name(field) for field in get_amendable_fields()]
    if len(labels) > 1:
        formatted = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    else:
        formatted = labels[0] if labels else "booking details"
    return f"Which booking detail would you like to change? You can change {formatted}."

def _amendment_values_reply(fields: List[str]) -> str:
    if not fields:
        return _amendment_choice_reply()
    if len(fields) == 1:
        return _friendly_prompt_for_field(fields[0])
    labels = [get_field_display_name(field).lower() for field in fields]
    return f"Please provide the updated {', '.join(labels[:-1])} and {labels[-1]}."

def _receipt_review_from_candidate(receipt: Dict[str, Any]) -> str:
    return (
        "Please review your updated booking details:\n"
        f"- Registration ID: {receipt.get('booking_id') or ''}\n"
        f"- Property: {receipt.get('property_title') or ''}\n"
        f"- Full name: {receipt.get('guest_name') or ''}\n"
        f"- Email: {receipt.get('guest_email') or ''}\n"
        f"- Phone: {receipt.get('guest_phone') or ''}\n"
        f"- Check-in: {_format_display_date(str(receipt.get('check_in') or ''))}\n"
        f"- Check-out: {_format_display_date(str(receipt.get('check_out') or ''))}\n"
        f"- Guests: {receipt.get('guests') or ''}\n"
        f"- Nights: {receipt.get('nights') or ''}\n"
        f"- Price per night: {_format_money(receipt.get('price_per_night'))}\n"
        f"- Total: {_format_money(receipt.get('total_amount'))}"
        + "\n\nPlease confirm if these updated booking details are correct."
    )

