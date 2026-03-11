# services/confirmation_helpers.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .config import (
    FIELD_MODIFICATION_PROMPTS,
    FIELD_PROMPTS,
    MODIFY_PHRASES,
    PROCEED_PHRASES,
    REQUIRED_FIELDS,
)
from .state_keys import SK

MODIFICATION_OPTIONS_REPLY = (
    "What would you like to modify - dates, guests, name, phone, email, or property?"
)
MODIFICATION_REASK_REPLY = (
    "Please specify what you'd like to change: dates, guests, name, phone, email, or property?"
)
POST_MODIFICATION_REPLY = (
    "Would you like to proceed to the updated receipt, or make another change?"
)
POST_CANCEL_CLARIFICATION_REPLY = (
    "Please specify: would you like to:\n"
    "1. Search for different properties, or\n"
    "2. Modify your current requirements?\n\n"
    "Just tell me what you'd prefer!"
)
RESTART_SEARCH_REPLY = (
    "Sure - let's explore more options. Tell me what you're looking for "
    "(city, budget, dates, beds, amenities)."
)
PROPERTY_RESTART_REPLY = (
    "Okay, let's search for a different property. What should I look for "
    "(city, budget, amenities, beds)?"
)
PREVIOUS_RESULTS_REPLY = "Sure! Here are your previous results:"


def _render_receipt(persisted: Dict[str, Any]) -> str:
    """Render the booking summary receipt."""
    selected_property = persisted.get(SK.selected_property) or {}
    title = selected_property.get("title", "Property")
    city = (selected_property.get("city") or "").title()
    price_per_night = float(selected_property.get("price_per_night") or 0)

    try:
        ci = datetime.strptime(persisted.get("check_in", ""), "%Y-%m-%d")
        co = datetime.strptime(persisted.get("check_out", ""), "%Y-%m-%d")
        nights = max(1, (co - ci).days)
    except Exception:
        nights = 1

    total = int(price_per_night * nights)

    return (
        "BOOKING SUMMARY\n\n"
        "Guest Information\n"
        f"- Name: {persisted.get('name')}\n"
        f"- Phone: {persisted.get('phone')}\n"
        f"- Email: {persisted.get('email')}\n\n"
        "Property Details\n"
        f"- {title}\n"
        f"- Location: {city}\n"
        f"- Price per night: ${int(price_per_night)}\n\n"
        "Booking Details\n"
        f"- Check-in: {persisted.get('check_in')}\n"
        f"- Check-out: {persisted.get('check_out')}\n"
        f"- Number of nights: {nights}\n"
        f"- Number of guests: {persisted.get('guests')}\n\n"
        f"TOTAL AMOUNT: ${total}\n\n"
        "Would you like to confirm this booking?\n"
        "Reply yes to confirm and proceed with payment, or no to cancel."
    )


def _next_missing_field(persisted: Dict[str, Any]) -> Optional[str]:
    """Return the first missing required field, or None if all are satisfied."""
    for field in REQUIRED_FIELDS:
        if not persisted.get(field):
            return field
    return None


def _ask_for_field(field: str, persisted: Dict[str, Any]) -> Dict[str, Any]:
    """Return a response asking for the given field."""
    persisted[SK.awaiting_field] = field
    return {
        "reply": FIELD_PROMPTS.get(field, f"Please provide your {field}."),
        "filters": persisted,
        "tool_result": {"ok": False, "need": [field]},
    }


def _ask_for_modification_field(field: str, persisted: Dict[str, Any]) -> Dict[str, Any]:
    """Ask for an updated value of a specific field."""
    persisted[SK.awaiting_field] = field
    return {
        "reply": FIELD_MODIFICATION_PROMPTS.get(
            field,
            FIELD_PROMPTS.get(field, f"Please provide your {field}."),
        ),
        "filters": persisted,
        "tool_result": {"ok": False, "need": [field]},
    }


def _try_show_receipt(persisted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if all required fields are present. If so, show receipt.
    Otherwise, ask for the next missing field.
    """
    missing = _next_missing_field(persisted)
    if missing:
        return _ask_for_field(missing, persisted)

    receipt = _render_receipt(persisted)
    persisted[SK.awaiting_field] = None
    persisted[SK.receipt_shown] = True
    return {
        "reply": receipt,
        "tool_result": {"ok": False, "need": ["final_confirmation"], "show_receipt": True},
        "filters": persisted,
    }


def normalize_date_value(value: str) -> Optional[str]:
    """Return YYYY-MM-DD or None for unsupported date strings."""
    text = (value or "").strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _set_date_field(persisted: Dict[str, Any], field: str, value: str) -> bool:
    normalized = normalize_date_value(value)
    if not normalized:
        return False
    persisted[field] = normalized
    return True


def _clear_selection_state(target: Dict[str, Any]) -> None:
    for key in (
        SK.recent_selection_index,
        SK.recent_property_id,
        SK.selected_property,
        SK.awaiting_selection_confirm,
        SK.awaiting_field,
        SK.receipt_shown,
        SK.awaiting_post_mod_choice,
        SK.awaiting_post_cancel_choice,
        SK.modifying_dates,
    ):
        target.pop(key, None)


def build_restart_filters(
    persisted: Dict[str, Any],
    *,
    include_results: bool = True,
) -> Dict[str, Any]:
    keep_keys = {
        "location",
        "city",
        "budget",
        "amenities",
        "beds",
        "property_type",
        *REQUIRED_FIELDS,
    }
    if include_results:
        keep_keys.update({"results_index_map", "last_results", "results"})

    reset = {key: value for key, value in persisted.items() if key in keep_keys}
    _clear_selection_state(reset)
    return reset


def _render_previous_results(persisted: Dict[str, Any]) -> Optional[str]:
    last_results = persisted.get("last_results") or persisted.get("results") or []
    idx_map = persisted.get("results_index_map") or {}
    if not (last_results and idx_map):
        return None

    lines: List[str] = []
    for number in sorted(idx_map.keys()):
        property_id = idx_map[number]
        prop = next((item for item in last_results if item.get("id") == property_id), None)
        if not prop:
            continue
        title = prop.get("title", "Property")
        city = (prop.get("city") or "").title()
        try:
            price = int(float(prop.get("price_per_night") or 0))
        except Exception:
            price = 0
        lines.append(f"{number}. {title} - {city} - about ${price}/night")

    if not lines:
        return None
    return PREVIOUS_RESULTS_REPLY + "\n\n" + "\n".join(lines) + "\n\nReply with a number to choose."


def handle_final_confirmation(
    user_text: str,
    persisted: Dict[str, Any],
    _is_yes,
    _is_no,
) -> Optional[Dict[str, Any]]:
    """
    Handle the yes/no response after receipt is shown.
    Returns None if not in receipt_shown state.
    """
    if not persisted.get(SK.receipt_shown):
        return None

    if _is_yes(user_text):
        persisted.pop(SK.awaiting_post_mod_choice, None)
        persisted.pop(SK.awaiting_post_cancel_choice, None)
        persisted.pop(SK.receipt_shown, None)
        persisted.pop(SK.awaiting_field, None)
        persisted.pop(SK.modifying_dates, None)
        return {
            "reply": "Perfect! Creating your booking now...",
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
        persisted.pop(SK.receipt_shown, None)
        persisted.pop(SK.awaiting_post_mod_choice, None)
        persisted[SK.awaiting_field] = "modification_choice"
        return {
            "reply": (
                "No problem, the booking has been cancelled. "
                + MODIFICATION_OPTIONS_REPLY
            ),
            "filters": persisted,
            "tool_result": {"ok": False, "cancelled": True, "need": ["modification"]},
        }

    tl = (user_text or "").lower().strip()
    if any(phrase in tl for phrase in ["total bill", "total", "bill", "receipt", "show total"]):
        receipt = _render_receipt(persisted)
        return {
            "reply": receipt,
            "tool_result": {"ok": False, "need": ["final_confirmation"], "show_receipt": True},
            "filters": persisted,
        }

    return None


def handle_post_modification_choice(user_text: str, persisted: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    After a modification, handle the proceed/modify-more decision.
    Returns None if not in awaiting_post_mod_choice state.
    """
    if not persisted.get(SK.awaiting_post_mod_choice):
        return None

    tl = (user_text or "").lower().strip()
    if tl == "yes" or any(phrase in tl for phrase in PROCEED_PHRASES):
        persisted.pop(SK.awaiting_post_mod_choice, None)
        persisted.pop(SK.awaiting_post_cancel_choice, None)
        return _try_show_receipt(persisted)

    if any(phrase in tl for phrase in MODIFY_PHRASES):
        persisted.pop(SK.awaiting_post_mod_choice, None)
        persisted.pop(SK.awaiting_post_cancel_choice, None)
        persisted[SK.awaiting_field] = "modification_choice"
        return {
            "reply": MODIFICATION_OPTIONS_REPLY,
            "filters": persisted,
            "tool_result": {"ok": False, "need": ["modification"]},
        }

    if not persisted.get(SK.modifying_dates):
        return _try_show_receipt(persisted)

    return {
        "reply": POST_MODIFICATION_REPLY,
        "filters": persisted,
        "tool_result": {"ok": False, "need": ["post_mod_choice"]},
    }


def handle_restart_search_request(
    user_text: str,
    persisted: Dict[str, Any],
    *,
    parsed_dates: List[str],
    wants_property_search_request: Callable[[str], bool],
) -> Optional[Dict[str, Any]]:
    awaiting_field = persisted.get(SK.awaiting_field)
    is_answering_dates = awaiting_field in {"check_in", "check_out"} and bool(parsed_dates)
    if (
        not wants_property_search_request(user_text)
        or is_answering_dates
        or persisted.get(SK.awaiting_selection_confirm)
    ):
        return None

    return {
        "reply": RESTART_SEARCH_REPLY,
        "filters": build_restart_filters(persisted, include_results=True),
        "tool_result": {"ok": False, "need": ["restart"]},
    }


def handle_property_selection(
    user_text: str,
    persisted: Dict[str, Any],
    sel: Optional[int],
    format_property_full: Callable[[Dict[str, Any]], str],
) -> Optional[Dict[str, Any]]:
    """
    Handle numeric selection of a property from results.
    Returns None if no valid selection is made.
    """
    awaiting_guests = persisted.get(SK.awaiting_field) == "guests"

    if awaiting_guests and user_text.strip().isdigit():
        return None
    if sel is None or awaiting_guests:
        return None

    idx_map = persisted.get("results_index_map") or {}
    property_id = idx_map.get(sel)
    results = persisted.get("last_results") or persisted.get("results") or []
    chosen = next((item for item in results if item.get("id") == property_id), None)

    if not chosen:
        max_option = max(idx_map.keys()) if idx_map else 4
        options = ", ".join(str(i) for i in range(1, max_option + 1))
        return {
            "reply": f"Sorry, I couldn't find that option. Please choose from: {options}.",
            "tool_result": {"ok": False, "need": ["property_selection"]},
            "filters": persisted,
        }

    card = format_property_full(chosen)
    persisted.update(
        {
            SK.recent_selection_index: sel,
            SK.recent_property_id: property_id,
            SK.selected_property: chosen,
            SK.awaiting_selection_confirm: True,
            SK.awaiting_field: None,
        }
    )

    if all(persisted.get(field) for field in REQUIRED_FIELDS):
        return _try_show_receipt(persisted)

    return {
        "reply": f"Selected:\n\n{card}\n\nWould you like to book this one?",
        "tool_result": {"ok": False, "need": ["booking_confirmation"], "property_id": property_id},
        "filters": persisted,
    }


def handle_selection_confirm(
    user_text: str,
    persisted: Dict[str, Any],
    _is_yes,
    _is_no,
    *,
    wants_previous_results: Optional[Callable[[str], bool]] = None,
    wants_property_search_request: Optional[Callable[[str], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Handle yes/no response after a property is selected.
    Returns None if not in awaiting_selection_confirm state.
    """
    if not persisted.get(SK.awaiting_selection_confirm):
        return None

    wants_previous = bool(wants_previous_results and wants_previous_results(user_text))
    wants_search = bool(
        wants_property_search_request and wants_property_search_request(user_text)
    )

    if _is_no(user_text) or wants_previous or wants_search:
        listing = _render_previous_results(persisted)
        for key in (
            SK.recent_property_id,
            SK.recent_selection_index,
            SK.selected_property,
            SK.awaiting_selection_confirm,
            SK.awaiting_field,
        ):
            persisted.pop(key, None)

        if listing:
            return {
                "reply": listing,
                "tool_result": {"ok": False, "need": ["property_selection"]},
                "filters": persisted,
            }
        return {
            "reply": "No previous results found. What would you like to search for? (city, property type, budget)",
            "tool_result": {"ok": False, "need": ["restart"]},
            "filters": persisted,
        }

    tl = (user_text or "").strip().lower()
    if _is_yes(user_text) or tl in {"yes sure", "sure yes", "yes please", "yes pls", "sure"}:
        persisted[SK.awaiting_selection_confirm] = False
        return _try_show_receipt(persisted)

    return {
        "reply": "Please reply with yes or no to continue.",
        "tool_result": {"ok": False, "need": ["booking_confirmation"]},
        "filters": persisted,
    }


def route_requested_modifications(
    persisted: Dict[str, Any],
    requested_fields: List[str],
    *,
    parsed_name: Optional[str] = None,
    parsed_phone: Optional[str] = None,
    parsed_email: Optional[str] = None,
    parsed_guests: Optional[int] = None,
    parsed_dates: Optional[List[str]] = None,
    allow_inline_apply: bool = False,
) -> Dict[str, Any]:
    """Route one or more requested modifications using a shared path."""
    parsed_dates = parsed_dates or []
    requested = list(dict.fromkeys(requested_fields))

    if not requested:
        persisted[SK.awaiting_field] = "modification_choice"
        return {
            "reply": MODIFICATION_OPTIONS_REPLY,
            "filters": persisted,
            "tool_result": {"ok": False, "need": ["modification"]},
        }

    if "location" in requested or "property" in requested:
        return {
            "reply": PROPERTY_RESTART_REPLY,
            "filters": build_restart_filters(persisted, include_results=True),
            "tool_result": {"ok": False, "need": ["restart"]},
        }

    if allow_inline_apply:
        applied_any = False
        need_next: Optional[str] = None

        if "dates" in requested:
            if len(parsed_dates) >= 2:
                ci_ok = _set_date_field(persisted, "check_in", parsed_dates[0])
                co_ok = _set_date_field(persisted, "check_out", parsed_dates[1])
                if ci_ok and co_ok:
                    applied_any = True
                else:
                    need_next = need_next or "check_in"
            else:
                need_next = need_next or "check_in"

        if "check_in" in requested and "dates" not in requested:
            if parsed_dates and _set_date_field(persisted, "check_in", parsed_dates[0]):
                applied_any = True
            else:
                need_next = need_next or "check_in"

        if "check_out" in requested and "dates" not in requested:
            if parsed_dates and _set_date_field(persisted, "check_out", parsed_dates[-1]):
                applied_any = True
            else:
                need_next = need_next or "check_out"

        if "guests" in requested:
            if parsed_guests is not None:
                persisted["guests"] = int(parsed_guests)
                applied_any = True
            else:
                need_next = need_next or "guests"

        if "name" in requested:
            if parsed_name:
                persisted["name"] = parsed_name
                applied_any = True
            else:
                need_next = need_next or "name"

        if "phone" in requested:
            if parsed_phone:
                persisted["phone"] = parsed_phone
                applied_any = True
            else:
                need_next = need_next or "phone"

        if "email" in requested:
            if parsed_email:
                persisted["email"] = parsed_email
                applied_any = True
            else:
                need_next = need_next or "email"

        if need_next:
            return _ask_for_modification_field(need_next, persisted)

        if applied_any:
            persisted[SK.awaiting_field] = None
            persisted.pop(SK.modifying_dates, None)
            persisted[SK.awaiting_post_mod_choice] = True
            return {
                "reply": "All set. Would you like to proceed to the updated receipt, or make another change?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["post_mod_choice"]},
            }

    if "dates" in requested or ("check_in" in requested and "check_out" in requested):
        persisted["check_in"] = None
        persisted["check_out"] = None
        persisted[SK.modifying_dates] = True
        return _ask_for_modification_field("check_in", persisted)

    if "check_in" in requested:
        persisted["check_in"] = None
        persisted[SK.modifying_dates] = True
        return _ask_for_modification_field("check_in", persisted)

    if "check_out" in requested:
        persisted["check_out"] = None
        persisted[SK.modifying_dates] = True
        return _ask_for_modification_field("check_out", persisted)

    if "guests" in requested:
        persisted["guests"] = None
        return _ask_for_modification_field("guests", persisted)

    if "name" in requested:
        persisted["name"] = None
        return _ask_for_modification_field("name", persisted)

    if "phone" in requested:
        persisted["phone"] = None
        return _ask_for_modification_field("phone", persisted)

    if "email" in requested:
        persisted["email"] = None
        return _ask_for_modification_field("email", persisted)

    persisted[SK.awaiting_field] = "modification_choice"
    return {
        "reply": MODIFICATION_REASK_REPLY,
        "filters": persisted,
        "tool_result": {"ok": False, "need": ["modification"]},
    }


def handle_post_cancel_choice(
    user_text: str,
    persisted: Dict[str, Any],
    *,
    requested_fields: List[str],
    wants_property_search_request: Callable[[str], bool],
    wants_modification: Callable[[str], bool],
) -> Optional[Dict[str, Any]]:
    """Handle search-again vs modification after a cancellation."""
    if not persisted.get(SK.awaiting_post_cancel_choice) or persisted.get(SK.awaiting_field):
        return None

    if wants_property_search_request(user_text):
        persisted.pop(SK.awaiting_post_cancel_choice, None)
        persisted.pop(SK.awaiting_post_mod_choice, None)
        return {
            "reply": PROPERTY_RESTART_REPLY,
            "filters": build_restart_filters(persisted, include_results=True),
            "tool_result": {"ok": False, "need": ["restart"]},
        }

    if wants_modification(user_text):
        persisted.pop(SK.awaiting_post_cancel_choice, None)
        persisted.pop(SK.awaiting_post_mod_choice, None)
        return route_requested_modifications(persisted, requested_fields)

    return {
        "reply": POST_CANCEL_CLARIFICATION_REPLY,
        "filters": persisted,
        "tool_result": {"ok": False, "need": ["clarification"]},
    }


def capture_awaited_field(
    persisted: Dict[str, Any],
    user_text: str,
    *,
    parsed_name: Optional[str],
    parsed_phone: Optional[str],
    parsed_email: Optional[str],
    parsed_guests: Optional[int],
    parsed_dates: List[str],
    resolve_name_candidate: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Any]:
    """
    Capture explicitly awaited fields and return a small state result.

    Result keys:
    - handled: whether an awaiting-field branch ran
    - updated: whether state was updated
    - response: optional reply payload to return immediately
    """
    awaited = (persisted.get(SK.awaiting_field) or "").strip()
    result = {"handled": False, "updated": False, "response": None}

    if awaited == "email":
        if parsed_email:
            persisted["email"] = parsed_email
            persisted[SK.awaiting_field] = None
            result.update({"handled": True, "updated": True})
        return result

    if awaited == "phone":
        if parsed_phone:
            persisted["phone"] = parsed_phone
            persisted[SK.awaiting_field] = None
            result.update({"handled": True, "updated": True})
        return result

    if awaited == "name":
        candidate = parsed_name
        if not candidate and resolve_name_candidate:
            candidate = resolve_name_candidate(user_text)
        if candidate:
            persisted["name"] = candidate
            persisted[SK.awaiting_field] = None
            result.update({"handled": True, "updated": True})
        return result

    if awaited == "guests":
        if parsed_guests is not None:
            persisted["guests"] = int(parsed_guests)
            persisted[SK.awaiting_field] = None
            result.update({"handled": True, "updated": True})
            return result
        if user_text.strip().isdigit():
            persisted["guests"] = int(user_text.strip())
            persisted[SK.awaiting_field] = None
            result.update({"handled": True, "updated": True})
        return result

    if awaited == "check_in":
        if not parsed_dates:
            return result
        result["handled"] = True
        if not _set_date_field(persisted, "check_in", parsed_dates[0]):
            result["response"] = _ask_for_modification_field("check_in", persisted)
            return result
        result["updated"] = True
        if persisted.get(SK.modifying_dates):
            result["response"] = _ask_for_modification_field("check_out", persisted)
            return result
        persisted[SK.awaiting_field] = None
        return result

    if awaited == "check_out":
        if not parsed_dates:
            return result
        result["handled"] = True
        if not _set_date_field(persisted, "check_out", parsed_dates[-1]):
            result["response"] = _ask_for_modification_field("check_out", persisted)
            return result
        persisted[SK.awaiting_field] = None
        result["updated"] = True
        if persisted.pop(SK.modifying_dates, None):
            persisted[SK.awaiting_post_mod_choice] = True
            result["response"] = {
                "reply": "Dates updated. Do you want to proceed to the total bill for payment or make more changes?",
                "filters": persisted,
                "tool_result": {"ok": False, "need": ["post_mod_choice"]},
            }
        return result

    return result


def _render_updated_receipt(persisted: Dict[str, Any]) -> Dict[str, Any]:
    persisted[SK.receipt_shown] = True
    persisted[SK.awaiting_field] = None
    persisted.pop(SK.awaiting_post_mod_choice, None)
    persisted.pop(SK.awaiting_post_cancel_choice, None)
    return {
        "reply": _render_receipt(persisted),
        "tool_result": {"ok": False, "need": ["final_confirmation"], "show_receipt": True},
        "filters": persisted,
    }


def handle_inline_receipt_updates(
    persisted: Dict[str, Any],
    *,
    parsed_name: Optional[str],
    parsed_phone: Optional[str],
    parsed_email: Optional[str],
    parsed_guests: Optional[int],
    parsed_dates: List[str],
) -> Optional[Dict[str, Any]]:
    """Apply direct updates while the receipt is visible and re-render it."""
    if not persisted.get(SK.receipt_shown) or persisted.get(SK.awaiting_field):
        return None

    updated = False
    if parsed_name and parsed_name != persisted.get("name"):
        persisted["name"] = parsed_name
        updated = True
    if parsed_phone and parsed_phone != persisted.get("phone"):
        persisted["phone"] = parsed_phone
        updated = True
    if parsed_email and parsed_email != persisted.get("email"):
        persisted["email"] = parsed_email
        updated = True
    if parsed_guests is not None and parsed_guests != persisted.get("guests"):
        persisted["guests"] = int(parsed_guests)
        updated = True

    if len(parsed_dates) >= 2:
        ci_ok = _set_date_field(persisted, "check_in", parsed_dates[0])
        co_ok = _set_date_field(persisted, "check_out", parsed_dates[1])
        updated = updated or (ci_ok and co_ok)
    elif len(parsed_dates) == 1:
        if not persisted.get("check_in"):
            return _ask_for_modification_field("check_in", persisted)
        if not persisted.get("check_out"):
            return _ask_for_modification_field("check_out", persisted)

    if updated:
        return _render_updated_receipt(persisted)
    return None
