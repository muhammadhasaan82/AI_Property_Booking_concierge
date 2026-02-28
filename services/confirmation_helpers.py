# services/confirmation_helpers.py
# Modular helper functions extracted from the monolithic confirmation_agent.
# Each handler returns Optional[Dict] — if it handles the case, returns the response;
# otherwise returns None and the next handler is tried.

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime

from .config import (
    REQUIRED_FIELDS,
    FIELD_PROMPTS,
    PROCEED_PHRASES,
    MODIFY_PHRASES,
)


def _render_receipt(persisted: Dict[str, Any]) -> str:
    """Render the booking summary receipt."""
    selected_property = persisted.get("selected_property") or {}
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

    return f"""📋 **BOOKING SUMMARY**

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



def _next_missing_field(persisted: Dict[str, Any]) -> Optional[str]:
    """Return the first missing required field, or None if all are satisfied."""
    for field in REQUIRED_FIELDS:
        if not persisted.get(field):
            return field
    return None


def _ask_for_field(field: str, persisted: Dict[str, Any]) -> Dict[str, Any]:
    """Return a response asking for the given field."""
    persisted["awaiting_field"] = field
    return {
        "reply": FIELD_PROMPTS.get(field, f"Please provide your {field}."),
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
    persisted["receipt_shown"] = True
    return {
        "reply": receipt,
        "tool_result": {"ok": False, "need": ["final_confirmation"], "show_receipt": True},
        "filters": persisted,
    }


def handle_final_confirmation(user_text: str, persisted: Dict[str, Any],
                               _is_yes, _is_no) -> Optional[Dict[str, Any]]:
    """
    Handle the yes/no response after receipt is shown.
    Returns None if not in receipt_shown state.
    """
    if not persisted.get("receipt_shown"):
        return None

    if _is_yes(user_text):
        persisted.pop("awaiting_post_mod_choice", None)
        persisted.pop("awaiting_post_cancel_choice", None)
        persisted.pop("receipt_shown", None)
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
        persisted.pop("receipt_shown", None)
        persisted.pop("awaiting_post_mod_choice", None)
        persisted["awaiting_field"] = "modification_choice"
        return {
            "reply": "No problem, the booking has been cancelled. What would you like to modify — dates, guests, name, phone, email, or property?",
            "filters": persisted,
            "tool_result": {"ok": False, "cancelled": True, "need": ["modification"]},
        }

    # Re-render receipt if user asks for it
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
    if not persisted.get("awaiting_post_mod_choice"):
        return None

    tl = (user_text or "").lower().strip()

    if tl.strip() == "yes" or any(p in tl for p in PROCEED_PHRASES):
        persisted.pop("awaiting_post_mod_choice", None)
        persisted.pop("awaiting_post_cancel_choice", None)
        return _try_show_receipt(persisted)

    if any(p in tl for p in MODIFY_PHRASES):
        persisted.pop("awaiting_post_mod_choice", None)
        persisted.pop("awaiting_post_cancel_choice", None)
        persisted["awaiting_field"] = "modification_choice"
        return {
            "reply": "What would you like to modify — dates, guests, name, phone, email, or property?",
            "filters": persisted,
            "tool_result": {"ok": False, "need": ["modification"]},
        }

    # Default: proceed to receipt
    if not persisted.get("modifying_dates"):
        return _try_show_receipt(persisted)

    return {
        "reply": "Would you like to proceed to the updated receipt, or make another change?",
        "filters": persisted,
        "tool_result": {"ok": False, "need": ["post_mod_choice"]},
    }


def handle_property_selection(
    user_text: str,
    persisted: Dict[str, Any],
    sel: Optional[int],
    _format_property_full,
) -> Optional[Dict[str, Any]]:
    """
    Handle numeric selection of a property from results.
    Returns None if no valid selection is made.
    """
    awaiting_guests = persisted.get("awaiting_field") == "guests"

    # If we're waiting for guests and got a number, treat as guest count
    import re
    if awaiting_guests and re.match(r"^\s*\d+\s*$", user_text.strip()):
        return None

    if sel is None or awaiting_guests:
        return None

    idx_map = persisted.get("results_index_map") or {}
    prop_id = idx_map.get(sel)
    results = persisted.get("last_results") or persisted.get("results") or []
    chosen = next((p for p in results if p.get("id") == prop_id), None)

    if not chosen:
        max_option = max(idx_map.keys()) if idx_map else 4
        options_str = ", ".join(str(i) for i in range(1, max_option + 1))
        return {
            "reply": f"Sorry, I couldn't find that option. Please choose from: {options_str}.",
            "tool_result": {"ok": False, "need": ["property_selection"]},
            "filters": persisted,
        }

    card = _format_property_full(chosen)
    persisted.update({
        "recent_selection_index": sel,
        "recent_property_id": prop_id,
        "selected_property": chosen,
        "awaiting_selection_confirm": True,
        "awaiting_field": None,
    })

    # If all fields present, show receipt immediately
    if all(persisted.get(k) for k in REQUIRED_FIELDS):
        return _try_show_receipt(persisted)

    return {
        "reply": f"🏠 Selected:\n\n{card}\n\nWould you like to book this one? (yes/no)",
        "tool_result": {"ok": False, "need": ["booking_confirmation"], "property_id": prop_id},
        "filters": persisted,
    }


def handle_selection_confirm(
    user_text: str,
    persisted: Dict[str, Any],
    _is_yes,
    _is_no,
) -> Optional[Dict[str, Any]]:
    """
    Handle yes/no response after a property is selected.
    Returns None if not in awaiting_selection_confirm state.
    """
    if not persisted.get("awaiting_selection_confirm"):
        return None

    tl = (user_text or "").strip().lower()
    if tl in {"yes please", "yes pls", "sure please", "pls yes", "yup please", "yeah please"}:
        user_text = "yes"

    if _is_no(user_text):
        for k in ["recent_property_id", "recent_selection_index", "selected_property",
                   "awaiting_selection_confirm", "awaiting_field"]:
            persisted.pop(k, None)
        return {
            "reply": "No worries — thanks for visiting! Have a lovely day ✨",
            "tool_result": {"ok": False, "end": True},
            "filters": persisted,
        }

    if _is_yes(user_text) or tl in {"yes sure", "sure yes", "yes please", "yes pls", "sure"}:
        persisted["awaiting_selection_confirm"] = False
        return _try_show_receipt(persisted)

    return {
        "reply": "Please reply with yes or no to continue.",
        "tool_result": {"ok": False, "need": ["booking_confirmation"]},
        "filters": persisted,
    }
