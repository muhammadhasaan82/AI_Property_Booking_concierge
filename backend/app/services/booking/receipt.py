from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.config.agent_config_loader import cfg
from app.services.booking.constants import (
    BOOKING_ID_PATTERN,
    RECEIPT_TO_SUCCESSFUL_BOOKING_COLUMNS,
    SUCCESSFUL_BOOKING_TO_RECEIPT_KEYS,
)
from app.services.booking.formatting import _format_display_date, _format_money, _receipt_reply
from app.agents.tools.helpers import _coerce_float

def receipt_updates_to_successful_booking_columns(updates: Dict[str, Any]) -> Dict[str, Any]:
    return {
        db_key: updates[receipt_key]
        for receipt_key, db_key in RECEIPT_TO_SUCCESSFUL_BOOKING_COLUMNS.items()
        if receipt_key in updates
    }

def successful_booking_row_to_receipt(row: Dict[str, Any]) -> Dict[str, Any]:
    receipt: Dict[str, Any] = {}
    for db_key, receipt_key in SUCCESSFUL_BOOKING_TO_RECEIPT_KEYS.items():
        if db_key in row:
            receipt[receipt_key] = row.get(db_key)
    if row.get("booking_id") is not None:
        receipt["booking_id"] = str(row.get("booking_id"))
    if row.get("payment_url") is not None:
        receipt["payment_url"] = row.get("payment_url")
    if not receipt.get("status"):
        receipt["status"] = cfg.booking_confirmed_status
    for key in ("check_in", "check_out"):
        value = receipt.get(key)
        if hasattr(value, "isoformat"):
            receipt[key] = value.isoformat()
    for key in ("price_per_night", "total_amount"):
        value = receipt.get(key)
        if value is not None and not isinstance(value, (int, float, str)):
            try:
                receipt[key] = float(value)
            except Exception:
                pass
    return receipt

def _store_current_receipt(soft_state: Dict[str, Any], receipt: Dict[str, Any]) -> None:
    if not isinstance(soft_state, dict) or not isinstance(receipt, dict):
        return
    soft_state["booking_receipt"] = dict(receipt)
    if receipt.get("booking_id"):
        soft_state["booking_registration_id"] = str(receipt["booking_id"])
    soft_state["booking_status"] = receipt.get("status") or cfg.booking_confirmed_status
    soft_state["booking_stage"] = "confirmed"
    soft_state["active_flow"] = "booking"
    soft_state["last_presented_view"] = "booking_receipt"

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

