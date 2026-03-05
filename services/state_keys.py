# services/state_keys.py
"""
Typed string constants for all state-machine filter keys.

Instead of raw string literals like "awaiting_post_mod_choice" scattered
across 30+ call-sites, import SK and use SK.awaiting_post_mod_choice.
A typo on a typed attribute raises AttributeError at import time,
not a silent state-machine divergence at runtime.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class _StateKeys:
    # --- Awaiting user input ---
    awaiting_field: str = "awaiting_field"
    awaiting_selection_confirm: str = "awaiting_selection_confirm"
    awaiting_post_mod_choice: str = "awaiting_post_mod_choice"
    awaiting_post_cancel_choice: str = "awaiting_post_cancel_choice"

    # --- Flow flags ---
    receipt_shown: str = "receipt_shown"
    modifying_dates: str = "modifying_dates"

    # --- Property selection ---
    selected_property: str = "selected_property"
    recent_selection_index: str = "recent_selection_index"
    recent_property_id: str = "recent_property_id"


SK = _StateKeys()
