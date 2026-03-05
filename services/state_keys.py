# services/state_keys.py
"""Typed constants for state-machine keys used across the codebase."""

from __future__ import annotations


class _StateKeys:
    # Awaiting user input
    AWAITING_FIELD = "awaiting_field"
    AWAITING_SELECTION_CONFIRM = "awaiting_selection_confirm"
    AWAITING_POST_MOD_CHOICE = "awaiting_post_mod_choice"
    AWAITING_POST_CANCEL_CHOICE = "awaiting_post_cancel_choice"
    AWAITING_UNAVAILABLE_CITY_CHOICE = "awaiting_unavailable_city_choice"
    AWAITING_CITY_SELECTION = "awaiting_city_selection"
    AWAITING_PROPERTY_TYPE_CHOICE = "awaiting_property_type_choice"

    # Flow flags
    RECEIPT_SHOWN = "receipt_shown"
    MODIFYING_DATES = "modifying_dates"
    FAQ_ANSWERED = "faq_answered"
    FAQ_RESUME_INTENT = "faq_resume_intent"

    # Property selection
    SELECTED_PROPERTY = "selected_property"
    RECENT_SELECTION_INDEX = "recent_selection_index"
    RECENT_PROPERTY_ID = "recent_property_id"

    # Compatibility aliases for existing call-sites
    @property
    def awaiting_field(self) -> str:
        return self.AWAITING_FIELD

    @property
    def awaiting_selection_confirm(self) -> str:
        return self.AWAITING_SELECTION_CONFIRM

    @property
    def awaiting_post_mod_choice(self) -> str:
        return self.AWAITING_POST_MOD_CHOICE

    @property
    def awaiting_post_cancel_choice(self) -> str:
        return self.AWAITING_POST_CANCEL_CHOICE

    @property
    def awaiting_unavailable_city_choice(self) -> str:
        return self.AWAITING_UNAVAILABLE_CITY_CHOICE

    @property
    def awaiting_city_selection(self) -> str:
        return self.AWAITING_CITY_SELECTION

    @property
    def awaiting_property_type_choice(self) -> str:
        return self.AWAITING_PROPERTY_TYPE_CHOICE

    @property
    def receipt_shown(self) -> str:
        return self.RECEIPT_SHOWN

    @property
    def modifying_dates(self) -> str:
        return self.MODIFYING_DATES

    @property
    def faq_answered(self) -> str:
        return self.FAQ_ANSWERED

    @property
    def faq_resume_intent(self) -> str:
        return self.FAQ_RESUME_INTENT

    @property
    def selected_property(self) -> str:
        return self.SELECTED_PROPERTY

    @property
    def recent_selection_index(self) -> str:
        return self.RECENT_SELECTION_INDEX

    @property
    def recent_property_id(self) -> str:
        return self.RECENT_PROPERTY_ID


STATE_KEYS = _StateKeys()
SK = STATE_KEYS
