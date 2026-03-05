# tests/test_confirmation_helpers.py
# Unit tests for the modular confirmation_agent helpers.

import pytest
from services.confirmation_helpers import (
    _render_receipt,
    _next_missing_field,
    _ask_for_field,
    _try_show_receipt,
    handle_final_confirmation,
    handle_post_modification_choice,
    handle_property_selection,
    handle_selection_confirm,
    REQUIRED_FIELDS,
    FIELD_PROMPTS,
)
from services.state_keys import SK


# --- Helpers for mocking intent detection ---
def _mock_is_yes(text: str) -> bool:
    return text.strip().lower() in {"yes", "y", "yeah", "yep", "sure", "ok"}


def _mock_is_no(text: str) -> bool:
    return text.strip().lower() in {"no", "n", "nah", "nope", "cancel"}


COMPLETE_FILTERS = {
    "name": "John Doe",
    "phone": "+15551234567",
    "email": "john@example.com",
    "check_in": "2027-06-01",
    "check_out": "2027-06-05",
    "guests": 2,
    SK.selected_property: {
        "title": "Beach Villa",
        "city": "miami",
        "price_per_night": 200,
    },
    SK.recent_property_id: "p1",
}


class TestRenderReceipt:
    def test_renders_guest_info(self):
        receipt = _render_receipt(COMPLETE_FILTERS)
        assert "John Doe" in receipt
        assert "+15551234567" in receipt
        assert "john@example.com" in receipt

    def test_renders_property_details(self):
        receipt = _render_receipt(COMPLETE_FILTERS)
        assert "Beach Villa" in receipt
        assert "Miami" in receipt
        assert "$200" in receipt

    def test_renders_booking_details(self):
        receipt = _render_receipt(COMPLETE_FILTERS)
        assert "2027-06-01" in receipt
        assert "2027-06-05" in receipt
        assert "4" in receipt  # nights
        assert "$800" in receipt  # total

    def test_renders_confirmation_prompt(self):
        receipt = _render_receipt(COMPLETE_FILTERS)
        assert "confirm" in receipt.lower()


class TestNextMissingField:
    def test_complete_returns_none(self):
        assert _next_missing_field(COMPLETE_FILTERS) is None

    def test_missing_name(self):
        filters = {**COMPLETE_FILTERS, "name": None}
        assert _next_missing_field(filters) == "name"

    def test_missing_email(self):
        filters = {k: v for k, v in COMPLETE_FILTERS.items() if k != "email"}
        assert _next_missing_field(filters) == "email"

    def test_empty_filters(self):
        assert _next_missing_field({}) == "name"


class TestAskForField:
    def test_asks_for_name(self):
        persisted = {}
        result = _ask_for_field("name", persisted)
        assert "name" in result["reply"].lower()
        assert persisted[SK.awaiting_field] == "name"

    def test_asks_for_email(self):
        persisted = {}
        result = _ask_for_field("email", persisted)
        assert "email" in result["reply"].lower()


class TestTryShowReceipt:
    def test_shows_receipt_when_complete(self):
        persisted = {**COMPLETE_FILTERS}
        result = _try_show_receipt(persisted)
        assert "BOOKING SUMMARY" in result["reply"]
        assert persisted.get(SK.receipt_shown) is True

    def test_asks_for_missing_field(self):
        persisted = {k: v for k, v in COMPLETE_FILTERS.items() if k != "phone"}
        result = _try_show_receipt(persisted)
        assert "phone" in result["reply"].lower()
        assert persisted[SK.awaiting_field] == "phone"


class TestHandleFinalConfirmation:
    def test_returns_none_without_receipt(self):
        persisted = {**COMPLETE_FILTERS}
        result = handle_final_confirmation("yes", persisted, _mock_is_yes, _mock_is_no)
        assert result is None

    def test_confirms_booking_on_yes(self):
        persisted = {**COMPLETE_FILTERS, SK.receipt_shown: True}
        result = handle_final_confirmation("yes", persisted, _mock_is_yes, _mock_is_no)
        assert result is not None
        assert result["tool_result"]["ready_for_booking"] is True
        assert result["booking_args"]["property_id"] == "p1"

    def test_cancels_on_no(self):
        persisted = {**COMPLETE_FILTERS, SK.receipt_shown: True}
        result = handle_final_confirmation("no", persisted, _mock_is_yes, _mock_is_no)
        assert result is not None
        assert result["tool_result"]["cancelled"] is True

    def test_re_renders_receipt_on_total_request(self):
        persisted = {**COMPLETE_FILTERS, SK.receipt_shown: True}
        result = handle_final_confirmation("show total", persisted, _mock_is_yes, _mock_is_no)
        assert result is not None
        assert "BOOKING SUMMARY" in result["reply"]


class TestHandlePostModification:
    def test_returns_none_without_flag(self):
        result = handle_post_modification_choice("proceed", {})
        assert result is None

    def test_proceeds_to_receipt(self):
        persisted = {**COMPLETE_FILTERS, SK.awaiting_post_mod_choice: True}
        result = handle_post_modification_choice("yes", persisted)
        assert result is not None
        assert "BOOKING SUMMARY" in result["reply"]

    def test_allows_more_modifications(self):
        persisted = {**COMPLETE_FILTERS, SK.awaiting_post_mod_choice: True}
        result = handle_post_modification_choice("modify", persisted)
        assert result is not None
        assert "modify" in result["reply"].lower()


class TestHandleSelectionConfirm:
    def test_returns_none_without_flag(self):
        result = handle_selection_confirm("yes", {}, _mock_is_yes, _mock_is_no)
        assert result is None

    def test_confirms_and_shows_receipt(self):
        persisted = {**COMPLETE_FILTERS, SK.awaiting_selection_confirm: True}
        result = handle_selection_confirm("yes", persisted, _mock_is_yes, _mock_is_no)
        assert result is not None
        assert "BOOKING SUMMARY" in result["reply"]

    def test_declines_selection(self):
        persisted = {**COMPLETE_FILTERS, SK.awaiting_selection_confirm: True}
        result = handle_selection_confirm("no", persisted, _mock_is_yes, _mock_is_no)
        assert result is not None
        assert "end" in str(result.get("tool_result", {}))
