# tests/test_confirmation_helpers.py
# Unit tests for the modular confirmation helpers.

from services.confirmation_helpers import (
    FIELD_PROMPTS,
    REQUIRED_FIELDS,
    _ask_for_field,
    _next_missing_field,
    _render_receipt,
    _try_show_receipt,
    build_restart_filters,
    capture_awaited_field,
    handle_final_confirmation,
    handle_inline_receipt_updates,
    handle_post_modification_choice,
    handle_property_selection,
    handle_selection_confirm,
    normalize_date_value,
    route_requested_modifications,
)
from services.state_keys import SK


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
        "id": "p1",
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
        assert "4" in receipt
        assert "$800" in receipt

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
        assert result["reply"] == FIELD_PROMPTS["name"]
        assert persisted[SK.awaiting_field] == "name"

    def test_asks_for_email(self):
        persisted = {}
        result = _ask_for_field("email", persisted)
        assert result["reply"] == FIELD_PROMPTS["email"]


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

    def test_cancels_on_no_and_opens_modification_choice(self):
        persisted = {**COMPLETE_FILTERS, SK.receipt_shown: True}
        result = handle_final_confirmation("no", persisted, _mock_is_yes, _mock_is_no)
        assert result is not None
        assert result["tool_result"]["cancelled"] is True
        assert persisted[SK.awaiting_field] == "modification_choice"

    def test_re_renders_receipt_on_total_request(self):
        persisted = {**COMPLETE_FILTERS, SK.receipt_shown: True}
        result = handle_final_confirmation("show total", persisted, _mock_is_yes, _mock_is_no)
        assert result is not None
        assert "BOOKING SUMMARY" in result["reply"]


class TestHandlePostModificationChoice:
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
        assert persisted[SK.awaiting_field] == "modification_choice"
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

    def test_declines_selection_and_reuses_previous_results(self):
        persisted = {
            **COMPLETE_FILTERS,
            SK.awaiting_selection_confirm: True,
            "last_results": [COMPLETE_FILTERS[SK.selected_property]],
            "results_index_map": {1: "p1"},
        }
        result = handle_selection_confirm("no", persisted, _mock_is_yes, _mock_is_no)
        assert result is not None
        assert result["tool_result"]["need"] == ["property_selection"]
        assert "Reply with a number to choose" in result["reply"]
        assert SK.selected_property not in persisted


class TestRestartFilters:
    def test_preserves_required_fields_and_clears_selection_state(self):
        persisted = {
            **COMPLETE_FILTERS,
            "city": "miami",
            "budget": 350,
            "results": [COMPLETE_FILTERS[SK.selected_property]],
            "last_results": [COMPLETE_FILTERS[SK.selected_property]],
            "results_index_map": {1: "p1"},
            SK.awaiting_selection_confirm: True,
            SK.awaiting_field: "email",
            SK.receipt_shown: True,
        }
        reset = build_restart_filters(persisted, include_results=True)
        assert reset["name"] == "John Doe"
        assert reset["budget"] == 350
        assert "results" in reset
        assert SK.selected_property not in reset
        assert SK.awaiting_selection_confirm not in reset
        assert SK.awaiting_field not in reset
        assert SK.receipt_shown not in reset


class TestRouteRequestedModifications:
    def test_routes_dates_through_shared_prompt_path(self):
        persisted = {**COMPLETE_FILTERS}
        result = route_requested_modifications(persisted, ["dates"])
        assert result["tool_result"]["need"] == ["check_in"]
        assert persisted[SK.awaiting_field] == "check_in"
        assert persisted["check_in"] is None
        assert persisted["check_out"] is None

    def test_inline_apply_updates_and_sets_post_mod_choice(self):
        persisted = {**COMPLETE_FILTERS}
        result = route_requested_modifications(
            persisted,
            ["email"],
            parsed_email="updated@example.com",
            allow_inline_apply=True,
        )
        assert persisted["email"] == "updated@example.com"
        assert persisted[SK.awaiting_post_mod_choice] is True
        assert result["tool_result"]["need"] == ["post_mod_choice"]

    def test_property_restart_preserves_user_details(self):
        persisted = {
            **COMPLETE_FILTERS,
            "city": "miami",
            "last_results": [COMPLETE_FILTERS[SK.selected_property]],
            "results_index_map": {1: "p1"},
            SK.awaiting_selection_confirm: True,
        }
        result = route_requested_modifications(persisted, ["property"])
        filters = result["filters"]
        assert result["tool_result"]["need"] == ["restart"]
        assert filters["name"] == "John Doe"
        assert SK.selected_property not in filters
        assert SK.awaiting_selection_confirm not in filters


class TestCaptureAwaitedField:
    def test_normalizes_slash_dates(self):
        persisted = {SK.awaiting_field: "check_in"}
        result = capture_awaited_field(
            persisted,
            "2027/06/10",
            parsed_name=None,
            parsed_phone=None,
            parsed_email=None,
            parsed_guests=None,
            parsed_dates=["2027/06/10"],
        )
        assert result["updated"] is True
        assert persisted["check_in"] == "2027-06-10"
        assert persisted[SK.awaiting_field] is None

    def test_malformed_dates_do_not_crash_and_ask_again(self):
        persisted = {**COMPLETE_FILTERS, SK.awaiting_field: "check_in"}
        result = capture_awaited_field(
            persisted,
            "2027.06.10",
            parsed_name=None,
            parsed_phone=None,
            parsed_email=None,
            parsed_guests=None,
            parsed_dates=["2027.06.10"],
        )
        assert result["response"] is not None
        assert "check-in" in result["response"]["reply"].lower()
        assert persisted["check_in"] == "2027-06-01"


class TestInlineReceiptUpdates:
    def test_direct_field_update_rerenders_receipt(self):
        persisted = {**COMPLETE_FILTERS, SK.receipt_shown: True}
        result = handle_inline_receipt_updates(
            persisted,
            parsed_name=None,
            parsed_phone="+15557654321",
            parsed_email=None,
            parsed_guests=None,
            parsed_dates=[],
        )
        assert result is not None
        assert "BOOKING SUMMARY" in result["reply"]
        assert "+15557654321" in result["reply"]
        assert persisted["phone"] == "+15557654321"


class TestNormalizeDateValue:
    def test_rejects_unsupported_format(self):
        assert normalize_date_value("2027.06.01") is None

    def test_accepts_iso_and_slash(self):
        assert normalize_date_value("2027-06-01") == "2027-06-01"
        assert normalize_date_value("2027/06/01") == "2027-06-01"


def test_required_fields_contract_still_has_core_slots():
    assert REQUIRED_FIELDS == ["name", "phone", "email", "check_in", "check_out", "guests"]
