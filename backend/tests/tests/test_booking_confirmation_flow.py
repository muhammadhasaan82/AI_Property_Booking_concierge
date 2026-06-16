from __future__ import annotations

import copy
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.state import booking_state as booking_state_module

from app.agents.tools.search import search_properties, select_property
from app.config.conversation_shortcuts_loader import match_shortcut
from app.services import adk_runner


@pytest.fixture(autouse=True)
def stable_booking_reference_date(monkeypatch):
    """
    Set the BOOKING_REFERENCE_DATE environment variable to 2026-06-01 to make booking-date handling deterministic in tests.
    
    This pytest fixture ensures booking-related logic that depends on the current/reference date behaves consistently across test runs by fixing the reference date to June 1, 2026.
    """
    monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")


class _Ctx:
    def __init__(self):
        """
        Initialize the context with an empty soft-state container.
        
        Sets `self.state` to a dictionary with a single key `"soft_state"` initialized to an empty dictionary.
        """
        self.state = {"soft_state": {}}


def _make_props(count: int) -> list[dict]:
    return [
        {
            "id": f"apt-{idx}",
            "title": f"Apartment {idx}",
            "city": "New York",
            "price_per_night": 100 + idx,
            "property_type": "Apartment",
            "bedrooms": 1,
            "bathrooms": 1,
            "rating": 4.0,
            "reviews_count": idx,
            "amenities": ["wifi"],
            "description": f"Apartment {idx} in New York",
        }
        for idx in range(1, count + 1)
    ]


async def _run_booking_confirmation(monkeypatch, message: str, option_number: int = 3):
    """
    Run a simulated booking-confirmation ADK turn against a patched test environment and return the reply and captured state.
    
    Parameters:
        message (str): User utterance to send to the ADK turn.
        option_number (int): 1-based index of the search result to select before starting the booking flow.
    
    Returns:
        tuple: A 6-tuple containing:
            - full_reply_text (str): Concatenated assistant reply produced by the ADK turn.
            - snapshot (dict): In-memory session snapshot used and updated during the turn.
            - saved_states (list[dict]): Sequence of `state` objects that were passed to the fake save_session_snapshot.
            - search_result (dict): The mocked search results returned by the patched search.
            - selected (dict): The property selection result produced before starting the booking flow.
            - route_pre_adk (AsyncMock): The AsyncMock used for `route_pre_adk` so callers can inspect await count and return value.
    """
    ctx = _Ctx()
    fake = _make_props(8)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        search_result = await search_properties(
            city="New York",
            property_type="apartment",
            tool_context=ctx,
        )
        selected = await select_property(option_number=option_number, tool_context=ctx)

    assert selected["status"] == "property_details"
    assert ctx.state["soft_state"]["last_presented_view"] == "property_details"

    snapshot = {
        "state": {"soft_state": ctx.state["soft_state"]},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-booking-confirm",
            "last_update_time": 1.0,
        },
    }
    saved_states: list[dict] = []

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        saved_states.append(state)
        snapshot["state"] = state

    def fail_get_runner():
        raise AssertionError("ADK runner must not be invoked for booking confirmation")

    route_pre_adk = AsyncMock(return_value={"reply": "generic concierge greeting"})
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "maybe_handle_direct_property_search", AsyncMock(return_value=None))
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "_get_runner", fail_get_runner)
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-booking-confirm", "s-booking-confirm", message):
        chunks.append(chunk)

    return "".join(chunks), snapshot, saved_states, search_result, selected, route_pre_adk


async def _run_booking_followup_turn(monkeypatch, snapshot: dict, saved_states: list[dict], message: str):
    """
    Run a follow-up booking shortcut turn using a provided session snapshot and record any saved soft-state mutations.
    
    Parameters:
        monkeypatch: pytest monkeypatch fixture used to patch adk_runner behaviors for the test.
        snapshot (dict): The session snapshot to be returned by patched `get_session_snapshot` (will be deep-copied).
        saved_states (list[dict]): Mutable list to which each saved `state` deep-copy will be appended when `save_session_snapshot` is called.
        message (str): The user message to send to `adk_runner.run_adk_turn`.
    
    Returns:
        tuple:
            - full_reply_text (str): The concatenated reply chunks produced by the turn.
            - route_pre_adk (AsyncMock): The AsyncMock instance patched in for `adk_runner.route_pre_adk` (useful for await-count assertions).
    """
    async def fake_get_session_snapshot(_session_id):
        return copy.deepcopy(snapshot)

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        """
        Save a deep-copied session snapshot into the shared in-memory test structures.
        
        Parameters:
            session_id (str): Ignored by this fake; included to match the real API.
            history (list): Conversation history to deep-copy into snapshot["history"].
            state (dict): Session state to deep-copy into snapshot["state"] and to append to `saved_states`.
            metadata (dict): Metadata to deep-copy and merge into snapshot["meta"].
        """
        state_to_save = copy.deepcopy(state)
        snapshot["state"] = state_to_save
        snapshot["history"] = copy.deepcopy(history)
        snapshot["meta"].update(copy.deepcopy(metadata or {}))
        saved_states.append(state_to_save)

    def fail_get_runner():
        """
        Prevent accidental invocation of the ADK runner during active booking shortcut turns.
        
        Raises:
            AssertionError: always raised with the message "ADK runner must not be invoked for active booking shortcut turns".
        """
        raise AssertionError("ADK runner must not be invoked for active booking shortcut turns")

    route_pre_adk = AsyncMock(return_value={"reply": "generic concierge greeting"})
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "maybe_handle_direct_property_search", AsyncMock(return_value=None))
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "_get_runner", fail_get_runner)
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-booking-confirm", "s-booking-confirm", message):
        chunks.append(chunk)

    return "".join(chunks), route_pre_adk


def _snapshot_from_soft_state(soft_state: dict, *, user_id: str) -> dict:
    return {
        "state": {"soft_state": copy.deepcopy(soft_state)},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": user_id,
            "last_update_time": 1.0,
        },
    }


async def _run_live_booking_turn(monkeypatch, snapshot: dict, message: str):
    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = copy.deepcopy(state)
        snapshot["history"] = copy.deepcopy(history)
        snapshot["meta"].update(copy.deepcopy(metadata or {}))

    def fail_get_runner():
        raise AssertionError("ADK runner must not be invoked for deterministic booking priority tests")

    route_pre_adk = AsyncMock(return_value={"reply": "generic concierge greeting"})
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "_get_runner", fail_get_runner)
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-booking-live", "s-booking-live", message):
        chunks.append(chunk)

    return "".join(chunks), route_pre_adk


@pytest.mark.asyncio
async def test_yes_after_property_details_starts_booking_details_collection(monkeypatch):
    reply, snapshot, saved_states, search_result, selected, route_pre_adk = (
        await _run_booking_confirmation(monkeypatch, "yeah sure", option_number=3)
    )

    selected_property = selected["property"]
    soft_state = snapshot["state"]["soft_state"]

    assert "generic concierge greeting" not in reply
    assert "Please provide" in reply
    assert "Full name" in reply
    assert selected_property["title"] in reply
    route_pre_adk.assert_not_awaited()
    assert soft_state["booking_stage"] == "collecting_details"
    assert soft_state["booking_property_id"] == selected_property["id"]
    assert soft_state["last_presented_view"] == "booking_details_request"
    assert soft_state["booking_selected_property"]["id"] == selected_property["id"]
    assert soft_state["booking_required_fields"]
    assert soft_state["booking_property_id"] == soft_state["last_selected_property_id"]
    assert soft_state["visible_results"]
    assert soft_state["option_map"]
    assert soft_state["all_search_results"]
    assert soft_state["all_search_results"][2]["id"] == search_result["properties"][2]["id"]
    assert saved_states


@pytest.mark.asyncio
async def test_booking_confirmation_sequence_persists_correct_dates_across_invalid_fix_and_confirm(monkeypatch):
    start_reply, snapshot, saved_states, _search_result, selected, route_pre_adk = (
        await _run_booking_confirmation(monkeypatch, "yeah sure", option_number=7)
    )

    assert "Please provide" in start_reply
    assert route_pre_adk.await_count == 0
    assert snapshot["state"]["soft_state"]["booking_stage"] == "collecting_details"
    assert snapshot["state"]["soft_state"]["booking_selected_property"]["id"] == selected["property"]["id"]

    invalid_reply, route_pre_adk = await _run_booking_followup_turn(
        monkeypatch,
        snapshot,
        saved_states,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2025, we are around 4 guests",
    )

    invalid_soft_state = saved_states[-1]["soft_state"]
    invalid_booking_state = invalid_soft_state["booking_state"]

    assert route_pre_adk.await_count == 0
    assert "cannot be earlier than your check-in date" in invalid_reply
    assert invalid_soft_state["booking_stage"] == "collecting_details"
    assert invalid_soft_state["awaiting_field"] == "check_out"
    assert invalid_booking_state["check_in"] == "2026-06-02"
    assert invalid_booking_state["guest_name"] == "Jane Doe"
    assert invalid_booking_state["guest_email"] == "jane@example.com"
    assert invalid_booking_state["guest_phone"] == "03001234567"
    assert invalid_booking_state["guests"] == 4
    assert "check_out" not in invalid_booking_state

    review_reply, route_pre_adk = await _run_booking_followup_turn(
        monkeypatch,
        snapshot,
        saved_states,
        "11 june 2026",
    )

    review_soft_state = saved_states[-1]["soft_state"]
    review = review_soft_state["booking_review"]

    assert route_pre_adk.await_count == 0
    assert "Please confirm if everything is correct." in review_reply
    assert review_soft_state["booking_stage"] == "awaiting_confirmation"
    assert review["check_in"] == "2026-06-02"
    assert review["check_out"] == "2026-06-11"
    assert review["guests"] == 4

    receipt_reply, route_pre_adk = await _run_booking_followup_turn(
        monkeypatch,
        snapshot,
        saved_states,
        "yes",
    )

    confirmed_soft_state = saved_states[-1]["soft_state"]

    assert route_pre_adk.await_count == 0
    assert "Your booking is confirmed." in receipt_reply
    assert confirmed_soft_state["booking_stage"] == "confirmed"
    assert confirmed_soft_state["booking_receipt"]["check_in"] == "2026-06-02"
    assert confirmed_soft_state["booking_receipt"]["check_out"] == "2026-06-11"


@pytest.mark.asyncio
async def test_book_this_property_after_details_starts_booking(monkeypatch):
    reply, snapshot, _saved_states, _search_result, selected, route_pre_adk = (
        await _run_booking_confirmation(
            monkeypatch,
            "I want to book this property",
            option_number=3,
        )
    )

    selected_property = selected["property"]
    soft_state = snapshot["state"]["soft_state"]

    assert "Please provide" in reply
    assert selected_property["title"] in reply
    route_pre_adk.assert_not_awaited()
    assert soft_state["booking_stage"] == "collecting_details"
    assert soft_state["booking_property_id"] == selected_property["id"]


@pytest.mark.asyncio
async def test_affirmative_without_property_details_does_not_start_booking(monkeypatch):
    snapshot = {
        "state": {
            "soft_state": {
                "last_presented_view": "property_list",
                "last_selected_property_id": "apt-1",
                "visible_results": [{"id": "apt-1", "title": "Apartment 1"}],
            }
        },
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-no-booking-confirm",
            "last_update_time": 1.0,
        },
    }

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    def fail_get_runner():
        raise AssertionError("ADK runner should not be reached when pre-router handles fallthrough")

    route_pre_adk = AsyncMock(return_value={"reply": "normal fallthrough"})
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "maybe_handle_direct_property_search", AsyncMock(return_value=None))
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "_get_runner", fail_get_runner)
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-no-booking-confirm", "s-no-booking-confirm", "yeah sure"):
        chunks.append(chunk)

    assert "".join(chunks) == "normal fallthrough"
    route_pre_adk.assert_awaited_once()
    soft_state = snapshot["state"]["soft_state"]
    assert "booking_stage" not in soft_state
    assert "booking_property_id" not in soft_state


def test_booking_confirm_shortcut_loaded_from_yaml():
    state = {
        "last_presented_view": "property_details",
        "last_selected_property_id": "apt-1",
    }

    match = match_shortcut("yeah sure", state)

    assert match is not None
    assert match.action == "start_booking_for_selected_property"
    assert match_shortcut("continue", state).action == "start_booking_for_selected_property"


@pytest.mark.asyncio
async def test_booking_confirmation_preserves_search_state(monkeypatch):
    _reply, snapshot, _saved_states, search_result, selected, _route_pre_adk = (
        await _run_booking_confirmation(monkeypatch, "book this property", option_number=3)
    )

    soft_state = snapshot["state"]["soft_state"]

    assert len(soft_state["visible_results"]) == len(search_result["properties"])
    assert set(soft_state["option_map"]) == {str(i) for i in range(1, 9)}
    assert len(soft_state["all_search_results"]) == 8
    assert soft_state["last_selected_property_id"] == selected["property"]["id"]
    assert soft_state["booking_property_id"] == selected["property"]["id"]
    assert soft_state["booking_stage"] == "collecting_details"


def test_start_booking_for_selected_property_restores_top_level_keys_after_helpers():
    """Top-level booking keys must survive canonical booking_state helper side effects."""
    selected_property = {
        "id": "apt-7",
        "title": "Apartment 7",
        "city": "New York",
        "price_per_night": 107,
    }
    soft_state = {
        "last_selected_property_id": selected_property["id"],
        "last_presented_view": "property_details",
        "visible_results": [selected_property],
        "option_map": {"7": {"property_id": selected_property["id"]}},
    }

    real_update = booking_state_module.update_booking_state

    def update_then_strip_top_level(soft_state_arg, updates):
        state = real_update(soft_state_arg, updates)
        soft_state_arg.pop("booking_stage", None)
        soft_state_arg.pop("booking_property_id", None)
        soft_state_arg.pop("booking_required_fields", None)
        return state

    with patch.object(
        booking_state_module,
        "update_booking_state",
        side_effect=update_then_strip_top_level,
    ):
        payload = adk_runner._start_booking_for_selected_property(soft_state)

    assert payload is not None
    assert payload["status"] == "booking_details_required"
    assert soft_state["booking_stage"] == "collecting_details"
    assert soft_state["booking_property_id"] == selected_property["id"]
    assert soft_state["last_presented_view"] == "booking_details_request"
    assert soft_state["booking_selected_property"]["id"] == selected_property["id"]
    assert soft_state["booking_required_fields"]
    assert soft_state["booking_property_id"] == soft_state["last_selected_property_id"]

_TEST_RECEIPT = {
    "booking_id": "BK-20260607-C48A4AFA",
    "property_title": "Apartment 3",
    "guest_name": "Jane Doe",
    "guest_email": "jane@example.com",
    "guest_phone": "03001234567",
    "check_in": "2026-06-02",
    "check_out": "2026-06-11",
    "nights": 9,
    "guests": 4,
    "price_per_night": 103.0,
    "total_amount": 927.0,
    "status": "confirmed",
}


def _build_status_check_snapshot(soft_state_overrides: dict | None = None) -> dict:
    """Build a minimal session snapshot for booking-status tests."""
    soft_state = dict(soft_state_overrides) if soft_state_overrides else {}
    return {
        "state": {"soft_state": soft_state},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-status-check",
            "last_update_time": 1.0,
        },
    }


async def _run_status_check_turn(
    monkeypatch,
    snapshot: dict,
    message: str,
) -> tuple[str, AsyncMock]:
    """
    Run a single turn through the booking-status handler and return the reply and route_pre_adk mock.

    Patches adk_runner so the turn stops after the booking_status_check step
    (route_pre_adk and _get_runner both raise if reached).
    """
    route_pre_adk = AsyncMock(return_value={"reply": "generic fallback"})

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history
        snapshot["meta"].update(metadata or {})

    def fail_get_runner():
        raise AssertionError("ADK runner must not be invoked for booking-status lookup")

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "maybe_handle_direct_property_search", AsyncMock(return_value=None))
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "_get_runner", fail_get_runner)
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-status-check", "s-status-check", message):
        chunks.append(chunk)

    return "".join(chunks), route_pre_adk


@pytest.mark.asyncio
async def test_booking_id_lookup_uses_current_session_receipt(monkeypatch):
    """
    Seed soft_state.booking_receipt and verify that a "my booking ID is BK-..." message
    returns deterministic status without invoking route_pre_adk.
    """
    snapshot = _build_status_check_snapshot({"booking_receipt": dict(_TEST_RECEIPT)})

    reply, route_pre_adk = await _run_status_check_turn(
        monkeypatch,
        snapshot,
        "My booking ID is BK-20260607-C48A4AFA",
    )

    route_pre_adk.assert_not_awaited()
    assert "BK-20260607-C48A4AFA" in reply
    assert "Jane Doe" in reply
    assert "jane@example.com" in reply
    assert "June 2, 2026" in reply
    assert "June 11, 2026" in reply
    assert "Apartment 3" in reply
    assert "confirmed" in reply
    assert "$927.00" in reply


@pytest.mark.asyncio
async def test_booking_status_without_id_uses_latest_session_receipt(monkeypatch):
    """
    When the user asks for booking status without providing an ID and a receipt
    exists in the session, return the latest receipt's status.
    """
    snapshot = _build_status_check_snapshot({"booking_receipt": dict(_TEST_RECEIPT)})

    reply, route_pre_adk = await _run_status_check_turn(
        monkeypatch,
        snapshot,
        "I want to check my booking status",
    )

    route_pre_adk.assert_not_awaited()
    assert "BK-20260607-C48A4AFA" in reply
    assert "Jane Doe" in reply
    assert "Apartment 3" in reply
    assert "confirmed" in reply


@pytest.mark.asyncio
async def test_booking_status_without_receipt_asks_for_booking_id(monkeypatch):
    """
    When the user asks for booking status but there is no receipt in the session,
    the system should ask the user for their booking ID.
    """
    snapshot = _build_status_check_snapshot({})

    reply, route_pre_adk = await _run_status_check_turn(
        monkeypatch,
        snapshot,
        "I want to check my booking status",
    )

    route_pre_adk.assert_not_awaited()
    lower_reply = reply.lower()
    assert "registration id" in lower_reply or "booking id" in lower_reply
    assert "BK-" in reply


@pytest.mark.asyncio
async def test_booking_status_wrong_id_not_found(monkeypatch):
    """
    When a booking ID is provided that doesn't match any receipt in the session
    (and DB is unreachable/mock), return a not-found message without invoking ADK.
    """
    snapshot = _build_status_check_snapshot({"booking_receipt": dict(_TEST_RECEIPT)})

    with (
        patch(
            "app.services.booking.get_booking_status",
            AsyncMock(return_value={"ok": False, "error": "not found"}),
        ),
        patch(
            "app.observability.db_logging.get_successful_booking_status",
            AsyncMock(return_value=None),
        ),
    ):
        reply, route_pre_adk = await _run_status_check_turn(
            monkeypatch,
            snapshot,
            "My booking ID is BK-20260607-UNKNOWN",
        )

    route_pre_adk.assert_not_awaited()
    assert "wasn't found" in reply.lower() or "not found" in reply.lower()


@pytest.mark.asyncio
async def test_booking_status_falls_back_to_successful_bookings(monkeypatch):
    """
    When session has no receipt, bookings lookup misses, but successful_bookings
    has the confirmed row, return persisted status instead of not-found.
    """
    snapshot = _build_status_check_snapshot({})
    booking_id = "BK-20260607-C48A4AFA"
    successful_row = {
        "booking_id": booking_id,
        "status": "confirmed",
        "check_in": "2026-06-02",
        "check_out": "2026-06-11",
        "user_name": "Jane Doe",
        "user_email": "jane@example.com",
        "property_title": "Apartment 3",
        "city": "Test City",
        "payment_url": "https://pay.example.com",
    }

    with (
        patch(
            "app.services.booking.get_booking_status",
            AsyncMock(return_value={"ok": False, "error": "not found"}),
        ),
        patch(
            "app.observability.db_logging.get_successful_booking_status",
            AsyncMock(return_value=successful_row),
        ),
    ):
        reply, route_pre_adk = await _run_status_check_turn(
            monkeypatch,
            snapshot,
            f"My booking ID is {booking_id}",
        )

    route_pre_adk.assert_not_awaited()
    lower_reply = reply.lower()
    assert "wasn't found" not in lower_reply
    assert "not found" not in lower_reply
    assert booking_id in reply
    assert "Jane Doe" in reply
    assert "jane@example.com" in reply
    assert "Apartment 3" in reply
    assert "confirmed" in lower_reply
    assert "June 2, 2026" in reply
    assert "June 11, 2026" in reply


@pytest.mark.asyncio
async def test_booking_status_both_db_lookups_fail_not_found(monkeypatch):
    """
    When session has no receipt and both bookings and successful_bookings lookups
    fail, return the not-found template.
    """
    snapshot = _build_status_check_snapshot({})

    with (
        patch(
            "app.services.booking.get_booking_status",
            AsyncMock(return_value={"ok": False, "error": "not found"}),
        ),
        patch(
            "app.observability.db_logging.get_successful_booking_status",
            AsyncMock(return_value=None),
        ),
    ):
        reply, route_pre_adk = await _run_status_check_turn(
            monkeypatch,
            snapshot,
            "My booking ID is BK-20260607-NOTFOUND",
        )

    route_pre_adk.assert_not_awaited()
    assert "wasn't found" in reply.lower() or "not found" in reply.lower()


@pytest.mark.asyncio
async def test_confirm_then_lookup_booking_id_e2e(monkeypatch):
    """
    End-to-end: create a booking through the shortcut flow, capture the generated
    registration ID, then send "My booking ID is <id>" and verify it's found.
    """
    reply, snapshot, saved_states, _search_result, _selected, route_pre_adk = (
        await _run_booking_confirmation(monkeypatch, "yeah sure", option_number=3)
    )
    assert route_pre_adk.await_count == 0
    assert snapshot["state"]["soft_state"]["booking_stage"] == "collecting_details"

    review_reply, route_pre_adk = await _run_booking_followup_turn(
        monkeypatch,
        snapshot,
        saved_states,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, we are around 4 guests",
    )
    assert route_pre_adk.await_count == 0
    assert "Please confirm" in review_reply

    receipt_reply, route_pre_adk = await _run_booking_followup_turn(
        monkeypatch,
        snapshot,
        saved_states,
        "yes",
    )
    assert route_pre_adk.await_count == 0
    assert "Your booking is confirmed." in receipt_reply

    soft_state = snapshot["state"]["soft_state"]
    registration_id = soft_state["booking_registration_id"]
    assert registration_id
    assert registration_id.startswith("BK-")

    status_reply, route_pre_adk = await _run_status_check_turn(
        monkeypatch,
        snapshot,
        f"My booking ID is {registration_id}",
    )

    assert route_pre_adk.await_count == 0
    assert registration_id in status_reply
    assert "confirmed" in status_reply
    assert "Jane Doe" in status_reply


@pytest.mark.asyncio
async def test_lookup_booking_id_cross_session(monkeypatch):
    """
    Regression test:
    - confirm booking in one session (persists to DB)
    - create a fresh empty soft_state/new session
    - ask status using the booking ID (using raw ID and spelling typos like 'boking id')
    - assert status is found from persistent storage and matches receipt details.
    """
    reply, snapshot, saved_states, _search_result, _selected, route_pre_adk = (
        await _run_booking_confirmation(monkeypatch, "yeah sure", option_number=3)
    )
    assert route_pre_adk.await_count == 0

    review_reply, route_pre_adk = await _run_booking_followup_turn(
        monkeypatch,
        snapshot,
        saved_states,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, we are around 4 guests",
    )
    assert route_pre_adk.await_count == 0

    receipt_reply, route_pre_adk = await _run_booking_followup_turn(
        monkeypatch,
        snapshot,
        saved_states,
        "yes",
    )
    assert route_pre_adk.await_count == 0
    assert "Your booking is confirmed." in receipt_reply

    soft_state = snapshot["state"]["soft_state"]
    registration_id = soft_state["booking_registration_id"]
    assert registration_id
    assert registration_id.startswith("BK-")
    persisted_receipt = dict(soft_state["booking_receipt"])
    successful_row = {
        "booking_id": registration_id,
        "payment_url": persisted_receipt.get("payment_url"),
    }
    for receipt_key, db_key in {
        "guest_name": "user_name",
        "guest_email": "user_email",
        "guest_phone": "user_phone",
        "check_in": "check_in",
        "check_out": "check_out",
        "guests": "guests",
        "nights": "nights",
        "price_per_night": "price_per_night",
        "total_amount": "total_amount",
        "property_title": "property_title",
        "city": "city",
        "status": "status",
    }.items():
        if receipt_key in persisted_receipt:
            successful_row[db_key] = persisted_receipt[receipt_key]

    fresh_snapshot = _build_status_check_snapshot({})

    queries = [
        f"My boking ID is {registration_id}",
        f"booking id {registration_id}",
        f"{registration_id}",
    ]

    with (
        patch(
            "app.services.booking.persistence.get_booking_status",
            AsyncMock(return_value={"ok": False, "error": "not found"}),
        ),
        patch(
            "app.observability.db_logging.get_successful_booking_status",
            AsyncMock(return_value=successful_row),
        ),
    ):
        for query in queries:
            status_reply, route_pre_adk = await _run_status_check_turn(
                monkeypatch,
                fresh_snapshot,
                query,
            )
            assert route_pre_adk.await_count == 0
            assert registration_id in status_reply
            assert "confirmed" in status_reply.lower()
            assert "Jane Doe" in status_reply
            assert "jane@example.com" in status_reply
            assert "June 2, 2026" in status_reply
            assert "June 11, 2026" in status_reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rust_result",
    [
        {"fallback": True},
        {"result": {"results": list(reversed(_make_props(8)))}},
    ],
)
async def test_seeded_search_paths_preserve_booking_flow_to_receipt(monkeypatch, rust_result):
    ctx = _Ctx()
    fake = _make_props(8)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value=rust_result,
    ):
        search_result = await search_properties(
            city="New York",
            property_type="apartment",
            tool_context=ctx,
        )
        selected = await select_property(option_number=1, tool_context=ctx)

    assert search_result["status"] == "properties_found"
    assert search_result["properties"]
    assert selected["status"] == "property_details"

    snapshot = _snapshot_from_soft_state(ctx.state["soft_state"], user_id="u-seeded-booking")

    reply_1, route_pre_adk = await _run_live_booking_turn(monkeypatch, snapshot, "yes please")
    assert "Please provide" in reply_1
    assert route_pre_adk.await_count == 0
    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["booking_stage"] == "collecting_details"
    assert soft_state["active_flow"] == "booking"
    assert soft_state["booking_selected_property"]["id"] == selected["property"]["id"]

    snapshot = copy.deepcopy(snapshot)

    reply_2, route_pre_adk = await _run_live_booking_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, we are around 4 guests",
    )
    assert "Please confirm if everything is correct." in reply_2
    assert "I found" not in reply_2
    assert route_pre_adk.await_count == 0
    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["booking_stage"] == "awaiting_confirmation"
    assert soft_state["active_flow"] == "booking"
    assert soft_state["booking_review"]["guest_name"] == "Jane Doe"
    assert soft_state["booking_review"]["property_id"] == selected["property"]["id"]

    snapshot = copy.deepcopy(snapshot)

    with patch("app.observability.db_logging.insert_successful_booking", new=AsyncMock()):
        reply_3, route_pre_adk = await _run_live_booking_turn(monkeypatch, snapshot, "confirm")

    assert route_pre_adk.await_count == 0
    assert "confirmed" in reply_3.lower()
    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["booking_stage"] == "confirmed"
    assert soft_state["active_flow"] == "booking"
    assert soft_state["booking_receipt"]["property_title"] == selected["property"]["title"]
    assert soft_state["booking_registration_id"].startswith("BK-")
