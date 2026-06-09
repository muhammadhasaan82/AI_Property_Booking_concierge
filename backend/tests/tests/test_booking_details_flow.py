from __future__ import annotations

import copy
import re
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.status_codes import Status
from app.agents.tools.search import search_properties, select_property
from app.services import adk_runner, booking_flow

@pytest.fixture(autouse=True)
def stable_booking_reference_date(monkeypatch):
    """
    Set a fixed booking reference date for tests.
    
    This fixture forces the BOOKING_REFERENCE_DATE environment variable to "2026-06-01" so date-dependent booking tests run deterministically.
    """
    monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")

class _Ctx:
    def __init__(self):
        """
        Initialize a mutable context container used by booking-related tools.
        
        Creates the `state` attribute initialized to `{"soft_state": {}}` for holding transient booking soft-state data.
        """
        self.state = {"soft_state": {}}


def _make_props(count: int) -> list[dict]:
    return [
        {
            "id": f"apt-{idx}",
            "title": f"Seattle Apartment {idx}",
            "city": "Seattle",
            "price_per_night": 100 + idx,
            "property_type": "Apartment",
            "bedrooms": 2,
            "bathrooms": 1,
            "rating": 4.2,
            "reviews_count": 20 + idx,
            "amenities": ["wifi", "pets"],
            "description": f"Apartment {idx} in Seattle",
            "occupancy_max": 5,
        }
        for idx in range(1, count + 1)
    ]


async def _seed_booking_snapshot(option_number: int = 10) -> tuple[dict, dict]:
    """
    Create a deterministic session snapshot for a seeded booking flow and return it alongside the selected property.
    
    Parameters:
        option_number (int): 1-based index of the property option to select from the generated fake results.
    
    Returns:
        tuple:
            snapshot (dict): Session snapshot with keys:
                - state: {'soft_state': ...} representing the booking tool's soft state.
                - history: list, initially empty.
                - meta: dict containing 'app_name', 'user_id', and 'last_update_time'.
            selected_property (dict): The `property` dictionary chosen from the synthetic dataset.
    """
    ctx = _Ctx()
    fake = _make_props(12)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        await search_properties(
            city="Seattle",
            property_type="apartment",
            tool_context=ctx,
        )
        selected = await select_property(option_number=option_number, tool_context=ctx)

    booking_flow.start_booking_for_selected_property(ctx.state["soft_state"])
    snapshot = {
        "state": {"soft_state": ctx.state["soft_state"]},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-booking-details",
            "last_update_time": 1.0,
        },
    }
    return snapshot, selected["property"]


async def _run_turn(
    monkeypatch,
    snapshot: dict,
    message: str,
    *,
    faq_answer: str | None = None,
    deepcopy_session_io: bool = False,
    saved_states: list[dict] | None = None,
):
    """
    Run a single booking ADK turn with session IO and ADK runner interactions stubbed for tests.
    
    Parameters:
        monkeypatch: pytest monkeypatch fixture used to replace ADK and booking_flow callables.
        snapshot (dict): Mutable session snapshot that will be used as the source of truth for the turn; it may be mutated in-place when the turn saves state/history/metadata.
        message (str): User message to feed into the ADK turn.
        faq_answer (str | None): If provided, `booking_flow.check_faq` is mocked to return this answer with a status of `Status.ANSWERED`.
        deepcopy_session_io (bool): If true, the helper will deep-copy session state/history/metadata when returning or saving snapshots to simulate isolated IO; otherwise references are used.
        saved_states (list[dict] | None): Optional list that, when provided, will receive deep-copied snapshots of the saved `state` each time `save_session_snapshot` is called.
    
    Returns:
        tuple[str, AsyncMock]: A tuple where the first element is the concatenated reply produced by streaming ADK output chunks, and the second element is the `route_pre_adk` AsyncMock used (so tests can assert whether it was awaited).
    """
    async def fake_get_session_snapshot(_session_id):
        return copy.deepcopy(snapshot) if deepcopy_session_io else snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        """
        Persist session pieces while preserving the original nested soft_state dict identity.

        Some tests keep a reference to snapshot["state"]["soft_state"] before the
        turn runs. Replacing snapshot["state"] with a new dict leaves that reference
        stale, so update the existing soft_state mapping in-place.
        """
        existing_state = snapshot.setdefault("state", {})
        existing_soft_state = existing_state.get("soft_state")
        new_soft_state = state.get("soft_state") if isinstance(state, dict) else {}

        if isinstance(existing_soft_state, dict) and isinstance(new_soft_state, dict):
            replacement_soft_state = copy.deepcopy(new_soft_state)
            existing_soft_state.clear()
            existing_soft_state.update(replacement_soft_state)
            existing_state["soft_state"] = existing_soft_state
        else:
            existing_state["soft_state"] = copy.deepcopy(new_soft_state) if deepcopy_session_io else new_soft_state

        if isinstance(state, dict):
            for key, value in state.items():
                if key != "soft_state":
                    existing_state[key] = copy.deepcopy(value) if deepcopy_session_io else value

        snapshot["history"] = copy.deepcopy(history) if deepcopy_session_io else history
        snapshot.setdefault("meta", {}).update(copy.deepcopy(metadata or {}) if deepcopy_session_io else (metadata or {}))

        if saved_states is not None:
            saved_states.append(copy.deepcopy(existing_state))

    route_pre_adk = AsyncMock(return_value={"reply": "generic fallback"})
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "maybe_handle_direct_property_search", AsyncMock(return_value=None))
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "_get_runner", lambda: (_ for _ in ()).throw(AssertionError("ADK runner must not be invoked")))
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)
    if faq_answer is not None:
        monkeypatch.setattr(
            booking_flow,
            "check_faq",
            AsyncMock(return_value={"status": Status.ANSWERED, "answer": faq_answer}),
        )

    chunks: list[str] = []
    async for chunk in adk_runner.run_adk_turn("u-booking-details", "s-booking-details", message):
        chunks.append(chunk)
    return "".join(chunks), route_pre_adk


@pytest.mark.asyncio
async def test_booking_collects_all_details_from_single_message(monkeypatch):
    snapshot, selected_property = await _seed_booking_snapshot()

    reply, route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, we are around 4 guests",
    )

    soft_state = snapshot["state"]["soft_state"]
    review = soft_state["booking_review"]

    assert route_pre_adk.await_count == 0
    assert soft_state["booking_stage"] == "awaiting_confirmation"
    assert review["property_id"] == selected_property["id"]
    assert review["guest_name"] == "Jane Doe"
    assert review["guest_email"] == "jane@example.com"
    assert review["guest_phone"] == "03001234567"
    assert review["check_in"] == "2026-06-02"
    assert review["check_out"] == "2026-06-11"
    assert review["guests"] == 4
    assert review["total"] == 9 * selected_property["price_per_night"]
    assert "Please confirm if everything is correct." in reply


@pytest.mark.asyncio
async def test_checkout_before_checkin_rejected(monkeypatch):
    snapshot, _selected_property = await _seed_booking_snapshot()

    reply, route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2025, we are around 4 guests",
    )

    soft_state = snapshot["state"]["soft_state"]
    booking_state = soft_state["booking_state"]

    assert route_pre_adk.await_count == 0
    assert "cannot be earlier than your check-in date" in reply
    assert soft_state["booking_stage"] == "collecting_details"
    assert soft_state["awaiting_field"] == "check_out"
    assert booking_state["guest_name"] == "Jane Doe"
    assert booking_state["guest_email"] == "jane@example.com"
    assert booking_state["guest_phone"] == "03001234567"
    assert booking_state["guests"] == 4
    assert booking_state["check_in"] == "2026-06-02"
    assert "check_out" not in booking_state


@pytest.mark.asyncio
async def test_details_message_with_explicit_checkin_checkout_ignores_awaiting_field_fallback(monkeypatch):
    """
    Verifies that explicit check-in and check-out dates in a user message override any pre-existing awaiting_field and that an invalid check-out is rejected.
    
    Sends a single combined details message while the soft state initially awaits `check_in`. The message includes explicit check-in (June 2, 2026) and a check-out date that is earlier (June 11, 2025). Asserts that the flow validates the provided dates (rejecting the earlier check-out), updates `awaiting_field` to `"check_out"`, keeps the booking stage at `"collecting_details"`, and persists parsed guest fields and `check_in` while leaving `check_out` absent.
    """
    snapshot, _selected_property = await _seed_booking_snapshot()
    snapshot["state"]["soft_state"]["awaiting_field"] = "check_in"

    reply, route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2025, we are around 4 guests",
    )

    soft_state = snapshot["state"]["soft_state"]
    booking_state = soft_state["booking_state"]

    assert route_pre_adk.await_count == 0
    assert "cannot be earlier than your check-in date" in reply
    assert soft_state["awaiting_field"] == "check_out"
    assert soft_state["booking_stage"] == "collecting_details"
    assert booking_state["check_in"] == "2026-06-02"
    assert booking_state["guest_name"] == "Jane Doe"
    assert booking_state["guest_email"] == "jane@example.com"
    assert booking_state["guest_phone"] == "03001234567"
    assert booking_state["guests"] == 4
    assert "check_out" not in booking_state


@pytest.mark.asyncio
async def test_active_booking_turn_persists_mutated_soft_state_to_saved_snapshot(monkeypatch):
    snapshot, _selected_property = await _seed_booking_snapshot()
    saved_states: list[dict] = []

    invalid_reply, route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2025, we are around 4 guests",
        deepcopy_session_io=True,
        saved_states=saved_states,
    )

    persisted_soft_state = saved_states[-1]["soft_state"]
    persisted_booking_state = persisted_soft_state["booking_state"]

    assert route_pre_adk.await_count == 0
    assert "cannot be earlier than your check-in date" in invalid_reply
    assert persisted_soft_state["booking_stage"] == "collecting_details"
    assert persisted_soft_state["awaiting_field"] == "check_out"
    assert persisted_booking_state["guest_name"] == "Jane Doe"
    assert persisted_booking_state["guest_email"] == "jane@example.com"
    assert persisted_booking_state["guest_phone"] == "03001234567"
    assert persisted_booking_state["check_in"] == "2026-06-02"
    assert persisted_booking_state["guests"] == 4
    assert "check_out" not in persisted_booking_state

    review_reply, route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "11 june 2026",
        deepcopy_session_io=True,
        saved_states=saved_states,
    )

    review_soft_state = saved_states[-1]["soft_state"]
    review = review_soft_state["booking_review"]

    assert route_pre_adk.await_count == 0
    assert "Please confirm if everything is correct." in review_reply
    assert review_soft_state["booking_stage"] == "awaiting_confirmation"
    assert review["check_in"] == "2026-06-02"
    assert review["check_out"] == "2026-06-11"
    assert review["guests"] == 4


@pytest.mark.asyncio
async def test_confirm_review_generates_receipt(monkeypatch):
    snapshot, _selected_property = await _seed_booking_snapshot()
    await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, we are around 4 guests",
    )

    reply, route_pre_adk = await _run_turn(monkeypatch, snapshot, "yes")
    soft_state = snapshot["state"]["soft_state"]

    assert route_pre_adk.await_count == 0
    assert soft_state["booking_stage"] == "confirmed"
    assert re.match(r"^BK-\d{8}-[A-F0-9]{8}$", soft_state["booking_registration_id"])
    assert soft_state["booking_registration_id"] in reply
    assert "Your booking is confirmed." in reply


@pytest.mark.asyncio
async def test_no_at_review_asks_what_to_modify(monkeypatch):
    snapshot, _selected_property = await _seed_booking_snapshot()
    await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, we are around 4 guests",
    )

    reply, _route_pre_adk = await _run_turn(monkeypatch, snapshot, "no")
    soft_state = snapshot["state"]["soft_state"]

    assert soft_state["booking_stage"] == "awaiting_modification_choice"
    assert "What would you like to change" in reply


@pytest.mark.asyncio
async def test_modify_choice_then_update_only_one_booking_field(monkeypatch):
    snapshot, _selected_property = await _seed_booking_snapshot()
    await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, we are around 4 guests",
    )
    before = dict(snapshot["state"]["soft_state"]["booking_review"])

    prompt_reply, _route_pre_adk = await _run_turn(monkeypatch, snapshot, "no")
    assert "What would you like to change" in prompt_reply

    field_reply, _route_pre_adk = await _run_turn(monkeypatch, snapshot, "phone")
    assert "phone number" in field_reply.lower()

    reply, _route_pre_adk = await _run_turn(monkeypatch, snapshot, "03112223333")
    after = snapshot["state"]["soft_state"]["booking_review"]

    assert snapshot["state"]["soft_state"]["booking_stage"] == "awaiting_confirmation"
    assert after["guest_phone"] == "03112223333"
    assert after["guest_name"] == before["guest_name"]
    assert after["guest_email"] == before["guest_email"]
    assert after["check_in"] == before["check_in"]
    assert after["check_out"] == before["check_out"]
    assert after["guests"] == before["guests"]
    assert "Please confirm if everything is correct." in reply


@pytest.mark.asyncio
async def test_update_only_one_booking_field(monkeypatch):
    snapshot, _selected_property = await _seed_booking_snapshot()
    await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, we are around 4 guests",
    )
    before = dict(snapshot["state"]["soft_state"]["booking_review"])

    reply, _route_pre_adk = await _run_turn(monkeypatch, snapshot, "change phone to 03112223333")
    after = snapshot["state"]["soft_state"]["booking_review"]

    assert snapshot["state"]["soft_state"]["booking_stage"] == "awaiting_confirmation"
    assert after["guest_phone"] == "03112223333"
    assert after["guest_name"] == before["guest_name"]
    assert after["guest_email"] == before["guest_email"]
    assert after["check_in"] == before["check_in"]
    assert after["check_out"] == before["check_out"]
    assert "Please confirm if everything is correct." in reply


@pytest.mark.asyncio
async def test_change_property_from_review_returns_to_property_selection(monkeypatch):
    snapshot, _selected_property = await _seed_booking_snapshot()
    await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, we are around 4 guests",
    )

    reply, _route_pre_adk = await _run_turn(monkeypatch, snapshot, "change property")
    soft_state = snapshot["state"]["soft_state"]

    assert soft_state["booking_stage"] == "awaiting_property_reselection"
    assert soft_state["last_presented_view"] == "property_list"
    assert soft_state["visible_results"]
    assert soft_state["option_map"]
    assert "Seattle Apartment 1" in reply


@pytest.mark.asyncio
async def test_faq_during_booking_preserves_booking_context(monkeypatch):
    snapshot, selected_property = await _seed_booking_snapshot()

    reply, _route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "are pets allowed?",
        faq_answer="Pets are allowed at selected properties when listed in the amenities.",
    )
    soft_state = snapshot["state"]["soft_state"]

    assert "Pets are allowed" in reply
    assert "Would you like to continue your booking?" in reply
    assert soft_state["booking_stage"] == "collecting_details"
    assert soft_state["booking_property_id"] == selected_property["id"]

    continue_reply, _route_pre_adk = await _run_turn(monkeypatch, snapshot, "continue booking please")
    assert ("Please provide" in continue_reply) or ("full name" in continue_reply.lower())


def test_available_cities_faq_lists_dataset_cities(monkeypatch):
    monkeypatch.setattr(
        booking_flow,
        "get_all_available_cities",
        lambda: {"status": Status.CITIES_FOUND, "cities": ["Seattle", "New York", "Dubai"]},
    )

    payload = booking_flow.list_available_cities_payload()

    assert payload["status"] == Status.CITIES_FOUND
    assert "Seattle" in payload["deterministic_reply"]
    assert "New York" in payload["deterministic_reply"]
    assert "Dubai" in payload["deterministic_reply"]


@pytest.mark.asyncio
async def test_available_cities_during_booking_preserves_booking_context(monkeypatch):
    snapshot, selected_property = await _seed_booking_snapshot()

    reply, route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "in which cities currently the service is available?",
    )
    soft_state = snapshot["state"]["soft_state"]

    assert "Seattle" in reply
    assert soft_state["booking_stage"] == "collecting_details"
    assert soft_state["booking_property_id"] == selected_property["id"]
    route_pre_adk.assert_not_awaited()


@pytest.mark.asyncio
async def test_booking_details_e2e_smoke(monkeypatch):
    snapshot, _selected_property = await _seed_booking_snapshot()

    invalid_reply, _route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2025, we are around 4 guests",
    )
    assert "cannot be earlier than your check-in date" in invalid_reply

    review_reply, _route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "11 june 2026",
    )
    assert "Please confirm if everything is correct." in review_reply

    receipt_reply, _route_pre_adk = await _run_turn(monkeypatch, snapshot, "yes")
    assert "Registration ID:" in receipt_reply
    assert snapshot["state"]["soft_state"]["booking_stage"] == "confirmed"

@pytest.mark.asyncio
async def test_shared_checkin_checkout_date_list_maps_first_and_second_dates(monkeypatch):
    """When both check-in and check-out labels precede a shared date list like
    "check-in and check-out are 2 june 2026 and 11 june 2026", the first date
    maps to check_in and the second to check_out.  The awaiting_field fallback
    must not override explicit labels."""
    snapshot, _selected_property = await _seed_booking_snapshot()
    soft_state = snapshot["state"]["soft_state"]
    soft_state["awaiting_field"] = "check_in"

    reply, route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in and check-out are 2 june 2026 and 11 june 2026, we are around 4 guests",
    )

    review = soft_state["booking_review"]

    assert route_pre_adk.await_count == 0
    assert soft_state["booking_stage"] == "awaiting_confirmation"
    assert review["check_in"] == "2026-06-02"
    assert review["check_out"] == "2026-06-11"
    assert "Please confirm if everything is correct." in reply


@pytest.mark.asyncio
async def test_invalid_checkin_takes_priority_over_checkout_order(monkeypatch):
    """When check-in is before today, it must be reported even if check-out is also invalid."""
    monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-10")
    snapshot, _selected_property = await _seed_booking_snapshot()

    reply, route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2025, we are around 4 guests",
    )

    soft_state = snapshot["state"]["soft_state"]
    booking_state = soft_state["booking_state"]

    assert route_pre_adk.await_count == 0
    assert "Check-in date must be today or later" in reply
    assert soft_state["awaiting_field"] == "check_in"
    assert soft_state["booking_stage"] == "collecting_details"
    assert booking_state["guest_name"] == "Jane Doe"
    assert booking_state["guest_email"] == "jane@example.com"
    assert booking_state["guest_phone"] == "03001234567"
    assert booking_state["guests"] == 4
    assert "check_in" not in booking_state
    assert "check_out" not in booking_state


@pytest.mark.asyncio
async def test_valid_checkin_invalid_checkout_preserves_checkin(monkeypatch):
    """When check-in is valid but check-out is before check-in, preserve check-in and ask for check-out."""
    monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")
    snapshot, _selected_property = await _seed_booking_snapshot()

    reply, route_pre_adk = await _run_turn(
        monkeypatch,
        snapshot,
        "my full name is Jane Doe, email is jane@example.com, number 03001234567, check-in date would 2nd of june, 2026 and check out shall be around 11 june 2025, we are around 4 guests",
    )

    soft_state = snapshot["state"]["soft_state"]
    booking_state = soft_state["booking_state"]

    assert route_pre_adk.await_count == 0
    assert "cannot be earlier than your check-in date" in reply
    assert soft_state["awaiting_field"] == "check_out"
    assert soft_state["booking_stage"] == "collecting_details"
    assert booking_state["check_in"] == "2026-06-02"
    assert "check_out" not in booking_state


@pytest.mark.asyncio
async def test_faq_during_booking_can_resume_booking_stage(monkeypatch):
    snapshot, _selected_property = await _seed_booking_snapshot()
    soft_state = snapshot["state"]["soft_state"]
    before_stage = soft_state["booking_stage"]
    before_awaiting_field = soft_state["awaiting_field"]

    with patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})):
        faq_reply, faq_route_pre_adk = await _run_turn(
            monkeypatch,
            snapshot,
            "what is the refund policy if i cancel before 5 days of check-in?",
        )

        assert faq_route_pre_adk.await_count == 0
        assert "40%" in faq_reply
        assert "Would you like to continue your booking" in faq_reply
        assert soft_state["faq_interruption"]["resume_target"] == "booking_flow"
        assert soft_state["booking_stage"] == before_stage
        assert soft_state["awaiting_field"] == before_awaiting_field

        resume_reply, resume_route_pre_adk = await _run_turn(
            monkeypatch,
            snapshot,
            "sure",
        )

    assert resume_route_pre_adk.await_count == 0
    assert soft_state.get("faq_interruption") is None
    assert soft_state["booking_stage"] == before_stage
    assert soft_state["awaiting_field"] == before_awaiting_field
    assert resume_reply
