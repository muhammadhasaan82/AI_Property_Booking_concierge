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
    """Keep date-based booking tests deterministic without test-runner checks in app code."""
    monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")

class _Ctx:
    def __init__(self):
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
    async def fake_get_session_snapshot(_session_id):
        return copy.deepcopy(snapshot) if deepcopy_session_io else snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        state_to_save = copy.deepcopy(state) if deepcopy_session_io else state
        history_to_save = copy.deepcopy(history) if deepcopy_session_io else history
        metadata_to_save = copy.deepcopy(metadata or {}) if deepcopy_session_io else (metadata or {})
        snapshot["state"] = state_to_save
        snapshot["history"] = history_to_save
        snapshot["meta"].update(metadata_to_save)
        if saved_states is not None and isinstance(state_to_save, dict):
            saved_states.append(copy.deepcopy(state_to_save))

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
