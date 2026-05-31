from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.state import booking_state as booking_state_module

from app.agents.tools.search import search_properties, select_property
from app.config.conversation_shortcuts_loader import match_shortcut
from app.services import adk_runner


class _Ctx:
    def __init__(self):
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
