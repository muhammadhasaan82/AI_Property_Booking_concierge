import copy
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import adk_runner
from app.agents.status_codes import Status

def _make_la_villas(count: int = 14):
    rows = []
    for idx in range(1, count + 1):
        rows.append(
            {
                "id": f"la_villa_{idx}",
                "title": f"LA Villa {idx}",
                "city": "Los Angeles",
                "price_per_night": 200.0 + idx * 10,
                "property_type": "Villa",
                "bedrooms": 3,
                "bathrooms": 2,
                "rating": 4.5,
                "amenities": ["pool", "wifi"],
                "description": f"Beautiful villa in Los Angeles {idx}",
            }
        )
    return rows

async def _run_adk_turn_with_snapshot(monkeypatch, snapshot: dict, message: str) -> tuple[str, AsyncMock]:
    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        existing_state = snapshot.setdefault("state", {})
        existing_soft_state = existing_state.get("soft_state")
        new_soft_state = state.get("soft_state") if isinstance(state, dict) else {}

        if isinstance(existing_soft_state, dict) and isinstance(new_soft_state, dict):
            replacement_soft_state = copy.deepcopy(new_soft_state)
            existing_soft_state.clear()
            existing_soft_state.update(replacement_soft_state)
            existing_state["soft_state"] = existing_soft_state
        else:
            existing_state["soft_state"] = copy.deepcopy(new_soft_state)

        if isinstance(state, dict):
            for key, value in state.items():
                if key != "soft_state":
                    existing_state[key] = value

        snapshot["history"] = history
        snapshot.setdefault("meta", {}).update(metadata or {})

    route_pre_adk = AsyncMock(side_effect=AssertionError("route_pre_adk must not be called"))
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "_get_runner", lambda: (_ for _ in ()).throw(AssertionError("ADK runner must not be invoked")))
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)

    chunks: list[str] = []
    async for chunk in adk_runner.run_adk_turn("u-regression", "s-regression", message):
        chunks.append(chunk)
    return "".join(chunks), route_pre_adk

@pytest.mark.asyncio
async def test_v25_smoke_deterministic_turns(monkeypatch):
    fake = _make_la_villas(14)
    snapshot = {
        "state": {},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-regression",
            "last_update_time": 1.0,
        },
    }

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}), \
         patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})):

        # Turn 1: search villas in Los Angeles
        reply_1, route_1 = await _run_adk_turn_with_snapshot(monkeypatch, snapshot, "show me villas in Los Angeles")
        assert "LA Villa 1" in reply_1
        assert "LA Villa 5" in reply_1
        route_1.assert_not_awaited()

        # Turn 2: option 5 selects the actual fifth result from stored search state
        reply_2, route_2 = await _run_adk_turn_with_snapshot(monkeypatch, snapshot, "5")
        assert "LA Villa 5" in reply_2
        assert "Want to book this one?" in reply_2
        route_2.assert_not_awaited()
        
        soft_state = snapshot["state"]["soft_state"]
        assert soft_state["last_presented_view"] == "property_details"
        assert soft_state["last_selected_property_id"] == "la_villa_5"

        # Turn 3: yes/book starts deterministic booking
        reply_3, route_3 = await _run_adk_turn_with_snapshot(monkeypatch, snapshot, "yes please")
        assert "I'll help you book LA Villa 5" in reply_3 or "Please provide" in reply_3
        route_3.assert_not_awaited()
        assert soft_state["booking_stage"] == "collecting_details"

        # Turn 4: invalid reversed dates asks for correction and preserves valid fields
        reply_4, route_4 = await _run_adk_turn_with_snapshot(
            monkeypatch,
            snapshot,
            "no. of guest are 4, check-out date 23 june and check in date is 13 july, also phone number is 123456789, email is abc@example.com and my name is ABC"
        )
        assert "cannot be earlier than your check-in date" in reply_4
        assert soft_state["awaiting_field"] == "check_out"
        booking_state = soft_state["booking_state"]
        assert booking_state["guest_name"] == "ABC"
        assert booking_state["guest_email"] == "abc@example.com"
        assert booking_state["guest_phone"] == "123456789"
        assert booking_state["guests"] == 4
        assert booking_state["check_in"] == "2026-07-13"
        assert "check_out" not in booking_state
        route_4.assert_not_awaited()

        # Turn 5: valid date completion reaches review and does not ask for check-in again
        # July 13 is check-in, so check-out is July 24
        reply_5, route_5 = await _run_adk_turn_with_snapshot(monkeypatch, snapshot, "24 july")
        assert "Please confirm if everything is correct." in reply_5
        assert soft_state["booking_stage"] == "awaiting_confirmation"
        review = soft_state["booking_review"]
        assert review["check_in"] == "2026-07-13"
        assert review["check_out"] == "2026-07-24"
        assert review["guest_name"] == "ABC"
        route_5.assert_not_awaited()
