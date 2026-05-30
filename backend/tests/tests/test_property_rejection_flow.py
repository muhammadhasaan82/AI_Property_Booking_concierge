from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.tools.search import search_properties, select_property
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


@pytest.mark.asyncio
async def test_no_after_property_details_returns_full_menu(monkeypatch):
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
        selected = await select_property(option_number=3, tool_context=ctx)

    assert selected["status"] == "property_details"
    assert ctx.state["soft_state"]["last_presented_view"] == "property_details"

    snapshot = {
        "state": {"soft_state": ctx.state["soft_state"]},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-reject",
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
        raise AssertionError("ADK runner must not be invoked for property rejection")

    route_pre_adk = AsyncMock(return_value={"reply": "generic concierge greeting"})
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "_get_runner", fail_get_runner)

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-reject", "s-reject", "no"):
        chunks.append(chunk)

    reply = "".join(chunks)
    assert "Apartment 1" in reply
    assert "Apartment 8" in reply
    assert "generic concierge greeting" not in reply
    route_pre_adk.assert_not_awaited()

    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["last_presented_view"] == "property_list"
    assert soft_state["last_rejected_property_id"] == search_result["properties"][2]["id"]
    assert len(soft_state["visible_results"]) == 8
    assert set(soft_state["option_map"]) == {str(i) for i in range(1, 9)}
    assert saved_states
