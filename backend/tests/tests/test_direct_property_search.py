"""Deterministic pre-ADK property search bypasses LLM and persists soft_state."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services import adk_runner
from app.services.direct_property_search import (
    extract_property_type_from_message,
    is_clear_direct_property_search,
)


def _make_apartment(idx: int) -> dict:
    return {
        "id": f"apt-{idx}",
        "title": f"Apartment {idx}",
        "city": "New York",
        "price_per_night": 100 + idx,
        "property_type": "Apartment",
        "bedrooms": 1,
        "bathrooms": 1,
        "rating": 4.0 + (idx % 3) * 0.1,
        "reviews_count": 100 - idx,
        "amenities": ["wifi"],
        "description": f"Apartment {idx} in New York",
    }


def _apartments(count: int) -> list[dict]:
    return [_make_apartment(i) for i in range(1, count + 1)]




def _block_adk_and_prerouter(monkeypatch):
    def fail_runner():
        raise AssertionError("ADK runner must not be invoked for direct property search")

    route_pre_adk = AsyncMock(side_effect=AssertionError("route_pre_adk must not be called"))
    monkeypatch.setattr(adk_runner, "_get_runner", fail_runner)
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda m: (m, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda m: m)
    return route_pre_adk


@pytest.mark.asyncio
async def test_direct_property_search_bypasses_adk_and_persists_state(monkeypatch):
    fake = _apartments(15)
    snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_session_snapshot(_sid):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history

    route_pre_adk = _block_adk_and_prerouter(monkeypatch)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-direct",
            "s-direct",
            "I am looking for an apartment in New York",
        ):
            chunks.append(chunk)

    reply = "".join(chunks)
    assert "Apartment 1" in reply
    assert "Apartment 15" in reply
    assert "15" in reply
    assert "apartment" in reply.lower()
    assert "Charming Studio" not in reply
    assert "Greenwich Village" not in reply
    route_pre_adk.assert_not_awaited()

    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["last_presented_view"] == "property_list"
    assert len(soft_state["visible_results"]) == 15
    assert len(soft_state["all_search_results"]) == 15
    assert set(soft_state["option_map"].keys()) == {str(i) for i in range(1, 16)}
    assert soft_state.get("last_search")
    assert soft_state.get("last_filters")
    assert soft_state.get("active_property_options_map")
    assert soft_state.get("active_property_options_shown_count") == 15
    assert soft_state.get("active_property_options_total_found") == 15


@pytest.mark.asyncio
async def test_direct_property_search_then_option_then_no_live_style(monkeypatch):
    fake = _apartments(8)
    snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_session_snapshot(_sid):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history

    _block_adk_and_prerouter(monkeypatch)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-flow",
            "s-flow",
            "I am looking for an apartment in New York",
        ):
            chunks.append(chunk)

        assert len(snapshot["state"]["soft_state"]["all_search_results"]) == 8
        soft_state = snapshot["state"]["soft_state"]
        expected_title = soft_state["visible_results"][2]["title"]

        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-flow",
            "s-flow",
            "please show me option 3",
        ):
            chunks.append(chunk)

    details_reply = "".join(chunks)
    assert expected_title in details_reply
    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["last_presented_view"] == "property_details"
    assert soft_state.get("last_selected_property_id")
    assert len(soft_state["visible_results"]) == 8

    route_pre_adk = AsyncMock(return_value={"reply": "generic concierge greeting"})
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        chunks = []
        async for chunk in adk_runner.run_adk_turn("u-flow", "s-flow", "no"):
            chunks.append(chunk)

    menu_reply = "".join(chunks)
    assert "Apartment 1" in menu_reply
    assert "Apartment 8" in menu_reply
    assert "generic concierge greeting" not in menu_reply
    route_pre_adk.assert_not_awaited()
    assert snapshot["state"]["soft_state"]["last_presented_view"] == "property_list"


@pytest.mark.asyncio
async def test_direct_search_no_hallucinated_voice_output(monkeypatch):
    fake = _apartments(8)
    snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_session_snapshot(_sid):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    _block_adk_and_prerouter(monkeypatch)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)

    async def fail_voice(*_a, **_k):
        raise AssertionError("_render_voice_from_router_output must not be called")

    monkeypatch.setattr(adk_runner, "_render_voice_from_router_output", fail_voice)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-voice",
            "s-voice",
            "show apartments in New York",
        ):
            chunks.append(chunk)

    assert "Apartment 1" in "".join(chunks)


def test_direct_search_uses_config_taxonomy():
    assert extract_property_type_from_message("flat in New York") == "apartment"
    assert is_clear_direct_property_search("flat in New York", {}) is True


@pytest.mark.asyncio
async def test_flat_search_resolves_via_taxonomy_in_runner(monkeypatch):
    fake = _apartments(5)
    snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_session_snapshot(_sid):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    _block_adk_and_prerouter(monkeypatch)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        chunks = []
        async for chunk in adk_runner.run_adk_turn("u-flat", "s-flat", "flat in New York"):
            chunks.append(chunk)

    assert "apartment" in "".join(chunks).lower()
    assert snapshot["state"]["soft_state"]["last_filters"]["property_type"] == "apartment"
