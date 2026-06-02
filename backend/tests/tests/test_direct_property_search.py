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


def _property(
    *,
    property_id: str,
    title: str,
    city: str,
    property_type: str,
    bedrooms: int,
    bathrooms: int = 1,
    price: int = 150,
    rating: float = 4.5,
    reviews: int = 100,
    amenities: list[str] | None = None,
    occupancy_max: int | None = None,
) -> dict:
    return {
        "id": property_id,
        "title": title,
        "city": city,
        "price_per_night": price,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "rating": rating,
        "reviews_count": reviews,
        "amenities": amenities or ["wifi"],
        "description": f"{title} in {city}",
        "occupancy_max": occupancy_max if occupancy_max is not None else max(bedrooms * 2, 2),
    }




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


@pytest.mark.asyncio
async def test_two_bedroom_apartment_query_filters_bedrooms(monkeypatch):
    fake = [
        _property(property_id="ny-apt-1", title="1BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=1, price=120),
        _property(property_id="ny-apt-2", title="2BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=2, price=180, rating=4.9, reviews=220),
        _property(property_id="ny-apt-4", title="4BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=4, price=260),
        _property(property_id="ny-condo-2", title="2BR Condo in New York", city="New York", property_type="Condo", bedrooms=2, price=190),
        _property(property_id="sea-apt-2", title="2BR Apartment in Seattle", city="Seattle", property_type="Apartment", bedrooms=2, price=170),
    ]
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
        async for chunk in adk_runner.run_adk_turn(
            "u-filtered",
            "s-filtered",
            "show me some 2 bedrooms apartment in new york city",
        ):
            chunks.append(chunk)

    reply = "".join(chunks)
    assert "2BR Apartment in New York" in reply
    assert "1BR Apartment in New York" not in reply
    assert "4BR Apartment in New York" not in reply
    assert "2BR Condo in New York" not in reply
    assert "2BR Apartment in Seattle" not in reply

    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["visible_results"]
    assert all(item["bedrooms"] == 2 for item in soft_state["visible_results"])
    assert all(item["property_type"] == "Apartment" for item in soft_state["visible_results"])
    assert all(item["city"] == "New York" for item in soft_state["visible_results"])
    assert len(soft_state["option_map"]) == len(soft_state["visible_results"])


@pytest.mark.asyncio
async def test_two_bedroom_query_option_selection_uses_filtered_map(monkeypatch):
    fake = [
        _property(property_id="ny-apt-2a", title="2BR Apartment A in New York", city="New York", property_type="Apartment", bedrooms=2, price=180, rating=4.9, reviews=220),
        _property(property_id="ny-apt-2b", title="2BR Apartment B in New York", city="New York", property_type="Apartment", bedrooms=2, price=170, rating=4.8, reviews=180),
        _property(property_id="ny-apt-4", title="4BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=4, price=260),
    ]
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
        async for _chunk in adk_runner.run_adk_turn(
            "u-filter-select",
            "s-filter-select",
            "show me some 2 bedrooms apartment in new york city",
        ):
            pass

        soft_state = snapshot["state"]["soft_state"]
        expected_title = soft_state["visible_results"][0]["title"]
        expected_id = soft_state["visible_results"][0]["id"]

        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-filter-select",
            "s-filter-select",
            "option 1",
        ):
            chunks.append(chunk)

    reply = "".join(chunks)
    assert expected_title in reply
    assert snapshot["state"]["soft_state"]["last_selected_property_id"] == expected_id


@pytest.mark.asyncio
async def test_no_matches_for_exact_bedroom_filter(monkeypatch):
    fake = [
        _property(property_id="ny-apt-1", title="1BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=1),
        _property(property_id="ny-apt-2", title="2BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=2),
        _property(property_id="ny-apt-4", title="4BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=4),
    ]
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
        async for chunk in adk_runner.run_adk_turn(
            "u-no-match",
            "s-no-match",
            "9 bedroom apartment in New York",
        ):
            chunks.append(chunk)

    reply = "".join(chunks)
    assert "couldn't find any 9-bedroom apartments in New York".lower() in reply.lower()
    assert "1BR Apartment in New York" not in reply
    assert "2BR Apartment in New York" not in reply
    assert "4BR Apartment in New York" not in reply


@pytest.mark.asyncio
async def test_screenshot_query_two_bedrooms_apartment_new_york_does_not_return_all_apartments(monkeypatch):
    fake = [
        _property(property_id="ny-apt-0", title="0-bed Apartment in New York", city="New York", property_type="Apartment", bedrooms=0),
        _property(property_id="ny-apt-1", title="1BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=1),
        _property(property_id="ny-apt-2", title="2BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=2),
        _property(property_id="ny-apt-4", title="4BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=4),
        _property(property_id="ny-apt-5", title="5BR Apartment in New York", city="New York", property_type="Apartment", bedrooms=5),
    ]
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
        async for chunk in adk_runner.run_adk_turn(
            "u-screenshot",
            "s-screenshot",
            "show me some 2 bedrooms apartment in new york city",
        ):
            chunks.append(chunk)

    reply = "".join(chunks).lower()
    assert "4br apartment" not in reply
    assert "5br apartment" not in reply
    assert "1br apartment" not in reply
    assert "0-bed apartment" not in reply
    assert "2-bedroom apartments" in reply or "2 beds" in reply


@pytest.mark.asyncio
async def test_deterministic_property_render_bypasses_voice_llm_for_large_lists(monkeypatch):
    fake = _apartments(128)
    snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_session_snapshot(_sid):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    _block_adk_and_prerouter(monkeypatch)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)

    async def fail_voice(*_a, **_k):
        raise AssertionError("voice LLM must not be called for deterministic property render")

    monkeypatch.setattr(adk_runner, "_render_voice_from_router_output", fail_voice)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-large",
            "s-large",
            "apartment in New York",
        ):
            chunks.append(chunk)

    reply = "".join(chunks)
    assert "Apartment 1" in reply
    assert "Apartment 128" in reply
