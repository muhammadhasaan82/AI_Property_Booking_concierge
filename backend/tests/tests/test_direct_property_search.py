"""Deterministic pre-ADK property search bypasses LLM and persists soft_state."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import app.agents.tools.search as search_tool_module
from app.agents.tools.search import search_properties
from app.services.constraint_extractor import extract_dynamic_constraints
from app.services import adk_runner
from app.services.direct_property_search import (
    extract_property_type_from_message,
    is_clear_direct_property_search,
)
from app.services.property_type_normalizer import fuzzy_resolve_property_type, normalize_property_type


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


def _enable_paginated_search(monkeypatch, *, page_size: int = 2):
    monkeypatch.setattr(search_tool_module, "_search_display_pagination_enabled", lambda: True)
    monkeypatch.setattr(search_tool_module, "_search_display_mode", lambda: "paginated")
    monkeypatch.setattr(search_tool_module, "_search_display_max_inline_results", lambda: None)
    monkeypatch.setattr(search_tool_module, "_resolve_page_size", lambda: page_size)


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
    lower_reply = reply.lower()
    assert "exact match" in lower_reply or "couldn't find" in lower_reply
    assert "9-bedroom" in lower_reply or "9 bedroom" in lower_reply
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


def _seattle_villa_dataset() -> list[dict]:
    return [
        _property(
            property_id="sea-villa-2a",
            title="2BR Villa A in Seattle",
            city="Seattle",
            property_type="Villa",
            bedrooms=2,
            bathrooms=2,
            amenities=["wifi", "parking", "pool"],
            rating=4.9,
            reviews=200,
        ),
        _property(
            property_id="sea-villa-2b",
            title="2BR Villa B in Seattle",
            city="Seattle",
            property_type="Villa",
            bedrooms=2,
            bathrooms=1,
            amenities=["wifi", "parking"],
            rating=4.8,
            reviews=180,
        ),
        _property(
            property_id="sea-apt-2",
            title="2BR Apartment in Seattle",
            city="Seattle",
            property_type="Apartment",
            bedrooms=2,
            rating=4.5,
        ),
        _property(
            property_id="sea-condo-2",
            title="2BR Condo in Seattle",
            city="Seattle",
            property_type="Condo",
            bedrooms=2,
            rating=4.4,
        ),
        _property(
            property_id="sea-studio-2",
            title="2BR Studio in Seattle",
            city="Seattle",
            property_type="Studio",
            bedrooms=2,
            rating=4.3,
        ),
        _property(
            property_id="sea-house-2",
            title="2BR House in Seattle",
            city="Seattle",
            property_type="House",
            bedrooms=2,
            rating=4.2,
        ),
        _property(
            property_id="sea-villa-3",
            title="3BR Villa in Seattle",
            city="Seattle",
            property_type="Villa",
            bedrooms=3,
            rating=4.7,
        ),
    ]


def _seattle_refinement_dataset() -> list[dict]:
    return [
        _property(
            property_id="sea-villa-pet-1",
            title="Cedar Pool Villa",
            city="Seattle",
            property_type="Villa",
            bedrooms=3,
            bathrooms=2,
            price=190,
            amenities=["pool", "pet_friendly", "wifi"],
            rating=4.9,
            reviews=250,
        ),
        _property(
            property_id="sea-villa-pet-2",
            title="Harbor Pool Villa",
            city="Seattle",
            property_type="Villa",
            bedrooms=3,
            bathrooms=2,
            price=210,
            amenities=["pool", "pet_friendly", "parking"],
            rating=4.8,
            reviews=220,
        ),
        _property(
            property_id="sea-villa-pet-3",
            title="Lakeside Pool Villa",
            city="Seattle",
            property_type="Villa",
            bedrooms=3,
            bathrooms=3,
            price=230,
            amenities=["pool", "pet_friendly", "wifi", "parking"],
            rating=4.7,
            reviews=210,
        ),
        _property(
            property_id="sea-villa-pet-4",
            title="Ridge Pool Villa",
            city="Seattle",
            property_type="Villa",
            bedrooms=3,
            bathrooms=2,
            price=260,
            amenities=["pool", "pet_friendly"],
            rating=4.6,
            reviews=205,
        ),
        _property(
            property_id="sea-villa-pool-only",
            title="Pool Villa Without Pets",
            city="Seattle",
            property_type="Villa",
            bedrooms=3,
            bathrooms=2,
            price=170,
            amenities=["pool", "wifi"],
            rating=4.5,
        ),
        _property(
            property_id="sea-villa-2br",
            title="Two Bedroom Pool Villa",
            city="Seattle",
            property_type="Villa",
            bedrooms=2,
            bathrooms=2,
            price=160,
            amenities=["pool", "pet_friendly"],
            rating=4.4,
        ),
        _property(
            property_id="sea-ap-pet-1",
            title="Green Lake Apartment",
            city="Seattle",
            property_type="Apartment",
            bedrooms=3,
            bathrooms=2,
            price=140,
            amenities=["pool", "pet_friendly", "wifi"],
            rating=4.7,
            reviews=180,
        ),
        _property(
            property_id="sea-ap-pet-2",
            title="Pioneer Apartment",
            city="Seattle",
            property_type="Apartment",
            bedrooms=3,
            bathrooms=2,
            price=175,
            amenities=["pool", "pet_friendly", "parking"],
            rating=4.6,
            reviews=170,
        ),
        _property(
            property_id="sea-ap-pool-only",
            title="Waterfront Apartment",
            city="Seattle",
            property_type="Apartment",
            bedrooms=3,
            bathrooms=2,
            price=165,
            amenities=["pool", "wifi"],
            rating=4.5,
            reviews=160,
        ),
    ]


def test_fuzzy_property_type_normalizes_vila_typo():
    assert fuzzy_resolve_property_type("vila") == "villa"
    assert extract_property_type_from_message("i am looking for an vila in seattle of 2BR") == "villa"


@pytest.mark.parametrize(
    "message",
    [
        "2 bed vila seattle",
        "show me 2BR villas Seattle",
        "looking for 2 bedroom villa in Seattle",
        "need a villa with 2 beds in seattle",
        "villa in Seattle for 2 bedrooms",
    ],
)
def test_property_type_variant_queries_normalize_to_villa(message: str):
    assert extract_property_type_from_message(message) == "villa"


@pytest.mark.asyncio
async def test_vila_seattle_2br_filters_exact_villas_only(monkeypatch):
    fake = _seattle_villa_dataset()
    snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_session_snapshot(_sid):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    route_pre_adk = _block_adk_and_prerouter(monkeypatch)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-vila",
            "s-vila",
            "i am looking for an vila in seattle of 2BR",
        ):
            chunks.append(chunk)

    reply = "".join(chunks)
    route_pre_adk.assert_not_awaited()
    assert "2BR Villa A in Seattle" in reply
    assert "2BR Villa B in Seattle" in reply
    assert "2BR Apartment in Seattle" not in reply
    assert "2BR Condo in Seattle" not in reply
    assert "2BR Studio in Seattle" not in reply
    assert "2BR House in Seattle" not in reply
    assert "3BR Villa in Seattle" not in reply

    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["last_filters"]["city"] == "Seattle"
    assert soft_state["last_filters"]["property_type"] == "villa"
    assert soft_state["last_filters"]["bedrooms"] == 2
    assert soft_state["last_search"]["query_context"]["property_type"] == "villa"
    assert soft_state["last_search"]["query_context"]["bedrooms"] == 2
    visible = soft_state["visible_results"]
    assert visible
    assert all(item["city"] == "Seattle" for item in visible)
    assert all(normalize_property_type(item["property_type"]) == "villa" for item in visible)
    assert all(item["bedrooms"] == 2 for item in visible)


@pytest.mark.asyncio
async def test_vila_seattle_option_selection_uses_constrained_results(monkeypatch):
    fake = _seattle_villa_dataset()
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
            "u-vila-select",
            "s-vila-select",
            "i am looking for an vila in seattle of 2BR",
        ):
            pass

        soft_state = snapshot["state"]["soft_state"]
        expected_title = soft_state["visible_results"][0]["title"]
        expected_id = soft_state["visible_results"][0]["id"]

        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-vila-select",
            "s-vila-select",
            "1",
        ):
            chunks.append(chunk)

    reply = "".join(chunks)
    assert expected_title in reply
    assert snapshot["state"]["soft_state"]["last_selected_property_id"] == expected_id
    assert normalize_property_type(
        snapshot["state"]["soft_state"]["last_filters"]["property_type"]
    ) == "villa"


@pytest.mark.asyncio
async def test_amenity_constrained_villa_search_filters_exact_matches(monkeypatch):
    fake = _seattle_villa_dataset()
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
            "u-amenity",
            "s-amenity",
            "2BR villa in Seattle with wifi and parking",
        ):
            chunks.append(chunk)

    reply = "".join(chunks)
    assert "2BR Villa A in Seattle" in reply
    assert "2BR Villa B in Seattle" in reply
    assert "2BR Apartment in Seattle" not in reply

    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["last_filters"]["property_type"] == "villa"
    assert set(soft_state["last_filters"]["amenities"]) == {"wifi", "parking"}
    for item in soft_state["visible_results"]:
        amenities = {str(a).lower() for a in item.get("amenities") or []}
        assert "wifi" in amenities
        assert "parking" in amenities


@pytest.mark.asyncio
async def test_bathroom_and_guest_constraints_are_applied_and_persisted(monkeypatch):
    fake = [
        _property(
            property_id="sea-villa-2bath",
            title="2BR 2BA Villa in Seattle",
            city="Seattle",
            property_type="Villa",
            bedrooms=2,
            bathrooms=2,
            occupancy_max=6,
            rating=4.9,
        ),
        _property(
            property_id="sea-villa-1bath",
            title="2BR 1BA Villa in Seattle",
            city="Seattle",
            property_type="Villa",
            bedrooms=2,
            bathrooms=1,
            occupancy_max=6,
            rating=4.7,
        ),
        _property(
            property_id="ny-apt-4guest",
            title="2BR Apartment in New York",
            city="New York",
            property_type="Apartment",
            bedrooms=2,
            bathrooms=1,
            occupancy_max=2,
            rating=4.6,
        ),
        _property(
            property_id="ny-apt-6guest",
            title="2BR Apartment in New York for groups",
            city="New York",
            property_type="Apartment",
            bedrooms=2,
            bathrooms=2,
            occupancy_max=6,
            rating=4.8,
        ),
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
            "u-bath",
            "s-bath",
            "villa in Seattle with 2 bedrooms and 2 bathrooms",
        ):
            chunks.append(chunk)

        reply_bath = "".join(chunks)
        assert "2BR 2BA Villa in Seattle" in reply_bath
        assert "2BR 1BA Villa in Seattle" not in reply_bath
        bath_state = snapshot["state"]["soft_state"]
        assert bath_state["last_filters"]["bathrooms"] == 2

    guest_snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_guest_snapshot(_sid):
        return guest_snapshot

    async def fake_save_guest_snapshot(*, session_id, history, state, metadata):
        guest_snapshot["state"] = state

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_guest_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_guest_snapshot)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-guest",
            "s-guest",
            "apartment in New York for 4 guests",
        ):
            chunks.append(chunk)

    reply_guest = "".join(chunks)
    assert "2BR Apartment in New York for groups" in reply_guest
    assert "**2BR Apartment in New York** —" not in reply_guest
    guest_state = guest_snapshot["state"]["soft_state"]
    assert guest_state["last_filters"]["guests"] == 4
    assert len(guest_state["visible_results"]) == 1
    assert guest_state["visible_results"][0]["id"] == "ny-apt-6guest"


@pytest.mark.asyncio
async def test_no_exact_match_does_not_broaden_property_types(monkeypatch):
    fake = [
        _property(
            property_id="sea-villa-3",
            title="3BR Villa in Seattle",
            city="Seattle",
            property_type="Villa",
            bedrooms=3,
            amenities=["wifi"],
        ),
        _property(
            property_id="sea-apt-2",
            title="2BR Apartment in Seattle",
            city="Seattle",
            property_type="Apartment",
            bedrooms=2,
            amenities=["wifi", "parking"],
        ),
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
            "u-no-exact",
            "s-no-exact",
            "2BR villa in Seattle with wifi and parking",
        ):
            chunks.append(chunk)

    reply = "".join(chunks).lower()
    assert "exact match" in reply or "couldn't find" in reply
    assert "2br apartment in seattle" not in reply
    assert "3br villa in seattle" not in reply
    assert "relax" in reply


@pytest.mark.asyncio
async def test_typo_city_trace_logs_show_fuzzy_root_cause(monkeypatch, caplog):
    fake = [
        _property(
            property_id="la-villa-1",
            title="Los Angeles Villa One",
            city="Los Angeles",
            property_type="Villa",
            bedrooms=3,
        ),
    ]
    snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_session_snapshot(_sid):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    route_pre_adk = _block_adk_and_prerouter(monkeypatch)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)

    with caplog.at_level("INFO"), patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        chunks = []
        async for chunk in adk_runner.run_adk_turn(
            "u-typo-trace",
            "s-typo-trace",
            "i am looking a villa in los angellas",
        ):
            chunks.append(chunk)

    reply = "".join(chunks)
    route_pre_adk.assert_not_awaited()
    assert "Los Angeles Villa One" in reply
    assert "exact_city=None" in caplog.text
    assert "city_resolution_status=fuzzy" in caplog.text
    assert "canonical_city='Los Angeles'" in caplog.text
    assert "active_search_context=False" in caplog.text


@pytest.mark.asyncio
async def test_refinement_flow_preserves_dynamic_constraints_pagination_and_selection(monkeypatch):
    fake = _seattle_refinement_dataset()
    snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_session_snapshot(_sid):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history

    async def run_turn(message: str) -> str:
        chunks = []
        async for chunk in adk_runner.run_adk_turn("u-refine", "s-refine", message):
            chunks.append(chunk)
        return "".join(chunks)

    route_pre_adk = _block_adk_and_prerouter(monkeypatch)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    _enable_paginated_search(monkeypatch, page_size=2)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        reply = await run_turn("villa in Seattle")
        assert "Cedar Pool Villa" in reply
        assert "Green Lake Apartment" not in reply
        soft_state = snapshot["state"]["soft_state"]
        assert soft_state["last_filters"]["city"] == "Seattle"
        assert soft_state["last_filters"]["property_type"] == "villa"
        assert soft_state["current_page"] == 1

        reply = await run_turn("with pool")
        soft_state = snapshot["state"]["soft_state"]
        assert set(soft_state["last_filters"]["amenities"]) == {"pool"}
        assert all("pool" in {str(a).lower() for a in item.get("amenities") or []} for item in soft_state["all_search_results"])

        reply = await run_turn("cheaper")
        soft_state = snapshot["state"]["soft_state"]
        assert soft_state["last_sort_preferences"] == [{"field": "price_per_night", "direction": "asc"}]
        cheaper_prices = [item["price_per_night"] for item in soft_state["visible_results"]]
        assert cheaper_prices == sorted(cheaper_prices)

        reply = await run_turn("3 bedrooms")
        soft_state = snapshot["state"]["soft_state"]
        assert soft_state["last_dynamic_constraints"]["bedrooms"] == {"value": 3, "operator": "exact"}
        assert all(item["bedrooms"] == 3 for item in soft_state["all_search_results"])
        bedroom_prices = [item["price_per_night"] for item in soft_state["all_search_results"]]
        assert bedroom_prices == sorted(bedroom_prices)

        reply = await run_turn("pet friendly")
        soft_state = snapshot["state"]["soft_state"]
        assert set(soft_state["last_filters"]["amenities"]) == {"pool", "pet_friendly"}
        assert soft_state["last_sort_preferences"] == [{"field": "price_per_night", "direction": "asc"}]
        assert len(soft_state["all_search_results"]) == 4
        assert [item["id"] for item in soft_state["visible_results"]] == [
            "sea-villa-pet-1",
            "sea-villa-pet-2",
        ]

        reply = await run_turn("show more")
        soft_state = snapshot["state"]["soft_state"]
        assert "Lakeside Pool Villa" in reply
        assert "Ridge Pool Villa" in reply
        assert soft_state["current_page"] == 2
        assert [item["id"] for item in soft_state["visible_results"]] == [
            "sea-villa-pet-3",
            "sea-villa-pet-4",
        ]

        reply = await run_turn("no, apartments only")
        soft_state = snapshot["state"]["soft_state"]
        assert "Green Lake Apartment" in reply
        assert "Pioneer Apartment" in reply
        assert "Cedar Pool Villa" not in reply
        assert soft_state["last_filters"]["property_type"] == "apartment"
        assert soft_state["last_filters"]["city"] == "Seattle"
        assert soft_state["last_filters"]["bedrooms"] == 3
        assert set(soft_state["last_filters"]["amenities"]) == {"pool", "pet_friendly"}
        assert soft_state["last_sort_preferences"] == [{"field": "price_per_night", "direction": "asc"}]
        assert [item["id"] for item in soft_state["visible_results"]] == [
            "sea-ap-pet-1",
            "sea-ap-pet-2",
        ]

        expected_title = soft_state["visible_results"][1]["title"]
        expected_id = soft_state["visible_results"][1]["id"]
        reply = await run_turn("book option 2")
        soft_state = snapshot["state"]["soft_state"]
        assert expected_title in reply
        assert soft_state["last_selected_property_id"] == expected_id
        assert soft_state["last_presented_view"] == "property_details"

    route_pre_adk.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_tool_and_rust_search_paths_return_identical_filtered_results(monkeypatch):
    fake = _seattle_refinement_dataset()
    message = "villa in Seattle with pool cheaper 3 bedrooms pet friendly"
    snapshot = {"state": {}, "history": [], "meta": {}}

    async def fake_get_session_snapshot(_sid):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    route_pre_adk = _block_adk_and_prerouter(monkeypatch)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)

    constraints, plan = extract_dynamic_constraints(message, session_id="parity")
    common_kwargs = {
        "city": constraints.get("city"),
        "property_type": constraints.get("property_type"),
        "beds": constraints.get("bedrooms"),
        "beds_operator": constraints.get_operator("bedrooms", "exact"),
        "bathrooms": constraints.get("bathrooms"),
        "bathrooms_operator": constraints.get_operator("bathrooms", "exact"),
        "guests": constraints.get("occupancy_max"),
        "guests_operator": constraints.get_operator("occupancy_max", "min"),
        "budget": constraints.get("price_per_night"),
        "amenities": ",".join(constraints.get("amenities") or []),
        "sort_preferences": list(plan.sort_preferences),
    }

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        async for _chunk in adk_runner.run_adk_turn("u-parity", "s-parity", message):
            pass
        direct_ids = [item["id"] for item in snapshot["state"]["soft_state"]["all_search_results"]]

        ctx_tool = SimpleNamespace(state={"soft_state": {}})
        tool_result = await search_properties(
            **common_kwargs,
            search_path="tool",
            tool_context=ctx_tool,
        )
        tool_ids = [item["id"] for item in ctx_tool.state["soft_state"]["all_search_results"]]
        assert tool_ids == [prop["id"] for prop in tool_result["properties"]]

    rust_payload = {"result": {"results": list(reversed(fake))}}
    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value=rust_payload,
    ):
        ctx_rust = SimpleNamespace(state={"soft_state": {}})
        rust_result = await search_properties(
            **common_kwargs,
            search_path="rust",
            tool_context=ctx_rust,
        )
        rust_ids = [item["id"] for item in ctx_rust.state["soft_state"]["all_search_results"]]
        assert rust_ids == [prop["id"] for prop in rust_result["properties"]]

    expected_ids = [
        "sea-villa-pet-1",
        "sea-villa-pet-2",
        "sea-villa-pet-3",
        "sea-villa-pet-4",
    ]
    route_pre_adk.assert_not_awaited()
    assert direct_ids == expected_ids
    assert tool_ids == expected_ids
    assert rust_ids == expected_ids
