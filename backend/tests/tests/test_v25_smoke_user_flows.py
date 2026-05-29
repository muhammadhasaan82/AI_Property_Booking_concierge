"""
V2.5 smoke tests for deterministic user flows.

Stable deterministic tests:

uv run env PYTHONPATH=. pytest \
  tests/tests/test_e2e_booking_to_receipt_flow.py \
  tests/tests/test_v25_smoke_user_flows.py \
  -v --tb=short

Optional live tests, only if LLM quota is available:

V25_LIVE_LLM_TESTS=1 uv run env PYTHONPATH=. pytest tests/test_v2_chaos.py -v --tb=short
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.schemas.understanding_frame import UnderstandingFrame
from app.agents.status_codes import Status
from app.agents.tools import support
from app.agents.tools.search import search_properties
from app.security import policy_router
from app.services.adk_runner import _render_property_results_from_router_output
from app.services.property_type_normalizer import normalize_property_type, get_all_canonical_types
import app.services.dynamic_config as dynamic_config


class _Ctx(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(state={"soft_state": {}})


def _seed_dataset() -> list[dict]:
    return [
        {
            "id": "ny_ap_1",
            "city": "New York",
            "property_type": "Apartment",
            "price_per_night": 120.0,
            "bedrooms": 2,
            "bathrooms": 1,
            "rating": 4.6,
            "amenities": ["wifi"],
            "title": "Hudson Loft",
            "description": "Bright loft near the park.",
        },
        {
            "id": "ny_ap_2",
            "city": "New York",
            "property_type": "Apartment",
            "price_per_night": 130.0,
            "bedrooms": 1,
            "bathrooms": 1,
            "rating": 4.4,
            "amenities": ["wifi"],
            "title": "Midtown Flat",
            "description": "Close to transit.",
        },
        {
            "id": "ny_ap_3",
            "city": "New York",
            "property_type": "Apartment",
            "price_per_night": 135.0,
            "bedrooms": 1,
            "bathrooms": 1,
            "rating": 4.3,
            "amenities": ["gym"],
            "title": "Soho Suite",
            "description": "Compact and central.",
        },
        {
            "id": "ny_ap_4",
            "city": "New York",
            "property_type": "Apartment",
            "price_per_night": 150.0,
            "bedrooms": 2,
            "bathrooms": 2,
            "rating": 4.7,
            "amenities": ["wifi", "gym"],
            "title": "Chelsea Corner",
            "description": "Quiet block.",
        },
        {
            "id": "ny_ap_5",
            "city": "New York",
            "property_type": "Apartment",
            "price_per_night": 155.0,
            "bedrooms": 2,
            "bathrooms": 1,
            "rating": 4.5,
            "amenities": ["parking"],
            "title": "Broadway North",
            "description": "Near theaters.",
        },
        {
            "id": "ny_dup_1",
            "city": "New York",
            "property_type": "Duplex",
            "price_per_night": 220.0,
            "bedrooms": 3,
            "bathrooms": 2,
            "rating": 4.2,
            "amenities": ["parking"],
            "title": "Riverside Duplex",
            "description": "Quiet street.",
        },
        {
            "id": "ny_house_1",
            "city": "New York",
            "property_type": "House",
            "price_per_night": 260.0,
            "bedrooms": 3,
            "bathrooms": 2,
            "rating": 4.1,
            "amenities": ["garden"],
            "title": "Parkside House",
            "description": "Backyard space.",
        },
        {
            "id": "ny_studio_1",
            "city": "New York",
            "property_type": "Studio",
            "price_per_night": 90.0,
            "bedrooms": 0,
            "bathrooms": 1,
            "rating": 4.0,
            "amenities": ["wifi"],
            "title": "Studio Lane",
            "description": "Efficient layout.",
        },
        {
            "id": "dub_v_1",
            "city": "Dubai",
            "property_type": "Villa",
            "price_per_night": 420.0,
            "bedrooms": 4,
            "bathrooms": 3,
            "rating": 4.9,
            "amenities": ["pool"],
            "title": "Palm Retreat",
            "description": "Private pool and terrace.",
        },
        {
            "id": "dub_v_2",
            "city": "Dubai",
            "property_type": "villa",
            "price_per_night": 380.0,
            "bedrooms": 3,
            "bathrooms": 2,
            "rating": 4.8,
            "amenities": ["pool"],
            "title": "Marina Villa",
            "description": "Waterfront access.",
        },
        {
            "id": "dub_ap_1",
            "city": "Dubai",
            "property_type": "Apartment",
            "price_per_night": 210.0,
            "bedrooms": 2,
            "bathrooms": 1,
            "rating": 4.3,
            "amenities": ["wifi"],
            "title": "Downtown View",
            "description": "High floor views.",
        },
    ]


def _make_frame(intent: str, entities: dict | None = None, confidence: float = 0.9, **kwargs) -> UnderstandingFrame:
    return UnderstandingFrame(
        primary_intent=intent,
        confidence=confidence,
        entities=entities or {},
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "city, property_type",
    [
        ("New York", "apartments"),
        ("Dubai", "villa"),
        ("New York", "duplex"),
    ],
)
async def test_property_type_city_filters_and_total_found(city: str, property_type: str):
    ctx = _Ctx()
    dataset = _seed_dataset()

    with patch("app.components.search._DATASET", dataset), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}):
        result = await search_properties(
            city=city,
            property_type=property_type,
            max_results=3,
            tool_context=ctx,
        )

    assert result["status"] == Status.PROPERTIES_FOUND
    expected = [
        row
        for row in dataset
        if row["city"].lower() == city.lower()
        and normalize_property_type(row["property_type"]) == normalize_property_type(property_type)
    ]
    assert result["total_found"] == len(expected)
    page_size = result["pagination"]["page_size"]
    assert result["shown_count"] == min(len(expected), page_size)
    assert result["max_results"] == 3

    returned_titles = {prop["title"] for prop in result["properties"]}
    expected_titles = {row["title"] for row in expected}
    assert returned_titles <= expected_titles

    for prop in result["properties"]:
        assert prop["city"].lower() == city.lower()
        assert normalize_property_type(prop["property_type"]) == normalize_property_type(property_type)

    fake_titles = {"the grand", "city lights", "sky high", "luxury manhattan loft", "cozy studio"}
    assert not {t.lower() for t in returned_titles} & fake_titles


@pytest.mark.asyncio
async def test_missing_region_requires_city_clarification():
    ctx = _Ctx()
    rust_stub = AsyncMock(return_value={"fallback": True})

    with patch("app.agents.tools.rust_client.search_properties", rust_stub):
        result = await search_properties(
            city=None,
            property_type="apartment",
            tool_context=ctx,
        )

    assert result["status"] == Status.MISSING_CRITICAL_DATA
    assert "city" in result.get("missing", [])
    assert "city" in result.get("context", "").lower()
    assert "properties" not in result
    rust_stub.assert_not_awaited()


def test_policy_router_delegation_matrix():
    cases = [
        {
            "name": "property_search",
            "frame": _make_frame("search_property", entities={"city": "New York", "property_type": "apartment"}),
            "soft_state": {},
            "expected_action": "execute_tool",
            "expected_tool": "search_properties",
        },
        {
            "name": "booking_continuation",
            "frame": _make_frame("booking_continuation", entities={"property_id": "prop-001"}),
            "soft_state": {},
            "expected_action": "execute_tool",
            "expected_tool": "request_booking_details",
        },
        {
            "name": "faq_during_booking",
            "frame": _make_frame("faq", entities={"question": "What is check-in time?"}),
            "soft_state": {"pending_booking": {"property_id": "prop-001"}},
            "expected_action": "execute_tool",
            "expected_tool": "check_faq",
        },
        {
            "name": "booking_status",
            "frame": _make_frame("booking_status", entities={"booking_id": "BKG-123"}),
            "soft_state": {},
            "expected_action": "execute_tool",
            "expected_tool": "check_booking_status",
        },
        {
            "name": "city_list",
            "frame": _make_frame("city_list"),
            "soft_state": {},
            "expected_action": "execute_tool",
            "expected_tool": "get_all_available_cities",
        },
        {
            "name": "handoff",
            "frame": _make_frame("human_handoff"),
            "soft_state": {},
            "expected_action": "escalate",
            "expected_tool": "escalate_to_human",
        },
        {
            "name": "small_talk",
            "frame": _make_frame("small_talk"),
            "soft_state": {},
            "expected_action": "execute_tool",
            "expected_tool": "handle_small_talk",
        },
    ]

    for case in cases:
        decision = policy_router.decide(case["frame"], case["soft_state"])
        assert decision["action"] == case["expected_action"], case["name"]
        assert decision.get("tool_name") == case["expected_tool"], case["name"]


@pytest.mark.asyncio
async def test_faq_during_booking_preserves_context():
    ctx = _Ctx()
    ctx.state["soft_state"]["pending_booking"] = {
        "property_id": "prop-001",
        "property": "Hudson Loft",
        "guest_email": "jane@example.com",
        "check_in": "2026-05-01",
        "check_out": "2026-05-03",
        "guests": 2,
        "price_per_night": 150.0,
    }

    with patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})), \
         patch("app.components.faq_enhanced.enhanced_faq_agent", return_value={"reply": "No smoking."}):
        result = await support.check_faq(
            question="Can I smoke inside the property?",
            tool_context=ctx,
        )

    assert result["status"] == Status.ANSWERED
    assert result.get("context_flag") == "faq_answered"
    assert ctx.state["soft_state"].get("pending_booking") is not None


@pytest.mark.asyncio
async def test_search_output_feeds_deterministic_render():
    ctx = _Ctx()
    dataset = _seed_dataset()

    with patch("app.components.search._DATASET", dataset), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}):
        result = await search_properties(
            city="New York",
            property_type="apartment",
            max_results=2,
            tool_context=ctx,
        )

    rendered = _render_property_results_from_router_output(result)
    assert rendered

    for prop in result["properties"]:
        assert prop["title"] in rendered

    for fake in ("the grand", "city lights", "sky high", "luxury manhattan loft", "cozy studio"):
        assert fake not in rendered.lower()


def test_taxonomy_drives_property_types():
    types = get_all_canonical_types()
    assert "apartment" in types
    assert "villa" in types
    assert "duplex" in types


def test_empty_intent_catalog_is_accepted(monkeypatch):
    original = dynamic_config._load_yaml

    def fake_load_yaml(path):
        if str(path).endswith("intent_catalog.yaml"):
            return {}
        return original(path)

    monkeypatch.setattr(dynamic_config, "_load_yaml", fake_load_yaml)
    dynamic_config._cache.clear()

    catalog = dynamic_config.get_intent_catalog()
    assert isinstance(catalog.intents, dict)
    assert catalog.default_threshold >= 0.0
