from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.tools import search as search_tool
from app.agents.tools.search import search_properties, select_property
from app.agents.tools.support import check_faq
from app.services.adk_runner import _render_property_results_from_router_output


class _Ctx:
    def __init__(self):
        self.state = {"soft_state": {}}


def _make_apartment(idx: int, *, rating: float, reviews: int, price: int) -> dict:
    return {
        "id": f"apt-{idx}",
        "title": f"Apartment {idx}",
        "city": "New York",
        "price_per_night": price,
        "property_type": "Apartment",
        "bedrooms": 1,
        "bathrooms": 1,
        "rating": rating,
        "reviews_count": reviews,
        "amenities": ["wifi"],
        "description": f"Apartment {idx} in New York",
    }


def _apartments(count: int) -> list[dict]:
    rows = []
    for idx in range(1, count + 1):
        rows.append(
            _make_apartment(
                idx,
                rating=4.0 + (idx % 3) * 0.1,
                reviews=100 - idx,
                price=100 + idx,
            )
        )
    return rows


def _sort_expected(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda item: (
            -(item.get("rating") or 0),
            -(item.get("reviews_count") or 0),
            item.get("price_per_night") or 10**9,
        ),
    )


@pytest.mark.asyncio
async def test_filtered_search_lists_all_matching_apartments_no_pagination():
    ctx = _Ctx()
    apartments = [
        _make_apartment(1, rating=4.9, reviews=20, price=180),
        _make_apartment(2, rating=4.9, reviews=25, price=190),
        _make_apartment(3, rating=4.8, reviews=30, price=160),
        _make_apartment(4, rating=4.8, reviews=30, price=150),
        _make_apartment(5, rating=4.7, reviews=50, price=140),
        _make_apartment(6, rating=4.6, reviews=60, price=130),
        _make_apartment(7, rating=4.6, reviews=55, price=120),
        _make_apartment(8, rating=4.5, reviews=70, price=110),
    ]
    fake = apartments + [
        {
            "id": "house-1",
            "title": "House 1",
            "city": "New York",
            "price_per_night": 100,
            "property_type": "House",
            "bedrooms": 2,
            "bathrooms": 1,
            "rating": 5.0,
            "reviews_count": 999,
            "amenities": [],
            "description": "",
        }
    ]

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        result = await search_properties(
            city="New York",
            property_type="apartment",
            tool_context=ctx,
        )

    assert result["total_found"] == 8
    assert result["shown_count"] == 8
    assert result["has_more"] is False
    assert result["pagination"]["has_more"] is False
    assert result["pagination"]["has_next"] is False
    assert len(result["properties"]) == 8
    assert {
        item["property_type"].lower() for item in result["properties"]
    } == {"apartment"}
    assert {item["city"] for item in result["properties"]} == {"New York"}
    assert [item["id"] for item in result["properties"]] == [
        item["id"] for item in _sort_expected(apartments)
    ]
    assert set(ctx.state["soft_state"]["option_map"]) == {str(i) for i in range(1, 9)}

    rendered = _render_property_results_from_router_output(result).lower()
    assert "show more" not in rendered
    assert "next page" not in rendered


@pytest.mark.asyncio
async def test_no_fixed_limit_5_remains_for_filtered_search():
    ctx = _Ctx()
    fake = _apartments(12)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        result = await search_properties(
            city="New York",
            property_type="apartment",
            tool_context=ctx,
        )

    assert result["shown_count"] == 12
    assert result["shown_count"] != 5


@pytest.mark.asyncio
async def test_option_selection_works_above_5():
    ctx = _Ctx()
    fake = _apartments(8)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        result = await search_properties(
            city="New York",
            property_type="apartment",
            tool_context=ctx,
        )
        selected = await select_property(option_number=7, tool_context=ctx)

    assert selected["status"] == "property_details"
    assert selected["property"]["id"] == result["properties"][6]["id"]


@pytest.mark.asyncio
async def test_faq_then_option_above_5_preserves_context():
    ctx = _Ctx()
    fake = _apartments(8)

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ), patch(
        "app.agents.tools.rust_client.execute_tool",
        new=AsyncMock(return_value={"answer": "Pets vary by property."}),
    ):
        result = await search_properties(
            city="New York",
            property_type="apartment",
            tool_context=ctx,
        )
        faq = await check_faq(question="can i bring pets?", tool_context=ctx)
        selected = await select_property(option_number=7, tool_context=ctx)

    assert faq["status"] == "answered"
    assert selected["status"] == "property_details"
    assert selected["property"]["id"] == result["properties"][6]["id"]


@pytest.mark.asyncio
async def test_search_display_mode_can_be_changed_by_config(monkeypatch):
    ctx = _Ctx()
    fake = _apartments(12)
    monkeypatch.setattr(
        search_tool.cfg,
        "search_display",
        SimpleNamespace(
            mode="paginated",
            sort=[
                {"field": "rating", "direction": "desc", "missing_last": True},
                {"field": "reviews_count", "direction": "desc", "missing_last": True},
                {"field": "price_per_night", "direction": "asc", "missing_last": True},
            ],
            max_inline_results=5,
            pagination_enabled=True,
        ),
    )

    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        result = await search_properties(
            city="New York",
            property_type="apartment",
            tool_context=ctx,
        )

    assert result["total_found"] == 12
    assert result["shown_count"] == 5
    assert result["pagination"]["has_next"] is True
