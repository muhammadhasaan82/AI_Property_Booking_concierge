from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.tools.search import paginate_stored_results, search_properties, select_property
from app.agents.tools.support import check_faq


class _Ctx:
    def __init__(self):
        self.state = {"soft_state": {}}


def _make_props(kind: str, city: str, count: int, start: int = 1):
    rows = []
    for idx in range(start, start + count):
        rows.append(
            {
                "id": f"{kind}-{idx}",
                "title": f"{kind.title()} In {city} {idx}",
                "city": city,
                "price_per_night": 100 + idx,
                "property_type": kind.title(),
                "bedrooms": 1,
                "bathrooms": 1,
                "rating": 4.0,
                "amenities": ["wifi"],
                "description": f"{kind} in {city}",
            }
        )
    return rows


@pytest.mark.asyncio
async def test_search_faq_option_3_keeps_state():
    ctx = _Ctx()
    fake = _make_props("apartment", "New York", 9) + _make_props("house", "New York", 3)

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}), \
         patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"answer": "Pets vary by property."})):

        result = await search_properties(city="New York", property_type="apartment", tool_context=ctx)
        assert result["status"] == "properties_found"
        assert result["total_found"] == 9

        faq = await check_faq(question="can i bring pets?", tool_context=ctx)
        assert faq["status"] == "answered"

        selected = await select_property(option_number=3, tool_context=ctx)
        assert selected["status"] == "property_details"
        assert selected["property"]["id"] == "apartment-3"


@pytest.mark.asyncio
async def test_search_show_more_option_2_keeps_filters():
    ctx = _Ctx()
    fake = _make_props("apartment", "New York", 12) + _make_props("duplex", "New York", 4)

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}):

        result = await search_properties(city="New York", property_type="apartment", tool_context=ctx)
        assert result["status"] == "properties_found"
        assert result["total_found"] == 12
        page_size = result["pagination"]["page_size"]
        assert len(result["properties"]) == min(result["total_found"], page_size)

        next_page = paginate_stored_results(ctx.state["soft_state"], direction="next")
        assert next_page is not None
        assert next_page["status"] == "properties_found"
        assert next_page["pagination"]["current_page"] == 2
        assert next_page["query_context"]["city"] == "New York"
        assert next_page["query_context"]["property_type"] == "apartment"

        returned_types = {item["property_type"].lower() for item in next_page["properties"]}
        assert returned_types == {"apartment"}

        selected = await select_property(option_number=2, tool_context=ctx)
        assert selected["status"] == "property_details"
        assert selected["property"]["id"] == "apartment-7"


@pytest.mark.asyncio
async def test_apartment_search_returns_only_apartments():
    ctx = _Ctx()
    fake = (
        _make_props("apartment", "New York", 7)
        + _make_props("house", "New York", 2)
        + _make_props("townhouse", "New York", 2)
    )

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}):

        result = await search_properties(city="New York", property_type="apartment", tool_context=ctx)

    assert result["status"] == "properties_found"
    assert result["total_found"] == 7
    returned_types = {item["property_type"].lower() for item in result["properties"]}
    assert returned_types == {"apartment"}


@pytest.mark.asyncio
async def test_total_found_is_full_count_not_page_size():
    ctx = _Ctx()
    fake = _make_props("apartment", "New York", 15)

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}):

        result = await search_properties(city="New York", property_type="apartment", tool_context=ctx)

    page_size = result["pagination"]["page_size"]
    assert result["shown_count"] == min(result["total_found"], page_size)
    assert result["pagination"]["page_start"] == 1
    assert result["pagination"]["page_end"] == min(result["total_found"], page_size)


@pytest.mark.asyncio
async def test_faq_does_not_clear_soft_state_search_keys():
    ctx = _Ctx()
    fake = _make_props("apartment", "New York", 6)

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}), \
         patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"answer": "Refunds depend on fare rules."})):

        await search_properties(city="New York", property_type="apartment", tool_context=ctx)
        before = dict(ctx.state["soft_state"])

        faq = await check_faq(question="what is the refund policy?", tool_context=ctx)
        assert faq["status"] == "answered"

        after = ctx.state["soft_state"]

    for key in (
        "active_flow",
        "last_filters",
        "all_search_results",
        "current_page",
        "page_size",
        "visible_results",
        "option_map",
    ):
        assert key in after
        assert before[key] == after[key]