from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.tools.search import paginate_stored_results, search_properties, select_property
from app.agents.tools.support import check_faq
from app.services import adk_runner


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


def _snapshot_from_ctx(ctx: _Ctx) -> dict:
    return {
        "state": {"soft_state": ctx.state["soft_state"]},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-faq-resume",
            "last_update_time": 1.0,
        },
    }


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
        assert result["shown_count"] == 12
        assert result["pagination"]["has_more"] is False

        next_page = paginate_stored_results(ctx.state["soft_state"], direction="next")
        assert next_page is not None
        assert next_page["status"] == "properties_found"
        assert next_page["deterministic_reply"] == "All matching properties are already shown."

        returned_types = {item["property_type"].lower() for item in result["properties"]}
        assert returned_types == {"apartment"}

        selected = await select_property(option_number=7, tool_context=ctx)
        assert selected["status"] == "property_details"
        assert selected["property"]["id"] == result["properties"][6]["id"]


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

    assert result["shown_count"] == result["total_found"]
    assert result["pagination"]["page_start"] == 1
    assert result["pagination"]["page_end"] == result["total_found"]


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


@pytest.mark.asyncio
async def test_faq_during_property_list_answers_policy_and_sets_resume_target():
    ctx = _Ctx()
    fake = _make_props("apartment", "New York", 6)

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}), \
         patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})):
        await search_properties(city="New York", property_type="apartment", tool_context=ctx)
        faq = await check_faq(
            question="first of all let me know the refund policy i return the booking before 5 days of check-in?",
            tool_context=ctx,
        )

    soft_state = ctx.state["soft_state"]
    interruption = soft_state["faq_interruption"]

    assert faq["status"] == "answered"
    assert "40%" in faq["answer"]
    assert "third-party" in faq["answer"].lower()
    assert "bank or card processing fees" in faq["answer"].lower()
    assert interruption["active"] is True
    assert interruption["resume_target"] == "property_menu"
    assert interruption["last_faq_intent"] == "refund_policy"
    assert soft_state["last_visible_results"] == soft_state["visible_results"]
    assert soft_state["last_search_filters"] == soft_state["last_filters"]
    assert "Would you like to continue with the property list you were viewing" in faq["deterministic_reply"]


@pytest.mark.asyncio
async def test_sure_after_faq_returns_previous_property_menu():
    ctx = _Ctx()
    fake = _make_props("apartment", "New York", 6)
    snapshot = _snapshot_from_ctx(ctx)

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history
        snapshot["meta"].update(metadata or {})

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}), \
         patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})):
        await search_properties(city="New York", property_type="apartment", tool_context=ctx)
        await check_faq(
            question="what is the refund policy if i cancel before 5 days of check-in?",
            tool_context=ctx,
        )
        with patch.object(adk_runner, "get_session_snapshot", fake_get_session_snapshot), \
             patch.object(adk_runner, "save_session_snapshot", fake_save_session_snapshot):
            payload = await adk_runner._maybe_handle_faq_resume_turn(
                session_id="s-faq-property-menu",
                message="sure",
            )

    assert payload is not None
    assert payload["status"] == "properties_found"
    assert "Apartment In New York 1" in payload["deterministic_reply"]
    assert "Reply with an option number to see full details" in payload["deterministic_reply"]
    assert "faq_interruption" not in snapshot["state"]["soft_state"]


@pytest.mark.asyncio
async def test_second_faq_after_faq_answers_new_policy_question_not_resume():
    ctx = _Ctx()
    fake = _make_props("apartment", "New York", 6)
    snapshot = _snapshot_from_ctx(ctx)

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history
        snapshot["meta"].update(metadata or {})

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}), \
         patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})):
        await search_properties(city="New York", property_type="apartment", tool_context=ctx)
        await check_faq(question="what is the refund policy before 5 days of check-in?", tool_context=ctx)
        with patch.object(adk_runner, "get_session_snapshot", fake_get_session_snapshot), \
             patch.object(adk_runner, "save_session_snapshot", fake_save_session_snapshot):
            payload = await adk_runner._maybe_handle_faq_resume_turn(
                session_id="s-faq-second-question",
                message="how much refund of deposit will i got?",
            )

    interruption = snapshot["state"]["soft_state"]["faq_interruption"]

    assert payload is not None
    assert payload["status"] == "answered"
    assert "unpaid rent" in payload["answer"].lower()
    assert "late fees" in payload["answer"].lower()
    assert "10–14 business days" in payload["answer"]
    assert interruption["active"] is True
    assert interruption["resume_target"] == "property_menu"


@pytest.mark.asyncio
async def test_faq_during_selected_property_can_resume_selected_property():
    ctx = _Ctx()
    fake = _make_props("apartment", "New York", 6)
    snapshot = _snapshot_from_ctx(ctx)

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history
        snapshot["meta"].update(metadata or {})

    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}), \
         patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})):
        await search_properties(city="New York", property_type="apartment", tool_context=ctx)
        selected = await select_property(option_number=3, tool_context=ctx)
        faq = await check_faq(question="how much refund of deposit will i got?", tool_context=ctx)
        assert snapshot["state"]["soft_state"]["faq_interruption"]["resume_target"] == "selected_property"
        with patch.object(adk_runner, "get_session_snapshot", fake_get_session_snapshot), \
             patch.object(adk_runner, "save_session_snapshot", fake_save_session_snapshot):
            payload = await adk_runner._maybe_handle_faq_resume_turn(
                session_id="s-faq-selected-property",
                message="go back",
            )

    interruption = snapshot["state"]["soft_state"].get("faq_interruption")

    assert faq["status"] == "answered"
    assert snapshot["state"]["soft_state"]["selected_property"]["id"] == selected["property"]["id"]
    assert payload is not None
    assert payload["status"] == "property_details"
    assert selected["property"]["title"] in payload["deterministic_reply"]
    assert interruption is None


@pytest.mark.asyncio
async def test_refund_5_days_uses_40_percent_policy():
    ctx = _Ctx()
    with patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})):
        faq = await check_faq(
            question="refund before 5 days of check-in",
            tool_context=ctx,
        )

    answer = faq["answer"].lower()
    assert "40%" in faq["answer"]
    assert "4–6 days" in faq["answer"]
    assert "fees actually received" in answer


@pytest.mark.asyncio
async def test_deposit_refund_uses_deposit_policy_details():
    ctx = _Ctx()
    with patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})):
        faq = await check_faq(
            question="how much refund of deposit will i got?",
            tool_context=ctx,
        )

    answer = faq["answer"].lower()
    assert "unpaid rent" in answer
    assert "late fees" in answer
    assert "damages beyond ordinary wear" in answer
    assert "10–14 business days" in faq["answer"]


@pytest.mark.asyncio
async def test_no_vague_let_me_check_response_for_policy_faq():
    ctx = _Ctx()
    with patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})):
        faq = await check_faq(
            question="what is the refund policy before 5 days of check-in?",
            tool_context=ctx,
        )

    reply = str(faq.get("deterministic_reply") or faq.get("answer") or "").lower()
    assert "let me check our faq" not in reply
    assert "let me check the exact details" not in reply
