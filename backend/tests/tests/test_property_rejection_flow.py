from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


class _FakeRunner:
    def __init__(self, events):
        self._events = events

    async def run_async(self, **_kwargs):
        for event in self._events:
            yield event


class _FakeSessionService:
    def __init__(self, state):
        self._state = state

    async def get_session(self, **_kwargs):
        return SimpleNamespace(state=self._state)


def _make_event(*, author: str, tool_response: dict | None = None, text: str | None = None, final: bool = False):
    event = MagicMock()
    event.author = author
    event.content = MagicMock()
    parts = []
    if tool_response is not None:
        parts.append(
            MagicMock(function_response=MagicMock(response=tool_response), function_call=None, text=None)
        )
    if text is not None:
        parts.append(MagicMock(text=text, function_call=None, function_response=None))
    event.content.parts = parts
    event.is_final_response.return_value = final
    return event


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


@pytest.mark.asyncio
async def test_no_after_property_details_flat_state_compatibility(monkeypatch):
    """Proves that _maybe_handle_search_state_shortcut handles a flat compatibility state shape."""
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
    
    # Flat compatibility shape: the state dictionary itself has the soft_state fields directly
    flat_state = dict(ctx.state["soft_state"])
    snapshot = {
        "state": flat_state,
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-reject-flat",
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
    async for chunk in adk_runner.run_adk_turn("u-reject-flat", "s-reject-flat", "no"):
        chunks.append(chunk)

    reply = "".join(chunks)
    assert "Apartment 1" in reply
    assert "Apartment 8" in reply
    assert "generic concierge greeting" not in reply
    route_pre_adk.assert_not_awaited()

    # Verify that the mutated state has been persisted nested in "soft_state"
    assert "soft_state" in snapshot["state"]
    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["last_presented_view"] == "property_list"
    assert soft_state["last_rejected_property_id"] == search_result["properties"][2]["id"]
    assert len(soft_state["visible_results"]) == 8
    assert saved_states


@pytest.mark.asyncio
async def test_run_adk_turn_select_property_persists_soft_state(monkeypatch):
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

    snapshot = {
        "state": {"soft_state": dict(ctx.state["soft_state"])},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-adk-persist",
            "last_update_time": 1.0,
        },
    }

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history
        snapshot["meta"].update(metadata or {})

    selection_state = {
        "soft_state": {
            "last_presented_view": "property_details",
            "last_selected_property_id": search_result["properties"][2]["id"],
        }
    }

    events = [
        _make_event(author="triage_router", tool_response={"status": "property_details"}, final=False),
        _make_event(author="concierge_voice", text="details", final=True),
    ]

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "_get_session_service", lambda: _FakeSessionService(selection_state))
    monkeypatch.setattr(adk_runner, "_get_runner", lambda: _FakeRunner(events))
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)
    monkeypatch.setattr(
        adk_runner,
        "_build_invocation_state_delta",
        AsyncMock(return_value={"user_cognitive_context": "", "soft_state": {}}),
    )
    monkeypatch.setattr(adk_runner, "_maybe_handle_search_state_shortcut", AsyncMock(return_value=None))

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-adk-persist", "s-adk-persist", "please show me option 3"):
        chunks.append(chunk)

    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["last_presented_view"] == "property_details"
    assert soft_state["last_selected_property_id"] == search_result["properties"][2]["id"]
    assert soft_state["visible_results"]
    assert soft_state["option_map"]
    assert soft_state["all_search_results"]


@pytest.mark.asyncio
async def test_run_adk_turn_rejection_after_selection_returns_menu(monkeypatch):
    ctx = _Ctx()
    fake = _make_props(8)
    real_shortcut = adk_runner._maybe_handle_search_state_shortcut
    with patch("app.components.search._DATASET", fake), patch(
        "app.agents.tools.rust_client.search_properties",
        return_value={"fallback": True},
    ):
        search_result = await search_properties(
            city="New York",
            property_type="apartment",
            tool_context=ctx,
        )

    snapshot = {
        "state": {},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-adk-flow",
            "last_update_time": 1.0,
        },
    }

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history
        snapshot["meta"].update(metadata or {})

    search_events = [
        _make_event(author="triage_router", tool_response=search_result, final=False),
        _make_event(author="concierge_voice", text="menu", final=True),
    ]
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(
        adk_runner,
        "maybe_handle_direct_property_search",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        adk_runner,
        "_get_session_service",
        lambda: _FakeSessionService({"soft_state": dict(ctx.state["soft_state"])}),
    )
    monkeypatch.setattr(adk_runner, "_get_runner", lambda: _FakeRunner(search_events))
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)
    monkeypatch.setattr(
        adk_runner,
        "_build_invocation_state_delta",
        AsyncMock(return_value={"user_cognitive_context": "", "soft_state": {}}),
    )
    monkeypatch.setattr(adk_runner, "_maybe_handle_search_state_shortcut", AsyncMock(return_value=None))

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-adk-flow", "s-adk-flow", "I am looking for an apartment in New York"):
        chunks.append(chunk)

    selection_state = {
        "soft_state": {
            "last_presented_view": "property_details",
            "last_selected_property_id": search_result["properties"][2]["id"],
        }
    }
    events = [
        _make_event(author="triage_router", tool_response={"status": "property_details"}, final=False),
        _make_event(author="concierge_voice", text="details", final=True),
    ]

    monkeypatch.setattr(adk_runner, "_get_session_service", lambda: _FakeSessionService(selection_state))
    monkeypatch.setattr(adk_runner, "_get_runner", lambda: _FakeRunner(events))

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-adk-flow", "s-adk-flow", "please show me option 3"):
        chunks.append(chunk)

    monkeypatch.setattr(adk_runner, "_maybe_handle_search_state_shortcut", real_shortcut)

    def fail_get_runner():
        raise AssertionError("ADK runner must not be invoked for property rejection")

    route_pre_adk = AsyncMock(return_value={"reply": "generic concierge greeting"})
    monkeypatch.setattr(adk_runner, "_get_runner", fail_get_runner)
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-adk-flow", "s-adk-flow", "no"):
        chunks.append(chunk)

    reply = "".join(chunks)
    assert "Apartment 1" in reply
    assert "Apartment 8" in reply
    assert "generic concierge greeting" not in reply
    route_pre_adk.assert_not_awaited()

    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["last_presented_view"] == "property_list"
    assert soft_state["last_rejected_property_id"] == search_result["properties"][2]["id"]


@pytest.mark.asyncio
async def test_run_adk_turn_search_result_persists_visible_results_from_router_output(monkeypatch):
    """
    Regression: live path must persist visible_results / option_map / all_search_results
    into Redis soft_state even when updated_session.state["soft_state"] is empty.

    This mirrors the real failure:
      User: "I am looking for an apartment in New York"
      GET /debug/session -> soft_state: {}   <- bug; should contain search context

    The test wires a fake ADK runner whose triage_router event carries a
    `properties_found` payload.  The fake session service returns an empty
    soft_state (simulating that the ADK in-memory session did not propagate
    tool_context mutations).  After run_adk_turn the Redis snapshot must contain
    all navigable search keys.
    """
    props = _make_props(15)
    search_router_output = {
        "status": "properties_found",
        "properties": props,
        "total_found": 15,
        "shown_count": 15,
        "pagination": {
            "current_page": 1,
            "page_size": 15,
            "has_more": False,
        },
    }
    search_result = search_router_output

    snapshot: dict = {
        "state": {},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-live",
            "last_update_time": 1.0,
        },
    }

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history
        snapshot["meta"].update(metadata or {})

    # ADK session state has an empty soft_state -- simulates the live bug.
    adk_session_state = {
        "final_reply": "",
        "router_output": "",
        "soft_state": {},
    }

    events = [
        _make_event(author="triage_router", tool_response=search_router_output, final=False),
        _make_event(author="concierge_voice", text="Here are 15 apartments...", final=True),
    ]

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(
        adk_runner,
        "maybe_handle_direct_property_search",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(adk_runner, "_get_session_service", lambda: _FakeSessionService(adk_session_state))
    monkeypatch.setattr(adk_runner, "_get_runner", lambda: _FakeRunner(events))
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)
    monkeypatch.setattr(
        adk_runner,
        "_build_invocation_state_delta",
        AsyncMock(return_value={"user_cognitive_context": "", "soft_state": {}}),
    )
    monkeypatch.setattr(adk_runner, "_maybe_handle_search_state_shortcut", AsyncMock(return_value=None))

    chunks: list[str] = []
    async for chunk in adk_runner.run_adk_turn("u-live", "s-live", "I am looking for an apartment in New York"):
        chunks.append(chunk)

    soft_state = snapshot["state"].get("soft_state", {})

    # Core navigation context must be present.
    assert soft_state.get("visible_results"), "visible_results must be populated from router_output"
    assert len(soft_state["visible_results"]) == 15
    assert soft_state.get("all_search_results"), "all_search_results must be populated from router_output"
    assert len(soft_state["all_search_results"]) == 15
    assert soft_state.get("option_map"), "option_map must be populated from router_output"
    assert set(soft_state["option_map"].keys()) == {str(i) for i in range(1, 16)}
    assert soft_state.get("last_presented_view") == "property_list"

    # Confirm option_map values reference the correct property IDs.
    expected_id_1 = str(search_result["properties"][0].get("id") or "")
    assert soft_state["option_map"]["1"]["property_id"] == expected_id_1
