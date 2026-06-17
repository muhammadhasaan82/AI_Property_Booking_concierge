"""Service coverage guard: block unsupported regions before ADK/search."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import service_coverage_loader as scl
from app.config.service_coverage_loader import (
    _ServiceCoverageRouter,
    evaluate_message_coverage,
    load_service_coverage,
)
from app.services import adk_runner


def _default_raw() -> dict:
    return {
        "version": "1.0",
        "coverage": {
            "enabled": True,
            "supported_countries": ["United States"],
            "supported_country_codes": ["US", "USA"],
            "unsupported_region_response": (
                "This application is only available for the United States region only."
            ),
            "allow_dataset_regions_outside_supported_market": False,
        },
        "city_country_map": {
            "New York": "United States",
            "Lahore": "Pakistan",
            "Karachi": "Pakistan",
            "Islamabad": "Pakistan",
            "Dubai": "United Arab Emirates",
        },
        "region_aliases": {
            "United States": ["united states", "usa", "us", "america"],
            "Pakistan": ["pakistan", "pk"],
            "United Arab Emirates": ["uae", "united arab emirates", "dubai"],
        },
    }


def _install_router(raw: dict | None = None) -> _ServiceCoverageRouter:
    router = _ServiceCoverageRouter(raw or _default_raw())
    scl._set_router_for_tests(router)
    return router


@pytest.fixture(autouse=True)
def _restore_router():
    original = scl._router
    yield
    scl._set_router_for_tests(original)


def _adk_raises_if_called(monkeypatch):
    async def _boom(*_args, **_kwargs):
        raise AssertionError("ADK runner must not be invoked for blocked regions")

    fake_runner = type("FakeRunner", (), {"run_async": staticmethod(_boom)})()
    monkeypatch.setattr(adk_runner, "_get_runner", lambda: fake_runner)
    monkeypatch.setattr(
        adk_runner,
        "_get_session_service",
        lambda: type("S", (), {"get_session": staticmethod(AsyncMock(return_value=None))})(),
    )
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda m: (m, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda m: m)
    monkeypatch.setattr(adk_runner, "route_pre_adk", AsyncMock(return_value=None))
    monkeypatch.setattr(
        adk_runner,
        "_maybe_handle_search_state_shortcut",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        adk_runner,
        "_build_invocation_state_delta",
        AsyncMock(return_value={"user_cognitive_context": "", "soft_state": {}}),
    )
    monkeypatch.setattr(
        adk_runner,
        "get_session_snapshot",
        AsyncMock(return_value={"state": {"soft_state": {}}, "history": [], "meta": {}}),
    )
    monkeypatch.setattr(
        adk_runner,
        "save_session_snapshot",
        AsyncMock(),
    )


@pytest.mark.asyncio
async def test_lahore_blocked_before_adk(monkeypatch):
    _install_router()
    expected = load_service_coverage().unsupported_region_response
    _adk_raises_if_called(monkeypatch)

    chunks = []
    async for chunk in adk_runner.run_adk_turn(
        "smoke-user-lahore-search",
        "smoke-session-lahore-search",
        "I am looking for properties in Lahore",
    ):
        chunks.append(chunk)

    assert "".join(chunks) == expected


@pytest.mark.asyncio
async def test_pakistan_blocked(monkeypatch):
    _install_router()
    expected = load_service_coverage().unsupported_region_response
    _adk_raises_if_called(monkeypatch)

    chunks = []
    async for chunk in adk_runner.run_adk_turn(
        "u-pk",
        "s-pk",
        "Do you have properties in Pakistan?",
    ):
        chunks.append(chunk)

    assert "".join(chunks) == expected


@pytest.mark.asyncio
async def test_new_york_allowed_past_guard(monkeypatch):
    _install_router()

    async def fake_run_async(*_args, **_kwargs):
        event = MagicMock()
        event.author = "concierge_voice"
        event.content = MagicMock()
        event.content.parts = [MagicMock(text="Found options in New York.", function_call=None, function_response=None)]
        event.is_final_response.return_value = True
        yield event

    fake_runner = type("FakeRunner", (), {"run_async": staticmethod(fake_run_async)})()
    monkeypatch.setattr(adk_runner, "_get_runner", lambda: fake_runner)
    monkeypatch.setattr(
        adk_runner,
        "_get_session_service",
        lambda: type("S", (), {"get_session": staticmethod(AsyncMock(return_value=None))})(),
    )
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda m: (m, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda m: m)
    monkeypatch.setattr(adk_runner, "route_pre_adk", AsyncMock(return_value=None))
    monkeypatch.setattr(
        adk_runner,
        "_maybe_handle_search_state_shortcut",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        adk_runner,
        "_build_invocation_state_delta",
        AsyncMock(return_value={"user_cognitive_context": "", "soft_state": {}}),
    )
    monkeypatch.setattr(
        adk_runner,
        "get_session_snapshot",
        AsyncMock(return_value={"state": {"soft_state": {}}, "history": [], "meta": {}}),
    )

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u-ny", "s-ny", "I am looking for properties in New York"):
        chunks.append(chunk)

    assert "New York" in "".join(chunks)


def test_config_driven_pakistan_supported_unblocks_lahore():
    raw = _default_raw()
    raw["coverage"]["supported_countries"] = ["United States", "Pakistan"]
    _install_router(raw)

    decision = evaluate_message_coverage("I am looking for properties in Lahore")
    assert decision.blocked is False
    assert decision.supported is True


@pytest.mark.asyncio
async def test_no_booking_state_mutation_on_block(monkeypatch):
    _install_router()
    _adk_raises_if_called(monkeypatch)

    soft_state = {
        "selected_property_id": "prop-123",
        "pending_booking": {"city": "Boston", "status": "collecting"},
    }
    saved_states: list[dict] = []

    async def fake_get_session_snapshot(_sid):
        return {"state": {"soft_state": dict(soft_state)}, "history": [], "meta": {}}

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        saved_states.append(state)

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)

    async for _chunk in adk_runner.run_adk_turn("u-block", "s-block", "properties in Lahore"):
        pass

    assert soft_state["selected_property_id"] == "prop-123"
    assert soft_state["pending_booking"] == {"city": "Boston", "status": "collecting"}

    if saved_states:
        persisted = saved_states[-1].get("soft_state", {})
        assert persisted.get("selected_property_id") == "prop-123"
        assert persisted.get("pending_booking") == {"city": "Boston", "status": "collecting"}


def test_chat_message_endpoint_blocked_shape(monkeypatch):
    _install_router()
    expected = load_service_coverage().unsupported_region_response
    _adk_raises_if_called(monkeypatch)

    from app.main import app

    response = TestClient(app).post(
        "/api/v1/chat/message",
        json={
            "user_id": "smoke-user-lahore-search",
            "session_id": "smoke-session-lahore-search",
            "message": "I am looking for properties in Lahore",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == expected
    assert body["user_id"] == "smoke-user-lahore-search"
    assert body["session_id"] == "smoke-session-lahore-search"


def test_debug_config_includes_service_coverage():
    from app.main import app

    response = TestClient(app).get("/debug/config")
    assert response.status_code == 200
    coverage = response.json().get("service_coverage")
    assert coverage is not None
    assert coverage["enabled"] is True
    assert "United States" in coverage["supported_countries"]
    assert "allow_dataset_regions_outside_supported_market" in coverage


@pytest.mark.asyncio
async def test_unsupported_region_followup_flow(monkeypatch):
    _install_router()

    snapshot = {
        "state": {"soft_state": {}},
        "history": [],
        "meta": {
            "app_name": "ai_concierge",
            "user_id": "u-coverage-followup",
            "last_update_time": 1.0,
        },
    }

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history
        snapshot["meta"].update(metadata or {})

    async def _boom(*_args, **_kwargs):
        raise AssertionError("ADK runner must not be invoked in this follow-up flow")

    fake_runner = type("FakeRunner", (), {"run_async": staticmethod(_boom)})()
    monkeypatch.setattr(adk_runner, "_get_runner", lambda: fake_runner)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda m: (m, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda m: m)

    chunks = []
    async for chunk in adk_runner.run_adk_turn(
        "u-coverage-followup",
        "s-coverage-followup",
        "I am looking for condo in Lahore",
    ):
        chunks.append(chunk)

    reply1 = "".join(chunks)
    assert "United States" in reply1

    soft_state1 = snapshot["state"]["soft_state"]
    assert soft_state1.get("service_coverage_stage") == "awaiting_city_list_confirmation"
    assert soft_state1.get("last_unsupported_region") == "Pakistan"

    chunks = []
    async for chunk in adk_runner.run_adk_turn(
        "u-coverage-followup",
        "s-coverage-followup",
        "yes",
    ):
        chunks.append(chunk)

    reply2 = "".join(chunks)
    assert "Which city do you want to book with?" in reply2
    assert "New York" in reply2

    soft_state2 = snapshot["state"]["soft_state"]
    assert soft_state2.get("service_coverage_stage") == "awaiting_supported_city_choice"

