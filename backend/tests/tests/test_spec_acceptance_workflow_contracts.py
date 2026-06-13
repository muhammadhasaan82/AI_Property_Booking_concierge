"""Acceptance tests generated from specs/booking_workflow.md and specs/acceptance_tests.md."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import adk_runner
from app.services.direct_property_search import maybe_handle_direct_property_search


def _property(property_id: str, title: str) -> dict:
    return {
        "id": property_id,
        "title": title,
        "city": "Seattle",
        "price_per_night": 120,
        "property_type": "Apartment",
        "bedrooms": 2,
        "bathrooms": 1,
        "rating": 4.4,
        "reviews_count": 25,
        "amenities": ["wifi", "parking"],
        "occupancy_max": 4,
        "description": f"{title} in Seattle",
    }


@pytest.mark.asyncio
async def test_active_booking_blocks_pagination_shortcut_outside_property_reselection(monkeypatch):
    initial_soft_state = {
        "booking_stage": "collecting_details",
        "booking_property_id": "apt-1",
        "all_search_results": [_property("apt-1", "Apartment 1"), _property("apt-2", "Apartment 2"), _property("apt-3", "Apartment 3")],
        "visible_results": [_property("apt-1", "Apartment 1"), _property("apt-2", "Apartment 2")],
        "last_filters": {"city": "Seattle", "property_type": "apartment"},
        "current_page": 1,
        "page_size": 2,
        "option_map": {"1": {"property_id": "apt-1"}, "2": {"property_id": "apt-2"}},
        "active_property_options_map": {"1": {"property_id": "apt-1"}, "2": {"property_id": "apt-2"}},
        "last_search": {
            "pagination": {
                "current_page": 1,
                "page_size": 2,
                "page_start": 1,
                "page_end": 2,
                "total_pages": 2,
                "has_more": True,
                "has_next": True,
                "has_prev": False,
                "pagination_enabled": True,
            }
        },
    }
    snapshot = {"state": {"soft_state": initial_soft_state}, "history": [], "meta": {}}

    async def fake_get_snapshot(_session_id):
        return snapshot

    async def fake_save_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_snapshot)

    payload = await adk_runner._maybe_handle_search_state_shortcut(
        session_id="s-booking-shortcut",
        message="show more",
    )

    assert payload is None
    soft_state = snapshot["state"]["soft_state"]
    assert soft_state["current_page"] == 1
    assert [item["id"] for item in soft_state["visible_results"]] == ["apt-1", "apt-2"]


@pytest.mark.asyncio
async def test_property_reselection_stage_allows_selection_shortcut(monkeypatch):
    visible_results = [_property("apt-1", "Apartment 1"), _property("apt-2", "Apartment 2")]
    snapshot = {
        "state": {
            "soft_state": {
                "booking_stage": "awaiting_property_reselection",
                "booking_property_id": "apt-1",
                "last_presented_view": "property_list",
                "visible_results": visible_results,
                "all_search_results": list(visible_results),
                "option_map": {
                    "1": {"property_id": "apt-1"},
                    "2": {"property_id": "apt-2"},
                },
                "active_property_options_map": {
                    "1": {"property_id": "apt-1"},
                    "2": {"property_id": "apt-2"},
                },
                "last_search": {
                    "properties": [
                        dict(visible_results[0], number=1),
                        dict(visible_results[1], number=2),
                    ]
                },
            }
        },
        "history": [],
        "meta": {},
    }

    async def fake_get_snapshot(_session_id):
        return snapshot

    async def fake_save_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_snapshot)

    payload = await adk_runner._maybe_handle_search_state_shortcut(
        session_id="s-reselect",
        message="option 2",
    )

    assert payload is not None
    assert payload["status"] == "property_details"
    assert payload["property"]["id"] == "apt-2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "I want to check my booking status in Seattle",
        "what is the cancellation policy in Seattle?",
        "cancel my booking in Seattle",
    ],
)
async def test_booking_status_and_cancellation_intents_are_not_intercepted_by_direct_search(message: str):
    snapshot = {"state": {"soft_state": {}}, "history": [], "meta": {}}

    async def fake_get_snapshot(_session_id):
        return snapshot

    async def fake_save_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    payload = await maybe_handle_direct_property_search(
        message,
        "s-routing",
        get_snapshot=fake_get_snapshot,
        save_snapshot=fake_save_snapshot,
    )

    assert payload is None


@pytest.mark.asyncio
async def test_booking_status_turn_stays_deterministic_and_bypasses_adk(monkeypatch):
    snapshot = {
        "state": {
            "soft_state": {
                "booking_receipt": {
                    "booking_id": "BK-20260612-ABCDEF12",
                    "property_title": "Apartment 2",
                    "guest_name": "Jane Doe",
                    "guest_email": "jane@example.com",
                    "guest_phone": "03001234567",
                    "check_in": "2026-06-20",
                    "check_out": "2026-06-22",
                    "nights": 2,
                    "guests": 2,
                    "price_per_night": 120.0,
                    "total_amount": 240.0,
                    "status": "confirmed",
                }
            }
        },
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-status-spec",
            "last_update_time": 1.0,
        },
    }

    async def fake_get_snapshot(_session_id):
        return snapshot

    async def fake_save_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_snapshot)
    payload = await adk_runner._maybe_handle_booking_status_check(
        session_id="s-status-spec",
        message="My booking ID is BK-20260612-ABCDEF12",
    )

    assert payload is not None
    reply = payload["deterministic_reply"]
    assert "BK-20260612-ABCDEF12" in reply
    assert "Apartment 2" in reply
    assert "confirmed" in reply
