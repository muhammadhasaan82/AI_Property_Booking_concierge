from __future__ import annotations

import copy
from unittest.mock import AsyncMock, patch
import pytest

from app.services import adk_runner
from app.services.booking_flow import _receipt_reply

_TEST_RECEIPT = {
    "booking_id": "BK-20260601-12345678",
    "status": "confirmed",
    "check_in": "2026-06-02",
    "check_out": "2026-06-11",
    "user_name": "Jane Doe",
    "user_email": "jane@example.com",
    "property_title": "Apartment 3",
    "city": "Test City",
    "payment_url": "https://pay.example.com",
    "guests": 2,
    "nights": 9,
    "price_per_night": 100.0,
    "total_amount": 900.0,
}

def _build_cancellation_snapshot(soft_state_data: dict) -> dict:
    return {
        "state": {"soft_state": soft_state_data},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-cancellation",
            "last_update_time": 1.0,
        },
    }

async def _run_cancellation_turn(
    monkeypatch,
    snapshot: dict,
    message: str,
    *,
    mock_update_sb=None,
    mock_update_b=None,
):
    """
    Run a single turn through run_adk_turn and return the reply plus routing mocks.
    """
    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = state
        snapshot["history"] = history
        snapshot["meta"].update(metadata or {})

    def fail_get_runner():
        raise AssertionError("ADK runner must not be invoked for booking-cancellation")

    direct_search = AsyncMock(return_value=None)
    route_pre_adk = AsyncMock(return_value=None)

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "maybe_handle_direct_property_search", direct_search)
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "_get_runner", fail_get_runner)
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)

    from contextlib import ExitStack

    chunks = []
    with ExitStack() as stack:
        if mock_update_sb is not None:
            stack.enter_context(patch("app.observability.db_logging.update_successful_booking", mock_update_sb))
        if mock_update_b is not None:
            stack.enter_context(patch("app.services.booking.update_booking_status", mock_update_b))
        async for chunk in adk_runner.run_adk_turn("u-cancellation", "s-cancellation", message):
            chunks.append(chunk)

    return "".join(chunks), direct_search, route_pre_adk

@pytest.mark.asyncio
async def test_delete_request_without_id_asks_for_booking_id(monkeypatch):
    """
    1. delete request without ID asks for booking ID
    """
    snapshot = _build_cancellation_snapshot({})
    reply, _, _ = await _run_cancellation_turn(monkeypatch, snapshot, "delete booking")
    
    assert "please provide your booking registration id" in reply.lower()
    assert snapshot["state"]["soft_state"].get("booking_cancellation_pending") is True
    assert snapshot["state"]["soft_state"].get("booking_cancellation_stage") == "awaiting_id"

@pytest.mark.asyncio
async def test_delete_request_with_current_receipt_asks_for_confirmation(monkeypatch):
    """
    2. delete request with current receipt asks for confirmation
    """
    snapshot = _build_cancellation_snapshot({
        "booking_receipt": dict(_TEST_RECEIPT)
    })
    
    with (
        patch("app.observability.db_logging.get_successful_booking_status", AsyncMock(return_value=_TEST_RECEIPT)),
        patch("app.services.booking.get_booking_status", AsyncMock(return_value={"ok": True, "status": "confirmed"}))
    ):
        reply, _, _ = await _run_cancellation_turn(monkeypatch, snapshot, "cancel booking")
        
    assert "Are you sure you want to cancel this booking?" in reply
    assert snapshot["state"]["soft_state"].get("booking_cancellation_pending") is True
    assert snapshot["state"]["soft_state"].get("booking_cancellation_stage") == "awaiting_confirmation"
    assert snapshot["state"]["soft_state"].get("booking_cancellation_id") == "BK-20260601-12345678"

@pytest.mark.asyncio
async def test_screenshot_flow_cancellation_success(monkeypatch):
    """
    3. screenshot flow: delete request -> booking ID -> yes delete -> DB status updated to cancelled
    """
    snapshot = _build_cancellation_snapshot({})
    
    reply1, _, _ = await _run_cancellation_turn(monkeypatch, snapshot, "delete booking")
    assert "please provide your booking registration id" in reply1.lower()
    assert snapshot["state"]["soft_state"].get("booking_cancellation_stage") == "awaiting_id"
    
    with (
        patch("app.observability.db_logging.get_successful_booking_status", AsyncMock(return_value=_TEST_RECEIPT)),
        patch("app.services.booking.get_booking_status", AsyncMock(return_value={"ok": True, "status": "confirmed"}))
    ):
        reply2, _, _ = await _run_cancellation_turn(monkeypatch, snapshot, "BK-20260601-12345678")
    assert "Are you sure you want to cancel this booking?" in reply2
    assert snapshot["state"]["soft_state"].get("booking_cancellation_stage") == "awaiting_confirmation"
    
    mock_update_sb = AsyncMock(return_value=True)
    mock_update_b = AsyncMock(return_value={"ok": True})
    reply3, _, _ = await _run_cancellation_turn(
        monkeypatch,
        snapshot,
        "yes delete this booking please",
        mock_update_sb=mock_update_sb,
        mock_update_b=mock_update_b,
    )

    assert "successfully cancelled" in reply3
    mock_update_sb.assert_awaited_once_with("BK-20260601-12345678", {"status": "cancelled"})
    mock_update_b.assert_awaited_once_with("BK-20260601-12345678", "", "cancelled")
    
    assert "booking_cancellation_pending" not in snapshot["state"]["soft_state"]
    assert "booking_cancellation_stage" not in snapshot["state"]["soft_state"]
    assert "booking_cancellation_id" not in snapshot["state"]["soft_state"]

@pytest.mark.asyncio
async def test_rejection_keeps_booking_unchanged(monkeypatch):
    """
    4. rejection keeps booking unchanged
    """
    snapshot = _build_cancellation_snapshot({
        "booking_cancellation_pending": True,
        "booking_cancellation_stage": "awaiting_confirmation",
        "booking_cancellation_id": "BK-20260601-12345678",
        "booking_cancellation_receipt": dict(_TEST_RECEIPT),
    })
    
    mock_update_sb = AsyncMock(return_value=True)
    mock_update_b = AsyncMock(return_value={"ok": True})
    reply, _, _ = await _run_cancellation_turn(
        monkeypatch,
        snapshot,
        "no",
        mock_update_sb=mock_update_sb,
        mock_update_b=mock_update_b,
    )

    assert "kept your booking unchanged" in reply
    mock_update_sb.assert_not_awaited()
    mock_update_b.assert_not_awaited()
    
    assert "booking_cancellation_pending" not in snapshot["state"]["soft_state"]
    assert "booking_cancellation_stage" not in snapshot["state"]["soft_state"]

@pytest.mark.asyncio
async def test_yes_delete_does_not_route_to_search(monkeypatch):
    """
    5. “yes delete this booking please” must not route to property search
    """
    snapshot = _build_cancellation_snapshot({
        "booking_cancellation_pending": True,
        "booking_cancellation_stage": "awaiting_confirmation",
        "booking_cancellation_id": "BK-20260601-12345678",
        "booking_cancellation_receipt": dict(_TEST_RECEIPT),
    })
    
    mock_update_sb = AsyncMock(return_value=True)
    mock_update_b = AsyncMock(return_value={"ok": True})
    reply, direct_search, route_pre_adk = await _run_cancellation_turn(
        monkeypatch,
        snapshot,
        "yes delete this booking please",
        mock_update_sb=mock_update_sb,
        mock_update_b=mock_update_b,
    )

    assert "successfully cancelled" in reply
    direct_search.assert_not_awaited()
    route_pre_adk.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "do not cancel",
        "don't cancel",
        "dont delete",
        "do not delete",
        "keep it",
        "nevermind",
    ],
)
async def test_negated_cancellation_is_rejection(monkeypatch, message):
    snapshot = _build_cancellation_snapshot({
        "booking_cancellation_pending": True,
        "booking_cancellation_stage": "awaiting_confirmation",
        "booking_cancellation_id": "BK-20260601-12345678",
        "booking_cancellation_receipt": dict(_TEST_RECEIPT),
    })

    mock_update_sb = AsyncMock(return_value=True)
    mock_update_b = AsyncMock(return_value={"ok": True})
    reply, _, _ = await _run_cancellation_turn(
        monkeypatch,
        snapshot,
        message,
        mock_update_sb=mock_update_sb,
        mock_update_b=mock_update_b,
    )

    assert "kept your booking unchanged" in reply
    mock_update_sb.assert_not_awaited()
    mock_update_b.assert_not_awaited()
    assert snapshot["state"]["soft_state"].get("booking_cancellation_stage") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "I do not want to cancel my booking",
        "I don't want to cancel my booking",
        "dont cancel my booking",
        "do not delete my booking",
        "keep my booking",
        "never mind, don't cancel",
    ],
)
async def test_negated_cancellation_without_confirmation_does_not_start_flow(message):
    from app.services.booking.cancellation import handle_booking_cancellation_turn

    soft_state = {
        "booking_receipt": dict(_TEST_RECEIPT),
        "booking_cancellation_pending": True,
        "booking_cancellation_id": "BK-20260601-12345678",
        "booking_cancellation_receipt": dict(_TEST_RECEIPT),
    }

    payload = await handle_booking_cancellation_turn(message, soft_state)

    assert payload["status"] == "cancellation_not_requested"
    assert payload["deterministic_reply"] == (
        "Okay, I will not cancel your booking. Your booking remains unchanged."
    )
    assert "booking_cancellation_pending" not in soft_state
    assert "booking_cancellation_stage" not in soft_state
    assert "booking_cancellation_id" not in soft_state
    assert "booking_cancellation_receipt" not in soft_state
    assert soft_state["booking_receipt"]["status"] == "confirmed"


@pytest.mark.asyncio
async def test_db_failure_keeps_pending_cancellation_state(monkeypatch):
    snapshot = _build_cancellation_snapshot({
        "booking_cancellation_pending": True,
        "booking_cancellation_stage": "awaiting_confirmation",
        "booking_cancellation_id": "BK-20260601-12345678",
        "booking_cancellation_receipt": dict(_TEST_RECEIPT),
    })

    mock_update_sb = AsyncMock(return_value=False)
    mock_update_b = AsyncMock(return_value={"ok": False})
    reply, _, _ = await _run_cancellation_turn(
        monkeypatch,
        snapshot,
        "yes",
        mock_update_sb=mock_update_sb,
        mock_update_b=mock_update_b,
    )

    assert "try confirming the cancellation again" in reply.lower()
    assert snapshot["state"]["soft_state"].get("booking_cancellation_stage") == "awaiting_confirmation"
    assert snapshot["state"]["soft_state"].get("booking_cancellation_id") == "BK-20260601-12345678"
