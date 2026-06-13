from __future__ import annotations

import copy
from unittest.mock import AsyncMock, patch

import pytest

from app.config.booking_schema_loader import get_amendable_fields
from app.services import adk_runner
from app.services.booking_flow import (
    handle_booking_amendment_turn,
    handle_booking_status_check,
    _extract_amendment_name_value,
    _extract_amendment_updates,
    _sanitize_message_for_amendment_extraction,
)


@pytest.fixture(autouse=True)
def stable_booking_reference_date(monkeypatch):
    monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")


def _receipt(**overrides):
    base = {
        "booking_id": "BK-20260607-C48A4AFA",
        "property_title": "Apartment 3",
        "guest_name": "Jane Doe",
        "guest_email": "jane@example.com",
        "guest_phone": "03001234567",
        "check_in": "2026-06-10",
        "check_out": "2026-06-25",
        "nights": 15,
        "guests": 2,
        "price_per_night": 103.0,
        "total_amount": 1545.0,
        "status": "confirmed",
    }
    base.update(overrides)
    return base


def _confirmed_soft_state(**receipt_overrides):
    receipt = _receipt(**receipt_overrides)
    return {
        "active_flow": "booking",
        "booking_stage": "confirmed",
        "booking_status": "confirmed",
        "booking_registration_id": receipt["booking_id"],
        "booking_receipt": receipt,
        "last_presented_view": "booking_receipt",
    }


@pytest.mark.asyncio
async def test_confirmed_booking_change_name_asks_only_for_name():
    soft_state = _confirmed_soft_state()

    payload = await handle_booking_amendment_turn(
        "I want to change my name in this booking",
        soft_state,
    )

    assert payload is not None
    assert payload["missing_fields"] == ["guest_name"]
    assert "full name" in payload["deterministic_reply"].lower()
    assert "email" not in payload["deterministic_reply"].lower()
    assert "phone" not in payload["deterministic_reply"].lower()
    assert soft_state["booking_receipt"]["guest_email"] == "jane@example.com"


@pytest.mark.asyncio
async def test_exact_screenshot_phrase_asks_only_for_name():
    soft_state = _confirmed_soft_state()

    payload = await handle_booking_amendment_turn(
        "I want make my name in this booking",
        soft_state,
    )

    assert payload is not None
    assert payload["missing_fields"] == ["guest_name"]
    assert "full name" in payload["deterministic_reply"].lower()
    assert "email" not in payload["deterministic_reply"].lower()
    assert "phone" not in payload["deterministic_reply"].lower()


@pytest.mark.asyncio
async def test_confirmed_booking_change_phone_inline_shows_updated_review_only():
    soft_state = _confirmed_soft_state()

    payload = await handle_booking_amendment_turn(
        "change my phone to 03009998877",
        soft_state,
    )

    assert payload is not None
    assert payload["status"] == "review_pending"
    assert payload["receipt"]["booking_id"] == "BK-20260607-C48A4AFA"
    assert payload["receipt"]["guest_phone"] == "03009998877"
    assert payload["receipt"]["guest_name"] == "Jane Doe"
    assert payload["receipt"]["guest_email"] == "jane@example.com"
    assert payload["deterministic_reply"].startswith("Please review your updated booking details:")
    assert "Please confirm if these updated booking details are correct." in payload["deterministic_reply"]
    assert "Your booking is confirmed" not in payload["deterministic_reply"]
    assert "What full name" not in payload["deterministic_reply"]
    assert "What email" not in payload["deterministic_reply"]


@pytest.mark.asyncio
async def test_confirmed_booking_change_checkin_and_guests_recomputes_total():
    soft_state = _confirmed_soft_state()

    payload = await handle_booking_amendment_turn(
        "change check-in to 20-06-2026 and guests to 3",
        soft_state,
    )

    assert payload is not None
    receipt = payload["receipt"]
    assert receipt["booking_id"] == "BK-20260607-C48A4AFA"
    assert receipt["check_in"] == "2026-06-20"
    assert receipt["check_out"] == "2026-06-25"
    assert receipt["guests"] == 3
    assert receipt["nights"] == 5
    assert receipt["total_amount"] == 515.0
    assert receipt["guest_name"] == "Jane Doe"


@pytest.mark.asyncio
async def test_confirmed_booking_change_email_and_phone_asks_only_those_fields():
    soft_state = _confirmed_soft_state()

    payload = await handle_booking_amendment_turn(
        "change my email and phone",
        soft_state,
    )

    assert payload is not None
    assert payload["missing_fields"] == ["guest_email", "guest_phone"]
    lower = payload["deterministic_reply"].lower()
    assert "email" in lower
    assert "phone" in lower
    assert "full name" not in lower
    assert "check-in" not in lower
    assert "guests" not in lower


@pytest.mark.asyncio
async def test_confirmed_booking_amend_name_confirm_persists_user_name():
    soft_state = _confirmed_soft_state()

    review = await handle_booking_amendment_turn(
        "change name to Mary Smith",
        soft_state,
    )
    assert review is not None
    assert review["receipt"]["guest_name"] == "Mary Smith"

    with patch(
        "app.observability.db_logging.update_successful_booking",
        new=AsyncMock(return_value=True),
    ) as update_successful_booking:
        confirmed = await handle_booking_amendment_turn("confirm", soft_state)

    assert confirmed is not None
    assert confirmed["receipt"]["booking_id"] == "BK-20260607-C48A4AFA"
    assert soft_state["booking_receipt"]["guest_name"] == "Mary Smith"
    update_successful_booking.assert_awaited_once()
    booking_id, updates = update_successful_booking.await_args.args
    assert booking_id == "BK-20260607-C48A4AFA"
    assert updates == {"user_name": "Mary Smith"}


@pytest.mark.asyncio
async def test_new_session_status_lookup_then_amend_phone_and_status_returns_updated_phone():
    booking_id = "BK-20260607-C48A4AFA"
    original_row = {
        "booking_id": booking_id,
        "status": "confirmed",
        "check_in": "2026-06-10",
        "check_out": "2026-06-25",
        "user_name": "Jane Doe",
        "user_email": "jane@example.com",
        "user_phone": "03001234567",
        "property_title": "Apartment 3",
        "guests": 2,
        "nights": 15,
        "price_per_night": 103.0,
        "total_amount": 1545.0,
    }
    soft_state = {}

    with (
        patch("app.services.booking.get_booking_status", AsyncMock(return_value={"ok": False})),
        patch("app.observability.db_logging.get_successful_booking_status", AsyncMock(return_value=original_row)),
    ):
        status = await handle_booking_status_check(f"My booking ID is {booking_id}", soft_state)

    assert status is not None
    assert soft_state["booking_receipt"]["guest_phone"] == "03001234567"

    await handle_booking_amendment_turn("change my phone to 03009998877", soft_state)
    with patch("app.observability.db_logging.update_successful_booking", new=AsyncMock(return_value=True)):
        await handle_booking_amendment_turn("confirm", soft_state)

    updated_row = dict(original_row)
    updated_row["user_phone"] = "03009998877"
    fresh_soft_state = {}
    with (
        patch("app.services.booking.get_booking_status", AsyncMock(return_value={"ok": False})),
        patch("app.observability.db_logging.get_successful_booking_status", AsyncMock(return_value=updated_row)),
    ):
        updated_status = await handle_booking_status_check(f"booking id {booking_id}", fresh_soft_state)

    assert updated_status is not None
    assert "03009998877" in updated_status["deterministic_reply"]
    assert fresh_soft_state["booking_receipt"]["booking_id"] == booking_id


@pytest.mark.asyncio
async def test_direct_amendment_by_booking_id_from_fresh_session_fetches_receipt():
    booking_id = "BK-20260607-C48A4AFA"
    db_row = {
        "booking_id": booking_id,
        "status": "confirmed",
        "check_in": "2026-06-10",
        "check_out": "2026-06-25",
        "user_name": "Jane Doe",
        "user_email": "jane@example.com",
        "user_phone": "03001234567",
        "property_title": "Apartment 3",
        "guests": 2,
        "nights": 15,
        "price_per_night": 103.0,
        "total_amount": 1545.0,
    }
    soft_state = {}

    with patch(
        "app.observability.db_logging.get_successful_booking_status",
        AsyncMock(return_value=db_row),
    ) as get_successful_booking_status:
        payload = await handle_booking_amendment_turn(
            f"change my phone for {booking_id}",
            soft_state,
        )

    get_successful_booking_status.assert_awaited_once_with(booking_id)
    assert payload is not None
    assert payload["missing_fields"] == ["guest_phone"]
    assert soft_state["booking_receipt"]["booking_id"] == booking_id
    assert soft_state["booking_receipt"]["guest_name"] == "Jane Doe"


@pytest.mark.asyncio
async def test_db_update_false_does_not_confirm_amendment_or_replace_receipt():
    soft_state = _confirmed_soft_state()
    original_receipt = copy.deepcopy(soft_state["booking_receipt"])

    review = await handle_booking_amendment_turn(
        "change name to Mary Smith",
        soft_state,
    )
    assert review is not None
    assert soft_state["booking_stage"] == "awaiting_amendment_confirmation"

    with patch(
        "app.observability.db_logging.update_successful_booking",
        new=AsyncMock(return_value=False),
    ):
        failed = await handle_booking_amendment_turn("confirm", soft_state)

    assert failed is not None
    assert failed["status"] == "error"
    assert "couldn't save" in failed["deterministic_reply"].lower()
    assert soft_state["booking_stage"] == "awaiting_amendment_confirmation"
    assert "pending_booking_amendment" in soft_state
    assert soft_state["booking_receipt"] == original_receipt


@pytest.mark.asyncio
async def test_invalid_amendment_checkout_before_checkin_preserves_unrelated_fields():
    soft_state = _confirmed_soft_state()
    before = copy.deepcopy(soft_state["booking_receipt"])

    payload = await handle_booking_amendment_turn(
        "change check-out to 01-06-2026",
        soft_state,
    )

    assert payload is not None
    assert payload["missing_fields"] == ["check_out"]
    assert "check-out" in payload["deterministic_reply"].lower()
    assert soft_state["booking_receipt"] == before
    assert soft_state["pending_booking_amendment"]["base_receipt"]["guest_name"] == "Jane Doe"


@pytest.mark.asyncio
async def test_singular_checkin_date_does_not_ask_for_checkout():
    soft_state = _confirmed_soft_state()

    payload = await handle_booking_amendment_turn(
        "change check-in date to 20-06-2026",
        soft_state,
    )

    assert payload is not None
    assert payload["status"] == "review_pending"
    assert payload["receipt"]["check_in"] == "2026-06-20"
    assert payload["receipt"]["check_out"] == "2026-06-25"
    assert payload["receipt"]["nights"] == 5
    assert payload["receipt"]["total_amount"] == 515.0
    assert "missing_fields" not in payload
    assert "And the check-out date?" not in payload["deterministic_reply"]


@pytest.mark.asyncio
async def test_unclear_this_booking_amendment_asks_choice_not_search():
    soft_state = _confirmed_soft_state()

    payload = await handle_booking_amendment_turn(
        "I want to change something in this booking",
        soft_state,
    )

    assert payload is not None
    assert "Which booking detail would you like to change?" in payload["deterministic_reply"]
    assert "supported city" not in payload["deterministic_reply"].lower()


def test_property_is_not_post_confirmation_amendable_until_reselection_is_complete():
    assert "property_id" not in get_amendable_fields()


def test_sanitize_message_for_amendment_extraction_strips_booking_id():
    booking_id = "BK-20260607-C48A4AFA"
    sanitized = _sanitize_message_for_amendment_extraction(f"change my phone for {booking_id}")
    assert booking_id not in sanitized
    assert sanitized == "change my phone for"


def test_extract_amendment_name_field_only_requests_value():
    for message in (
        "I want to change my name in this booking",
        "I want make my name in this booking",
    ):
        assert _extract_amendment_name_value(message) is None
        updates, errors = _extract_amendment_updates(message, ["guest_name"])
        assert updates == {}
        assert errors == {}


def test_extract_amendment_name_inline_value_is_captured():
    updates, errors = _extract_amendment_updates("change name to Mary Smith", ["guest_name"])
    assert updates == {"guest_name": "Mary Smith"}
    assert errors == {}


def test_extract_amendment_phone_ignores_booking_id_digits():
    booking_id = "BK-20260607-C48A4AFA"
    updates, errors = _extract_amendment_updates(
        f"change my phone for {booking_id}",
        ["guest_phone"],
    )
    assert updates == {}
    assert errors == {}


def test_extract_amendment_phone_inline_value_is_captured():
    updates, errors = _extract_amendment_updates(
        "change my phone to 03009998877",
        ["guest_phone"],
    )
    assert updates == {"guest_phone": "03009998877"}
    assert errors == {}


def test_extract_amendment_compact_date_and_guests_update():
    updates, errors = _extract_amendment_updates(
        "change check-in to 20-06-2026 and guests to 3",
        ["check_in", "guests"],
    )
    assert errors == {}
    assert updates["check_in"] == "2026-06-20"
    assert updates["guests"] == 3


@pytest.mark.asyncio
async def test_this_booking_with_context_is_not_routed_to_property_search(monkeypatch):
    snapshot = {
        "state": {"soft_state": _confirmed_soft_state()},
        "history": [],
        "meta": {
            "app_name": adk_runner.APP_NAME,
            "user_id": "u-amendment",
            "last_update_time": 1.0,
        },
    }

    async def fake_get_session_snapshot(_session_id):
        return snapshot

    async def fake_save_session_snapshot(*, session_id, history, state, metadata):
        snapshot["state"] = copy.deepcopy(state)
        snapshot["history"] = copy.deepcopy(history)
        snapshot["meta"].update(copy.deepcopy(metadata or {}))

    route_pre_adk = AsyncMock(return_value={"reply": "unsupported city fallback"})
    direct_search = AsyncMock(return_value={"deterministic_reply": "I couldn't confidently match 'this booking' to a supported city."})

    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)
    monkeypatch.setattr(adk_runner, "save_session_snapshot", fake_save_session_snapshot)
    monkeypatch.setattr(adk_runner, "maybe_handle_direct_property_search", direct_search)
    monkeypatch.setattr(adk_runner, "route_pre_adk", route_pre_adk)
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda msg: (msg, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda msg: msg)

    chunks = []
    async for chunk in adk_runner.run_adk_turn(
        "u-amendment",
        "s-amendment",
        "I want to change something in this booking",
    ):
        chunks.append(chunk)

    reply = "".join(chunks)
    assert "Which booking detail would you like to change?" in reply
    assert "supported city" not in reply.lower()
    direct_search.assert_not_awaited()
    route_pre_adk.assert_not_awaited()
