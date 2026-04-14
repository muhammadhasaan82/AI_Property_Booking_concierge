import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.status_codes import Status
from app.agents.tools import booking


def _tool_context_with_property(property_id: str):
    return SimpleNamespace(state={"soft_state": {"last_selected_property_id": property_id}})


def _base_booking_args():
    return {
        "property_title": "Test Property",
        "guest_name": "Jane Doe",
        "guest_email": "jane@example.com",
        "guest_phone": "5551234567",
        "check_in": "2026-05-01",
        "check_out": "2026-05-03",
        "guests": 2,
        "price_per_night": 120.0,
    }


@pytest.mark.parametrize("property_id_value", [None, ""])
def test_review_booking_details_uses_soft_state_property_id(property_id_value):
    tool_context = _tool_context_with_property("prop-123")
    args = _base_booking_args()

    result = asyncio.run(
        booking.review_booking_details(
            property_id=property_id_value,
            tool_context=tool_context,
            **args,
        )
    )

    assert result["status"] != Status.MISSING_CRITICAL_DATA
    assert result["summary"]["property_id"] == "prop-123"


@pytest.mark.parametrize("property_id_value", [None, ""])
def test_process_v2_booking_uses_soft_state_property_id(property_id_value):
    tool_context = _tool_context_with_property("prop-123")
    args = _base_booking_args()

    with patch("app.observability.db_logging.insert_successful_booking", new=AsyncMock()):
        result = asyncio.run(
            booking.process_v2_booking(
                property_id=property_id_value,
                tool_context=tool_context,
                **args,
            )
        )

    assert result["status"] == Status.BOOKING_CONFIRMED
