"""
E2E booking to receipt flow tests.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.status_codes import Status
from app.agents.tools import booking
from app.agents.tools.search import search_properties, select_property
from app.services.property_type_normalizer import normalize_property_type


class _Ctx(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(state={"soft_state": {}})


def _seed_dataset() -> list[dict]:
    return [
        {
            "id": "ny_ap_1",
            "city": "New York",
            "property_type": "Apartment",
            "price_per_night": 140.0,
            "bedrooms": 2,
            "bathrooms": 1,
            "rating": 4.7,
            "amenities": ["wifi"],
            "title": "Hudson Loft",
            "description": "Bright loft near the park.",
        },
        {
            "id": "ny_ap_2",
            "city": "New York",
            "property_type": "Apartment",
            "price_per_night": 160.0,
            "bedrooms": 1,
            "bathrooms": 1,
            "rating": 4.4,
            "amenities": ["wifi", "gym"],
            "title": "Midtown Flat",
            "description": "Walkable to transit.",
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
    ]


@pytest.mark.asyncio
async def test_booking_to_receipt_flow_preserves_context():
    ctx = _Ctx()
    dataset = _seed_dataset()

    with patch("app.components.search._DATASET", dataset), \
         patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}):
        search_result = await search_properties(
            city="New York",
            property_type="apartment",
            tool_context=ctx,
        )

        assert search_result["status"] == Status.PROPERTIES_FOUND
        expected_titles = {
            row["title"]
            for row in dataset
            if row["city"].lower() == "new york"
            and normalize_property_type(row["property_type"]) == "apartment"
        }
        returned_titles = {prop["title"] for prop in search_result["properties"]}
        assert returned_titles <= expected_titles
        assert search_result["total_found"] == len(expected_titles)

        selected = await select_property(option_number=1, tool_context=ctx)
        assert selected["status"] == Status.PROPERTY_DETAILS
        selected_id = selected["property"]["id"]
        assert selected_id == search_result["properties"][0]["id"]
        assert ctx.state["soft_state"]["last_selected_property_id"] == selected_id

        missing_fields = ["guest_email", "check_in", "check_out", "guests", "price_per_night"]
        gather = await booking.request_booking_details(
            missing_fields=missing_fields,
            tool_context=ctx,
        )
        assert gather["status"] == Status.GATHERING_INFO
        assert gather["missing_fields"][0] in missing_fields
        assert ctx.state["soft_state"].get("awaiting_field") in missing_fields

        review = await booking.request_booking_details(
            property_title=selected["property"]["title"],
            guest_name="Jane Doe",
            guest_email="jane@example.com",
            guest_phone="5551234567",
            check_in="2026-05-01",
            check_out="2026-05-03",
            guests=2,
            price_per_night=150.0,
            tool_context=ctx,
        )
        assert review["status"] == Status.REVIEW_PENDING
        summary = review["summary"]
        assert summary["property_id"] == selected_id
        assert summary["property"] == selected["property"]["title"]
        assert summary["check_in"] == "2026-05-01"
        assert summary["check_out"] == "2026-05-03"
        assert summary["guests"] == 2

        with patch("app.observability.db_logging.insert_successful_booking", new=AsyncMock()):
            receipt = await booking.process_v2_booking(
                property_title=selected["property"]["title"],
                guest_name="Jane Doe",
                guest_email="jane@example.com",
                guest_phone="5551234567",
                check_in="2026-05-01",
                check_out="2026-05-03",
                guests=2,
                price_per_night=150.0,
                tool_context=ctx,
            )

        assert receipt["status"] == Status.BOOKING_CONFIRMED
        payload = receipt["receipt"]
        assert payload["property_title"] == selected["property"]["title"]
        assert payload["guest_email"] == "jane@example.com"
        assert payload["check_in"] == "2026-05-01"
        assert payload["check_out"] == "2026-05-03"
        assert payload["guests"] == 2
        assert payload.get("booking_id")


@pytest.mark.asyncio
async def test_correction_flow_updates_pending_booking():
    ctx = _Ctx()
    ctx.state["soft_state"]["pending_booking"] = {
        "property": "Hudson Loft",
        "property_id": "ny_ap_1",
        "guest_name": "Jane Doe",
        "guest_email": "jane@example.com",
        "guest_phone": "5551234567",
        "check_in": "2026-05-01",
        "check_out": "2026-05-03",
        "guests": 2,
        "price_per_night": 150.0,
        "total": 300.0,
    }

    amended = await booking.request_booking_details(
        check_in="2026-05-02",
        tool_context=ctx,
    )

    assert amended["status"] == Status.AMENDMENT_ACKNOWLEDGED
    assert "check_in" in amended["updated_fields"]
    assert ctx.state["soft_state"]["pending_booking"]["check_in"] == "2026-05-02"
    assert "booking_id" not in amended.get("current_state", {})

    pending = ctx.state["soft_state"]["pending_booking"]
    review = await booking.review_booking_details(
        property_id=pending.get("property_id"),
        property_title=pending.get("property") or pending.get("property_title"),
        guest_name=pending.get("guest_name"),
        guest_email=pending.get("guest_email"),
        guest_phone=pending.get("guest_phone"),
        check_in=pending.get("check_in"),
        check_out=pending.get("check_out"),
        guests=pending.get("guests"),
        price_per_night=pending.get("price_per_night"),
        tool_context=ctx,
    )

    assert review["status"] == Status.REVIEW_PENDING
    assert review["summary"]["check_in"] == "2026-05-02"
