"""Acceptance tests generated from specs/search_workflow.md and specs/acceptance_tests.md."""
from __future__ import annotations

from unittest.mock import patch

import pytest

import app.agents.tools.search as search_tool_module
from app.agents.tools.search import paginate_stored_results
from app.services.direct_property_search import _extract_soft_ranking_terms
from app.services.property_query_constraints import extract_property_search_query


def _find_constraint(query, field: str):
    return next((item for item in query.constraints if item.field == field), None)


def _property(
    *,
    property_id: str,
    title: str,
    city: str,
    property_type: str,
    bedrooms: int,
    bathrooms: int = 1,
    price: int = 150,
    amenities: list[str] | None = None,
    occupancy_max: int = 2,
    rating: float = 4.5,
    reviews_count: int = 10,
    description: str = "",
):
    return {
        "id": property_id,
        "title": title,
        "city": city,
        "price_per_night": price,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "rating": rating,
        "reviews_count": reviews_count,
        "amenities": amenities or ["wifi"],
        "occupancy_max": occupancy_max,
        "description": description or f"{title} in {city}",
    }


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("apartment in Seattle under $300", 300),
        ("villa in Seattle budget $250", 250),
        ("condo in Seattle less than $400", 400),
    ],
)
def test_budget_phrases_with_dollar_sign_normalize_to_price_max(message: str, expected: int):
    query = extract_property_search_query(message)

    price = _find_constraint(query, "price_per_night")
    assert price is not None
    assert price.operator == "max"
    assert price.value == expected


def test_guest_capacity_phrase_maps_to_minimum_occupancy():
    query = extract_property_search_query("apartment in Seattle for 4 guests")

    guests = _find_constraint(query, "occupancy_max")
    assert guests is not None
    assert guests.operator == "min"
    assert guests.value == 4


def test_comma_and_semicolon_amenity_inputs_normalize_identically():
    comma_terms = search_tool_module._split_amenity_input("wifi, parking")
    semicolon_terms = search_tool_module._split_amenity_input("wifi; parking")

    assert comma_terms == ["wifi", "parking"]
    assert semicolon_terms == comma_terms


def test_unknown_vibe_amenity_becomes_soft_term_not_hard_filter():
    soft_terms = _extract_soft_ranking_terms(
        "villa in Seattle with wifi, parking, cozy",
        ["wifi", "parking"],
    )

    query = extract_property_search_query("villa in Seattle with wifi, parking, cozy")
    amenity_values = [constraint.value for constraint in query.constraints if constraint.field == "amenities"]

    assert set(amenity_values) == {"wifi", "parking"}
    assert soft_terms == "cozy"


def test_pagination_state_keys_remain_consistent(monkeypatch):
    all_results = [
        _property(
            property_id=f"prop-{idx}",
            title=f"Property {idx}",
            city="Seattle",
            property_type="Villa",
            bedrooms=2,
            amenities=["wifi", "parking"],
            occupancy_max=4,
        )
        for idx in range(1, 6)
    ]
    soft_state = {
        "all_search_results": list(all_results),
        "visible_results": all_results[:2],
        "last_filters": {"city": "Seattle", "property_type": "villa"},
        "current_page": 1,
        "page_size": 2,
        "last_search": {
            "pagination": {
                "current_page": 1,
                "page_size": 2,
                "page_start": 1,
                "page_end": 2,
                "total_pages": 3,
                "has_more": True,
                "has_next": True,
                "has_prev": False,
                "pagination_enabled": True,
            }
        },
    }

    monkeypatch.setattr(search_tool_module, "_search_display_pagination_enabled", lambda: True)
    monkeypatch.setattr(search_tool_module, "_search_display_mode", lambda: "paginated")
    monkeypatch.setattr(search_tool_module, "_search_display_max_inline_results", lambda: None)

    payload = paginate_stored_results(soft_state, direction="next")

    assert payload is not None
    assert soft_state["current_page"] == 2
    assert soft_state["page_size"] == 2
    assert payload["shown_count"] == len(payload["properties"]) == len(soft_state["visible_results"]) == len(soft_state["option_map"])
    assert soft_state["active_property_options_shown_count"] == payload["shown_count"]
    assert soft_state["active_property_options_total_found"] == payload["total_found"]
    assert soft_state["active_property_options_map"] == soft_state["option_map"]
