from __future__ import annotations

import pytest

from app.services.property_query_constraints import extract_property_search_query


def _find_constraint(query, field: str):
    return next((item for item in query.constraints if item.field == field), None)


def test_extracts_exact_bedrooms_from_classified_query():
    query = extract_property_search_query("show me some 2 bedrooms apartment in new york city")

    assert query.city == "New York"
    assert query.property_type == "Apartment"
    bedrooms = _find_constraint(query, "bedrooms")
    assert bedrooms is not None
    assert bedrooms.operator == "exact"
    assert bedrooms.value == 2


@pytest.mark.parametrize(
    "message",
    [
        "2 bedroom apartment in New York",
        "2 bedrooms apartment in New York",
        "2 bed apartment in New York",
        "2 beds apartment in New York",
        "2br apartment in New York",
        "2 br apartment in New York",
    ],
)
def test_extracts_bedroom_variants(message: str):
    query = extract_property_search_query(message)

    bedrooms = _find_constraint(query, "bedrooms")
    assert bedrooms is not None
    assert bedrooms.operator == "exact"
    assert bedrooms.value == 2


def test_extracts_min_bedrooms():
    query = extract_property_search_query("at least 3 bedrooms apartment in Seattle")

    bedrooms = _find_constraint(query, "bedrooms")
    assert bedrooms is not None
    assert bedrooms.operator == "min"
    assert bedrooms.value == 3


def test_extracts_bathrooms():
    query = extract_property_search_query("2 bedroom 2 bathroom apartment in New York")

    bedrooms = _find_constraint(query, "bedrooms")
    bathrooms = _find_constraint(query, "bathrooms")
    assert bedrooms is not None
    assert bedrooms.operator == "exact"
    assert bedrooms.value == 2
    assert bathrooms is not None
    assert bathrooms.operator == "exact"
    assert bathrooms.value == 2


def test_extracts_guest_capacity():
    query = extract_property_search_query("apartment in Seattle for 4 guests")

    guests = _find_constraint(query, "occupancy_max")
    assert guests is not None
    assert guests.operator == "min"
    assert guests.value == 4


def test_extracts_price_max():
    query = extract_property_search_query("2 bedroom apartment in New York under 300")

    price = _find_constraint(query, "price_per_night")
    assert price is not None
    assert price.operator == "max"
    assert price.value == 300


def test_extracts_amenity_pet_friendly():
    query = extract_property_search_query("pet friendly apartment in Seattle")

    amenities = _find_constraint(query, "amenities")
    assert amenities is not None
    assert amenities.operator == "contains"
    assert amenities.value == "pet_friendly"
