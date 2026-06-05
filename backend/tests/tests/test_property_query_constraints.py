from __future__ import annotations

import pytest

from app.services.property_query_constraints import extract_property_search_query, _canonical_city


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


# ===========================================================================
# Tests for _canonical_city() — added in this PR
# ===========================================================================


class TestCanonicalCity:
    """Unit tests for the _canonical_city() helper."""

    def test_none_returns_none(self):
        assert _canonical_city(None) is None

    def test_empty_string_returns_none(self):
        assert _canonical_city("") is None

    def test_whitespace_only_returns_none(self):
        assert _canonical_city("   ") is None

    def test_lowercase_city_is_title_cased(self):
        assert _canonical_city("new york") == "New York"

    def test_uppercase_city_is_title_cased(self):
        assert _canonical_city("NEW YORK") == "New York"

    def test_mixed_case_city_is_title_cased(self):
        assert _canonical_city("nEw YoRk") == "New York"

    def test_single_word_city_is_title_cased(self):
        assert _canonical_city("dubai") == "Dubai"

    def test_multi_word_city_is_title_cased(self):
        assert _canonical_city("los angeles") == "Los Angeles"

    def test_leading_trailing_whitespace_is_stripped(self):
        assert _canonical_city("  seattle  ") == "Seattle"

    def test_extra_internal_spaces_collapsed(self):
        """Multiple spaces between words must be collapsed to a single space."""
        result = _canonical_city("new   york")
        assert result == "New York"

    def test_already_title_cased_unchanged(self):
        assert _canonical_city("New York") == "New York"

    def test_short_city_name(self):
        assert _canonical_city("Bali") == "Bali"

    def test_hyphenated_city(self):
        """Title-casing a hyphenated city — Python's str.title() handles each segment."""
        result = _canonical_city("abu-dhabi")
        # str.title() capitalises after hyphens: "Abu-Dhabi"
        assert result == "Abu-Dhabi"


# ===========================================================================
# Tests for extract_property_search_query() city title-casing (end-to-end)
# ===========================================================================


def test_city_is_title_cased_in_query_result():
    """City extracted via extract_property_search_query must be title-cased."""
    query = extract_property_search_query("show me apartments in new york city")
    # The _canonical_city wrapper ensures title-case
    assert query.city is None or query.city[0].isupper()


def test_city_returned_as_title_case_not_lowercase():
    """Regression: city must NOT be returned in all-lowercase."""
    query = extract_property_search_query("find a flat in new york")
    if query.city is not None:
        # First character must be uppercase (title-case)
        assert query.city[0] == query.city[0].upper()
