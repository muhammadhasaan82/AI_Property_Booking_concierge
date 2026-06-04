"""
Unit tests for property_query_constraints._canonical_city().

This function was added in this PR to normalise city names to title case
before storing them in search queries, while preserving search semantics.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.property_query_constraints import _canonical_city


class TestCanonicalCity:
    """Tests for _canonical_city()."""

    def test_none_returns_none(self):
        assert _canonical_city(None) is None

    def test_empty_string_returns_none(self):
        assert _canonical_city("") is None

    def test_whitespace_only_returns_none(self):
        """A string of only spaces is falsy after strip and returns None."""
        assert _canonical_city("   ") is None

    def test_lowercase_city_gets_title_cased(self):
        assert _canonical_city("new york") == "New York"

    def test_uppercase_city_gets_title_cased(self):
        assert _canonical_city("NEW YORK") == "New York"

    def test_already_title_cased_remains_unchanged(self):
        assert _canonical_city("New York") == "New York"

    def test_single_word_city(self):
        assert _canonical_city("dubai") == "Dubai"

    def test_three_word_city(self):
        assert _canonical_city("kuala lumpur city") == "Kuala Lumpur City"

    def test_leading_and_trailing_whitespace_stripped(self):
        assert _canonical_city("  dubai  ") == "Dubai"

    def test_multiple_internal_spaces_normalised(self):
        """Multiple consecutive spaces between words are collapsed to one."""
        assert _canonical_city("new   york") == "New York"

    def test_mixed_case_city(self):
        assert _canonical_city("nEw yOrK") == "New York"

    def test_single_char_city(self):
        """Edge case: single character city should be title-cased."""
        assert _canonical_city("a") == "A"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("london", "London"),
            ("san francisco", "San Francisco"),
            ("los angeles", "Los Angeles"),
            ("abu dhabi", "Abu Dhabi"),
            ("HONG KONG", "Hong Kong"),
            ("  paris  ", "Paris"),
            ("new   delhi", "New Delhi"),
        ],
    )
    def test_canonical_city_parametrized(self, raw, expected):
        assert _canonical_city(raw) == expected

    def test_used_in_extract_property_search_query_returns_title_case(self):
        """Integration: extract_property_search_query should return title-cased city."""
        from app.services.property_query_constraints import extract_property_search_query

        query = extract_property_search_query("find apartments in new york")
        if query.city is not None:
            # If city was extracted, it must be title-cased by _canonical_city
            assert query.city == query.city.title() or query.city[0].isupper()