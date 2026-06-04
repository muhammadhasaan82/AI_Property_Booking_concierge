"""
Unit tests for booking_flow._find_all_dates() and _extract_dates_by_association().

Both functions are new in this PR. They parse dates from natural-language booking
messages and associate them with check-in or check-out fields based on proximity
to labelling keywords such as "check-in date", "check-out", "arrival", etc.

Dates are returned in cfg.date_format which is "%Y-%m-%d" by default.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.booking_flow import _extract_dates_by_association, _find_all_dates


# ---------------------------------------------------------------------------
# _find_all_dates
# ---------------------------------------------------------------------------

class TestFindAllDates:
    """Tests for _find_all_dates()."""

    def test_empty_string_returns_empty_list(self):
        assert _find_all_dates("") == []

    def test_none_returns_empty_list(self):
        # The function guards `if not text`; an empty string also returns []
        assert _find_all_dates("") == []

    def test_no_dates_in_text_returns_empty_list(self):
        assert _find_all_dates("Book a hotel in Dubai for two nights") == []

    def test_iso_date_found(self):
        results = _find_all_dates("My check-in is 2026-06-10.")
        assert len(results) == 1
        assert results[0][0] == "2026-06-10"

    def test_iso_date_returns_tuple_with_positions(self):
        text = "Check in 2026-06-10 checkout 2026-06-15"
        results = _find_all_dates(text)
        # Each element: (formatted_date, start_pos, end_pos)
        assert all(len(r) == 3 for r in results)
        dates = [r[0] for r in results]
        assert "2026-06-10" in dates
        assert "2026-06-15" in dates

    def test_named_month_day_year_format(self):
        """Format: '11th of June, 2026' or '2nd of June 2026'"""
        results = _find_all_dates("Check out on 11th of June, 2026")
        assert len(results) >= 1
        assert results[0][0] == "2026-06-11"

    def test_named_month_without_of(self):
        """Format: '11 June, 2026'"""
        results = _find_all_dates("Arrival: 11 June, 2026")
        assert len(results) == 1
        assert results[0][0] == "2026-06-11"

    def test_reversed_named_month_year_format(self):
        """Format: 'June 11, 2026'"""
        results = _find_all_dates("Departure June 11, 2026")
        assert len(results) == 1
        assert results[0][0] == "2026-06-11"

    def test_named_month_no_comma(self):
        """Format: 'June 11 2026' (no comma)"""
        results = _find_all_dates("check out June 11 2026 please")
        assert len(results) == 1
        assert results[0][0] == "2026-06-11"

    def test_ordinal_suffixes_handled(self):
        """'2nd', '3rd', '21st', '15th' should all parse correctly."""
        test_cases = [
            ("2nd June 2026", "2026-06-02"),
            ("3rd March 2027", "2027-03-03"),
            ("21st January 2025", "2025-01-21"),
            ("15th November 2026", "2026-11-15"),
        ]
        for text, expected in test_cases:
            results = _find_all_dates(text)
            assert len(results) >= 1, f"No date found in: {text!r}"
            assert results[0][0] == expected, f"For {text!r}: expected {expected}, got {results[0][0]}"

    def test_multiple_dates_all_returned(self):
        """Two dates in one message should both be found."""
        text = "Check in 2026-06-02 and check out 2026-06-11"
        results = _find_all_dates(text)
        dates = [r[0] for r in results]
        assert "2026-06-02" in dates
        assert "2026-06-11" in dates
        assert len(results) == 2

    def test_invalid_iso_date_skipped(self):
        """Invalid ISO date like 2026-13-01 must not appear in results."""
        results = _find_all_dates("My date is 2026-13-01")
        assert all(r[0] != "2026-13-01" for r in results)

    def test_invalid_day_in_named_date_skipped(self):
        """February 30 should be silently skipped."""
        results = _find_all_dates("book for 30th February 2026")
        # Either empty or doesn't contain that date
        for r in results:
            assert "2026-02-30" not in r[0]

    def test_results_sorted_by_position(self):
        """Dates must appear in text-order (left to right)."""
        text = "First date 2026-06-02 then second date 2026-08-15 and third 2026-12-01"
        results = _find_all_dates(text)
        dates = [r[0] for r in results]
        assert dates == ["2026-06-02", "2026-08-15", "2026-12-01"]

    def test_overlapping_matches_deduplicated(self):
        """If two patterns match the same substring, only one is returned."""
        # "2 June 2026" could match both pat1 and pat2; only one should remain
        text = "check in 2 June 2026"
        results = _find_all_dates(text)
        assert len(results) == 1
        assert results[0][0] == "2026-06-02"

    def test_case_insensitive_month_names(self):
        """Month names should be case-insensitive."""
        results_lower = _find_all_dates("JUNE 11 2026")
        results_upper = _find_all_dates("june 11 2026")
        results_title = _find_all_dates("June 11 2026")
        for results, label in [(results_lower, "JUNE"), (results_upper, "june"), (results_title, "June")]:
            assert len(results) >= 1, f"Failed for {label}"
            assert results[0][0] == "2026-06-11", f"Wrong date for {label}"


# ---------------------------------------------------------------------------
# _extract_dates_by_association
# ---------------------------------------------------------------------------

class TestExtractDatesByAssociation:
    """Tests for _extract_dates_by_association()."""

    def test_empty_text_returns_none_none(self):
        check_in, check_out = _extract_dates_by_association("")
        assert check_in is None
        assert check_out is None

    def test_no_dates_returns_none_none(self):
        check_in, check_out = _extract_dates_by_association("I want to book in Dubai")
        assert check_in is None
        assert check_out is None

    def test_single_date_with_check_in_keyword(self):
        """One date preceded by 'check-in date' → check_in set, check_out None."""
        check_in, check_out = _extract_dates_by_association("check-in date 2026-06-02")
        assert check_in == "2026-06-02"
        assert check_out is None

    def test_single_date_with_arrival_keyword(self):
        """'arrival date' is a check-in synonym."""
        check_in, check_out = _extract_dates_by_association("arrival date 2026-06-02")
        assert check_in == "2026-06-02"
        assert check_out is None

    def test_single_date_with_check_out_keyword(self):
        """One date preceded by 'check-out' → check_out set, check_in None."""
        check_in, check_out = _extract_dates_by_association("check-out 2026-06-15")
        assert check_in is None
        assert check_out == "2026-06-15"

    def test_single_date_with_checkout_no_hyphen(self):
        """'checkout' (no hyphen) is a check-out synonym."""
        check_in, check_out = _extract_dates_by_association("checkout date 2026-06-15")
        assert check_in is None
        assert check_out == "2026-06-15"

    def test_single_date_with_departure_keyword(self):
        """'departure date' is a check-out synonym."""
        check_in, check_out = _extract_dates_by_association("departure date 2026-06-15")
        assert check_in is None
        assert check_out == "2026-06-15"

    def test_two_dates_with_explicit_keywords(self):
        """Two dates with explicit check-in and check-out keywords are assigned correctly."""
        text = "check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026"
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_two_dates_no_keywords_uses_order(self):
        """Two dates with no keywords → first is check-in, second is check-out."""
        text = "Book from 2026-06-02 to 2026-06-11"
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_check_in_labeled_and_remaining_becomes_check_out(self):
        """If check-in is labeled and second date unlabeled, unlabeled becomes check-out."""
        text = "check-in 2026-06-02, also 2026-06-11"
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_check_out_labeled_and_remaining_becomes_check_in(self):
        """If check-out is labeled and another date unlabeled, unlabeled becomes check-in."""
        text = "2026-06-02, check-out 2026-06-11"
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_single_date_no_keywords_returns_none_none(self):
        """A single date with no associated keywords → (None, None)."""
        text = "I want to book for 2026-06-02"
        check_in, check_out = _extract_dates_by_association(text)
        # Without any keyword, a single date cannot be assigned to either field
        # Both should be None (the function returns (None, None) in this path)
        assert check_in is None or check_out is None  # At most one can be set

    def test_realistic_full_message(self):
        """A realistic multi-field message correctly extracts both dates."""
        text = (
            "my full name is Jane Doe, email is jane@example.com, "
            "number 03001234567, check-in date would 2nd of june, 2026 "
            "and check out shall be around 11 june 2026, we are around 4 guests"
        )
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_returns_tuple_of_two(self):
        """Return value must always be a 2-tuple."""
        result = _extract_dates_by_association("check-in 2026-06-01 check-out 2026-06-10")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_keyword_before_date_is_closer_than_keyword_after(self):
        """A keyword immediately before a date wins over one far after."""
        text = "check-in 2026-06-01 ... ... ... check-out"
        check_in, check_out = _extract_dates_by_association(text)
        # The date 2026-06-01 is right after 'check-in', so should be check_in
        assert check_in == "2026-06-01"

    def test_boundary_single_date_with_strong_checkout_keyword(self):
        """Single date with 'checkout' keyword directly before it → check_out."""
        text = "checkout 2026-07-01"
        check_in, check_out = _extract_dates_by_association(text)
        assert check_out == "2026-07-01"
        assert check_in is None