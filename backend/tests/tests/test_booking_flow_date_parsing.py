"""
Unit tests for the date-parsing helpers added to booking_flow.py in this PR:
  - _find_all_dates(text)          — extract all date spans from free text
  - _extract_dates_by_association(text) — map dates to check_in / check_out labels

Scope: purely the two new private helpers; integration with _extract_updates_from_message
is exercised by test_booking_details_flow.py.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.booking_flow import _find_all_dates, _extract_dates_by_association


def _dates(text: str) -> List[str]:
    """Return just the date strings from _find_all_dates for easy comparison."""
    return [d for d, _, _ in _find_all_dates(text)]


class TestFindAllDates:

    def test_empty_string_returns_empty_list(self):
        assert _find_all_dates("") == []

    def test_none_like_empty_text_returns_empty_list(self):
        assert _find_all_dates("   ") == []

    def test_text_without_dates_returns_empty_list(self):
        assert _find_all_dates("I would like to book a nice apartment please") == []

    def test_iso_date_detected(self):
        result = _dates("check in on 2026-06-15")
        assert "2026-06-15" in result

    def test_iso_date_invalid_month_skipped(self):
        """Month 13 is invalid; the date must be silently skipped."""
        result = _dates("date is 2026-13-01")
        assert result == []

    def test_iso_date_invalid_day_skipped(self):
        """Day 32 is invalid; the date must be silently skipped."""
        result = _dates("date is 2026-02-30")
        assert result == []

    def test_day_of_month_year_format(self):
        result = _dates("check in on 2nd of June, 2026")
        assert "2026-06-02" in result

    def test_ordinal_suffix_st_detected(self):
        result = _dates("arriving 1st January 2026")
        assert "2026-01-01" in result

    def test_ordinal_suffix_nd_detected(self):
        result = _dates("departing 22nd March 2027")
        assert "2027-03-22" in result

    def test_ordinal_suffix_rd_detected(self):
        result = _dates("leaving 3rd July 2026")
        assert "2026-07-03" in result

    def test_ordinal_suffix_th_detected(self):
        result = _dates("arrive on 15th August 2026")
        assert "2026-08-15" in result

    def test_month_day_year_format(self):
        result = _dates("checkout June 11, 2026")
        assert "2026-06-11" in result

    def test_month_day_year_without_comma(self):
        result = _dates("arrive June 2 2026")
        assert "2026-06-02" in result

    def test_day_month_without_year(self, monkeypatch):
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")
        result = _dates("check-out date 13 july and check in date is 23 june")
        assert "2026-06-23" in result
        assert "2026-07-13" in result

    def test_month_day_without_year(self, monkeypatch):
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")
        result = _dates("arrive july 13 and leave august 2")
        assert "2026-07-13" in result
        assert "2026-08-02" in result

    def test_two_iso_dates_in_text(self):
        result = _dates("check in 2026-06-02, check out 2026-06-11")
        assert "2026-06-02" in result
        assert "2026-06-11" in result
        assert len(result) == 2

    def test_mixed_format_two_dates(self):
        result = _dates("check-in 2nd of June 2026, checkout June 11, 2026")
        assert "2026-06-02" in result
        assert "2026-06-11" in result

    def test_order_preserved_by_position(self):
        """Dates must be returned in document order (left to right)."""
        result = _dates("first 2026-06-02 then 2026-06-11 and finally 2026-12-25")
        assert result[0] == "2026-06-02"
        assert result[1] == "2026-06-11"
        assert result[2] == "2026-12-25"

    def test_overlapping_matches_deduped(self):
        """When ISO and natural-language patterns match the same span, only one
        entry should appear (last_end filter prevents double-counting)."""

        result = _dates("arrive 2026-06-02 please")
        assert result.count("2026-06-02") == 1

    def test_returns_tuple_with_position_info(self):
        text = "check in 2026-06-02 checkout 2026-06-11"
        raw = _find_all_dates(text)
        assert len(raw) == 2
        for item in raw:
            assert len(item) == 3
            date_str, start, end = item
            assert isinstance(date_str, str)
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start < end


class TestExtractDatesByAssociation:

    def test_empty_text_returns_none_none(self):
        assert _extract_dates_by_association("") == (None, None)

    def test_text_without_dates_returns_none_none(self):
        assert _extract_dates_by_association("I want to book a hotel") == (None, None)

    def test_single_date_with_checkin_label_assigns_check_in(self):
        check_in, check_out = _extract_dates_by_association(
            "check-in date would be 2nd of June, 2026"
        )
        assert check_in == "2026-06-02"
        assert check_out is None

    def test_single_date_with_arrival_label_assigns_check_in(self):
        check_in, check_out = _extract_dates_by_association(
            "arrival date is June 2nd, 2026"
        )
        assert check_in == "2026-06-02"
        assert check_out is None

    def test_single_date_with_checkout_label_assigns_check_out(self):
        check_in, check_out = _extract_dates_by_association(
            "check out shall be 11 june 2026"
        )
        assert check_out == "2026-06-11"
        assert check_in is None

    def test_single_date_with_departure_label_assigns_check_out(self):
        check_in, check_out = _extract_dates_by_association(
            "departure date is 2026-06-15"
        )
        assert check_out == "2026-06-15"
        assert check_in is None

    def test_two_dates_with_both_labels_assigned_correctly(self):
        text = (
            "check-in date would 2nd of june, 2026 "
            "and check out shall be around 11 june 2026"
        )
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_iso_dates_with_both_labels(self):
        text = "check-in 2026-06-02 check-out 2026-06-11"
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_two_dates_no_labels_use_positional_order(self):
        """Without any check-in/check-out labels the first date becomes check_in
        and the second becomes check_out."""
        check_in, check_out = _extract_dates_by_association(
            "I want to stay from 2026-06-02 to 2026-06-11"
        )
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_only_checkout_label_with_two_dates_fills_check_in_from_remainder(self):
        """When only check-out is labelled, the other date should become check_in."""
        text = "2026-06-02 and then check out on 2026-06-11"
        check_in, check_out = _extract_dates_by_association(text)
        assert check_out == "2026-06-11"
        assert check_in == "2026-06-02"

    def test_only_checkin_label_with_two_dates_fills_check_out_from_remainder(self):
        """When only check-in is labelled, the other date should become check_out."""
        text = "check-in 2026-06-02 and then I'll leave 2026-06-11"
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_realistic_sentence_all_fields(self):
        """Mirrors the message used in test_booking_details_flow: both labels present."""
        text = (
            "my full name is Jane Doe, email is jane@example.com, number 03001234567, "
            "check-in date would 2nd of june, 2026 and check out shall be around 11 june 2026, "
            "we are around 4 guests"
        )
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11"

    def test_realistic_sentence_invalid_checkout_year(self):
        """A check-out date with the wrong year (2025) is still parsed; validation
        is done elsewhere — _extract_dates_by_association just classifies."""
        text = (
            "check-in date would 2nd of june, 2026 and check out shall be around 11 june 2025"
        )
        check_in, check_out = _extract_dates_by_association(text)
        assert check_in == "2026-06-02"
        assert check_out == "2026-06-11" or check_out == "2025-06-11"

    @pytest.mark.parametrize("label", [
        "check-out",
        "check out",
        "checkout",
        "departure",
    ])
    def test_checkout_label_variants_detected(self, label: str):
        text = f"I will {label} on 2026-07-20"
        _, check_out = _extract_dates_by_association(text)
        assert check_out == "2026-07-20"

    @pytest.mark.parametrize("label", [
        "check-in",
        "check in",
        "arrival",
    ])
    def test_checkin_label_variants_detected(self, label: str):
        text = f"My {label} date is 2026-07-10"
        check_in, _ = _extract_dates_by_association(text)
        assert check_in == "2026-07-10"