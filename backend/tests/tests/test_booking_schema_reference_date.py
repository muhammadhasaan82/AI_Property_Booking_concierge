"""
Unit tests for booking_schema_loader._reference_today() and the related
validate_field date logic that was changed in this PR.

Scope:
  - _reference_today(): env-var override, invalid value fallback, default today
  - validate_field "date" type with not_before="today" using a controlled reference date
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helpers — import the target functions
# ---------------------------------------------------------------------------

from app.config.booking_schema_loader import _reference_today, validate_field


# ===========================================================================
# Tests for _reference_today()
# ===========================================================================


class TestReferenceToday:
    """Covers the _reference_today() helper added in this PR."""

    def test_returns_date_today_when_env_not_set(self, monkeypatch):
        """Without env var, _reference_today() must equal date.today()."""
        monkeypatch.delenv("BOOKING_REFERENCE_DATE", raising=False)
        result = _reference_today()
        assert result == date.today()

    def test_returns_parsed_date_when_env_valid(self, monkeypatch):
        """A valid BOOKING_REFERENCE_DATE env var must be parsed and returned."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2030-03-15")
        result = _reference_today()
        assert result == date(2030, 3, 15)

    def test_returns_today_when_env_is_invalid_format(self, monkeypatch):
        """An unparseable env var must log a warning and fall back to date.today()."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "not-a-date")
        result = _reference_today()
        assert result == date.today()

    def test_returns_today_when_env_is_empty_string(self, monkeypatch):
        """An empty string must be treated as 'not set' and return date.today()."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "")
        result = _reference_today()
        assert result == date.today()

    def test_returns_today_when_env_has_only_whitespace(self, monkeypatch):
        """A whitespace-only string must fall back to date.today()."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "   ")
        result = _reference_today()
        assert result == date.today()

    def test_returns_parsed_date_with_leading_trailing_whitespace(self, monkeypatch):
        """Env var with surrounding whitespace must still parse correctly."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "  2026-01-01  ")
        result = _reference_today()
        assert result == date(2026, 1, 1)

    def test_rejects_wrong_format_like_dd_mm_yyyy(self, monkeypatch):
        """A date in DD-MM-YYYY format (wrong separator / order) must fall back."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "15-03-2030")
        result = _reference_today()
        assert result == date.today()

    def test_returns_min_valid_date(self, monkeypatch):
        """Boundary: smallest sensible valid date must parse correctly."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2000-01-01")
        result = _reference_today()
        assert result == date(2000, 1, 1)

    def test_env_var_month_and_day_zero_padded(self, monkeypatch):
        """Zero-padded month and day must parse correctly."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")
        result = _reference_today()
        assert result == date(2026, 6, 1)


# ===========================================================================
# Tests for validate_field "date" type using _reference_today
# ===========================================================================


class TestValidateFieldDateWithReferenceDate:
    """
    validate_field for date-type fields now uses _reference_today() instead of
    date.today() when checking not_before="today".  These tests verify that a
    fixed reference date is honoured correctly.
    """

    # We patch _reference_today inside the module so the validator uses our stub.
    _MODULE = "app.config.booking_schema_loader._reference_today"

    def _ref(self, d: date):
        return patch(self._MODULE, return_value=d)

    def test_date_equal_to_reference_is_accepted(self):
        """A date equal to the reference date must pass not_before='today'."""
        ref = date(2026, 6, 1)
        with self._ref(ref):
            ok, err = validate_field("check_in", "2026-06-01")
        assert ok, f"Expected ok but got error: {err}"
        assert err is None

    def test_date_after_reference_is_accepted(self):
        """A date after the reference date must pass not_before='today'."""
        ref = date(2026, 6, 1)
        with self._ref(ref):
            ok, err = validate_field("check_in", "2026-07-15")
        assert ok, f"Expected ok but got error: {err}"

    def test_date_before_reference_is_rejected(self):
        """A date before the reference date must fail not_before='today'."""
        ref = date(2026, 6, 1)
        with self._ref(ref):
            ok, err = validate_field("check_in", "2026-05-31")
        assert not ok
        assert err is not None

    def test_date_far_in_past_is_rejected(self):
        """A date clearly in the past must always fail the not_before check."""
        ref = date(2026, 6, 1)
        with self._ref(ref):
            ok, err = validate_field("check_in", "2020-01-01")
        assert not ok
        assert err is not None

    def test_date_check_out_after_check_in_accepted(self):
        """check_out after check_in must pass the after_field constraint."""
        ref = date(2026, 6, 1)
        with self._ref(ref):
            ok, err = validate_field(
                "check_out",
                "2026-06-10",
                current_state={"check_in": "2026-06-05"},
            )
        assert ok, f"Expected ok but got: {err}"

    def test_date_check_out_before_check_in_rejected(self):
        """check_out on or before check_in must fail the after_field constraint."""
        ref = date(2026, 6, 1)
        with self._ref(ref):
            ok, err = validate_field(
                "check_out",
                "2026-06-04",
                current_state={"check_in": "2026-06-05"},
            )
        assert not ok
        assert err is not None

    def test_date_check_out_equal_to_check_in_rejected(self):
        """check_out equal to check_in must also fail (must be strictly after)."""
        ref = date(2026, 6, 1)
        with self._ref(ref):
            ok, err = validate_field(
                "check_out",
                "2026-06-05",
                current_state={"check_in": "2026-06-05"},
            )
        assert not ok
        assert err is not None

    def test_invalid_date_string_is_rejected(self):
        """A completely invalid date string must return False with an error."""
        with self._ref(date(2026, 6, 1)):
            ok, err = validate_field("check_in", "not-a-date")
        assert not ok
        assert err is not None

    def test_reference_date_set_in_future_allows_past_relative_date(self):
        """
        If the reference date is far in the future, a date that is 'today' in
        real time but 'past' relative to the reference date must be rejected.
        """
        far_future_ref = date(2099, 1, 1)
        with self._ref(far_future_ref):
            ok, err = validate_field("check_in", "2026-06-01")
        assert not ok
        assert err is not None