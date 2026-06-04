"""
Unit tests for booking_schema_loader._reference_today() and its integration
with validate_field() date validation.

The _reference_today() function was added in this PR to allow deterministic
date references in tests and smoke runs via the BOOKING_REFERENCE_DATE env var.
"""
from __future__ import annotations

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.config.booking_schema_loader import _reference_today


class TestReferenceToday:
    """Tests for the _reference_today() helper function."""

    def test_returns_date_today_when_env_not_set(self, monkeypatch):
        """Without env var, _reference_today() returns date.today()."""
        monkeypatch.delenv("BOOKING_REFERENCE_DATE", raising=False)
        result = _reference_today()
        assert result == date.today()

    def test_returns_date_today_when_env_empty(self, monkeypatch):
        """Empty string env var falls back to date.today()."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "")
        result = _reference_today()
        assert result == date.today()

    def test_returns_date_today_when_env_whitespace_only(self, monkeypatch):
        """Whitespace-only env var is treated as empty and falls back to date.today()."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "   ")
        result = _reference_today()
        assert result == date.today()

    def test_returns_parsed_date_when_valid_env_set(self, monkeypatch):
        """A valid BOOKING_REFERENCE_DATE value is parsed and returned."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")
        result = _reference_today()
        assert result == date(2026, 6, 1)

    def test_returns_parsed_date_for_different_valid_dates(self, monkeypatch):
        """Various valid date strings are parsed correctly."""
        test_cases = [
            ("2025-01-15", date(2025, 1, 15)),
            ("2030-12-31", date(2030, 12, 31)),
            ("2024-02-29", date(2024, 2, 29)),  # leap year
        ]
        for raw, expected in test_cases:
            monkeypatch.setenv("BOOKING_REFERENCE_DATE", raw)
            result = _reference_today()
            assert result == expected, f"Expected {expected} for {raw!r}, got {result}"

    def test_falls_back_to_today_on_invalid_date_format(self, monkeypatch):
        """An invalid date string logs a warning and falls back to date.today()."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "not-a-date")
        result = _reference_today()
        assert result == date.today()

    def test_falls_back_to_today_on_wrong_format(self, monkeypatch):
        """A date in non-YYYY-MM-DD format is treated as invalid."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "01/06/2026")
        result = _reference_today()
        assert result == date.today()

    def test_falls_back_to_today_on_impossible_date(self, monkeypatch):
        """An impossible date like month 13 falls back to date.today()."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-13-01")
        result = _reference_today()
        assert result == date.today()

    def test_returns_date_object_not_datetime(self, monkeypatch):
        """The returned value is a date, not a datetime."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-15")
        result = _reference_today()
        assert type(result) is date


class TestReferenceTodayIntegrationWithValidateField:
    """Integration tests verifying _reference_today() is used by validate_field()."""

    def test_check_in_date_before_reference_today_is_rejected(self, monkeypatch):
        """A check_in date earlier than the reference date must fail validation."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")
        from app.config.booking_schema_loader import validate_field

        ok, err = validate_field("check_in", "2026-05-31")
        assert ok is False
        assert err is not None

    def test_check_in_date_equal_to_reference_today_is_accepted(self, monkeypatch):
        """A check_in date equal to the reference date must pass validation."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")
        from app.config.booking_schema_loader import validate_field

        ok, _err = validate_field("check_in", "2026-06-01")
        assert ok is True

    def test_check_in_date_after_reference_today_is_accepted(self, monkeypatch):
        """A check_in date after the reference date must pass validation."""
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")
        from app.config.booking_schema_loader import validate_field

        ok, _err = validate_field("check_in", "2026-06-10")
        assert ok is True

    def test_changing_reference_date_affects_validation(self, monkeypatch):
        """Changing BOOKING_REFERENCE_DATE changes whether a date passes validation."""
        from app.config.booking_schema_loader import validate_field

        # Date "2026-06-05" is valid when reference is 2026-06-01
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-06-01")
        ok_before, _ = validate_field("check_in", "2026-06-05")
        assert ok_before is True

        # The same date is invalid when reference is 2026-07-01
        monkeypatch.setenv("BOOKING_REFERENCE_DATE", "2026-07-01")
        ok_after, _ = validate_field("check_in", "2026-06-05")
        assert ok_after is False