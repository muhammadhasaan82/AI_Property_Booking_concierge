"""
Unit tests for adk_runner._state_with_persisted_soft_state().

This function was added in this PR to fix a bug where soft_state mutations
were not being persisted correctly across session saves. It:
  1. Filters out temp: prefixed keys from the state
  2. Replaces any pre-existing "soft_state" key with the provided soft_state
  3. Normalises soft_state through _jsonable() to ensure JSON-serializability
  4. Guards against non-dict soft_state by falling back to {}
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.adk_runner import _state_with_persisted_soft_state


class TestStateWithPersistedSoftState:
    """Tests for _state_with_persisted_soft_state()."""

    def test_normal_state_and_soft_state_are_combined(self):
        """Regular state keys survive and soft_state is set correctly."""
        state = {"user_id": "u1", "booking_step": "city"}
        soft_state = {"booking_stage": "collecting_details", "city": "Dubai"}

        result = _state_with_persisted_soft_state(state, soft_state)

        assert result["user_id"] == "u1"
        assert result["booking_step"] == "city"
        assert result["soft_state"] == soft_state

    def test_temp_prefixed_keys_are_filtered_out(self):
        """Keys starting with 'temp:' must be excluded from the result."""
        state = {
            "keep_me": "yes",
            "temp:scratch": "remove",
            "temp:internal": {"data": True},
        }
        soft_state = {"stage": "done"}

        result = _state_with_persisted_soft_state(state, soft_state)

        assert "keep_me" in result
        assert "temp:scratch" not in result
        assert "temp:internal" not in result

    def test_existing_soft_state_in_state_is_replaced(self):
        """An old 'soft_state' key in the state dict must be replaced by the new one."""
        old_soft_state = {"booking_stage": "old_stage", "stale_key": True}
        state = {"soft_state": old_soft_state, "other": "value"}
        new_soft_state = {"booking_stage": "collecting_details"}

        result = _state_with_persisted_soft_state(state, new_soft_state)

        assert result["soft_state"] == new_soft_state
        assert "stale_key" not in result["soft_state"]
        assert result["other"] == "value"

    def test_non_dict_soft_state_becomes_empty_dict(self):
        """If soft_state is not a dict (e.g. None, list), it is normalised to {}."""
        state = {"foo": "bar"}

        for bad_soft_state in [None, "string", 42, ["a", "b"]]:
            result = _state_with_persisted_soft_state(state, bad_soft_state)
            assert result["soft_state"] == {}, (
                f"Expected {{}} for soft_state={bad_soft_state!r}, got {result['soft_state']!r}"
            )

    def test_none_state_returns_only_soft_state(self):
        """If state is None (non-dict), result contains only soft_state."""
        soft_state = {"booking_stage": "confirmed"}

        result = _state_with_persisted_soft_state(None, soft_state)

        assert result == {"soft_state": soft_state}

    def test_non_dict_state_returns_only_soft_state(self):
        """If state is not a dict, result contains only soft_state."""
        result = _state_with_persisted_soft_state("not a dict", {"k": "v"})
        assert "soft_state" in result
        assert result["soft_state"] == {"k": "v"}

    def test_dict_keys_in_soft_state_are_stringified(self):
        """_jsonable() converts integer dict keys to strings."""
        state = {}
        soft_state = {1: "one", 2: "two"}

        result = _state_with_persisted_soft_state(state, soft_state)

        # After _jsonable, integer keys become strings
        assert "1" in result["soft_state"]
        assert "2" in result["soft_state"]

    def test_soft_state_nested_values_preserved(self):
        """Nested data within soft_state is preserved after normalisation."""
        state = {"ctx": "ok"}
        soft_state = {
            "booking_state": {
                "guest_name": "Jane",
                "guests": 2,
                "check_in": "2026-06-02",
            },
            "booking_stage": "awaiting_confirmation",
        }

        result = _state_with_persisted_soft_state(state, soft_state)

        assert result["soft_state"]["booking_state"]["guest_name"] == "Jane"
        assert result["soft_state"]["booking_state"]["guests"] == 2
        assert result["soft_state"]["booking_stage"] == "awaiting_confirmation"

    def test_empty_state_and_empty_soft_state(self):
        """Both state and soft_state being empty dicts yields {'soft_state': {}}."""
        result = _state_with_persisted_soft_state({}, {})
        assert result == {"soft_state": {}}

    def test_result_is_a_new_dict_not_the_original(self):
        """The returned dict must not be the same object as the input state."""
        state = {"key": "value"}
        soft_state = {"stage": "init"}

        result = _state_with_persisted_soft_state(state, soft_state)

        assert result is not state

    def test_temp_key_and_soft_state_key_both_stripped(self):
        """Both temp: keys and the old soft_state are stripped before assignment."""
        state = {
            "user": "alice",
            "temp:cache": {"big": "data"},
            "soft_state": {"old": True},
        }
        new_soft_state = {"new": True}

        result = _state_with_persisted_soft_state(state, new_soft_state)

        assert "user" in result
        assert "temp:cache" not in result
        assert result["soft_state"] == {"new": True}
        assert "old" not in result["soft_state"]