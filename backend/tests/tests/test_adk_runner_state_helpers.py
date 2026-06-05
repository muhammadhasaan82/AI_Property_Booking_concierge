"""
Unit tests for adk_runner._state_with_persisted_soft_state() added in this PR.

The function:
  1. Filters out temp: keys via _filter_persistent_state
  2. Removes any existing "soft_state" key from the filtered result
  3. Normalises the supplied soft_state (non-dict → {}, via _jsonable)
  4. Injects it back as persisted_state["soft_state"]
  5. Returns the merged dict
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.adk_runner import _state_with_persisted_soft_state


class TestStateWithPersistedSoftState:
    """Unit tests for _state_with_persisted_soft_state()."""

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_basic_merge_of_state_and_soft_state(self):
        """Normal state dict + soft_state dict produces a merged result."""
        state = {"user_id": "u1", "booking_ref": "B1"}
        soft = {"booking_stage": "collecting_details", "last_presented_view": "property_list"}

        result = _state_with_persisted_soft_state(state, soft)

        assert result["user_id"] == "u1"
        assert result["booking_ref"] == "B1"
        assert result["soft_state"] == soft

    def test_soft_state_key_is_always_present_in_result(self):
        """Even with minimal inputs, result must always contain 'soft_state'."""
        result = _state_with_persisted_soft_state({}, {})
        assert "soft_state" in result

    def test_old_soft_state_in_state_is_replaced(self):
        """Any pre-existing soft_state key inside state must be overwritten."""
        state = {
            "user_id": "u2",
            "soft_state": {"stale_key": "stale_value"},
        }
        new_soft = {"fresh_key": "fresh_value"}

        result = _state_with_persisted_soft_state(state, new_soft)

        assert result["soft_state"] == {"fresh_key": "fresh_value"}
        assert "stale_key" not in result["soft_state"]

    # ------------------------------------------------------------------
    # Normalisation of invalid soft_state types
    # ------------------------------------------------------------------

    def test_none_soft_state_normalises_to_empty_dict(self):
        """soft_state=None must produce soft_state={} in the result."""
        result = _state_with_persisted_soft_state({"k": "v"}, None)
        assert result["soft_state"] == {}

    def test_list_soft_state_normalises_to_empty_dict(self):
        """soft_state as a list (wrong type) must normalise to {}."""
        result = _state_with_persisted_soft_state({"k": "v"}, ["item1", "item2"])
        assert result["soft_state"] == {}

    def test_string_soft_state_normalises_to_empty_dict(self):
        """soft_state as a plain string must normalise to {}."""
        result = _state_with_persisted_soft_state({"k": "v"}, "not-a-dict")
        assert result["soft_state"] == {}

    def test_integer_soft_state_normalises_to_empty_dict(self):
        """soft_state as an integer must normalise to {}."""
        result = _state_with_persisted_soft_state({"k": "v"}, 42)
        assert result["soft_state"] == {}

    # ------------------------------------------------------------------
    # Normalisation of invalid state types
    # ------------------------------------------------------------------

    def test_none_state_produces_only_soft_state_key(self):
        """state=None should be treated as an empty dict by _filter_persistent_state."""
        soft = {"booking_stage": "confirmed"}
        result = _state_with_persisted_soft_state(None, soft)
        assert result["soft_state"] == soft

    def test_non_dict_state_produces_only_soft_state_key(self):
        """Non-dict state must not crash; result must contain soft_state."""
        result = _state_with_persisted_soft_state("broken-state", {"x": 1})
        assert result["soft_state"] == {"x": 1}

    # ------------------------------------------------------------------
    # Temp key filtering
    # ------------------------------------------------------------------

    def test_temp_keys_are_stripped_from_state(self):
        """Keys prefixed with 'temp:' must be excluded from the result."""
        state = {
            "user_id": "u3",
            "temp:scratch": "discard-me",
            "temp:another": 999,
            "booking_ref": "B2",
        }
        result = _state_with_persisted_soft_state(state, {})

        assert "user_id" in result
        assert "booking_ref" in result
        assert "temp:scratch" not in result
        assert "temp:another" not in result

    def test_non_temp_keys_survive_filtering(self):
        """Regular keys (no 'temp:' prefix) must be preserved in the output."""
        state = {"a": 1, "b": 2, "c": [1, 2, 3]}
        result = _state_with_persisted_soft_state(state, {"stage": "done"})

        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == [1, 2, 3]
        assert result["soft_state"]["stage"] == "done"

    # ------------------------------------------------------------------
    # _jsonable pass-through
    # ------------------------------------------------------------------

    def test_jsonable_conversion_applied_to_nested_soft_state(self):
        """Nested dict values inside soft_state survive _jsonable serialisation."""
        soft = {
            "booking_state": {
                "check_in": "2026-06-02",
                "check_out": "2026-06-11",
                "guests": 4,
            },
            "booking_stage": "awaiting_confirmation",
        }
        result = _state_with_persisted_soft_state({}, soft)
        assert result["soft_state"]["booking_state"]["check_in"] == "2026-06-02"
        assert result["soft_state"]["booking_state"]["guests"] == 4

    def test_empty_state_and_empty_soft_state(self):
        """Both empty inputs must produce a result with an empty soft_state."""
        result = _state_with_persisted_soft_state({}, {})
        assert result == {"soft_state": {}}

    # ------------------------------------------------------------------
    # Regression: soft_state must not bleed into next call
    # ------------------------------------------------------------------

    def test_successive_calls_do_not_share_soft_state_reference(self):
        """Each call must return an independent dict — no shared mutable state."""
        soft1 = {"stage": "a"}
        soft2 = {"stage": "b"}

        result1 = _state_with_persisted_soft_state({}, soft1)
        result2 = _state_with_persisted_soft_state({}, soft2)

        assert result1["soft_state"]["stage"] == "a"
        assert result2["soft_state"]["stage"] == "b"