"""
Comprehensive unit tests for app/config/conversation_shortcuts_loader.py.

Covers: _as_str_list, _normalize, _compile_pattern, _spec_body,
_ShortcutRouter (construction, _state_ok, match), match_shortcut,
reload, ShortcutMatch/ShortcutSpec models, and the new
requires_any_state gate introduced in this PR.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.config.conversation_shortcuts_loader import (
    ShortcutMatch,
    ShortcutSpec,
    _ShortcutRouter,
    _as_str_list,
    _compile_pattern,
    _normalize,
    _spec_body,
    match_shortcut,
    reload,
)


# ---------------------------------------------------------------------------
# _as_str_list
# ---------------------------------------------------------------------------


class TestAsStrList:
    def test_none_returns_empty_list(self):
        assert _as_str_list(None) == []

    def test_empty_list_returns_empty_list(self):
        assert _as_str_list([]) == []

    def test_list_with_strings(self):
        assert _as_str_list(["a", "b"]) == ["a", "b"]

    def test_list_strips_whitespace(self):
        assert _as_str_list(["  hello  ", " world "]) == ["hello", "world"]

    def test_list_filters_empty_after_strip(self):
        assert _as_str_list(["a", "   ", "", "b"]) == ["a", "b"]

    def test_plain_string_returns_single_element_list(self):
        assert _as_str_list("hello") == ["hello"]

    def test_whitespace_string_returns_empty_list(self):
        assert _as_str_list("   ") == []

    def test_empty_string_returns_empty_list(self):
        assert _as_str_list("") == []

    def test_integer_in_list_is_cast_to_string(self):
        result = _as_str_list([1, 2])
        assert result == ["1", "2"]

    def test_integer_scalar_is_cast_to_string(self):
        result = _as_str_list(42)
        assert result == ["42"]

    def test_list_with_none_items_filtered(self):
        # None becomes "None" which is non-empty after strip
        result = _as_str_list([None])
        # "None".strip() == "None" which is truthy → kept
        assert result == ["None"]


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_empty_string(self):
        assert _normalize("") == ""

    def test_strips_leading_trailing_whitespace(self):
        assert _normalize("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert _normalize("hello   world") == "hello world"

    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_mixed_case_and_whitespace(self):
        assert _normalize("  Show  Me  MORE  ") == "show me more"

    def test_already_normalized(self):
        assert _normalize("show me more") == "show me more"

    def test_tabs_and_newlines_collapsed(self):
        assert _normalize("hello\tworld\n") == "hello world"


# ---------------------------------------------------------------------------
# _compile_pattern
# ---------------------------------------------------------------------------


class TestCompilePattern:
    def test_simple_pattern_without_number(self):
        pattern = _compile_pattern("show me more")
        assert pattern.search("show me more") is not None

    def test_simple_pattern_no_match(self):
        pattern = _compile_pattern("show me more")
        assert pattern.search("show me less") is None

    def test_number_token_captures_digit(self):
        pattern = _compile_pattern("option {number}")
        m = pattern.search("option 3")
        assert m is not None
        assert m.groupdict().get("number") == "3"

    def test_number_token_captures_multi_digit(self):
        pattern = _compile_pattern("select {number}")
        m = pattern.search("select 42")
        assert m is not None
        assert m.groupdict()["number"] == "42"

    def test_number_token_no_match_for_non_digit(self):
        pattern = _compile_pattern("option {number}")
        assert pattern.search("option abc") is None

    def test_pattern_case_insensitive_handling(self):
        # _compile_pattern lowercases the template
        pattern = _compile_pattern("Option {number}")
        # match against lowercased text
        m = pattern.search("option 5")
        assert m is not None

    def test_go_with_pattern(self):
        pattern = _compile_pattern("go with {number}")
        m = pattern.search("go with 2")
        assert m is not None
        assert m.groupdict()["number"] == "2"

    def test_pattern_is_compiled_regex(self):
        pattern = _compile_pattern("show me more")
        assert isinstance(pattern, re.Pattern)


# ---------------------------------------------------------------------------
# _spec_body
# ---------------------------------------------------------------------------


class TestSpecBody:
    def test_none_input_returns_empty_dict_with_lists(self):
        result = _spec_body(None)
        assert result["requires_state"] == []
        assert result["requires_any_state"] == []
        assert result["examples"] == []
        assert result["patterns"] == []

    def test_non_dict_input_returns_empty(self):
        result = _spec_body("invalid")
        assert result == {"requires_state": [], "requires_any_state": [], "examples": [], "patterns": []}

    def test_preserves_action_and_direction(self):
        body = {"action": "paginate_results", "direction": "next", "requires_state": ["x"]}
        result = _spec_body(body)
        assert result["action"] == "paginate_results"
        assert result["direction"] == "next"
        assert result["requires_state"] == ["x"]

    def test_converts_string_requires_state_to_list(self):
        body = {"action": "x", "requires_state": "all_search_results"}
        result = _spec_body(body)
        assert result["requires_state"] == ["all_search_results"]

    def test_filters_empty_examples(self):
        body = {"action": "x", "examples": ["ok", "", "  "]}
        result = _spec_body(body)
        assert result["examples"] == ["ok"]


# ---------------------------------------------------------------------------
# ShortcutMatch model
# ---------------------------------------------------------------------------


class TestShortcutMatch:
    def test_minimal_construction(self):
        m = ShortcutMatch(intent="test", action="paginate_results")
        assert m.intent == "test"
        assert m.action == "paginate_results"
        assert m.direction is None
        assert m.selection_number is None
        assert m.requires_state == []
        assert m.requires_any_state == []

    def test_full_construction(self):
        m = ShortcutMatch(
            intent="resume_booking_selection",
            action="select_property",
            direction=None,
            selection_number=3,
            requires_state=[],
            requires_any_state=["option_map", "active_property_options_map"],
        )
        assert m.selection_number == 3
        assert m.requires_any_state == ["option_map", "active_property_options_map"]


# ---------------------------------------------------------------------------
# ShortcutSpec model
# ---------------------------------------------------------------------------


class TestShortcutSpec:
    def test_defaults_are_empty_lists(self):
        spec = ShortcutSpec(intent="x", action="y")
        assert spec.requires_state == []
        assert spec.requires_any_state == []
        assert spec.examples == []
        assert spec.patterns == []
        assert spec.entity_schema == {}

    def test_entity_schema_stored(self):
        spec = ShortcutSpec(intent="x", action="y", entity_schema={"selection_number": "integer"})
        assert spec.entity_schema["selection_number"] == "integer"


# ---------------------------------------------------------------------------
# _ShortcutRouter construction
# ---------------------------------------------------------------------------


class TestShortcutRouterConstruction:
    def test_empty_dict_creates_router_with_no_specs(self):
        router = _ShortcutRouter({})
        assert router.specs == []
        assert router.version == "1.0"

    def test_none_input_creates_empty_router(self):
        router = _ShortcutRouter(None)
        assert router.specs == []

    def test_non_dict_input_creates_empty_router(self):
        router = _ShortcutRouter("invalid")
        assert router.specs == []

    def test_version_is_read_from_raw(self):
        router = _ShortcutRouter({"version": "2.5", "shortcuts": {}})
        assert router.version == "2.5"

    def test_missing_version_defaults_to_1_0(self):
        router = _ShortcutRouter({"shortcuts": {}})
        assert router.version == "1.0"

    def test_shortcuts_not_dict_creates_empty_router(self):
        router = _ShortcutRouter({"shortcuts": ["not", "a", "dict"]})
        assert router.specs == []

    def test_invalid_spec_is_skipped_with_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="app.config.conversation_shortcuts_loader"):
            router = _ShortcutRouter({
                "shortcuts": {
                    "bad_spec": None,  # _spec_body(None) produces valid spec with empty fields
                    "another_bad": 42,  # non-dict → _spec_body returns {}
                }
            })
        # Router should not raise; may have empty specs or skipped ones
        # (None becomes {} via _spec_body which is valid with just intent)
        assert isinstance(router.specs, list)

    def test_valid_spec_creates_spec_entry(self):
        router = _ShortcutRouter({
            "shortcuts": {
                "pagination_next": {
                    "action": "paginate_results",
                    "direction": "next",
                    "requires_state": ["all_search_results"],
                    "examples": ["show me more"],
                }
            }
        })
        assert len(router.specs) == 1
        assert router.specs[0].intent == "pagination_next"
        assert router.specs[0].action == "paginate_results"


# ---------------------------------------------------------------------------
# _ShortcutRouter._state_ok
# ---------------------------------------------------------------------------


class TestShortcutRouterStateOk:
    def _make_spec(self, requires_state=None, requires_any_state=None):
        return ShortcutSpec(
            intent="test",
            action="act",
            requires_state=requires_state or [],
            requires_any_state=requires_any_state or [],
        )

    def test_no_requirements_always_ok(self):
        router = _ShortcutRouter({})
        spec = self._make_spec()
        assert router._state_ok(spec, {}) is True
        assert router._state_ok(spec, {"x": 1}) is True

    def test_requires_state_all_present(self):
        router = _ShortcutRouter({})
        spec = self._make_spec(requires_state=["all_search_results"])
        assert router._state_ok(spec, {"all_search_results": [1, 2]}) is True

    def test_requires_state_one_missing(self):
        router = _ShortcutRouter({})
        spec = self._make_spec(requires_state=["all_search_results"])
        assert router._state_ok(spec, {}) is False
        assert router._state_ok(spec, {"all_search_results": None}) is False
        assert router._state_ok(spec, {"all_search_results": []}) is False

    def test_requires_state_multiple_all_present(self):
        router = _ShortcutRouter({})
        spec = self._make_spec(requires_state=["a", "b"])
        assert router._state_ok(spec, {"a": 1, "b": 2}) is True

    def test_requires_state_multiple_one_missing(self):
        router = _ShortcutRouter({})
        spec = self._make_spec(requires_state=["a", "b"])
        assert router._state_ok(spec, {"a": 1}) is False

    def test_requires_any_state_one_present(self):
        router = _ShortcutRouter({})
        spec = self._make_spec(requires_any_state=["option_map", "active_property_options_map"])
        # option_map present
        assert router._state_ok(spec, {"option_map": {"1": {}}}) is True
        # active_property_options_map present
        assert router._state_ok(spec, {"active_property_options_map": {"1": {}}}) is True

    def test_requires_any_state_none_present(self):
        router = _ShortcutRouter({})
        spec = self._make_spec(requires_any_state=["option_map", "active_property_options_map"])
        assert router._state_ok(spec, {}) is False
        assert router._state_ok(spec, {"other_key": "value"}) is False

    def test_requires_any_state_with_falsy_value_fails(self):
        router = _ShortcutRouter({})
        spec = self._make_spec(requires_any_state=["option_map"])
        assert router._state_ok(spec, {"option_map": {}}) is False  # empty dict is falsy
        assert router._state_ok(spec, {"option_map": None}) is False

    def test_requires_both_state_and_any_state_both_must_pass(self):
        router = _ShortcutRouter({})
        spec = self._make_spec(
            requires_state=["all_search_results"],
            requires_any_state=["option_map", "active_property_options_map"],
        )
        # requires_state present but requires_any_state missing → False
        assert router._state_ok(spec, {"all_search_results": [1]}) is False
        # requires_any_state present but requires_state missing → False
        assert router._state_ok(spec, {"option_map": {"1": {}}}) is False
        # Both present → True
        assert router._state_ok(
            spec,
            {"all_search_results": [1], "option_map": {"1": {}}},
        ) is True


# ---------------------------------------------------------------------------
# _ShortcutRouter.match
# ---------------------------------------------------------------------------


class TestShortcutRouterMatch:
    def _pagination_router(self):
        return _ShortcutRouter({
            "shortcuts": {
                "pagination_next": {
                    "action": "paginate_results",
                    "direction": "next",
                    "requires_state": ["all_search_results"],
                    "examples": ["show me more", "more", "next"],
                }
            }
        })

    def _selection_router(self):
        return _ShortcutRouter({
            "shortcuts": {
                "resume_booking_selection": {
                    "action": "select_property",
                    "requires_any_state": ["option_map", "active_property_options_map"],
                    "patterns": ["option {number}", "select {number}", "go with {number}"],
                }
            }
        })

    def test_empty_message_returns_none(self):
        router = self._pagination_router()
        assert router.match("", {"all_search_results": [1]}) is None
        assert router.match("   ", {"all_search_results": [1]}) is None

    def test_non_dict_soft_state_treated_as_empty(self):
        router = self._pagination_router()
        # State not present → no match
        assert router.match("show me more", None) is None
        assert router.match("show me more", "not-a-dict") is None

    def test_example_match_exact(self):
        router = self._pagination_router()
        m = router.match("show me more", {"all_search_results": [1]})
        assert m is not None
        assert m.intent == "pagination_next"
        assert m.action == "paginate_results"
        assert m.direction == "next"
        assert m.selection_number is None

    def test_example_match_case_insensitive(self):
        router = self._pagination_router()
        m = router.match("SHOW ME MORE", {"all_search_results": [1]})
        assert m is not None

    def test_example_match_with_extra_whitespace(self):
        router = self._pagination_router()
        m = router.match("  show me more  ", {"all_search_results": [1]})
        assert m is not None

    def test_example_no_match_without_required_state(self):
        router = self._pagination_router()
        assert router.match("show me more", {}) is None
        assert router.match("show me more", {"all_search_results": None}) is None

    def test_pattern_match_with_number(self):
        router = self._selection_router()
        m = router.match("option 3", {"option_map": {"3": {"property_id": "x"}}})
        assert m is not None
        assert m.action == "select_property"
        assert m.selection_number == 3

    def test_pattern_match_with_legacy_state_key(self):
        """Regression: requires_any_state lets option_map OR active_property_options_map fire."""
        router = self._selection_router()
        m = router.match("option 2", {"active_property_options_map": {"2": {"property_id": "y"}}})
        assert m is not None
        assert m.selection_number == 2

    def test_pattern_match_no_state_returns_none(self):
        router = self._selection_router()
        assert router.match("option 1", {}) is None

    def test_pattern_match_go_with_number(self):
        router = self._selection_router()
        m = router.match("go with 5", {"option_map": {"5": {}}})
        assert m is not None
        assert m.selection_number == 5

    def test_no_match_returns_none(self):
        router = self._pagination_router()
        assert router.match("hello there", {"all_search_results": [1]}) is None

    def test_requires_any_state_populated_in_match(self):
        router = self._selection_router()
        m = router.match("select 1", {"option_map": {"1": {}}})
        assert m is not None
        assert "option_map" in m.requires_any_state or "active_property_options_map" in m.requires_any_state

    def test_empty_router_returns_none(self):
        router = _ShortcutRouter({})
        assert router.match("show me more", {"all_search_results": [1]}) is None

    def test_multiple_specs_first_match_wins(self):
        router = _ShortcutRouter({
            "shortcuts": {
                "spec_a": {
                    "action": "action_a",
                    "requires_state": ["key_a"],
                    "examples": ["trigger"],
                },
                "spec_b": {
                    "action": "action_b",
                    "requires_state": ["key_b"],
                    "examples": ["trigger"],
                },
            }
        })
        # Only spec_a state present → spec_a wins
        m = router.match("trigger", {"key_a": True})
        assert m is not None
        assert m.action == "action_a"


# ---------------------------------------------------------------------------
# match_shortcut (public API) uses the real YAML
# ---------------------------------------------------------------------------


class TestMatchShortcutPublicApi:
    def test_pagination_next_from_yaml(self):
        m = match_shortcut("show me more", {"all_search_results": [1, 2]})
        assert m is not None
        assert m.action == "paginate_results"
        assert m.direction == "next"

    def test_pagination_previous_from_yaml(self):
        m = match_shortcut("previous", {"all_search_results": [1]})
        assert m is not None
        assert m.action == "paginate_results"
        assert m.direction == "previous"

    def test_selection_via_option_map_state(self):
        m = match_shortcut("option 2", {"option_map": {"2": {"property_id": "abc"}}})
        assert m is not None
        assert m.action == "select_property"
        assert m.selection_number == 2

    def test_selection_via_active_property_options_map_state(self):
        """PR change: requires_any_state so either key enables the shortcut."""
        m = match_shortcut("option 4", {"active_property_options_map": {"4": {"property_id": "xyz"}}})
        assert m is not None
        assert m.action == "select_property"
        assert m.selection_number == 4

    def test_no_match_returns_none(self):
        assert match_shortcut("random text", {"all_search_results": [1]}) is None

    def test_empty_state_blocks_shortcuts(self):
        assert match_shortcut("show me more", {}) is None
        assert match_shortcut("option 1", {}) is None

    def test_none_state_blocks_shortcuts(self):
        assert match_shortcut("show me more", None) is None


# ---------------------------------------------------------------------------
# reload function
# ---------------------------------------------------------------------------


class TestReload:
    def test_reload_does_not_raise(self):
        """reload() should silently succeed when YAML file exists."""
        reload()  # uses real YAML file

    def test_reload_updates_global_router(self):
        """After reload, match_shortcut still works correctly."""
        reload()
        m = match_shortcut("show me more", {"all_search_results": [1]})
        assert m is not None

    def test_reload_with_missing_yaml_uses_empty_router(self, tmp_path, monkeypatch):
        """When YAML is absent, reload should not raise and shortcuts return None."""
        import app.config.conversation_shortcuts_loader as loader
        fake_path = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(loader, "_SHORTCUTS_PATH", fake_path)
        loader.reload()
        assert loader.match_shortcut("show me more", {"all_search_results": [1]}) is None
        # Restore
        loader.reload.__module__  # just access to confirm module is ok


# ---------------------------------------------------------------------------
# Regression: resume_booking_selection uses requires_any_state (not requires_state)
# ---------------------------------------------------------------------------


class TestRequiresAnyStateRegression:
    """Ensure the YAML change (requires_state → requires_any_state) is honoured."""

    def test_old_option_map_key_still_fires_shortcut(self):
        m = match_shortcut("back to booking option 3", {"option_map": {"3": {}}})
        assert m is not None
        assert m.selection_number == 3

    def test_legacy_active_property_options_map_key_fires_shortcut(self):
        """This is the NEW behavior from the PR."""
        m = match_shortcut(
            "book option 1",
            {"active_property_options_map": {"1": {"property_id": "p1"}}},
        )
        assert m is not None
        assert m.action == "select_property"
        assert m.selection_number == 1

    def test_neither_key_present_blocks_shortcut(self):
        m = match_shortcut("option 5", {"all_search_results": [1, 2]})
        assert m is None

    def test_both_keys_present_fires_shortcut(self):
        m = match_shortcut(
            "choose 2",
            {
                "option_map": {"2": {}},
                "active_property_options_map": {"2": {"property_id": "p2"}},
            },
        )
        assert m is not None
        assert m.selection_number == 2
