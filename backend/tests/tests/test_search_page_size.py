"""
Comprehensive unit tests for page_size resolution and property ID resolution
in app/agents/tools/search.py (changes introduced in this PR).

Covers:
  - _resolve_page_size_max
  - _resolve_page_size
  - _resolve_page_size_from
  - _resolve_property_id_from_selection (option_map vs active_property_options_map)
  - _build_search_page_payload with page_size=None
  - paginate_stored_results page_size handling
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

import app.agents.tools.search as search_module
from app.agents.tools.search import (
    _build_search_page_payload,
    _resolve_page_size,
    _resolve_page_size_from,
    _resolve_page_size_max,
    _resolve_property_id_from_selection,
    paginate_stored_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results(n: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": str(i),
            "title": f"Property {i}",
            "city": "New York",
            "price_per_night": 100 + i,
            "bedrooms": 2,
            "bathrooms": 1,
            "property_type": "apartment",
            "rating": 4.5,
            "amenities": [],
            "number": i,
        }
        for i in range(1, n + 1)
    ]


def _patch_cfg(page_size: int = 5, page_size_max: int = 25):
    """Context manager that patches cfg attributes on the search module."""
    from unittest.mock import patch as _patch
    cfg = search_module.cfg
    return _patch.multiple(
        cfg.__class__,
        page_size=property(lambda self: page_size),
        page_size_max=property(lambda self: page_size_max),
    )


# ---------------------------------------------------------------------------
# _resolve_page_size_max
# ---------------------------------------------------------------------------


class TestResolvePageSizeMax:
    def test_default_from_cfg(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size_max() == 25

    def test_custom_cfg_value(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size_max", 10)
        assert _resolve_page_size_max() == 10

    def test_zero_value_clamped_to_1(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size_max", 0)
        assert _resolve_page_size_max() == 1

    def test_negative_value_clamped_to_1(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size_max", -5)
        assert _resolve_page_size_max() == 1

    def test_none_value_uses_default_25(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size_max", None)
        assert _resolve_page_size_max() == 25

    def test_returns_int(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size_max", 20)
        result = _resolve_page_size_max()
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# _resolve_page_size
# ---------------------------------------------------------------------------


class TestResolvePageSize:
    def test_returns_configured_value(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size() == 5

    def test_custom_page_size_within_max(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 8)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size() == 8

    def test_page_size_clamped_to_max(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 50)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size() == 25

    def test_none_page_size_uses_default_5(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", None)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size() == 5

    def test_zero_page_size_uses_default_5(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 0)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size() == 5

    def test_negative_page_size_uses_default_5(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", -3)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size() == 5

    def test_result_at_least_1(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 1)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 1)
        assert _resolve_page_size() == 1

    def test_returns_int(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert isinstance(_resolve_page_size(), int)


# ---------------------------------------------------------------------------
# _resolve_page_size_from
# ---------------------------------------------------------------------------


class TestResolvePageSizeFrom:
    def test_none_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size_from(None) == 5

    def test_zero_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size_from(0) == 5

    def test_negative_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size_from(-1) == 5

    def test_valid_positive_value_used(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size_from(7) == 7

    def test_value_above_max_clamped_to_max(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 10)
        assert _resolve_page_size_from(100) == 10

    def test_value_exactly_max_is_accepted(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size_from(25) == 25

    def test_value_1_is_accepted(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size_from(1) == 1

    def test_string_integer_is_coerced(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size_from("8") == 8

    def test_non_numeric_string_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert _resolve_page_size_from("abc") == 5

    def test_float_is_coerced_to_int(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        # _coerce_int handles floats
        result = _resolve_page_size_from(7.9)
        assert result == 7

    def test_result_always_at_least_1(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        # Even 0 returns at least 1 (via default path which gives 5)
        assert _resolve_page_size_from(0) >= 1


# ---------------------------------------------------------------------------
# _resolve_property_id_from_selection
# ---------------------------------------------------------------------------


class TestResolvePropertyIdFromSelection:
    def test_none_selection_returns_none(self):
        assert _resolve_property_id_from_selection(None, {}, {}) is None

    def test_option_map_resolution(self):
        soft_state = {
            "option_map": {
                "1": {"property_id": "prop-abc", "title": "Nice Place"},
                "2": {"property_id": "prop-xyz"},
            }
        }
        assert _resolve_property_id_from_selection(1, soft_state, None) == "prop-abc"
        assert _resolve_property_id_from_selection(2, soft_state, None) == "prop-xyz"

    def test_option_map_key_not_found_falls_through(self):
        soft_state = {"option_map": {"1": {"property_id": "abc"}}}
        # Number 5 not in option_map → falls through to next source
        result = _resolve_property_id_from_selection(5, soft_state, None)
        assert result is None

    def test_active_property_options_map_legacy_fallback(self):
        """PR change: active_property_options_map is now the second lookup."""
        soft_state = {
            "active_property_options_map": {
                "3": {"property_id": "legacy-prop"},
            }
        }
        result = _resolve_property_id_from_selection(3, soft_state, None)
        assert result == "legacy-prop"

    def test_option_map_takes_precedence_over_legacy(self):
        """option_map is checked before active_property_options_map."""
        soft_state = {
            "option_map": {"1": {"property_id": "from-option-map"}},
            "active_property_options_map": {"1": {"property_id": "from-legacy"}},
        }
        assert _resolve_property_id_from_selection(1, soft_state, None) == "from-option-map"

    def test_last_search_fallback(self):
        soft_state = {}
        last_search = {
            "properties": [
                {"number": 1, "id": "search-prop-1"},
                {"number": 2, "id": "search-prop-2"},
            ]
        }
        assert _resolve_property_id_from_selection(2, soft_state, last_search) == "search-prop-2"

    def test_last_search_not_found_returns_none(self):
        last_search = {"properties": [{"number": 1, "id": "p1"}]}
        assert _resolve_property_id_from_selection(9, {}, last_search) is None

    def test_none_soft_state_uses_last_search(self):
        last_search = {"properties": [{"number": 1, "id": "p1"}]}
        assert _resolve_property_id_from_selection(1, None, last_search) == "p1"

    def test_none_last_search_returns_none_when_no_soft_state_match(self):
        assert _resolve_property_id_from_selection(1, {}, None) is None

    def test_option_map_property_id_is_none_skips(self):
        soft_state = {
            "option_map": {"1": {"property_id": None}},
            "active_property_options_map": {"1": {"property_id": "legacy-id"}},
        }
        # option_map has property_id=None → should fall through to legacy
        result = _resolve_property_id_from_selection(1, soft_state, None)
        assert result == "legacy-id"

    def test_property_id_is_cast_to_str(self):
        soft_state = {"option_map": {"1": {"property_id": 42}}}
        result = _resolve_property_id_from_selection(1, soft_state, None)
        assert result == "42"
        assert isinstance(result, str)

    def test_empty_option_map_falls_through_to_last_search(self):
        soft_state = {"option_map": {}}
        last_search = {"properties": [{"number": 3, "id": "ls-prop"}]}
        assert _resolve_property_id_from_selection(3, soft_state, last_search) == "ls-prop"


# ---------------------------------------------------------------------------
# _build_search_page_payload – page_size=None uses dynamic default
# ---------------------------------------------------------------------------


class TestBuildSearchPagePayload:
    def test_page_size_none_uses_resolved_default(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        results = _make_results(10)
        payload, visible, option_map = _build_search_page_payload(
            results=results, filters={}, page=1, page_size=None
        )
        assert payload["pagination"]["page_size"] == 5
        assert len(visible) == 5

    def test_page_size_explicit_used(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        results = _make_results(12)
        payload, visible, _ = _build_search_page_payload(
            results=results, filters={}, page=1, page_size=3
        )
        assert payload["pagination"]["page_size"] == 3
        assert len(visible) == 3

    def test_page_size_clamped_to_max(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 4)
        results = _make_results(10)
        payload, visible, _ = _build_search_page_payload(
            results=results, filters={}, page=1, page_size=100
        )
        assert payload["pagination"]["page_size"] == 4
        assert len(visible) == 4

    def test_total_found_is_full_count(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        results = _make_results(15)
        payload, _, _ = _build_search_page_payload(results=results, filters={}, page=1)
        assert payload["total_found"] == 15

    def test_first_page_shown_count_equals_page_size(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        results = _make_results(12)
        payload, _, _ = _build_search_page_payload(results=results, filters={}, page=1)
        assert payload["shown_count"] == 5

    def test_second_page(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        results = _make_results(12)
        payload, visible, _ = _build_search_page_payload(results=results, filters={}, page=2)
        assert payload["pagination"]["current_page"] == 2
        assert len(visible) == 5

    def test_last_partial_page(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        results = _make_results(7)
        payload, visible, _ = _build_search_page_payload(results=results, filters={}, page=2)
        # Page 2: items 6-7 → 2 items
        assert len(visible) == 2
        assert payload["shown_count"] == 2

    def test_empty_results(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        payload, visible, option_map = _build_search_page_payload(
            results=[], filters={}, page=1
        )
        assert payload["total_found"] == 0
        assert visible == []
        assert option_map == {}

    def test_page_beyond_last_clamped_to_last_page(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        results = _make_results(10)
        payload, _, _ = _build_search_page_payload(results=results, filters={}, page=99)
        assert payload["pagination"]["current_page"] == 2  # 10 items / 5 per page = 2 pages

    def test_option_map_keys_are_strings(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        results = _make_results(3)
        _, _, option_map = _build_search_page_payload(results=results, filters={}, page=1)
        for key in option_map:
            assert isinstance(key, str)

    def test_pagination_start_end_correct(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        results = _make_results(12)
        payload, _, _ = _build_search_page_payload(results=results, filters={}, page=1)
        assert payload["pagination"]["page_start"] == 1
        assert payload["pagination"]["page_end"] == 5


# ---------------------------------------------------------------------------
# paginate_stored_results
# ---------------------------------------------------------------------------


class TestPaginateStoredResults:
    def test_non_dict_soft_state_returns_none(self):
        assert paginate_stored_results(None) is None
        assert paginate_stored_results("string") is None
        assert paginate_stored_results(42) is None

    def test_empty_all_search_results_returns_none(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {"all_search_results": []}
        assert paginate_stored_results(soft_state) is None

    def test_missing_all_search_results_returns_none(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        assert paginate_stored_results({}) is None

    def test_next_page_advances(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(10),
            "current_page": 1,
            "page_size": 5,
            "last_filters": {},
        }
        payload = paginate_stored_results(soft_state, direction="next")
        assert payload is not None
        assert payload["pagination"]["current_page"] == 2

    def test_previous_page_retreats(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(10),
            "current_page": 2,
            "page_size": 5,
            "last_filters": {},
        }
        payload = paginate_stored_results(soft_state, direction="previous")
        assert payload is not None
        assert payload["pagination"]["current_page"] == 1

    def test_previous_on_first_page_stays_at_1(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(10),
            "current_page": 1,
            "page_size": 5,
            "last_filters": {},
        }
        payload = paginate_stored_results(soft_state, direction="previous")
        assert payload is not None
        assert payload["pagination"]["current_page"] == 1

    def test_updates_soft_state_current_page(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(10),
            "current_page": 1,
            "page_size": 5,
        }
        paginate_stored_results(soft_state, direction="next")
        assert soft_state["current_page"] == 2

    def test_updates_soft_state_option_map(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(10),
            "current_page": 1,
            "page_size": 5,
        }
        paginate_stored_results(soft_state, direction="next")
        assert "option_map" in soft_state
        assert isinstance(soft_state["option_map"], dict)

    def test_updates_active_property_options_map(self, monkeypatch):
        """paginate_stored_results sets both option_map and active_property_options_map."""
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(10),
            "current_page": 1,
            "page_size": 5,
        }
        paginate_stored_results(soft_state)
        assert soft_state.get("active_property_options_map") is not None

    def test_page_size_from_soft_state_used(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(20),
            "current_page": 1,
            "page_size": 10,  # stored page size
        }
        payload = paginate_stored_results(soft_state, direction="next")
        assert payload is not None
        assert payload["pagination"]["page_size"] == 10

    def test_invalid_page_size_in_soft_state_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(10),
            "current_page": 1,
            "page_size": 0,  # invalid
        }
        payload = paginate_stored_results(soft_state, direction="next")
        assert payload is not None
        # Falls back to default page size of 5
        assert payload["pagination"]["page_size"] == 5

    def test_default_direction_is_next(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(10),
            "current_page": 1,
            "page_size": 5,
        }
        payload = paginate_stored_results(soft_state)  # no direction arg
        assert payload is not None
        assert payload["pagination"]["current_page"] == 2

    def test_source_is_memory(self, monkeypatch):
        monkeypatch.setattr(search_module.cfg, "page_size", 5)
        monkeypatch.setattr(search_module.cfg, "page_size_max", 25)
        soft_state = {
            "all_search_results": _make_results(10),
            "current_page": 1,
            "page_size": 5,
        }
        payload = paginate_stored_results(soft_state, direction="next")
        assert payload is not None
        assert payload.get("source") is not None


# ---------------------------------------------------------------------------
# Regression: duplicate option_map lookup removed in _resolve_property_id_from_selection
# ---------------------------------------------------------------------------


class TestResolvePropertyIdRegression:
    """Verify the PR fix: duplicate soft_state.get('option_map') was removed."""

    def test_first_option_map_lookup_succeeds(self):
        """option_map lookup must work after removing the duplicate line."""
        soft_state = {"option_map": {"2": {"property_id": "prop-2"}}}
        result = _resolve_property_id_from_selection(2, soft_state, None)
        assert result == "prop-2"

    def test_both_maps_in_state_option_map_wins(self):
        soft_state = {
            "option_map": {"1": {"property_id": "from-option-map"}},
            "active_property_options_map": {"1": {"property_id": "from-legacy"}},
        }
        assert _resolve_property_id_from_selection(1, soft_state, None) == "from-option-map"

    def test_only_legacy_map_in_state_legacy_used(self):
        soft_state = {"active_property_options_map": {"7": {"property_id": "legacy-7"}}}
        assert _resolve_property_id_from_selection(7, soft_state, None) == "legacy-7"