"""Regression tests: shortcut behavior + page size are config-driven, not hardcoded."""
from __future__ import annotations

from pathlib import Path

from app.config.conversation_shortcuts_loader import _ShortcutRouter, match_shortcut
from app.services.property_type_normalizer import normalize_property_type


def test_pagination_shortcut_loaded_from_yaml_triggers_when_state_present():
    m = match_shortcut("show me more", {"all_search_results": [1, 2, 3]})
    assert m is not None
    assert m.action == "paginate_results"
    assert m.direction == "next"


def test_shortcut_requires_state_gating():
    # No shortlist in state → must fall through to ADK/LLM (None).
    assert match_shortcut("show me more", {}) is None


def test_numbered_selection_shortcut_extracts_entity():
    m = match_shortcut("back to booking option 3", {"option_map": {"3": {"property_id": "x"}}})
    assert m is not None
    assert m.action == "select_property"
    assert m.selection_number == 3


def test_renaming_or_removing_shortcut_in_yaml_changes_behavior_without_python():
    # Build a router from a custom dict (simulates an edited YAML) — no Python change.
    custom = _ShortcutRouter(
        {
            "shortcuts": {
                "pagination_next": {
                    "action": "paginate_results",
                    "direction": "next",
                    "requires_state": ["all_search_results"],
                    "examples": ["advance"],  # renamed example set
                }
            }
        }
    )
    state = {"all_search_results": [1]}
    assert custom.match("advance", state).action == "paginate_results"
    # Old phrase no longer configured → no match.
    assert custom.match("show me more", state) is None
    # Empty config → nothing matches at all.
    assert _ShortcutRouter({}).match("advance", state) is None


def test_property_aliases_loaded_from_taxonomy_config():
    assert normalize_property_type("flat") == "apartment"
    assert normalize_property_type("villas") == "villa"


def test_no_hardcoded_phrase_list_remains_in_adk_runner():
    src = Path("app/services/adk_runner.py").read_text(encoding="utf-8")
    assert "_extract_pagination_direction" not in src
    assert "_extract_option_selection" not in src
    assert "show me more" not in src