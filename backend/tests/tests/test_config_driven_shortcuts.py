"""Regression tests: shortcut behavior + page size are config-driven, not hardcoded."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config.conversation_shortcuts_loader import (
    _ShortcutRouter,
    _matches_semantic_cues,
    _normalize,
    match_shortcut,
)
from app.services.faq_interruption import detect_resume_cue
from app.services.property_type_normalizer import normalize_property_type

_REJECTION_STATE = {
    "last_presented_view": "property_details",
    "visible_results": [{"id": "apt-1"}],
    "option_map": {"1": {"property_id": "apt-1"}},
    "all_search_results": [{"id": "apt-1"}],
}

def test_pagination_shortcut_loaded_from_yaml_triggers_when_state_present():
    """
    Verify that the pagination shortcut defined in configuration matches when the conversation state contains required pagination data.

    Calls match_shortcut with the phrase "show me more" and state containing `all_search_results`, and asserts the returned shortcut indicates `action == "paginate_results"` and `direction == "next"`.
    """
    m = match_shortcut("show me more", {"all_search_results": [1, 2, 3]})
    assert m is not None
    assert m.action == "paginate_results"
    assert m.direction == "next"


def test_shortcut_requires_state_gating():

    assert match_shortcut("show me more", {}) is None


def test_numbered_selection_shortcut_extracts_entity():
    m = match_shortcut("back to booking option 3", {"option_map": {"3": {"property_id": "x"}}})
    assert m is not None
    assert m.action == "select_property"
    assert m.selection_number == 3


def test_property_detail_rejection_shortcut_is_context_gated():
    state = {
        "last_presented_view": "property_details",
        "visible_results": [{"id": "x"}],
    }
    m = match_shortcut("no thanks", state)
    assert m is not None
    assert m.action == "return_to_previous_results"

    assert match_shortcut("no thanks", {"visible_results": [{"id": "x"}]}) is None


def test_booking_review_confirm_shortcut_loaded_from_yaml():
    state = {
        "booking_stage": "awaiting_confirmation",
        "booking_review": {"property_id": "apt-1"},
    }
    match = match_shortcut("looks good", state)
    assert match is not None
    assert match.action == "confirm_booking_review"


def test_resume_booking_shortcut_requires_active_booking_context():
    state = {
        "booking_stage": "collecting_details",
        "booking_property_id": "apt-1",
    }
    match = match_shortcut("continue booking please", state)
    assert match is not None
    assert match.action == "resume_booking_flow"

    assert match_shortcut("continue booking please", {"booking_stage": "collecting_details"}) is None
    assert match_shortcut("continue booking please", {"booking_property_id": "apt-1"}) is None


def test_renaming_or_removing_shortcut_in_yaml_changes_behavior_without_python():
   
    custom = _ShortcutRouter(
        {
            "shortcuts": {
                "pagination_next": {
                    "action": "paginate_results",
                    "direction": "next",
                    "requires_state": ["all_search_results"],
                    "examples": ["advance"],
                }
            }
        }
    )
    state = {"all_search_results": [1]}
    assert custom.match("advance", state).action == "paginate_results"
   
    assert custom.match("show me more", state) is None

    assert _ShortcutRouter({}).match("advance", state) is None


def test_property_aliases_loaded_from_taxonomy_config():
    assert normalize_property_type("flat") == "apartment"
    assert normalize_property_type("villas") == "villa"


def test_no_hardcoded_phrase_list_remains_in_adk_runner():
    src = Path("app/services/adk_runner.py").read_text(encoding="utf-8")
    assert "_extract_pagination_direction" not in src
    assert "_extract_option_selection" not in src
    assert "show me more" not in src


def test_faq_resume_cues_are_config_driven():
    assert detect_resume_cue("yes please") is True
    assert detect_resume_cue("go back") is True
    assert detect_resume_cue("refund policy") is False



@pytest.mark.parametrize(
    "raw, expected",
    [
        ("no, not this one", "no not this one"),
        ("NO, NOT THIS ONE", "no not this one"),
        ("  multiple   spaces\tand\ttabs ", "multiple spaces and tabs"),
        ("end-of-line;with:punctuation!", "end of line with punctuation"),
        ("I don't want this property", "i dont want this property"),
        ("I dont want this property", "i dont want this property"),
        ("don\u2019t want", "dont want"),
        ("hello???world", "hello world"),
    ],
)
def test_normalize_handles_punctuation_contractions_and_case(raw, expected):
    assert _normalize(raw) == expected


def test_matches_semantic_cues_is_content_free():
    cues = {"any": [["not", "this"]], "none": [["book"], ["yes"]]}
    assert _matches_semantic_cues("not this one", cues) is True
    assert _matches_semantic_cues("yes book it", cues) is False
    assert _matches_semantic_cues("something else", cues) is False
   
    assert _matches_semantic_cues("not this one", {}) is False
    assert _matches_semantic_cues("not this one", None) is False


def test_matches_semantic_cues_handles_single_token_groups():
    cues = {"any": [["skip"]], "none": []}
    assert _matches_semantic_cues("please skip this", cues) is True
    assert _matches_semantic_cues("don't skip", cues) is True
    assert _matches_semantic_cues("keep going", cues) is False




def test_property_detail_rejection_matches_semantic_cues():
    """YAML semantic cues must catch natural-language rejection variants
    without any phrase being hardcoded in Python.
    """
    for msg in (
        "no, not this one",
        "no not this one",
        "nah this is not good",
        "I don't want this property",
        "I dont want this property",
        "show me another option",
        "show me other options",
        "back to list",
        "back to menu",
        "skip this one",
        "next one",
    ):
        m = match_shortcut(msg, _REJECTION_STATE)
        assert m is not None, f"expected rejection match for: {msg!r}"
        assert m.action == "return_to_previous_results", f"wrong action for: {msg!r}"
        assert m.intent == "property_detail_rejected"


def test_semantic_cues_respect_negative_cues():
    """The 'none' group must override 'any' so confirmation phrases don't
    trigger the rejection flow.
    """
    
    for msg in (
        "yes book this one",
        "book it please",
        "yes please proceed",
        "sure, let's confirm",
    ):
        m = match_shortcut(msg, _REJECTION_STATE)

        assert m is None or m.action != "return_to_previous_results", (
            f"negative cue ignored for: {msg!r}"
        )


def test_semantic_cues_are_context_gated():
    """The same 'no' message in a different conversational context must
    not be classified as property rejection.
    """
    other_state = {
        "last_presented_view": "property_list",
        "visible_results": [{"id": "x"}],
    }
    for msg in ("no, not this one", "nah", "I don't want this property"):
        m = match_shortcut(msg, other_state)
        assert m is None or m.action != "return_to_previous_results", (
            f"context gate leaked for: {msg!r}"
        )


def test_semantic_cues_are_generic_no_property_specific_python():
    """Guard: the generic semantic-cue matcher must not contain any
    user-phrase hardcoding.  We restrict the inspection to the matcher
    function (so docstring / identifier strings like
    ``return_to_previous_results`` or ``property_details`` outside the
    matcher are out of scope) and check only exact user-phrase strings.
    """
    import inspect

    from app.config.conversation_shortcuts_loader import (
        _matches_semantic_cues as matcher,
    )

    matcher_src = inspect.getsource(matcher)


    forbidden_user_phrases = [
        "no, not this one",
        "not this one",
        "i don't want this property",
        "show me another option",
    ]
    for phrase in forbidden_user_phrases:
        assert phrase not in matcher_src, (
            f"semantic-cue matcher contains hardcoded user phrase: {phrase!r}"
        )


def test_property_rejection_shortcut_exposes_semantic_cues_in_match():
    """ShortcutMatch must surface the semantic_cues block so downstream
    consumers (and tests) can introspect which rules fired.
    """
    m = match_shortcut("no, not this one", _REJECTION_STATE)
    assert m is not None
    assert m.semantic_cues, "semantic_cues should be present in match result"
    assert any(group in m.semantic_cues.get("any", []) for group in (["no"],))
    assert any(group in m.semantic_cues.get("none", []) for group in (["book"],))
