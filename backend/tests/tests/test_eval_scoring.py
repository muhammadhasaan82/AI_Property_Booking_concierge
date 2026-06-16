from __future__ import annotations

from evaluation.eval_dataset import Expected, ScoringConfig
from evaluation.scoring import score_turn


def _score(expected: Expected, **overrides):
    kwargs = {
        "expected": expected,
        "actual_tool": expected.tool,
        "actual_intent": expected.intent,
        "actual_args": dict(expected.args),
        "response": "Found condos in Los Angeles for $9,405.00.",
        "soft_state": {"last_filters": {"city": "Los Angeles"}},
        "scoring": ScoringConfig(
            weights={"routing": 0.2, "args": 0.2, "state": 0.2, "response": 0.3, "safety": 0.1},
            pass_threshold=0.8,
        ),
    }
    kwargs.update(overrides)
    return score_turn(**kwargs)


def test_scoring_full_pass_for_matching_expected_output():
    result = _score(
        Expected(
            tool="search_properties",
            args={"city": "LA", "property_type": "condo"},
            response_contains=["Los Angeles", "$9405"],
            state_assertions={"last_filters.city": "Los Angeles"},
        ),
        actual_args={"city": "Los Angeles", "property_type": "Condo"},
    )
    assert result.passed
    assert result.score == 1.0
    assert result.failures == []


def test_scoring_reports_missing_required_text():
    result = _score(Expected(response_contains=["New York"]))
    assert not result.passed
    assert any("missing required text" in failure for failure in result.failures)


def test_scoring_reports_wrong_tool_and_args():
    result = _score(
        Expected(tool="search_properties", args={"city": "New York"}),
        actual_tool="check_faq",
        actual_args={"city": "Los Angeles"},
    )
    assert not result.passed
    assert any("expected tool" in failure for failure in result.failures)
    assert any("expected arg" in failure for failure in result.failures)


def test_scoring_reports_forbidden_text():
    result = _score(
        Expected(response_not_contains=["booking confirmed"]),
        response="Your booking confirmed.",
    )
    assert not result.passed
    assert any("forbidden text present" in failure for failure in result.failures)
