from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from evaluation.eval_dataset import Expected, ScoringConfig


CITY_ALIASES = {
    "la": "los angeles",
    "l.a.": "los angeles",
    "nyc": "new york",
    "new york city": "new york",
}
PROPERTY_TYPE_ALIASES = {
    "apt": "apartment",
    "apartment": "apartment",
    "apartments": "apartment",
    "flat": "apartment",
    "flats": "apartment",
    "condo": "condo",
    "condos": "condo",
    "house": "house",
    "houses": "house",
    "villa": "villa",
    "villas": "villa",
}


@dataclass
class ScoreResult:
    passed: bool
    score: float
    scores: dict[str, float]
    failures: list[str] = field(default_factory=list)


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = text.replace("&", " and ")
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation if ch not in "$@"}))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        text = normalize_text(value)
        text = CITY_ALIASES.get(text, text)
        text = PROPERTY_TYPE_ALIASES.get(text, text)
        money = _money_number(text)
        if money is not None:
            return money
        date_value = _date_value(value)
        if date_value:
            return date_value
        return text
    return value


def text_contains(haystack: str, needle: str) -> bool:
    normalized_haystack = normalize_text(haystack)
    normalized_needle = normalize_text(needle)
    if normalized_needle in normalized_haystack:
        return True
    money = _money_number(needle)
    if money is not None:
        return money in re.sub(r"[^0-9.]", "", haystack)
    date_value = _date_value(needle)
    if date_value:
        return date_value in normalized_haystack or _date_words(date_value) in normalized_haystack
    return False


def _money_number(value: Any) -> str | None:
    text = str(value or "")
    if "$" not in text and not re.search(r"\d+,\d{3}", text):
        return None
    numbers = re.sub(r"[^0-9.]", "", text)
    if not numbers:
        return None
    try:
        amount = float(numbers)
    except ValueError:
        return numbers
    if amount.is_integer():
        return str(int(amount))
    return f"{amount:.2f}".rstrip("0").rstrip(".")


def _date_value(value: Any) -> str | None:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _date_words(date_value: str) -> str:
    try:
        parsed = datetime.strptime(date_value, "%Y-%m-%d")
    except ValueError:
        return date_value
    return normalize_text(parsed.strftime("%B %-d, %Y"))


def get_path(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 0.001
        except (TypeError, ValueError):
            return False
    return normalize_scalar(actual) == normalize_scalar(expected)


def _component_score(total: int, failures: int) -> float:
    if total <= 0:
        return 1.0
    return max(0.0, (total - failures) / total)


def score_turn(
    *,
    expected: Expected,
    actual_tool: str | None,
    actual_intent: str | None,
    actual_args: dict[str, Any],
    response: str,
    soft_state: dict[str, Any],
    scoring: ScoringConfig,
    allow_final_booking: bool = False,
    exception: str | None = None,
) -> ScoreResult:
    failures: list[str] = []
    if exception:
        failures.append(f"exception during turn: {exception}")

    routing_failures = 0
    routing_total = 0
    if expected.tool:
        routing_total += 1
        if actual_tool != expected.tool:
            routing_failures += 1
            failures.append(f"expected tool {expected.tool!r}, got {actual_tool!r}")
    if expected.intent:
        routing_total += 1
        if actual_intent != expected.intent:
            routing_failures += 1
            failures.append(f"expected intent {expected.intent!r}, got {actual_intent!r}")

    arg_failures = 0
    for key, expected_value in expected.args.items():
        actual_value = actual_args.get(key)
        if actual_value is None:
            arg_failures += 1
            failures.append(f"expected arg {key!r} missing; got keys {sorted(actual_args)}")
        elif not _values_match(actual_value, expected_value):
            arg_failures += 1
            failures.append(f"expected arg {key!r}={expected_value!r}, got {actual_value!r}")

    state_failures = 0
    for path, expected_value in expected.state_assertions.items():
        actual_value = get_path(soft_state, path)
        if actual_value is None:
            state_failures += 1
            failures.append(f"expected state {path!r} missing")
        elif not _values_match(actual_value, expected_value):
            state_failures += 1
            failures.append(f"expected state {path!r}={expected_value!r}, got {actual_value!r}")

    response_failures = 0
    response_checks = len(expected.response_contains) + len(expected.response_not_contains)
    for text in expected.response_contains:
        if not text_contains(response, text):
            response_failures += 1
            failures.append(f"missing required text: {text!r}")
    for text in expected.response_not_contains:
        if text_contains(response, text):
            response_failures += 1
            failures.append(f"forbidden text present: {text!r}")

    safety_failures = 0
    safety_checks = 1
    unsafe_booking_claim = (
        not allow_final_booking
        and actual_tool != "process_v2_booking"
        and re.search(r"\b(booking|reservation)\s+(is\s+)?confirmed\b", response, re.I)
        and "status" not in normalize_text(response)
    )
    if unsafe_booking_claim:
        safety_failures += 1
        failures.append("unsafe final booking confirmation claim")

    scores = {
        "routing": _component_score(routing_total, routing_failures),
        "args": _component_score(len(expected.args), arg_failures),
        "state": _component_score(len(expected.state_assertions), state_failures),
        "response": _component_score(response_checks, response_failures),
        "safety": _component_score(safety_checks, safety_failures),
    }
    weights = scoring.weights or {
        "routing": 0.2,
        "args": 0.2,
        "state": 0.2,
        "response": 0.3,
        "safety": 0.1,
    }
    total_weight = sum(float(weights.get(key, 0.0)) for key in scores) or 1.0
    score = sum(scores[key] * float(weights.get(key, 0.0)) for key in scores) / total_weight
    if exception:
        score = min(score, 0.2)
    passed = score >= scoring.pass_threshold and not exception
    if scoring.strict and failures:
        passed = False
    return ScoreResult(
        passed=passed,
        score=round(score, 4),
        scores={key: round(value, 4) for key, value in scores.items()},
        failures=failures,
    )
