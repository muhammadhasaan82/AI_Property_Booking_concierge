from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evaluation.eval_dataset import EvalSample, Expected


def _assertions(expected: Expected) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []
    for text in expected.response_contains:
        assertions.append({"type": "contains", "value": text})
    for text in expected.response_not_contains:
        assertions.append({"type": "not-contains", "value": text})
    if expected.args or expected.state_assertions or expected.tool:
        assertions.append(
            {
                "type": "javascript",
                "value": "output && output.length > 0",
                "metadata": {
                    "expected_tool": expected.tool,
                    "expected_args": expected.args,
                    "state_assertions": expected.state_assertions,
                },
            }
        )
    return assertions


def export_promptfoo(samples: list[EvalSample], path: str | Path) -> None:
    tests: list[dict[str, Any]] = []
    for sample in samples:
        if sample.type == "single_turn":
            tests.append(
                {
                    "description": sample.id,
                    "vars": {"prompt": sample.prompt},
                    "assert": _assertions(sample.expected),
                    "metadata": {"tags": sample.tags, "type": sample.type},
                }
            )
        else:
            tests.append(
                {
                    "description": sample.id,
                    "vars": {"turns": [turn.user for turn in sample.turns]},
                    "assert": [
                        assertion
                        for turn in sample.turns
                        for assertion in _assertions(turn.expected)
                    ],
                    "metadata": {"tags": sample.tags, "type": sample.type},
                }
            )
    payload = {
        "description": "AI Property Booking Concierge golden eval export",
        "prompts": ["{{prompt}}"],
        "tests": tests,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
