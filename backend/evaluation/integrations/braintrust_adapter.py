from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.eval_dataset import EvalSample


def _sample_input(sample: EvalSample) -> Any:
    if sample.type == "single_turn":
        return sample.prompt
    return [turn.user for turn in sample.turns]


def _sample_expected(sample: EvalSample) -> Any:
    if sample.type == "single_turn":
        return sample.expected.__dict__
    return [turn.expected.__dict__ for turn in sample.turns]


def export_braintrust(samples: list[EvalSample], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for sample in samples:
            item = {
                "id": sample.id,
                "input": _sample_input(sample),
                "expected": _sample_expected(sample),
                "metadata": {
                    "tags": sample.tags,
                    "type": sample.type,
                    "initial_soft_state": sample.initial_soft_state,
                },
            }
            fh.write(json.dumps(item, default=str) + "\n")
