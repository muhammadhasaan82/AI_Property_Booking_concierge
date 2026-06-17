from __future__ import annotations

"""Compatibility wrapper for the v2 AI Report Card evaluator.

The production eval path is ``evaluation/v2_eval.py``. This module exists so
older automation that still calls this entrypoint lands on the same YAML
golden dataset and deterministic report-card scoring.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation import v2_eval


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for evaluation/v2_eval.py")
    parser.add_argument("--golden", "--dataset", dest="dataset", default=str(HERE / "golden_set.yaml"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--tag", "--tags", dest="tags")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--fail-under", type=float, default=0.85)
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))
    forwarded = [
        "v2_eval.py",
        "--dataset",
        str(args.dataset),
        "--fail-under",
        str(args.fail_under),
    ]
    if args.json:
        forwarded.append("--json")
    if args.out:
        forwarded.extend(["--out", args.out])
    if args.tags:
        forwarded.extend(["--tags", args.tags])
    if args.ci:
        forwarded.append("--ci")

    original_argv = sys.argv
    try:
        sys.argv = forwarded
        return v2_eval.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
