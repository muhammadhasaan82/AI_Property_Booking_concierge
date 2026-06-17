from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.eval_dataset import DEFAULT_DATASET_PATH, filter_samples, load_dataset
from evaluation.eval_runner import run_evaluation_sync
from evaluation.integrations.braintrust_adapter import export_braintrust
from evaluation.integrations.langfuse_eval import publish_results
from evaluation.integrations.promptfoo_adapter import export_promptfoo
from evaluation.judge import evaluate_case_with_judge
from evaluation.report_card import (
    build_report_card,
    render_terminal_report,
    results_to_json,
    write_json,
    write_markdown,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BACKEND / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Report Card eval runner")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to golden dataset YAML")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    parser.add_argument("--out", help="Write JSON results to this path")
    parser.add_argument("--tags", help="Comma-separated tag filter")
    parser.add_argument("--ci", action="store_true", help="Use CI exit-code semantics")
    parser.add_argument("--fail-under", type=float, default=1.0, help="Minimum pass rate, expressed 0.0-1.0")
    parser.add_argument("--judge", choices=["none", "llm"], default="none", help="Optional LLM-as-a-judge")
    parser.add_argument("--require-judge", action="store_true", help="Fail if --judge llm is unavailable")
    parser.add_argument("--report-card", action="store_true", help="Write latest Markdown report")
    parser.add_argument("--dry-run-booking", action="store_true", default=True, help="Stub booking writes")
    parser.add_argument("--strict-integrations", action="store_true", help="Fail on integration publishing errors")
    parser.add_argument("--export-braintrust", help="Export golden cases as Braintrust-compatible JSONL")
    parser.add_argument("--export-promptfoo", help="Export golden cases as promptfoo-compatible YAML")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_dataset(_path(args.dataset))
    samples = filter_samples(dataset.samples, args.tags)
    if not samples:
        print(f"No samples matched tags: {args.tags}", file=sys.stderr)
        return 2

    if args.export_braintrust:
        export_braintrust(samples, _path(args.export_braintrust))
    if args.export_promptfoo:
        export_promptfoo(samples, _path(args.export_promptfoo))

    started = time.strftime("%Y%m%d-%H%M%S")
    results = run_evaluation_sync(dataset, samples, dry_run_booking=args.dry_run_booking)

    judge_unavailable = False
    if args.judge != "none":
        for result in results:
            result.judge = evaluate_case_with_judge(result, mode=args.judge)
            if result.judge.get("judge_skipped"):
                judge_unavailable = True

    report = build_report_card(results, fail_under=args.fail_under)
    payload = results_to_json(results, report)
    payload["metadata"] = {
        "dataset": str(_path(args.dataset)),
        "tags": args.tags,
        "run_id": started,
        "judge": args.judge,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(render_terminal_report(report))

    if args.out:
        write_json(_path(args.out), payload)
    if args.report_card:
        write_markdown(BACKEND / "evaluation" / "eval_results" / "latest.md", report)
    if args.out and Path(args.out).name == "latest.json":
        write_markdown(Path(_path(args.out)).with_suffix(".md"), report)

    publish = publish_results(run_id=started, results=results, strict=args.strict_integrations)
    payload["integrations"] = {"langfuse": publish}

    if args.require_judge and judge_unavailable:
        print("LLM judge was required but unavailable", file=sys.stderr)
        return 1
    if args.ci:
        return 0 if report["ci"]["decision"] == "PASS" else 1
    return 0 if report["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
