"""
A/B comparison of two dispatcher models on the golden set.

Usage:
    python evaluation/eval_compare_models.py \
        --baseline openai/gpt-5-nano \
        --candidate ft:openai:gpt-5-nano:org:concierge-router:abcd \
        --out evaluation/eval_results/compare.json
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
V2_EVAL = _HERE / "v2_eval.py"

def run_suite_with_model(model: str, golden: str, out_path: Path) -> dict:
    env = os.environ.copy()
    env["ADK_DISPATCHER_MODEL"] = model
    env["POLICY_ROUTER_MODE"] = env.get("POLICY_ROUTER_MODE", "shadow")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(V2_EVAL),
        "--dataset", golden,
        "--json",
        "--out", str(out_path),
        "--fail-under", "0.0",
    ]
    print(f"\n=== Running with model={model} ===")
    res = subprocess.run(cmd, env=env, cwd=_BACKEND)
    if res.returncode not in (0, 1):
        raise RuntimeError(f"v2_eval.py failed (rc={res.returncode})")
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _metrics(payload: dict) -> dict:
    report = payload.get("report_card") or {}
    return {
        "pass_rate": float(report.get("pass_rate") or 0.0),
        "average_score": float(report.get("average_score") or 0.0),
        "deterministic_score": float(report.get("deterministic_score") or 0.0),
        "failed_cases": float(report.get("failed_cases") or 0.0),
        "partial_pass_cases": float(report.get("partial_pass_cases") or 0.0),
    }

def diff_metrics(a: dict, b: dict) -> dict:
    a_metrics = _metrics(a)
    b_metrics = _metrics(b)
    keys = (
        "pass_rate",
        "average_score",
        "deterministic_score",
        "failed_cases",
        "partial_pass_cases",
    )
    return {k: round((b_metrics[k] - a_metrics[k]), 4) for k in keys}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--golden", default=str(_HERE / "golden_set.yaml"))
    parser.add_argument("--out", default=str(_HERE / "eval_results" / "compare.json"))
    args = parser.parse_args()

    out_dir = Path(args.out).parent
    a_path = out_dir / "baseline.json"
    b_path = out_dir / "candidate.json"

    a_payload = run_suite_with_model(args.baseline, args.golden, a_path)
    b_payload = run_suite_with_model(args.candidate, args.golden, b_path)

    delta = diff_metrics(a_payload, b_payload)

    summary = {
        "baseline_model": args.baseline,
        "candidate_model": args.candidate,
        "baseline_metrics": _metrics(a_payload),
        "candidate_metrics": _metrics(b_payload),
        "delta_candidate_minus_baseline": delta,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Delta candidate - baseline ===")
    for k, v in delta.items():
        sing = "+" if v >= 0 else ""
        print(f"    {k:<32} {sing}{v}")
    print(f"\nFull report: {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
