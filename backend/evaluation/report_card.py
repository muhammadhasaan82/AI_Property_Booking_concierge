from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.eval_runner import CaseResult


CATEGORY_TAGS = [
    "search",
    "booking",
    "service_coverage",
    "faq",
    "cancellation",
    "amendment",
    "status",
    "safety",
    "robustness",
]


def _percent(value: float) -> float:
    return round(value * 100.0, 1)


def _latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "mean": round(statistics.mean(values), 2),
        "p50": round(statistics.median(values), 2),
        "p95": round(ordered[p95_index], 2),
        "max": round(max(values), 2),
    }


def build_report_card(results: list[CaseResult], *, fail_under: float = 1.0) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    average_score = statistics.mean([result.score for result in results]) if results else 0.0
    deterministic_score = average_score
    judge_scores = [
        float(result.judge["score"])
        for result in results
        if isinstance(result.judge, dict) and result.judge.get("score") is not None
    ]

    per_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORY_TAGS:
        subset = [result for result in results if category in result.tags]
        if not subset:
            per_category[category] = {"total": 0, "pass_rate": 0.0, "average_score": 0.0}
            continue
        per_category[category] = {
            "total": len(subset),
            "pass_rate": _percent(sum(1 for result in subset if result.passed) / len(subset)),
            "average_score": round(statistics.mean([result.score for result in subset]), 4),
        }

    pass_rate = passed / total if total else 0.0
    failures = [
        {
            "id": result.id,
            "score": result.score,
            "tags": result.tags,
            "failures": result.failures,
        }
        for result in results
        if not result.passed
    ]
    slowest = sorted(
        [{"id": result.id, "latency_ms": result.latency_ms, "score": result.score} for result in results],
        key=lambda item: item["latency_ms"],
        reverse=True,
    )[:10]
    top_regressions = sorted(failures, key=lambda item: item["score"])[:10]
    ci_pass = pass_rate >= fail_under and not failures
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": _percent(pass_rate),
        "average_score": round(average_score, 4),
        "deterministic_score": round(deterministic_score, 4),
        "llm_judge_score": round(statistics.mean(judge_scores), 4) if judge_scores else None,
        "per_category": per_category,
        "latency_ms": _latency([result.latency_ms for result in results]),
        "failures": failures,
        "top_regressions": top_regressions,
        "slowest_cases": slowest,
        "ci": {
            "decision": "PASS" if ci_pass else "FAIL",
            "fail_under": fail_under,
        },
    }


def results_to_json(results: list[CaseResult], report_card: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_card": report_card,
        "results": [asdict(result) for result in results],
    }


def render_terminal_report(report: dict[str, Any]) -> str:
    lines = [
        "AI Report Card",
        "==============",
        f"Cases: {report['passed_cases']}/{report['total_cases']} passed ({report['pass_rate']}%)",
        f"Average score: {report['average_score']:.4f}",
        f"Deterministic score: {report['deterministic_score']:.4f}",
        f"LLM judge score: {report['llm_judge_score'] if report['llm_judge_score'] is not None else 'skipped'}",
        (
            "Latency ms: "
            f"mean={report['latency_ms']['mean']} "
            f"p50={report['latency_ms']['p50']} "
            f"p95={report['latency_ms']['p95']} "
            f"max={report['latency_ms']['max']}"
        ),
        f"CI decision: {report['ci']['decision']} (fail-under {report['ci']['fail_under']:.2f})",
        "",
        "Category Scores",
    ]
    for tag, stats in report["per_category"].items():
        if stats["total"]:
            lines.append(
                f"- {tag}: {stats['pass_rate']}% pass, avg {stats['average_score']:.4f} ({stats['total']} cases)"
            )
    lines.append("")
    if report["failures"]:
        lines.append("Failures")
        for failure in report["failures"][:20]:
            first_reason = failure["failures"][0] if failure["failures"] else "unknown failure"
            lines.append(f"- {failure['id']} score={failure['score']:.4f}: {first_reason}")
    else:
        lines.append("Failures: none")
    lines.append("")
    lines.append("Slowest Cases")
    for item in report["slowest_cases"][:5]:
        lines.append(f"- {item['id']}: {item['latency_ms']} ms")
    return "\n".join(lines) + "\n"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# AI Report Card",
        "",
        f"- Cases: {report['passed_cases']}/{report['total_cases']} passed ({report['pass_rate']}%)",
        f"- Average score: `{report['average_score']:.4f}`",
        f"- Deterministic score: `{report['deterministic_score']:.4f}`",
        f"- LLM judge score: `{report['llm_judge_score'] if report['llm_judge_score'] is not None else 'skipped'}`",
        f"- CI decision: **{report['ci']['decision']}**",
        "",
        "## Categories",
        "",
        "| Category | Cases | Pass Rate | Avg Score |",
        "| --- | ---: | ---: | ---: |",
    ]
    for tag, stats in report["per_category"].items():
        lines.append(f"| {tag} | {stats['total']} | {stats['pass_rate']}% | {stats['average_score']:.4f} |")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['id']}` score `{failure['score']:.4f}`: {'; '.join(failure['failures'])}")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown_report(report), encoding="utf-8")
