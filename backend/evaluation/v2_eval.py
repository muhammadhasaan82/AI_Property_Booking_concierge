# evaluation/v2_eval.py
"""
V2 Evaluation Framework — Replaces legacy offline_eval.py

Evaluates the triage_router on two axes only:
  1. Tool Selection Accuracy  — did it call the right tool?
  2. Argument Extraction Accuracy — did it extract the right city / dates / type?

Runs an async batch loop against the live triage_router (stubs out downstream
I/O so no DB or Rust gateway is needed).

Usage (from backend/ directory):
    python evaluation/v2_eval.py
    python evaluation/v2_eval.py --json            # machine-readable output
    python evaluation/v2_eval.py --out eval_results/run.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

logging.basicConfig(level=logging.WARNING)
for _noisy in ("httpx", "litellm", "google", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════
# EVAL DATASET  (20 labelled prompts)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    """A single labelled evaluation sample."""
    id: str
    prompt: str
    expected_tool: str
    expected_args: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


EVAL_DATASET: List[EvalSample] = [
    # ── Property Search ───────────────────────────────────────────────────
    EvalSample(
        id="search_01",
        prompt="Find apartments in New York under $150",
        expected_tool="search_properties",
        expected_args={"city": "new york", "property_type": "apartment"},
        tags=["search", "alias"],
    ),
    EvalSample(
        id="search_02",
        prompt="Show me villas in Dubai with a pool",
        expected_tool="search_properties",
        expected_args={"city": "dubai", "property_type": "villa"},
        tags=["search"],
    ),
    EvalSample(
        id="search_03",
        prompt="I need an apt in NYC with wifi",
        expected_tool="search_properties",
        expected_args={"city": "new york", "property_type": "apartment"},
        tags=["search", "alias", "abbreviation"],
    ),
    EvalSample(
        id="search_04",
        prompt="Any 2 bedroom places in London for under 200 dollars?",
        expected_tool="search_properties",
        expected_args={"city": "london", "beds": 2},
        tags=["search", "beds"],
    ),
    EvalSample(
        id="search_05",
        prompt="luk for appartments in new york with pool max $200",
        expected_tool="search_properties",
        expected_args={"city": "new york", "property_type": "apartment"},
        tags=["search", "typo", "robustness"],
    ),
    EvalSample(
        id="search_06",
        prompt="I want a condo in Chicago, parking included",
        expected_tool="search_properties",
        expected_args={"city": "chicago", "property_type": "condo"},
        tags=["search"],
    ),
    # ── FAQ ───────────────────────────────────────────────────────────────
    EvalSample(
        id="faq_01",
        prompt="What is your cancellation policy?",
        expected_tool="check_faq",
        expected_args={},
        tags=["faq"],
    ),
    EvalSample(
        id="faq_02",
        prompt="Do you allow pets?",
        expected_tool="check_faq",
        expected_args={},
        tags=["faq"],
    ),
    EvalSample(
        id="faq_03",
        prompt="What time is check-in?",
        expected_tool="check_faq",
        expected_args={},
        tags=["faq"],
    ),
    EvalSample(
        id="faq_04",
        prompt="How do I pay? Do you take credit cards?",
        expected_tool="check_faq",
        expected_args={},
        tags=["faq", "payment"],
    ),
    # ── Booking — Off-Switch (missing data) ───────────────────────────────
    EvalSample(
        id="booking_off_01",
        prompt="I want to book option 3. My name is John.",
        expected_tool="request_booking_details",
        expected_args={},
        tags=["booking", "off_switch", "missing_data"],
    ),
    EvalSample(
        id="booking_off_02",
        prompt="Book the first listing. Email: alex@test.com.",
        expected_tool="request_booking_details",
        expected_args={},
        tags=["booking", "off_switch", "missing_data"],
    ),
    EvalSample(
        id="booking_off_03",
        prompt="Reserve option 2 for me please",
        expected_tool="request_booking_details",
        expected_args={},
        tags=["booking", "off_switch"],
    ),
    # ── Booking — Full (process_v2_booking) ───────────────────────────────
    EvalSample(
        id="booking_full_01",
        prompt=(
            "Book option 1 for Jane Doe, jane@test.com, phone 555-1234, "
            "check-in 2025-10-12, check-out 2025-10-15, 1 guest."
        ),
        expected_tool="process_v2_booking",
        expected_args={"guest_name": "jane doe", "guest_email": "jane@test.com", "guests": 1},
        tags=["booking", "full"],
    ),
    EvalSample(
        id="booking_full_02",
        prompt=(
            "Confirm booking: name Mike Smith, email mike@co.com, "
            "phone 555-9999, arriving 2025-11-01, leaving 2025-11-05, 2 guests."
        ),
        expected_tool="process_v2_booking",
        expected_args={"guest_name": "mike smith", "guest_email": "mike@co.com", "guests": 2},
        tags=["booking", "full"],
    ),
    # ── Booking Status ────────────────────────────────────────────────────
    EvalSample(
        id="status_01",
        prompt="Can you check my booking? ID is abc-123-xyz",
        expected_tool="check_booking_status",
        expected_args={},
        tags=["status"],
    ),
    EvalSample(
        id="status_02",
        prompt="What's the status of reservation BKG-999?",
        expected_tool="check_booking_status",
        expected_args={},
        tags=["status"],
    ),
    # ── City List ─────────────────────────────────────────────────────────
    EvalSample(
        id="cities_01",
        prompt="What cities do you have available?",
        expected_tool="get_all_available_cities",
        expected_args={},
        tags=["cities"],
    ),
    # ── Escalation ───────────────────────────────────────────────────────
    EvalSample(
        id="escalate_01",
        prompt="I need to speak to a real person NOW.",
        expected_tool="escalate_to_human",
        expected_args={},
        tags=["escalation"],
    ),
    EvalSample(
        id="escalate_02",
        prompt="None of this is working. Connect me with a human agent.",
        expected_tool="escalate_to_human",
        expected_args={},
        tags=["escalation"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# STUB LAYER — intercepts tool calls without touching real I/O
# ═══════════════════════════════════════════════════════════════════════════

_STUB_RESPONSES: Dict[str, dict] = {
    "search_properties": {
        "status": "success", "total_found": 1, "showing": 1,
        "properties": [{"number": 1, "title": "Stub Property", "city": "Stub City",
                        "price_per_night": 100, "bedrooms": 2,
                        "property_type": "apartment", "id": "stub-001"}],
        "instruction": "stub",
    },
    "check_faq": {"status": "answered", "answer": "stub faq", "source": "stub"},
    "check_booking_status": {"status": "found", "booking_status": "confirmed"},
    "request_booking_details": {"status": "gathering_info", "instruction": "stub"},
    "process_v2_booking": {
        "status": "booking_confirmed",
        "receipt": {"booking_id": "stub-uuid", "property": "Stub", "guest": "Stub",
                    "email": "stub@stub.com", "phone": "000", "dates": "n/a",
                    "nights": 1, "guests": 1, "price_per_night": "$0", "total": "$0"},
        "instruction": "stub",
    },
    "escalate_to_human": {"status": "handoff", "message": "stub"},
    "get_all_available_cities": {"status": "success", "cities_list": "Dubai, London, NYC"},
}

_last_capture: Dict[str, Any] = {}


def _make_eval_stub(tool_name: str):
    """Return an async stub that records name + args then returns a canned response."""
    async def _stub(**kwargs):
        _last_capture["name"] = tool_name
        _last_capture["args"] = dict(kwargs)
        return _STUB_RESPONSES.get(tool_name, {"status": "stub"})
    _stub.__name__ = tool_name
    return _stub


def _install_eval_stubs():
    """Patch triage_router tools for eval — runs once at startup."""
    from app.agents import adk_agents

    for fname in list(_STUB_RESPONSES.keys()):
        stub = _make_eval_stub(fname)
        if hasattr(adk_agents, fname):
            setattr(adk_agents, fname, stub)

    router = adk_agents.triage_router
    new_tools = []
    for t in router.tools:
        raw_name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if raw_name in _STUB_RESPONSES:
            new_tools.append(_make_eval_stub(raw_name))
        else:
            new_tools.append(t)
    router.tools = new_tools


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    sample_id: str
    prompt: str
    expected_tool: str
    actual_tool: Optional[str]
    tool_correct: bool
    args_correct: bool
    arg_failures: List[str]
    latency_ms: float
    tags: List[str]

    @property
    def fully_correct(self) -> bool:
        return self.tool_correct and self.args_correct


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE SAMPLE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

async def _evaluate_sample(sample: EvalSample, session_idx: int) -> EvalResult:
    from app.services.adk_runner import run_adk_turn

    _last_capture.clear()

    user_id = f"eval_user_{session_idx}"
    session_id = f"eval_session_{session_idx}"

    t0 = time.monotonic()
    async for _ in run_adk_turn(user_id, session_id, sample.prompt):
        pass
    latency_ms = (time.monotonic() - t0) * 1000.0

    actual_tool = _last_capture.get("name")
    actual_args = _last_capture.get("args", {})

    tool_correct = actual_tool == sample.expected_tool

    # Arg extraction check — only if tool was right
    arg_failures: List[str] = []
    if tool_correct and sample.expected_args:
        for key, expected in sample.expected_args.items():
            actual = actual_args.get(key)
            if actual is None:
                arg_failures.append(f"'{key}' missing (got keys: {list(actual_args.keys())})")
            elif isinstance(expected, str):
                if expected.lower() not in str(actual).lower():
                    arg_failures.append(f"'{key}': expected '{expected}', got '{actual}'")
            elif isinstance(expected, int):
                try:
                    if int(actual) != expected:
                        arg_failures.append(f"'{key}': expected {expected}, got {actual}")
                except (ValueError, TypeError):
                    arg_failures.append(f"'{key}': expected int {expected}, got '{actual}'")

    args_correct = len(arg_failures) == 0

    return EvalResult(
        sample_id=sample.id,
        prompt=sample.prompt,
        expected_tool=sample.expected_tool,
        actual_tool=actual_tool,
        tool_correct=tool_correct,
        args_correct=args_correct,
        arg_failures=arg_failures,
        latency_ms=latency_ms,
        tags=sample.tags,
    )


# ═══════════════════════════════════════════════════════════════════════════
# BATCH EVALUATOR (async loop — no concurrency to avoid session collisions)
# ═══════════════════════════════════════════════════════════════════════════

async def run_evaluation(dataset: List[EvalSample]) -> List[EvalResult]:
    _install_eval_stubs()
    results: List[EvalResult] = []
    for idx, sample in enumerate(dataset):
        result = await _evaluate_sample(sample, idx)
        results.append(result)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# METRICS CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[EvalResult]) -> Dict[str, Any]:
    total = len(results)
    tool_correct = sum(1 for r in results if r.tool_correct)
    args_correct = sum(1 for r in results if r.args_correct)
    fully_correct = sum(1 for r in results if r.fully_correct)
    latencies = [r.latency_ms for r in results]

    # Per-tag breakdown
    tag_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        for tag in r.tags:
            if tag not in tag_stats:
                tag_stats[tag] = {"total": 0, "tool_pass": 0, "full_pass": 0}
            tag_stats[tag]["total"] += 1
            if r.tool_correct:
                tag_stats[tag]["tool_pass"] += 1
            if r.fully_correct:
                tag_stats[tag]["full_pass"] += 1

    return {
        "total_samples": total,
        "tool_selection_accuracy": round(tool_correct / total * 100, 1) if total else 0,
        "arg_extraction_accuracy": round(args_correct / total * 100, 1) if total else 0,
        "full_pass_rate": round(fully_correct / total * 100, 1) if total else 0,
        "tool_correct": tool_correct,
        "args_correct": args_correct,
        "fully_correct": fully_correct,
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "min": round(min(latencies), 1) if latencies else 0,
            "max": round(max(latencies), 1) if latencies else 0,
            "p50": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
        },
        "per_tag": tag_stats,
        "failures": [
            {
                "id": r.sample_id,
                "expected": r.expected_tool,
                "actual": r.actual_tool,
                "arg_failures": r.arg_failures,
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in results
            if not r.fully_correct
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# TERMINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════

def _pct_colour(pct: float) -> str:
    if pct >= 90:
        return GREEN
    if pct >= 70:
        return YELLOW
    return RED


def print_report(results: List[EvalResult], metrics: Dict[str, Any]) -> None:
    W = 68

    print(f"\n{BOLD}{CYAN}{'═' * W}{RESET}")
    print(f"{BOLD}{CYAN}  V2 EVALUATION REPORT  —  {len(results)} samples{RESET}")
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}\n")

    # ── Per-sample table ──────────────────────────────────────────────────
    col_id      = 14
    col_expect  = 28
    col_actual  = 28
    col_ms      = 8

    header = (
        f"  {'ID':<{col_id}} {'EXPECTED':<{col_expect}} "
        f"{'ACTUAL':<{col_actual}} {'ms':>{col_ms}}  STATUS"
    )
    print(f"{DIM}{header}{RESET}")
    print(f"{DIM}  {'─' * (col_id)} {'─' * col_expect} {'─' * col_actual} {'─' * col_ms}  {'──────'}{RESET}")

    for r in results:
        status = f"{GREEN}✓ PASS{RESET}" if r.fully_correct else f"{RED}✗ FAIL{RESET}"
        actual_str = r.actual_tool or "—"
        actual_col = GREEN if r.tool_correct else RED
        print(
            f"  {r.sample_id:<{col_id}} "
            f"{r.expected_tool:<{col_expect}} "
            f"{actual_col}{actual_str:<{col_actual}}{RESET} "
            f"{r.latency_ms:>{col_ms}.0f}  {status}"
        )
        if r.arg_failures:
            for af in r.arg_failures:
                print(f"  {' ' * col_id}   {YELLOW}↳ arg: {af}{RESET}")

    # ── Aggregate metrics ─────────────────────────────────────────────────
    tool_pct = metrics["tool_selection_accuracy"]
    arg_pct  = metrics["arg_extraction_accuracy"]
    full_pct = metrics["full_pass_rate"]
    lat      = metrics["latency_ms"]

    print(f"\n{BOLD}{CYAN}{'─' * W}{RESET}")
    print(f"{BOLD}  METRICS{RESET}\n")
    print(f"  Tool Selection Accuracy  : "
          f"{_pct_colour(tool_pct)}{BOLD}{tool_pct:>5.1f}%{RESET}  "
          f"({metrics['tool_correct']}/{metrics['total_samples']})")
    print(f"  Argument Extraction Acc. : "
          f"{_pct_colour(arg_pct)}{BOLD}{arg_pct:>5.1f}%{RESET}  "
          f"({metrics['args_correct']}/{metrics['total_samples']})")
    print(f"  Full Pass Rate           : "
          f"{_pct_colour(full_pct)}{BOLD}{full_pct:>5.1f}%{RESET}  "
          f"({metrics['fully_correct']}/{metrics['total_samples']})")
    print()
    print(f"  Latency  mean={lat['mean']} ms  "
          f"min={lat['min']} ms  max={lat['max']} ms  "
          f"p50={lat['p50']} ms  p95={lat['p95']} ms")

    # ── Per-tag breakdown ─────────────────────────────────────────────────
    if metrics["per_tag"]:
        print(f"\n{BOLD}  PER-TAG BREAKDOWN{RESET}")
        for tag, stats in sorted(metrics["per_tag"].items()):
            t = stats["total"]
            tp = stats["tool_pass"]
            fp = stats["full_pass"]
            pct = tp / t * 100 if t else 0
            bar_filled = int(pct / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(f"  {tag:<18} [{_pct_colour(pct)}{bar}{RESET}] "
                  f"{_pct_colour(pct)}{tp}/{t} tool-correct{RESET}  "
                  f"{fp}/{t} full-pass")

    # ── Failure detail ────────────────────────────────────────────────────
    failures = metrics["failures"]
    if failures:
        print(f"\n{BOLD}  FAILURES ({len(failures)}){RESET}")
        for f in failures:
            print(f"  • {RED}{f['id']}{RESET}  expected={f['expected']}  "
                  f"actual={f['actual'] or '—'}  ({f['latency_ms']} ms)")
            for af in f["arg_failures"]:
                print(f"    {YELLOW}↳ {af}{RESET}")
    else:
        print(f"\n  {GREEN}{BOLD}All samples passed!{RESET}")

    print(f"{BOLD}{CYAN}{'═' * W}{RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

async def _main(args: argparse.Namespace) -> int:
    dataset = EVAL_DATASET

    if args.tags:
        filter_tags = {t.strip() for t in args.tags.split(",")}
        dataset = [s for s in dataset if set(s.tags) & filter_tags]
        if not dataset:
            print(f"{RED}No samples matched tags: {args.tags}{RESET}")
            return 1

    print(f"{CYAN}Running evaluation on {len(dataset)} samples…{RESET}")
    results = await run_evaluation(dataset)
    metrics = compute_metrics(results)

    if args.json:
        output = {
            "metrics": metrics,
            "results": [asdict(r) for r in results],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_report(results, metrics)

    if args.out:
        out_path = os.path.join(_HERE, args.out) if not os.path.isabs(args.out) else args.out
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"metrics": metrics, "results": [asdict(r) for r in results]},
                fh, indent=2, default=str,
            )
        print(f"{GREEN}Results saved → {out_path}{RESET}")

    fully_correct = metrics["fully_correct"]
    total = metrics["total_samples"]
    return 0 if fully_correct == total else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 Eval Framework — triage_router accuracy benchmark"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of human report",
    )
    parser.add_argument(
        "--out", metavar="FILE",
        help="Save JSON results to FILE (relative to evaluation/)",
    )
    parser.add_argument(
        "--tags", metavar="TAGS",
        help="Comma-separated list of tags to filter (e.g. 'search,faq')",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
