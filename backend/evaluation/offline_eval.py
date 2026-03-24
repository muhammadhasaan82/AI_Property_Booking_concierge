#!/usr/bin/env python3
"""
Offline Evaluation — measures intent accuracy and response quality
against a fixed golden set without hitting live APIs.

Usage:
    python evaluation/offline_eval.py
    python evaluation/offline_eval.py --output eval_results/run_001.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

GOLDEN_SET: List[Dict[str, Any]] = [
    {"input": "Find a 2-bedroom apartment in Miami under $200", "expected_intent": "property_search"},
    {"input": "What is the cancellation policy?", "expected_intent": "faq"},
    {"input": "Book property #3 for John Smith, john@example.com, Dec 10-15", "expected_intent": "confirmation"},
    {"input": "What is the status of booking 57015107-d414-409c-843e-b6a6b15d9b59?", "expected_intent": "status_update"},
    {"input": "Hello, I need help finding a place", "expected_intent": "greeting"},
    {"input": "I want to pay for my booking", "expected_intent": "payment"},
]


async def run_eval(output_path: str | None = None) -> Dict[str, Any]:
    from app.services.graph import run_chat_graph

    results = []
    correct = 0

    for case in GOLDEN_SET:
        try:
            state = await run_chat_graph(message=case["input"])
            got_intent = state.get("intent", "unknown")
            passed = got_intent == case["expected_intent"]
            if passed:
                correct += 1
            results.append({
                "input": case["input"],
                "expected": case["expected_intent"],
                "got": got_intent,
                "passed": passed,
                "reply_preview": (state.get("reply") or "")[:120],
            })
        except Exception as exc:
            results.append({
                "input": case["input"],
                "expected": case["expected_intent"],
                "got": "error",
                "passed": False,
                "error": str(exc),
            })

    accuracy = correct / len(GOLDEN_SET) if GOLDEN_SET else 0.0
    report = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(GOLDEN_SET),
        "results": results,
    }

    print(f"\nEval complete: {correct}/{len(GOLDEN_SET)} correct ({accuracy:.1%})\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['input'][:60]!r}")
        if not r["passed"]:
            print(f"         expected={r['expected']}  got={r['got']}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline intent accuracy evaluation")
    parser.add_argument("--output", default=None, help="JSON output path (optional)")
    args = parser.parse_args()
    asyncio.run(run_eval(args.output))


if __name__ == "__main__":
    main()
