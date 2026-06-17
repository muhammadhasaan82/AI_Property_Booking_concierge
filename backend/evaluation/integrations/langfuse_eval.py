from __future__ import annotations

import os
import re
from dataclasses import asdict
from typing import Any

from evaluation.eval_runner import CaseResult


EMAIL_RE = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")
BOOKING_RE = re.compile(r"\bB(?:K|KG)-[A-Z0-9-]+\b", re.I)


def redact(value: Any) -> Any:
    if isinstance(value, str):
        text = EMAIL_RE.sub("[redacted-email]", value)
        text = PHONE_RE.sub("[redacted-phone]", text)
        text = BOOKING_RE.sub("[redacted-booking-id]", text)
        return text
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def publish_results(
    *,
    run_id: str,
    results: list[CaseResult],
    strict: bool = False,
) -> dict[str, Any]:
    if os.getenv("EVAL_LANGFUSE_ENABLED") != "1":
        return {"skipped": True, "reason": "EVAL_LANGFUSE_ENABLED is not set"}
    try:
        from app.services.observability.langfuse_observer import get_observer

        observer = get_observer()
        for result in results:
            payload = redact(asdict(result))
            trace = observer.trace(
                name="eval_case",
                user_id="eval",
                session_id=f"eval-{run_id}",
                metadata={
                    "eval_run_id": run_id,
                    "sample_id": result.id,
                    "tags": result.tags,
                    "deterministic_score": result.score,
                    "judge_score": result.judge.get("score") if isinstance(result.judge, dict) else None,
                    "passed": result.passed,
                    "latency_ms": result.latency_ms,
                    "failures": redact(result.failures),
                },
            )
            trace.update(output=payload)
            trace.end()
        return {"published": len(results)}
    except Exception as exc:
        if strict:
            raise
        return {"skipped": True, "reason": f"langfuse error: {exc}"}
