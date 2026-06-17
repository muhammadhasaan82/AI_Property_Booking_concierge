from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from evaluation.eval_runner import CaseResult


def judge_enabled(mode: str) -> bool:
    return mode == "llm" and os.getenv("EVAL_ENABLE_LLM_JUDGE") == "1"


def evaluate_case_with_judge(result: CaseResult, *, mode: str = "none", timeout: float = 20.0) -> dict[str, Any]:
    if mode == "none":
        return {"judge_skipped": True, "reason": "judge disabled"}
    if not judge_enabled(mode):
        return {"judge_skipped": True, "reason": "EVAL_ENABLE_LLM_JUDGE is not set"}
    if not os.getenv("OPENAI_API_KEY"):
        return {"judge_skipped": True, "reason": "OPENAI_API_KEY is not set"}

    try:
        from openai import OpenAI
    except Exception as exc:
        return {"judge_skipped": True, "reason": f"openai package unavailable: {exc}"}

    prompt = {
        "task": "Evaluate this property-booking assistant result. Return strict JSON only.",
        "rubric": [
            "helpfulness",
            "conversational quality",
            "policy faithfulness",
            "refusal correctness",
            "matches expected user outcome",
            "no hallucinated booking confirmation",
            "no unsupported city claim",
        ],
        "case": asdict(result),
        "response_schema": {"score": 0.0, "passed": False, "reason": "", "rubric_flags": []},
    }
    try:
        client = OpenAI(timeout=timeout, max_retries=1)
        response = client.chat.completions.create(
            model=os.getenv("EVAL_LLM_JUDGE_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a strict eval judge. Output JSON only."},
                {"role": "user", "content": json.dumps(prompt, default=str)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
    except Exception as exc:
        return {"judge_skipped": True, "reason": f"judge error: {exc}"}

    score = parsed.get("score", 0.0)
    try:
        parsed["score"] = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        parsed["score"] = 0.0
    parsed["passed"] = bool(parsed.get("passed"))
    parsed["reason"] = str(parsed.get("reason") or "")
    flags = parsed.get("rubric_flags")
    parsed["rubric_flags"] = flags if isinstance(flags, list) else []
    return parsed
