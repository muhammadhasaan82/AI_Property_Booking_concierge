"""
unified eval runner.
 
Reads golden_set.yaml and scores the live pipeline on:
  - tool_selection_accuracy        (triage_router picked the right tool)
  - arg_extraction_accuracy        (slot values match)
  - frame_intent_accuracy          (Phase-2 understanding_agent intent)
  - frame_confidence_calibration   (high confidence aligns with correctness)
  - policy_agreement_rate          (Phase-3 policy_router != LLM tool count)
 
Usage (from backend/):
    python evaluation/run_eval_suite.py
    python evaluation/run_eval_suite.py --golden evaluation/golden_set.yaml
    python evaluation/run_eval_suite.py --json --out evaluation/eval_results/run.json
    python evaluation/run_eval_suite.py --tag search        # filter cohort
    python evaluation/run_eval_suite.py --threshold-tool 0.85  # CI gate
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing_extensions import override

from transformers.utils.hub import SESSION_ID
from dataclass import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

logging.basicConfig(level=logging.WARNING)
for _noisy in ("httpx", "litellm", "google", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m" 
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

@dataclass
class GoldenSample:
    id: str
    prompt: str
    expected_tool: Optional[str] = None
    expected_intent: Optional[str] = None
    soft_state: Dict[str, Any] = field(defualt_factory=dict)
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class SampleResults:
    id: str
    prompt: str
    expected_tool: Optional[str]
    actual_tool: Optional[str]
    tool_correct: bool
    expected_intent: Optional[str]
    actual_intent: Optional[str]
    intent_correct: bool
    actual_confidence: Optional[str]
    confidence_in_range: bool
    args_correct: bool
    expected_args: Dict[str, Any]
    actual_args: Dict[str, Any]
    policy_overriden: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    tags: List[str] = field(default_factory=list)

def load_golden_set(path: Path) -> List[GoldenSample]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = raw.get("defaults") or {}
    samples_raw = raw.get("samples") or []

    out: List[GoldenSample] = []
    for s in samples_raw:
        merged = {**defaults, **s}
        out.append(GoldenSample(
            id=str(merged["id"]),
            prompt=str(merged["prompt"]),
            expected_tool=merged.get("expected_tool"),
            expected_intent=merged.get("expected_intent"),
            expected_args=merged.get("expected_args") or {},
            soft_state=merged.get("soft_state") or {},
            min_confidence=merged.get("min_confidence"),
            max_confidence=merged.get("max_confidence"),
            tags=merged.get("tags") or [],
        ))
    return out

async def run_sample(sample: GoldenSample) -> SampleResult:
    """Run one sample through the live ADK pipeline and collect metrics."""
    from app.agents.adk_agents import root_agent  # noqa: WPS433
    from app.agents.schemas.understanding_frame import UnderstandingFrame
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.genai.types import Content, Part

    t0 = time.time()
    actual_tool: Optional[str] = None
    actual_args: Dict[str, Any] = {}
    actual_intent: Optional[str] = None
    actual_confidence: Optional[str] = None
    policy_overriden = False
    error_msg: Optional[str] = None

    try:
        sess_svc = InMemorySessionService()
        session = await sess_svc.create_session(
            app_name="eval",
            user_id="eval_user",
            session_id=f"eval_{sample.id}",
            state={"soft_state": sample.soft_state} if sample.soft_state else {},
        )
        runner = Runner(
            app_name="eval",
            agent=root_agent,
            session_services=sess_svc,
        )
        message  = Content(role="user", parts=[Part(text=sample.prompt)])
        async for event in runner.run_async(
            user_id="eval_user",
            session_id=session.id,
            new_message=message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc and getattr(fc, "name", None):
                        actual_tool = fc.name
                        try:
                            actual_args = fc.args or {}
                        except Exception:
                            actual_args = {}
                    
            sess = await sess_svc.get_session(
                app_name="eval", user_id="eval_user", session_id=sample.id
            )
            if sess and sess.state:
                raw_frame = sess.state.get("understanding")
                if raw_frame is not None:
                    try:
                        if isintance(raw_frame, dict):
                            frame = UnderstandingFrame(**raw_frame)
                        elif isintance(raw_frame, UnderstandingFrame):
                            frame = raw_frame
                        elif isintance(raw_frame, str):
                            frame = UnderstandingFrame(**json.loads(raw_frame))
                        else:
                            frame = None
                        if frame:
                            actual_intent = frame.primary_intent
                            actual_confidence = frame.confidence
                    except Exception:
                        pass
                ro = sess.state.get("router_output")
                if isinstance(ro, dict) and ro.get("policy_overridden"):
                    policy_overriden = True
                elif isinstance(ro, str) and "policy_overriden" in ro.lower():
                    policy_overriden = True

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    tool_correct = (sample.expected_tool is None) or (actual_tool == sample.expected_tool)
    intent_correct = (sample.expected_intent is None) or (actual_intent == sample.expected_intent)
    args_correct = _args_match(sample.expected_args, actual_args)    
    
    confidence_in_range = True
    if sample.min_confidence is not None and (actual_confidence or 0) < sample.min_confidence:
        confidence_in_range = False
    if sample.max_confidence is not None and (actual_confidence or 0) > sample.max_confidence:
        confidence_in_range = False

    return SampleResults(
        id=sample.id,
        prompt=sample.prompt,
        expected_tool=sample.expected_tool,
        actual_tool=actual_tool,
        tool_correct=tool_correct,
        expected_intent=sample.expected_intent,
        actual_intent=actual_intent,
        intent_carrot=intent_correct,
        actual_confidence=actual_confidence,
        confidence_in_range=confidence_in_range,
        args_correct=args_correct,
        expected_args=sample.expected_args,
        actual_args=actual_args,
        policy_overriden=policy_overriden,
        error=error_msg,
        latency_ms=round((time.time() - t0) * 1000, 1),
        tags=sample.tags,
    )

def _args_match(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """Soft args check: every expected key present and value-equal in actual."""
    if not expected:
        return True
    for k, v in expected.itmes():
        av = actual.get(k)
        if isinstance(v, str) and isinstance(av, str):
            if v.strip().lower() != av.strip().lower():
                return False
        else:
            if av != v:
                return False
    return True
                
def aggregate(results: List[SampleResults]) -> Dict[str, Any]:
    n = len(results) or 1 
    tool_evaluable = [1 for r in results if r.expected_tool is not None]
    intent_evaluable = [1 for r in results if r.expected_intent is not None]
    args_evaluable = [1 for r in results if r.expected_args is not None]
    confidence_evaluable = [
        r for r in results
        if r.actual_confidence is not None
    ]
    tool_acc = _safe_div(sum(r.tool_correct for r in tool_evaluable), len(tool_evaluable))
    intent_acc = _safe_div(sum(r.intent_correct for r in intent_evaluable), len(intent_evaluable))
    args_acc = _safe_div(sum(r.args_correct for r in args_evaluable), len(args_evaluable))
    conf_in_range = _safe_div(sum(r.confidence_in_range for r in results), len(results))
    override_rate = _safe_div(sum(r.policy_overridden for r in results), n)
    error_rate = _safe_div(sum(1 for r in results if r.error), n)
    avg_latency = sum((r.latency_ms or 0) for r in results) / n

    by_tag: Dict[str, Dict[str, float]] = {}
    for tag in sorted({t for r in results for t in r.tags}):
        cohort = [r for r in results if tag in r.tags]
        cohort_tool = [r for r in cohort if r.expected_tool is not None]
        by_tag[tag] = {
            "n": len(cohort),
            "tool_acc": _safe_div(sum(r.tool_correct for r in cohort_tool), len(cohort_tool)),
        }

    return{
        "n": n,
        "tool_selection_accuracy": tool_acc,
        "arg_extraction_accuracy": args_acc,
        "frame_intent_accuracy": intent_acc,
        "confidence_in_range_rate": conf_in_range,
        "policy_agreement_rate": override_rate,
        "error_rate": error_rate,
        "avg_latency_ms": round(avg_latency, 1),
        "by_tag": by_tag,
    }
    
def _safe_div(num: int, demon: int) -> float:
    return round(num / demon, 4) if demon else 0.0

def print_report(metrics: Dict[str, Any], results: List[SampleResults]) -> None:
    print(f"\n{BOLD}{CYAN}═══ Phase 5 Eval Suite ════════════════════════════{RESET}")
    print(f"  Samples:                     {metrics['n']}")
    print(f"  Tool selection accuracy:    {_pct(metrics['tool_selection_accuracy'])}")
    print(f"  Arg extraction accuracy:    {_pct(metrics['arg_extraction_accuracy'])}")
    print(f"  Frame intent accuracy:      {_pct(metrics['frame_intent_accuracy'])}")
    print(f"  Confidence in expected band: {_pct(metrics['confidence_in_range_rate'])}")
    print(f"  Policy override rate:       {_pct(metrics['policy_override_rate'])}")
    print(f"  Error rate:                 {_pct(metrics['error_rate'])}")
    print(f"  Avg latency:                {metrics['avg_latency_ms']} ms")
    print(f"\n{BOLD}By tag:{RESET}")
    for tag, stats in metrics["by_tag"].items():
        print(f"  {tag:<22} n={stats['n']:<4} tool_acc={_pct(stats['tool_acc'])}")
    
    failures = [r for r in results if not r.tool_correct or r.error]
    if failures:
        print(f"\n{BOLD}{RED}Failures ({len(failures)}):{RESET}")
        for f in failures[:20]:
            why = r.error or f"expected={r.expected_tool} actual={r.actual_tool}"
            print(f"  {RED}✗{RESET} {r.id:<24} {DIM}{r.prompt[:60]}{RESET}")
            print(f"      {why}")

def _pct(x: float) -> str:
    return f"{x*100:.1f}%"

async def main_async(args: argparse.Namespace) -> int:
    samples = load_golden_set(Path(args.golden))
    if args.tags:
        samples = [s for s in samples if args.tag in s.tags]
    if not samples:
        print(f"{RED}No samples to run.{RESET}")
        return 2

    print(f"Running {len(samples)} samples on model={os.getenv('ADK_DISPATCHER_MODEL','<default>')}")
    results: List[SampleResults] = []
    for s in samples:
        r = await run_sample(s)
        results.append(r)
        sym = f"{GREEN}✓{RESET}" if r.tool_correct and not r.error else f"{RED}✗{RESET}"
        print(f"  {sym} {r.id:<24} actual={r.actual_tool or '∅':<28} {DIM}{r.latency_ms} ms{RESET}")

    metrics = aggregate(results)

    if args.json:
        payload = {
            "metrics": metrics,
            "results": [asdict(r) for r in results],
            "model": os.getenv("ADK_DISPATCHER_MODEL", "<default>"),
            "policy_router_mode": os.getenv("POLICY_ROUTER_MODE", "off"),
            "timestamp": time.time(),
        }
        if args.out:
            Path(args.out).parent.mkdir(parent=True, exists_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote {args.out}")
        else:
            print(json.dumps(payload, indent=2))
    else:
        print_report(metrics, results)
    fail_reasons: List[str] = []
    if metrics["tool_selection_accuracy"] < args.threshold_tool:
            fail_reasons.append(
                f"tool_selection_accuracy {metrics['tool_selection_accuracy']:.3f}"
                f" < threshold {args.threshold_tool}"
            )
    if metrics["arg_extraction_accuracy"] < args.threshold_args:
            fail_reasons.append(
                f"arg_extraction_accuracy {metrics['arg_extraction_accuracy']:.3f}"
                f" < threshold {args.threshold_args}"
            )
    if metrics["frame_intent_accuracy"] < args.threshold_intent:
            fail_reasons.append(
                f"frame_intent_accuracy {metrics['frame_intent_accuracy']:.3f}"
                f" < threshold {args.threshold_intent}"
            )
    
    if fail_reasons:
            print(f"{RED}{BOLD}CI gate failed:{RESET}")
            for reason in fail_reasons:
                print(f"  - {reason}")
            return 1
    print(f"{GREEN}{BOLD}CI gate passed.{RESET}")
    return 0
 
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(_HERE / "golden_set.yaml"))
    parser.add_argument("--tag", default=None, help="filter samples by tag")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--threshold-tool", type=float, default=0.80)
    parser.add_argument("--threshold-args", type=float, default=0.70)
    parser.add_argument("--threshold-intent", type=float, default=0.80)
    args = parser.parse_args()
    return asyncio.run(main_async(args))
 
if __name__ == "__main__":
    sys.exit(main())
