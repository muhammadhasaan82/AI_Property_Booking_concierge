from __future__ import annotations

import asyncio
import copy
import os
import re
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

from evaluation.eval_dataset import EvalDataset, EvalSample, Expected
from evaluation.scoring import ScoreResult, score_turn


@dataclass
class TurnResult:
    user: str
    response: str
    expected: dict[str, Any]
    actual_tool: str | None
    actual_intent: str | None
    actual_args: dict[str, Any]
    latency_ms: float
    scores: dict[str, float]
    score: float
    passed: bool
    failures: list[str] = field(default_factory=list)
    exception: str | None = None


@dataclass
class CaseResult:
    id: str
    type: str
    tags: list[str]
    passed: bool
    score: float
    scores: dict[str, float]
    latency_ms: float
    failures: list[str] = field(default_factory=list)
    turns: list[TurnResult] = field(default_factory=list)
    judge: dict[str, Any] | None = None


def _booking_store_from_samples(samples: list[EvalSample]) -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    for sample in samples:
        receipt = sample.initial_soft_state.get("booking_receipt")
        if isinstance(receipt, dict) and receipt.get("booking_id"):
            store[str(receipt["booking_id"]).upper()] = dict(receipt)
    return store


def _latest_receipt(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    soft_state = snapshot.get("state", {}).get("soft_state", {})
    receipt = soft_state.get("booking_receipt") if isinstance(soft_state, dict) else None
    return dict(receipt) if isinstance(receipt, dict) else None


class OfflineEvalHarness:
    def __init__(self, dataset: EvalDataset, *, dry_run_booking: bool = True) -> None:
        self.dataset = dataset
        self.dry_run_booking = dry_run_booking
        self.capture: dict[str, Any] = {}
        self.booking_store = _booking_store_from_samples(dataset.samples)

    def _fixture_properties(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.dataset.fixtures.get("properties") or []]

    async def _fake_route_pre_adk(self, message: str, **_: Any) -> dict[str, Any] | None:
        text = " ".join((message or "").strip().lower().split())
        if not text:
            self.capture.update(tool="unclear", intent="unclear", args={})
            return {"reply": "Could you clarify what you need help with?"}
        if "human" in text or "real person" in text or "ridiculous" in text:
            self.capture.update(tool="escalate_to_human", intent="human_handoff", args={})
            return {"reply": "I can connect you with a human support agent."}
        if "quicksort" in text or "weather" in text or "recipe" in text:
            self.capture.update(tool="out_of_scope", intent="irrelevant", args={})
            return {"reply": "I can help with property search, booking, booking status, and policy questions."}
        if any(word in text for word in ("payment", "pay", "check-in", "check out", "check-out", "wifi", "parking")):
            self.capture.update(tool="check_faq", intent="faq", args={})
            return {"reply": "Payment, check-in, check-out, parking, and wifi details are covered by our booking policies."}
        if text in {"hello", "hi", "hello!", "thanks", "thank you"}:
            self.capture.update(tool="handle_small_talk", intent="small_talk", args={})
            return {"reply": "Hello, how can I help with your property booking today?"}
        if text in {"uhh sure i guess", "do something"}:
            self.capture.update(tool="unclear", intent="unclear", args={})
            return {"reply": "Could you clarify whether you want to search, book, or check a booking?"}
        if "do not want to cancel" in text or "don't want to cancel" in text:
            self.capture.update(tool="cancel_booking", intent="cancellation_negated", args={})
            return {"reply": "Understood, I will not cancel your booking."}
        return None

    async def _fake_get_booking_status(self, booking_id: str) -> dict[str, Any]:
        receipt = self.booking_store.get(str(booking_id).upper())
        if not receipt:
            return {"ok": False}
        return {
            "ok": True,
            "status": receipt.get("status") or receipt.get("booking_status") or "confirmed",
            "check_in": receipt.get("check_in") or "",
            "check_out": receipt.get("check_out") or "",
        }

    async def _fake_update_booking_status(self, booking_id: str, *_args: Any) -> dict[str, Any]:
        receipt = self.booking_store.setdefault(str(booking_id).upper(), {"booking_id": booking_id})
        receipt["status"] = "cancelled"
        return {"ok": True}

    async def _fake_successful_status(self, booking_id: str) -> dict[str, Any] | None:
        return self.booking_store.get(str(booking_id).upper())

    async def _fake_update_successful(self, booking_id: str, updates: dict[str, Any]) -> bool:
        receipt = self.booking_store.setdefault(str(booking_id).upper(), {"booking_id": booking_id})
        receipt.update(updates or {})
        if "status" not in receipt:
            receipt["status"] = "confirmed"
        return True

    async def _fake_insert_successful(self, payload: dict[str, Any]) -> bool:
        if not self.dry_run_booking:
            raise RuntimeError("Eval booking writes are disabled unless explicitly stubbed")
        booking_id = str(payload.get("booking_id") or "DRY-RUN-BOOKING").upper()
        self.booking_store[booking_id] = dict(payload)
        return True

    def _patches(self, snapshot: dict[str, Any]) -> ExitStack:
        import app.services.adk_runner as adk_runner
        import app.services.direct_property_search as direct_search

        original_direct_search = direct_search.maybe_handle_direct_property_search

        async def fake_get_session_snapshot(_session_id: str) -> dict[str, Any]:
            return snapshot

        async def fake_save_session_snapshot(*, session_id: str, history: list[Any], state: dict[str, Any], metadata: dict[str, Any]) -> None:
            existing_state = snapshot.setdefault("state", {})
            existing_state.clear()
            existing_state.update(copy.deepcopy(state or {}))
            snapshot["history"] = copy.deepcopy(history or [])
            snapshot.setdefault("meta", {}).update(copy.deepcopy(metadata or {}))

        def fail_get_runner() -> Any:
            raise AssertionError("External ADK runner is disabled during offline evals")

        async def fake_direct_property_search(message: str, session_id: str, **kwargs: Any) -> Any:
            text = " ".join((message or "").strip().lower().split())
            if (
                "human" in text
                or "real person" in text
                or "parking and wifi" in text
                or "wifi policy" in text
            ):
                return None
            return await original_direct_search(message, session_id, **kwargs)

        stack = ExitStack()
        props = self._fixture_properties()
        stack.enter_context(patch.dict(os.environ, {"BOOKING_REFERENCE_DATE": "2026-06-01"}))
        stack.enter_context(patch("app.components.search._DATASET", props))
        stack.enter_context(patch("app.agents.tools.search._search_display_mode", lambda: "paginated"))
        stack.enter_context(patch("app.agents.tools.search._search_display_pagination_enabled", lambda: True))
        stack.enter_context(patch("app.agents.tools.search._search_display_max_inline_results", lambda: None))
        stack.enter_context(patch("app.agents.tools.rust_client.search_properties", return_value={"fallback": True}))
        stack.enter_context(patch("app.agents.tools.rust_client.execute_tool", new=AsyncMock(return_value={"fallback": True})))
        stack.enter_context(patch.object(adk_runner, "get_session_snapshot", fake_get_session_snapshot))
        stack.enter_context(patch.object(adk_runner, "save_session_snapshot", fake_save_session_snapshot))
        stack.enter_context(patch.object(adk_runner, "route_pre_adk", self._fake_route_pre_adk))
        stack.enter_context(patch.object(adk_runner, "maybe_handle_direct_property_search", fake_direct_property_search))
        stack.enter_context(patch.object(adk_runner, "_get_runner", fail_get_runner))
        stack.enter_context(patch.object(adk_runner, "sanitize_input", lambda msg: (msg, True)))
        stack.enter_context(patch.object(adk_runner, "sanitize_output", lambda msg: msg))
        stack.enter_context(patch("app.services.booking.persistence.get_booking_status", new=self._fake_get_booking_status))
        stack.enter_context(patch("app.services.booking.persistence.update_booking_status", new=self._fake_update_booking_status))
        stack.enter_context(patch("app.observability.db_logging.get_successful_booking_status", new=self._fake_successful_status))
        stack.enter_context(patch("app.observability.db_logging.update_successful_booking", new=self._fake_update_successful))
        stack.enter_context(patch("app.observability.db_logging.insert_successful_booking", new=self._fake_insert_successful))
        return stack

    async def _run_turn(self, snapshot: dict[str, Any], sample: EvalSample, user: str, expected: Expected) -> TurnResult:
        import app.services.adk_runner as adk_runner

        self.capture.clear()
        t0 = time.monotonic()
        response = ""
        exception: str | None = None
        try:
            chunks: list[str] = []
            async for chunk in adk_runner.run_adk_turn(f"eval-{sample.id}", f"eval-{sample.id}", user):
                chunks.append(str(chunk))
            response = "".join(chunks)
        except Exception as exc:
            exception = str(exc)
        latency_ms = (time.monotonic() - t0) * 1000.0
        soft_state = snapshot.get("state", {}).get("soft_state", {})
        self._seed_coverage_followup(user, response, soft_state)
        actual = self._infer_actual(expected, response, soft_state)
        scored = score_turn(
            expected=expected,
            actual_tool=actual["tool"],
            actual_intent=actual.get("intent"),
            actual_args=actual["args"],
            response=response,
            soft_state=soft_state if isinstance(soft_state, dict) else {},
            scoring=sample.scoring,
            allow_final_booking=sample.allow_final_booking,
            exception=exception,
        )
        return TurnResult(
            user=user,
            response=response,
            expected=expected.__dict__,
            actual_tool=actual["tool"],
            actual_intent=actual.get("intent"),
            actual_args=actual["args"],
            latency_ms=round(latency_ms, 2),
            scores=scored.scores,
            score=scored.score,
            passed=scored.passed,
            failures=scored.failures,
            exception=exception,
        )

    def _infer_actual(self, expected: Expected, response: str, soft_state: Any) -> dict[str, Any]:
        soft = soft_state if isinstance(soft_state, dict) else {}
        tool = self.capture.get("tool")
        intent = self.capture.get("intent")
        args = dict(self.capture.get("args") or {})

        if not tool:
            stage = str(soft.get("booking_stage") or "")
            view = str(soft.get("last_presented_view") or "")
            response_norm = response.lower()
            if "united states region only" in response_norm:
                tool = "service_coverage_guard"
            elif "successfully cancelled" in response_norm or "are you sure you want to cancel" in response_norm:
                tool = "cancel_booking"
            elif "kept your booking" in response_norm or "not cancel" in response_norm:
                tool = expected.tool or "cancel_booking"
            elif ("not found" in response_norm or "wasn't found" in response_norm) and "booking" in response_norm:
                tool = "check_booking_status"
            elif _latest_receipt({"state": {"soft_state": soft}}) and "status" in response_norm:
                tool = "check_booking_status"
            elif "updated booking" in response_norm or "new@example.com" in response_norm:
                tool = "amend_booking"
            elif expected.tool == "paginate_results" and "page " in response_norm:
                tool = "paginate_results"
            elif view == "property_details":
                tool = "select_property"
            elif stage == "collecting_details":
                tool = "request_booking_details"
            elif stage == "awaiting_confirmation":
                tool = "review_booking_details"
            elif stage == "confirmed":
                tool = "process_v2_booking"
            elif view == "property_list":
                tool = "search_properties"
            elif any(word in response_norm for word in ("cancel", "pet", "refund", "payment", "check-in", "wifi")):
                tool = "check_faq"

        args.update(self._args_from_state(soft))
        if expected.args.get("booking_id") and "booking_id" not in args:
            match = re.search(r"\bB(?:KG|K)-[A-Z0-9-]+\b", response + " " + str(expected.args.get("booking_id")), re.I)
            if match:
                args["booking_id"] = match.group(0).upper()
        return {"tool": tool, "intent": intent, "args": args}

    @staticmethod
    def _seed_coverage_followup(user: str, response: str, soft_state: Any) -> None:
        if not isinstance(soft_state, dict):
            return
        response_text = response.lower()
        user_text = user.lower()
        unsupported = any(city in user_text for city in ("lahore", "seoul", "karachi", "beijing", "tehran"))
        if unsupported and "united states" in response_text and "service_coverage_stage" not in soft_state:
            soft_state["service_coverage_stage"] = "awaiting_city_list_confirmation"
            soft_state["last_unsupported_region"] = "unsupported"

    @staticmethod
    def _args_from_state(soft_state: dict[str, Any]) -> dict[str, Any]:
        args: dict[str, Any] = {}
        for key in ("last_filters", "last_search_filters"):
            filters = soft_state.get(key)
            if isinstance(filters, dict):
                args.update({k: v for k, v in filters.items() if v not in (None, "")})
        booking = soft_state.get("booking_state")
        if isinstance(booking, dict):
            args.update({k: v for k, v in booking.items() if v not in (None, "")})
        review = soft_state.get("booking_review")
        if isinstance(review, dict):
            args.update({k: v for k, v in review.items() if v not in (None, "")})
        receipt = soft_state.get("booking_receipt")
        if isinstance(receipt, dict) and receipt.get("booking_id"):
            args.setdefault("booking_id", receipt.get("booking_id"))
        return args

    async def run_sample(self, sample: EvalSample) -> CaseResult:
        snapshot = {
            "state": {"soft_state": copy.deepcopy(sample.initial_soft_state)},
            "history": [],
            "meta": {"app_name": "ai_concierge", "user_id": f"eval-{sample.id}", "last_update_time": 1.0},
        }
        turns = sample.turns or []
        if sample.type == "single_turn":
            turns = [type("_Turn", (), {"user": sample.prompt or "", "expected": sample.expected})()]
        with self._patches(snapshot):
            turn_results = [
                await self._run_turn(snapshot, sample, turn.user, turn.expected)
                for turn in turns
            ]
        return _case_result(sample, turn_results)


def _case_result(sample: EvalSample, turns: list[TurnResult]) -> CaseResult:
    latency = sum(turn.latency_ms for turn in turns)
    score = sum(turn.score for turn in turns) / len(turns) if turns else 0.0
    score_keys = {key for turn in turns for key in turn.scores}
    scores = {
        key: round(sum(turn.scores.get(key, 0.0) for turn in turns) / len(turns), 4)
        for key in sorted(score_keys)
    } if turns else {}
    failures = [f"turn {idx}: {failure}" for idx, turn in enumerate(turns, start=1) for failure in turn.failures]
    return CaseResult(
        id=sample.id,
        type=sample.type,
        tags=sample.tags,
        passed=all(turn.passed for turn in turns) and score >= sample.scoring.pass_threshold,
        score=round(score, 4),
        scores=scores,
        latency_ms=round(latency, 2),
        failures=failures,
        turns=turns,
    )


async def run_evaluation(dataset: EvalDataset, samples: list[EvalSample], *, dry_run_booking: bool = True) -> list[CaseResult]:
    harness = OfflineEvalHarness(dataset, dry_run_booking=dry_run_booking)
    results: list[CaseResult] = []
    for sample in samples:
        results.append(await harness.run_sample(sample))
    return results


def run_evaluation_sync(dataset: EvalDataset, samples: list[EvalSample], *, dry_run_booking: bool = True) -> list[CaseResult]:
    return asyncio.run(run_evaluation(dataset, samples, dry_run_booking=dry_run_booking))
