# tests/test_v2_chaos.py
"""
V2 Chaos Test Suite — Native Agentic Architecture

Feeds "messy" conversational inputs directly into run_adk_turn and inspects
which tool the triage_router triggered and what arguments it extracted.

Run from the backend/ directory:
    python -m pytest tests/test_v2_chaos.py -v
    # or standalone:
    python tests/test_v2_chaos.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap — allows running from backend/ or project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ---------------------------------------------------------------------------
# Logging: suppress noisy library output, keep test output clean
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Colours (no external deps)
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}{BOLD}PASS{RESET}"
FAIL = f"{RED}{BOLD}FAIL{RESET}"


# ═══════════════════════════════════════════════════════════════════════════
# INTERCEPT LAYER
# We monkey-patch the tool functions on the triage_router BEFORE the Runner
# executes them, so we can capture exactly what the LLM decided to call
# without hitting the database / Rust gateway / external APIs.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCapture:
    """Holds the details of a single intercepted tool call."""
    name: str
    args: Dict[str, Any]


_captured: List[ToolCapture] = []


def _make_stub(tool_name: str, return_value: dict):
    """Return an async stub that records its invocation then returns a canned response."""
    async def _stub(**kwargs):
        _captured.append(ToolCapture(name=tool_name, args=kwargs))
        return return_value
    _stub.__name__ = tool_name
    return _stub


def _install_stubs():
    """Replace every tool on the triage_router with a recording stub."""
    from app.agents import adk_agents

    stubs = {
        "search_properties": _make_stub("search_properties", {
            "status": "success",
            "total_found": 1,
            "showing": 1,
            "properties": [{"number": 1, "title": "Test Apt", "city": "New York",
                            "price_per_night": 120, "bedrooms": 2,
                            "property_type": "apartment", "id": "prop-001"}],
            "instruction": "stub response",
        }),
        "request_booking_details": _make_stub("request_booking_details", {
            "status": "gathering_info",
            "instruction": "stub — asking for missing info",
        }),
        "process_v2_booking": _make_stub("process_v2_booking", {
            "status": "booking_confirmed",
            "receipt": {"booking_id": "test-uuid", "property": "Test Property",
                        "guest": "Jane Doe", "email": "jane@test.com",
                        "phone": "555-0000", "dates": "2025-10-12 to 2025-10-15",
                        "nights": 3, "guests": 1,
                        "price_per_night": "$100.00", "total": "$300.00"},
            "instruction": "stub response",
        }),
        "check_faq": _make_stub("check_faq", {
            "status": "answered", "answer": "stub faq answer", "source": "stub",
        }),
        "check_booking_status": _make_stub("check_booking_status", {
            "status": "found", "booking_status": "confirmed",
        }),
        "escalate_to_human": _make_stub("escalate_to_human", {
            "status": "handoff", "message": "stub escalation",
        }),
        "get_all_available_cities": _make_stub("get_all_available_cities", {
            "status": "success", "cities_list": "Dubai, London, New York",
        }),
    }

    # Patch the module-level functions so ADK picks them up via the agent's tool list
    for func_name, stub in stubs.items():
        if hasattr(adk_agents, func_name):
            setattr(adk_agents, func_name, stub)

    # Also re-register tools on the live triage_router object
    # ADK resolves tools via the agent's .tools list at call-time
    router = adk_agents.triage_router
    new_tools = []
    for t in router.tools:
        raw_name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if raw_name in stubs:
            new_tools.append(stubs[raw_name])
        else:
            new_tools.append(t)
    router.tools = new_tools


# ═══════════════════════════════════════════════════════════════════════════
# TEST CASE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChaosTestCase:
    """Declarative test case for a single chaos input."""
    name: str
    user_message: str
    expected_tool: str
    expected_args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


CHAOS_CASES: List[ChaosTestCase] = [
    # ── Test 1: Alias resolution ──────────────────────────────────────────
    ChaosTestCase(
        name="alias_apt_nyc_wifi",
        description="Alias Test — 'apt' → apartment, 'NYC' → New York",
        user_message="Find an apt in NYC with wifi",
        expected_tool="search_properties",
        expected_args={
            "city": "new york",          # case-insensitive check
            "property_type": "apartment",
        },
    ),
    # ── Test 2: Alias — house synonym ────────────────────────────────────
    ChaosTestCase(
        name="alias_place_to_stay",
        description="Alias Test — 'place to stay' → search_properties",
        user_message="I need a place to stay in Dubai, something cheap under 100",
        expected_tool="search_properties",
        expected_args={"city": "dubai"},
    ),
    # ── Test 3: Off-Switch — missing data ────────────────────────────────
    ChaosTestCase(
        name="missing_data_partial_booking",
        description="Missing Data Test — partial intent must NOT trigger process_v2_booking",
        user_message="I want to book option 3. My name is John.",
        expected_tool="request_booking_details",
        expected_args={},
    ),
    # ── Test 4: Off-Switch — intent with only email ───────────────────────
    ChaosTestCase(
        name="missing_data_email_only",
        description="Missing Data Test — email only, dates/phone still missing",
        user_message="Book the first listing. Email: alex@test.com.",
        expected_tool="request_booking_details",
        expected_args={},
    ),
    # ── Test 5: Mind-change → full booking ───────────────────────────────
    ChaosTestCase(
        name="mind_change_full_booking",
        description="Mind-Change Test — name updated mid-sentence, all fields present → process_v2_booking",
        user_message=(
            "Book option 1. Name is John. "
            "Actually, put it under Jane Doe. "
            "Email is jane@test.com. Phone 555-1234. "
            "Arriving Oct 12 2025, leaving Oct 15 2025. Just me."
        ),
        expected_tool="process_v2_booking",
        expected_args={
            "guest_name": "jane doe",     # case-insensitive
            "guest_email": "jane@test.com",
            "guests": 1,
        },
    ),
    # ── Test 6: FAQ intent ────────────────────────────────────────────────
    ChaosTestCase(
        name="faq_cancellation_policy",
        description="FAQ Test — cancellation policy question",
        user_message="What is your cancellation policy?",
        expected_tool="check_faq",
        expected_args={},
    ),
    # ── Test 7: Booking status lookup ────────────────────────────────────
    ChaosTestCase(
        name="booking_status_lookup",
        description="Status Test — user provides booking ID",
        user_message="Can you check my booking? ID is abc-123-xyz",
        expected_tool="check_booking_status",
        expected_args={},
    ),
    # ── Test 8: City list request ─────────────────────────────────────────
    ChaosTestCase(
        name="city_list_request",
        description="City List Test — user asks what cities are available",
        user_message="What cities do you have available?",
        expected_tool="get_all_available_cities",
        expected_args={},
    ),
    # ── Test 9: Escalation ───────────────────────────────────────────────
    ChaosTestCase(
        name="escalation_frustrated_user",
        description="Escalation Test — user explicitly asks for human",
        user_message="This is ridiculous. I need to speak to a real person NOW.",
        expected_tool="escalate_to_human",
        expected_args={},
    ),
    # ── Test 10: Typo + alias combo ───────────────────────────────────────
    ChaosTestCase(
        name="typo_alias_combo",
        description="Robustness Test — typos and informal phrasing",
        user_message="luk for appartments in new york with pool, max budget $200",
        expected_tool="search_properties",
        expected_args={
            "city": "new york",
            "property_type": "apartment",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════

async def _run_single_case(
    case: ChaosTestCase,
    session_offset: int,
) -> tuple[bool, str, float, Optional[ToolCapture]]:
    """
    Execute one chaos test case.

    Returns (passed, failure_reason, latency_ms, captured_call).
    """
    from app.services.adk_runner import run_adk_turn

    _captured.clear()

    user_id = f"chaos_user_{session_offset}"
    session_id = f"chaos_session_{session_offset}"

    t0 = time.monotonic()
    chunks = []
    async for chunk in run_adk_turn(user_id, session_id, case.user_message):
        chunks.append(chunk)
    latency_ms = (time.monotonic() - t0) * 1000.0

    # Inspect what was captured
    if not _captured:
        return False, "No tool was called (LLM may have responded without tool use)", latency_ms, None

    # The FIRST tool call is what we assert on (triage_router fires tools in order)
    first_call = _captured[0]

    # 1. Tool name check
    if first_call.name != case.expected_tool:
        reason = (
            f"Expected tool '{case.expected_tool}', "
            f"got '{first_call.name}' with args={json.dumps(first_call.args, default=str)}"
        )
        return False, reason, latency_ms, first_call

    # 2. Argument assertions (case-insensitive string comparison for robustness)
    for key, expected_val in case.expected_args.items():
        actual_val = first_call.args.get(key)
        if actual_val is None:
            reason = f"Arg '{key}' missing from tool call. Got args={json.dumps(first_call.args, default=str)}"
            return False, reason, latency_ms, first_call
        if isinstance(expected_val, str):
            if expected_val.lower() not in str(actual_val).lower():
                reason = (
                    f"Arg '{key}': expected to contain '{expected_val}', "
                    f"got '{actual_val}'"
                )
                return False, reason, latency_ms, first_call
        elif isinstance(expected_val, int):
            try:
                if int(actual_val) != expected_val:
                    reason = f"Arg '{key}': expected {expected_val}, got {actual_val}"
                    return False, reason, latency_ms, first_call
            except (ValueError, TypeError):
                reason = f"Arg '{key}': expected int {expected_val}, got non-int '{actual_val}'"
                return False, reason, latency_ms, first_call

    return True, "", latency_ms, first_call


async def run_chaos_suite() -> bool:
    """Run all chaos test cases and print a formatted report. Returns True if all pass."""
    print(f"\n{BOLD}{CYAN}{'═' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  V2 AGENTIC CHAOS TEST SUITE{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 65}{RESET}\n")

    # Install stubs ONCE before any test runs
    _install_stubs()

    results = []
    for idx, case in enumerate(CHAOS_CASES):
        print(f"  {CYAN}[{idx + 1:02d}/{len(CHAOS_CASES)}]{RESET} {case.description}")
        print(f"        Input : {YELLOW}\"{case.user_message[:80]}{'...' if len(case.user_message) > 80 else ''}\"{RESET}")

        passed, reason, latency_ms, capture = await _run_single_case(case, idx)

        tool_info = (
            f"{capture.name}({json.dumps({k: v for k, v in capture.args.items()}, default=str)})"
            if capture else "—"
        )
        status_str = PASS if passed else FAIL
        print(f"        Tool  : {tool_info}")
        print(f"        Result: {status_str}  ({latency_ms:.0f} ms)")
        if not passed:
            print(f"        {RED}Reason: {reason}{RESET}")
        print()

        results.append((case.name, passed, latency_ms))

    # ── Summary ──────────────────────────────────────────────────────────
    total = len(results)
    passed_count = sum(1 for _, p, _ in results if p)
    failed_count = total - passed_count
    avg_latency = sum(l for _, _, l in results) / total if total else 0

    print(f"{BOLD}{CYAN}{'─' * 65}{RESET}")
    print(f"{BOLD}  RESULTS : {passed_count}/{total} passed  |  "
          f"avg latency {avg_latency:.0f} ms{RESET}")
    if failed_count:
        print(f"\n  {RED}{BOLD}FAILED CASES:{RESET}")
        for name, passed, _ in results:
            if not passed:
                print(f"    • {name}")
    print(f"{BOLD}{CYAN}{'═' * 65}{RESET}\n")

    return failed_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# pytest INTEGRATION
# Each chaos case becomes an individual pytest test so failures are isolated.
# ═══════════════════════════════════════════════════════════════════════════

import pytest  # noqa: E402  (imported here to keep module usable without pytest)


def pytest_configure(config):
    _install_stubs()


@pytest.mark.parametrize("case,idx", [(c, i) for i, c in enumerate(CHAOS_CASES)], ids=[c.name for c in CHAOS_CASES])
@pytest.mark.asyncio
async def test_chaos_case(case: ChaosTestCase, idx: int):
    passed, reason, latency_ms, capture = await _run_single_case(case, idx + 100)
    assert passed, f"{case.name} FAILED — {reason}  (latency={latency_ms:.0f}ms, captured={capture})"


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    success = asyncio.run(run_chaos_suite())
    sys.exit(0 if success else 1)
