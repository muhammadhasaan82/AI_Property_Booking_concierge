# services/checkout_graph.py
"""
Deterministic Vault — Isolated LangGraph checkout/payment flow.

This module extracts the strict booking state machine from the V1 graph.py
and exposes it as a single async function that the ADK can invoke as a tool.

Flow: confirmation → booking → payment
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..agents.agents import (
    booking_agent,
    confirmation_agent,
    payment_agent,
    status_agent,
)
from ..security.guardrails import sanitize_input, sanitize_output
from .state_keys import SK
from ..observability.tracing import span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema (subset of V1 ChatState, scoped to checkout)
# ---------------------------------------------------------------------------
class CheckoutState(TypedDict, total=False):
    user_text: str
    intent: str
    filters: Dict[str, Any]
    booking_args: Dict[str, Any]
    status_args: Dict[str, Any]
    payment_args: Dict[str, Any]
    results: List[Dict[str, Any]]
    tool_result: Dict[str, Any]
    reply: str


# ---------------------------------------------------------------------------
# Graph nodes (thin wrappers around V1 agents)
# ---------------------------------------------------------------------------
async def node_confirmation(state: CheckoutState) -> CheckoutState:
    async with span("checkout_confirmation"):
        out = await confirmation_agent(
            state.get("user_text", ""),
            state.get("filters", {}) or {},
        )
        result = {**state, **out}
        tool_result = out.get("tool_result", {})
        if tool_result.get("ready_for_booking"):
            result["intent"] = "booking"
        elif tool_result.get("show_receipt"):
            result["intent"] = "confirmation"
        if result.get("intent") == "booking":
            f = {**(result.get("filters") or {})}
            for k in [
                SK.awaiting_post_mod_choice,
                SK.awaiting_post_cancel_choice,
                SK.awaiting_field,
                SK.receipt_shown,
            ]:
                f.pop(k, None)
            result["filters"] = f
        return result


async def node_booking(state: CheckoutState) -> CheckoutState:
    async with span("checkout_booking"):
        booking_args_input = state.get("booking_args", {}) or {}
        out = await booking_agent(booking_args_input)
        result = {**state, **out}

        tool_result = out.get("tool_result", {})
        if tool_result.get("clear_booking_state"):
            result["payment_args"] = {
                "booking_id": tool_result.get("booking_id", ""),
                "payment_url": tool_result.get("payment_url", ""),
                "phone": booking_args_input.get("phone", ""),
            }
            filters = result.get("filters", {})
            for key in [
                SK.selected_property,
                SK.recent_property_id,
                SK.recent_selection_index,
                "name", "phone", "email", "check_in", "check_out", "guests",
                SK.receipt_shown,
                SK.awaiting_field,
                SK.awaiting_post_mod_choice,
                SK.awaiting_post_cancel_choice,
                SK.modifying_dates,
                SK.awaiting_selection_confirm,
            ]:
                filters.pop(key, None)
            result["filters"] = filters
        return result


async def node_payment(state: CheckoutState) -> CheckoutState:
    async with span("checkout_payment"):
        out = await payment_agent(state.get("payment_args", {}) or {})
        return {**state, **out}


async def node_status(state: CheckoutState) -> CheckoutState:
    async with span("checkout_status"):
        args = {**(state.get("status_args", {}) or {})}
        user_text = state.get("user_text", "")
        if not args.get("booking_id"):
            m = re.search(
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
                user_text or "",
            )
            if m:
                args["booking_id"] = m.group(0)
        out = await status_agent(user_text, args)
        return {**state, **out}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------
def _route_after_confirmation(state: CheckoutState) -> str:
    intent = state.get("intent", "confirmation")
    tool_result = state.get("tool_result", {})
    if intent == "booking":
        return "booking"
    if tool_result.get("end"):
        return END
    return END


def _route_after_booking(state: CheckoutState) -> str:
    tool_result = state.get("tool_result", {})
    if tool_result.get("clear_booking_state") and state.get("payment_args", {}).get("payment_url"):
        return "payment"
    return END


# ---------------------------------------------------------------------------
# Build the checkout graph
# ---------------------------------------------------------------------------
def build_checkout_graph():
    g = StateGraph(CheckoutState)

    g.add_node("confirmation", node_confirmation)
    g.add_node("booking", node_booking)
    g.add_node("payment", node_payment)
    g.add_node("status", node_status)

    g.set_entry_point("confirmation")

    g.add_conditional_edges("confirmation", _route_after_confirmation, {
        "booking": "booking",
        END: END,
    })

    g.add_conditional_edges("booking", _route_after_booking, {
        "payment": "payment",
        END: END,
    })

    g.add_edge("payment", END)
    g.add_edge("status", END)

    compile_kwargs: Dict[str, Any] = {}
    try:
        if "recursion_limit" in inspect.signature(g.compile).parameters:
            compile_kwargs["recursion_limit"] = 15
    except (TypeError, ValueError):
        pass

    return g.compile(**compile_kwargs)


CHECKOUT_APP = build_checkout_graph()


# ---------------------------------------------------------------------------
# Public entry point — called by the ADK trigger_checkout_flow tool
# ---------------------------------------------------------------------------
async def run_checkout_flow(
    user_text: str,
    filters: Optional[Dict[str, Any]] = None,
    booking_args: Optional[Dict[str, Any]] = None,
    status_args: Optional[Dict[str, Any]] = None,
    payment_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the deterministic checkout flow and return the result dict.

    Returns a dict with at minimum: {"reply": str, "filters": dict, ...}
    """
    user_text, _ = sanitize_input(user_text)

    state: CheckoutState = {
        "user_text": user_text,
        "filters": filters or {},
        "booking_args": booking_args or {},
        "status_args": status_args or {},
        "payment_args": payment_args or {},
    }

    result = await CHECKOUT_APP.ainvoke(state, config={"recursion_limit": 15})

    if result.get("reply"):
        result["reply"] = sanitize_output(result["reply"])
        result["instruction"] = "Stop calling tools. Pass this exact reply to the voice agent to present to the user."

    return result
