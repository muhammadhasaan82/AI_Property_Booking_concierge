# services/graph.py
from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List
import logging
import re
import inspect
from langgraph.graph import StateGraph, END
from .tracing import span
from .db_logging import log_chat
from .guardrails import sanitize_input, sanitize_output
from . import nlp_engine

logger = logging.getLogger(__name__)

from .agents import (
    triage_intent,
    greeting_agent, faq_agent, property_agent, booking_agent, status_agent, payment_agent,
    confirmation_agent
)
from .state_keys import SK

class ChatState(TypedDict, total=False):
    user_text: str
    intent: str
    filters: Dict[str, Any]
    booking_args: Dict[str, Any]
    status_args: Dict[str, Any]
    payment_args: Dict[str, Any]
    results: List[Dict[str, Any]]
    tool_result: Dict[str, Any]
    reply: str

def node_triage(state: ChatState) -> ChatState:
    with span("node_triage", {"user_text_len": len(state.get("user_text", ""))}):
        user_text = state.get("user_text", "")
        filters = state.get("filters", {}) or {}

        # --- Guardrails: sanitize input before any routing ---
        user_text, is_safe = sanitize_input(user_text)
        intent = triage_intent(user_text, filters)

        # Build execution context for policy evaluation
        ctx = {
            "intent": intent,
            "is_safe": is_safe,
            "filter_keys": set(filters.keys()),
            "has_cardinal_extraction": nlp_engine.has_cardinal_extraction(user_text),
            "has_booking_context": bool(
                filters.get(SK.recent_property_id) or
                filters.get("name") or filters.get("phone") or filters.get("email") or
                filters.get("check_in") or filters.get("check_out") or filters.get(SK.receipt_shown)
            ),
            SK.awaiting_field: filters.get(SK.awaiting_field),
            "has_field_data": False,
            "lacks_explicit_status_keywords": False,
            "no_selected_property": not filters.get(SK.selected_property),
            "no_awaiting_field": not filters.get(SK.awaiting_field),
            "no_active_selection": not any(filters.get(k) for k in [
                SK.awaiting_unavailable_city_choice,
                SK.awaiting_city_selection,
                SK.awaiting_property_type_choice,
                SK.receipt_shown,
                SK.awaiting_selection_confirm,
            ]),
        }

        # Lazy context enrichments (only evaluated if needed by policy)
        if ctx["has_booking_context"] and intent == "property_search":
            from services.agents import _parse_name, _parse_phone, _parse_email, _parse_dates, _parse_guests, _detect_requested_fields
            ctx["has_field_data"] = bool(
                _parse_name(user_text) or _parse_phone(user_text) or _parse_email(user_text) or
                _parse_dates(user_text) or _parse_guests(user_text) or _detect_requested_fields(user_text)
            )

        if ctx["has_booking_context"] and intent == "status_update":
            tl = user_text.lower()
            from services.dynamic_config import get_vocabulary
            explicit = any(
                k in tl for k in get_vocabulary().nlp_fallback.status_explicit_keywords
            )
            ctx["lacks_explicit_status_keywords"] = not explicit

        if intent == "confirmation" and ctx["no_selected_property"] and ctx["no_awaiting_field"] and ctx["no_active_selection"]:
            from services.agents import _parse_selection_index
            ctx["no_active_selection"] = _parse_selection_index(user_text) is None

        # --- Evaluate policies from dynamic_config ---
        from services.dynamic_config import get_routing_policies
        rp_config = get_routing_policies()
        
        target_route = "_from_intent"  # fallback
        reply_override = None
        extract_action = None

        for policy in rp_config.sorted_policies:
            cond = policy.condition
            match = True

            if cond.always:
                target_route = policy.route
                break

            if cond.is_safe is not None and cond.is_safe != ctx["is_safe"]: match = False
            if cond.intent is not None and cond.intent != ctx["intent"]: match = False
            if cond.intent_in is not None and ctx["intent"] not in cond.intent_in: match = False
            if cond.filter_key is not None and not filters.get(cond.filter_key): match = False
            if cond.any_filter_key is not None and not any(filters.get(k) for k in cond.any_filter_key): match = False
            if cond.has_context_key is not None and not filters.get(cond.has_context_key): match = False
            if cond.lacks_context_key is not None and filters.get(cond.lacks_context_key): match = False
            if cond.has_booking_context is not None and cond.has_booking_context != ctx["has_booking_context"]: match = False
            if cond.has_field_data is not None and cond.has_field_data != ctx["has_field_data"]: match = False
            if cond.awaiting_field_in is not None and ctx[SK.awaiting_field] not in cond.awaiting_field_in: match = False
            if cond.requires_cardinal_extraction is not None and cond.requires_cardinal_extraction != ctx["has_cardinal_extraction"]: match = False
            if cond.lacks_explicit_status_keywords is not None and cond.lacks_explicit_status_keywords != ctx["lacks_explicit_status_keywords"]: match = False
            if cond.no_selected_property is not None and cond.no_selected_property != ctx["no_selected_property"]: match = False
            if cond.no_awaiting_field is not None and cond.no_awaiting_field != ctx["no_awaiting_field"]: match = False
            if cond.no_active_selection is not None and cond.no_active_selection != ctx["no_active_selection"]: match = False

            if match:
                target_route = policy.route
                reply_override = policy.reply
                extract_action = policy.extract
                if policy.id == "faq_return_guard":
                    # Special inline action for FAQ return
                    try:
                        resume_intent = filters.get(SK.faq_resume_intent, "confirmation")
                        filters.pop(SK.faq_answered, None)
                        filters.pop(SK.faq_resume_intent, None)
                        if intent == "faq":
                            target_route = "faq"
                        elif intent in ["confirmation", "booking"] or nlp_engine.is_resume_request(user_text):
                            target_route = resume_intent
                        else:
                            target_route = "_from_intent"  # Let it fall through normally
                    except Exception:
                        logger.warning("[graph] faq_return_guard policy failed, falling through")
                        target_route = "_from_intent"
                break

        # --- Apply target route ---
        final_intent = intent if target_route == "_from_intent" else target_route
        out_state: ChatState = {**state, "intent": final_intent}
        
        if reply_override:
            out_state["reply"] = reply_override

        # Apply specific extractions requested by policy
        if extract_action == "booking_id":
            status_args = {**(state.get("status_args", {}) or {})}
            m = re.search(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", user_text or "")
            if m and not status_args.get("booking_id"):
                status_args["booking_id"] = m.group(0)
            out_state["status_args"] = status_args

        if out_state.get("filters") != filters or "filters" not in out_state:
            out_state["filters"] = filters

        return out_state
def node_greeting(state: ChatState) -> ChatState:
    with span("node_greeting"):
        out = greeting_agent(state.get("filters", {}) or {}, state.get("user_text", ""))
        return {**state, **out}

def node_faq(state: ChatState) -> ChatState:
    with span("node_faq"):
        # Prepare context for FAQ agent
        context = {}
        
        # Check if we're in the middle of a booking/property flow
        filters = state.get("filters", {})
        if filters and any(filters.get(k) for k in [SK.selected_property, "name", "phone", "email", "check_in", "check_out"]):
            context["in_booking_flow"] = True
            context["return_to"] = "confirmation"
            context["booking_state"] = filters
        elif filters and any(filters.get(k) for k in ["last_results", "results_index_map", "location", "city"]):
            context["in_booking_flow"] = True
            context["return_to"] = "property_search"
            context["booking_state"] = filters
        
        # Call FAQ agent with context
        out = faq_agent(state.get("user_text", ""), context)
        
        # Preserve state if FAQ was asked during booking/search.
        if out.get("preserve_context") and context.get("booking_state"):
            kept = {**context["booking_state"]}
            kept[SK.faq_answered] = True
            kept[SK.faq_resume_intent] = context.get("return_to", "confirmation")
            out["filters"] = kept
        
        return {**state, **out}

async def node_confirmation(state: ChatState) -> ChatState:
    async with span("node_confirmation"):
        out = await confirmation_agent(state.get("user_text", ""), state.get("filters", {}) or {})
        result = {**state, **out}
        tool_result = out.get("tool_result", {})
        if tool_result.get("ready_for_booking"):
            result["intent"] = "booking"
        elif tool_result.get("show_receipt"):
            result["intent"] = "confirmation"
        # If user confirmed and we're proceeding to booking, clear confirmation-only flags to avoid re-entry loops
        if result.get("intent") == "booking":
            f = {**(result.get("filters") or {})}
            for k in [SK.awaiting_post_mod_choice,SK.awaiting_post_cancel_choice,SK.awaiting_field,SK.receipt_shown]:
                f.pop(k, None)
            result["filters"] = f
        return result

async def node_property(state: ChatState) -> ChatState:
    async with span("node_property"):
        out = await property_agent(state.get("user_text", ""), state.get("filters", {}) or {})
        return {**state, **out}

async def node_booking(state: ChatState) -> ChatState:
    async with span("node_booking"):
        out = await booking_agent(state.get("booking_args", {}) or {})
        result = {**state, **out}
        
        # If booking was successful, clear all booking-related state
        if out.get("tool_result", {}).get("clear_booking_state"):
            # Clear all booking-related filters to prevent re-entry
            filters = result.get("filters", {})
            booking_keys_to_clear = [
                SK.selected_property, SK.recent_property_id, SK.recent_selection_index,
                "name", "phone", "email", "check_in", "check_out", "guests",
                SK.receipt_shown, SK.awaiting_field, SK.awaiting_post_mod_choice, 
                SK.awaiting_post_cancel_choice, SK.modifying_dates, SK.awaiting_selection_confirm
            ]
            for key in booking_keys_to_clear:
                filters.pop(key, None)
            result["filters"] = filters
            
        return result

async def node_status(state: ChatState) -> ChatState:
    async with span("node_status"):
        # Merge booking_id parsed from text if not provided in status_args
        args = {**(state.get("status_args", {}) or {})}
        user_text = state.get("user_text", "")
        if not args.get("booking_id"):
            m = re.search(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", user_text or "")
            if m:
                args["booking_id"] = m.group(0)
        out = await status_agent(user_text, args)
        return {**state, **out}

async def node_payment(state: ChatState) -> ChatState:
    async with span("node_payment"):
        out = await payment_agent(state.get("payment_args", {}) or {})
        return {**state, **out}

def node_handoff(state: ChatState) -> ChatState:
    from .agents import handoff_agent
    with span("node_handoff"):
        out = handoff_agent(state.get("user_text", ""), state.get("filters", {}) or {})
        return {**state, **out}

def node_availability(state: ChatState) -> ChatState:
    from .agents import availability_agent
    with span("node_availability"):
        out = availability_agent(state.get("filters", {}) or {})
        return {**state, **out}

def _route_from_intent(state: ChatState) -> str:
    intent = state.get("intent", "property_search")
    mapping = {
        "greeting": "greeting",
        "faq": "faq",
        "confirmation": "confirmation",
        "property_search": "property",
        "booking": "booking",
        "status_update": "status",
        "payment_link": "payment",
        "handoff": "handoff",
        "availability": "availability",
        "end": "end",
    }
    return mapping.get(intent, "property")

def _route_after_confirmation(state: ChatState) -> str:
    intent = state.get("intent", "confirmation")
    tool_result = state.get("tool_result", {})
    if intent == "booking":
        return "booking"
    if tool_result.get("need") == ["restart"]:
        return "property"
    if tool_result.get("end"):
        return END
    return END

def build_chat_graph():
    g = StateGraph(ChatState)

    g.add_node("triage", node_triage)
    g.add_node("greeting", node_greeting)
    g.add_node("faq", node_faq)
    g.add_node("confirmation", node_confirmation)
    g.add_node("handoff", node_handoff)
    g.add_node("availability", node_availability)
    g.add_node("property", node_property)
    g.add_node("booking", node_booking)
    g.add_node("status", node_status)
    # Simple end node returns END with a friendly message
    def node_end(state: ChatState) -> ChatState:
        return {**state, "intent": "end", "reply": "Thank you for booking with us! Have a wonderful stay!", "tool_result": {"end": True}}
    g.add_node("end", node_end)
    g.add_node("payment", node_payment)

    g.set_entry_point("triage")

    g.add_conditional_edges("triage", _route_from_intent, {
        "greeting": "greeting",
        "faq": "faq",
        "confirmation": "confirmation",
        "property": "property",
        "booking": "booking",
        "status": "status",
        "payment": "payment",
        "handoff": "handoff",
        "availability": "availability",
        "end": "end",
    })

    g.add_conditional_edges("confirmation", _route_after_confirmation, {
        "booking": "booking",
        "property": "property",
        END: END,
    })

    for node in ["greeting","faq","property","booking","status","payment","handoff","availability"]:
        g.add_edge(node, END)
    g.add_edge("end", END)

    compile_kwargs: Dict[str, Any] = {}
    try:
        if "recursion_limit" in inspect.signature(g.compile).parameters:
            compile_kwargs["recursion_limit"] = 25
    except (TypeError, ValueError):
        # Some LangGraph versions expose non-introspectable callables.
        pass

    return g.compile(**compile_kwargs)

APP = build_chat_graph()

async def run_chat_graph(
    message: str,
    filters: Optional[Dict[str, Any]] = None,
    booking_args: Optional[Dict[str, Any]] = None,
    status_args: Optional[Dict[str, Any]] = None,
    payment_args: Optional[Dict[str, Any]] = None,
) -> ChatState:
    # Clean the filters to ensure no stale booking context on first message
    clean_filters = filters or {}
    
    state: ChatState = {
        "user_text": message,
        "filters": clean_filters,
        "booking_args": booking_args or {},
        "status_args": status_args or {},
        "payment_args": payment_args or {},
    }
    result = await APP.ainvoke(state, config={"recursion_limit": 25})
    # --- Guardrails: sanitize output ---
    if result.get("reply"):
        result["reply"] = sanitize_output(result["reply"])
    # Best-effort chat logging (only if both sides present)
    try:
        user_msg = message or ""
        bot_resp = (result.get("reply") or "").strip()
        if user_msg and bot_resp:
            await log_chat(user_msg, bot_resp)
    except Exception:
        pass
    return result
