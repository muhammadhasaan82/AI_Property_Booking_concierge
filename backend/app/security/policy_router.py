"""
Phase 3: Deterministic policy router.
 
Pure Python. NO LLM calls. Reads YAML config + UnderstandingFrame + soft state
and emits a typed RouterDecision the runner can act on.
 
YAML inputs (loaded once at import time):
  - app/config/agent_policy.yaml    (intent rules, priorities, thresholds)
  - app/config/tool_registry.yaml   (tool metadata)
  - app/config/booking_schema.yaml  (booking field schema)
 
Operating modes (cfg.feature_policy_router_mode):
  off      : disabled; runner skips this module entirely.
  shadow   : decision is computed and logged, but no behavior change.
  enforce  : non-execute decisions replace router_output. Execute-tool
             disagreements are logged but not re-run (Phase 3 scope).
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, TypedDict, Union
# from torch._dynamo.utils import V
from app.config.agent_policy_loader import (
    policy,
    sorted_priorities,
    get_intent,
    IntentRule,
    PriorityCondition,
)
from app.config.booking_schema_loader import (
    get_required_fields as booking_required_fields,
    get_required_numeric_fields as booking_numeric_fields,
)
from app.agents.schemas.understanding_frame import UnderstandingFrame

logger = logging.getLogger(__name__)

PolicyAction = Literal["execute_tool", "ask_clarification", "fallback", "escalate", "block"]

class RouterDecision(TypedDict, total=False):
    action: PolicyAction
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    clarification_message: Optional[str]
    response_policy: Optional[str]
    effective_intent: str
    matched_priority_id: Optional[str]
    confidence: float
    reasoning: str

def _has_active_shortlist(soft_state: Dict[str, Any]) -> bool:
    last_search = soft_state.get("last_search")
    if not isinstance(last_search, dict):
        return False
    properties = last_search.get("properties") or []
    return bool(properties)

def _has_pending_booking(soft_state: Dict[str, Any]) -> bool:
    booking_state = soft_state.get("Booking_state") or {}
    pending = soft_state.get("pending_booking") or {}
    return bool(booking_state) or bool(pending)

def _has_awaiting_field(soft_state: Dict[str, Any]) -> bool:
    return bool(soft_state.get("awaiting_field"))

def _conditional_matches(
    frame: UnderstandingFrame,
    soft_state: Dict[str, Any],
    cond: PriorityCondition
) -> bool:
    """Evaluate a single PriorityCondition against frame + soft state."""
    if cond.primary_intent_in:
        if frame.primary_intent not in cond.primary_intent_in:
            return False

    if cond.has_active_shortlist is not None:
        if cond.has_active_shortlist != _has_active_shortlist(soft_state):
            return False
    
    if cond.has_pending_booking is not None:
        if cond.has_pending_booking != _has_awaiting_booking(soft_state):
            return False

    if cond.awaiting_field_present is not None:
        if cond.awaiting_field_present != _has_awaiting_field(soft_state):
            return False

    if cond.selection_number_present is not None:
        has_sel = frame.selection_number is not None
        if cond.selection_number_present != has_sel:
            return False

    if cond.explicit_status_present is True:
        if frame.primary_intent != "booking_status":
            return False

    if cond.faq_question_form is True:
        if frame.primary_intent != "faq":
            return False

    return True

def _resolve_effective_intent(
    frame: UnderstandingFrame,
    soft_state: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    """Apply priorities[] in descending order and return (intent, matched_rule_id)."""
    for rule in sorted_priorities():
        if _conditional_matches(frame, soft_state, rule.when):
            return rule.prefer_intent, rule.id
        return frame.primary_intent, None

def _value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True

def _check_required_fields(
    intent_rule: IntentRule,
    frame: UnderstandingFrame,
    soft_state: Dict[str, Any],
) -> List[str]:
    """Return the subset of Intent_rule.required_fields not present in frame/state."""
    last_search = soft_state.get("last_search") or {}
    booking_state = soft_state.get("booking_state") or {}

    missing: List[str] = []
    for field in intent_rule.required_fields:
        if _value_present(frame.entities.get(field)):
            continue
        if _value_present(last_search.get(field)):
            continue
        if _value_present(booking_state.get(field)):
            continue
        missing.append(field)
    return missing

def _check_required_context(
    intent_rule: IntentRule,
    soft_state: Dict[str, Any],
) -> List[str]:
    """Return the list of required context keys not satisfied by soft_state."""
    missing: List[str] = []
    for ctx in intent_rule.requires_context:
        if ctx == "active_shortlist" and not _has_active_shortlist(soft_state):
            missing.append(ctx)
        elif ctx == "pending_booking" and not _has_pending_booking(soft_state):
            missing.append(ctx)
    return missing

def _compute_booking_missing(
    frame: UnderstandingFrame,
    soft_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Hydrate booking state from frame.entities and return (state, missing_fields)."""
    state = dict(soft_state.get("booking_state") or {})
    for k, v in frame.entities.items():
        if _value_present(v):
            state[k] = v
    missing: List[str] = []
    for f in booking_required_fields():
        if not _value_present(state.get(f)):
            missing.append(f)
    for f in booking_numeric_fields():
        if not _value_present(state.get(f)):
            missing.append(f)
    return state, list(dict.fromkeys(missing))

def _build_tool_args(
    tool_name: str,
    frame: UnderstandingFrame,
    soft_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Build best-effort tool args from frame.entities + soft state."""
    args: Dict[str, Any] = {}
    
    if tool_name == "search_properties":
        for k in ("city","budget", "beds", "property_type", "amenities", "free_text", "max_results" ):
            if k in frame.entities and _value_present(frame.entities[k]):
                args[k] = frame.entities[k]

    elif tool_name == "select_property":
        if frame.selection_number is not None:
            args["option_number"] = frame.selection_number

    elif tool_name == "get_property_details":
        if "property_id" in frame.entities:
            args["property_id"] = frame.entities["property_id"]
        elif frame.selection_number is not None:
            args["option_number"] = frame.selection_number

    elif tool_name == "check_booking_status":
        if "booking_id" in frame.entities:
            args["booking_id"] = frame.entities["booking_id"]

    elif tool_name in ("request_booking_details", "review_booking_details", "process_v2_details"):
        for k in booking_required_fields() + booking_numberic_fields():
            if k in frame.entities and _value_present(frame.entities[k]):
                args[k] = frame.entities[k]
        return args

def _make_decision(
    *,
    action: PolicyAction,
    intent: str,
    matched_priority_id: Optional[str],
    confidence: float,
    reasoning: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
    clarification_message: Optional[str] = None,
    response_policy: Optional[str] = None,
) -> RouterDecision:
    return{
        "action": action,
        "tool_name": tool_name,
        "tool_args": tool_args or {},
        "clarification_message": clarification_message,
        "response_policy": response_policy,
        "effective_intent": intent,
        "matched_priority_id": matched_priority_id,
        "confidence": confidence,
        "reasoning": reasoning,
    }
def decide(
    frame: UnderstandingFrame,
    soft_state: Optional[Dict[str, Any]] = None,
) -> RouterDecision:
    """Compute a deterministic RouterDecision from the LLM frame + soft state."""
    soft_state = soft_state or {}
    
    effective_intent, matched_priority_id = _resolve_effective_intent(frame, soft_state)

    intent_rule = get_intent(effective_intent)
    if intent_rule is None:
        return _make_decision(
            action="fallback",
            intent=effective_intent,
            matched_priority_id=matched_priority_id,
            confidence=frame.confidence,
            reasoning=f"No rule defined for intent={effective_intent}.",
            response_policy="cascual_interaction",
        )

    if intent_rule.block_other_intents and effective_intent == "human_handoff":
        return _make_decision(
            action="escalate",
            intent=effective_intent,
            matched_priority_id=matched_priority_id,
            confidence=frame.confidence,
            reasoning="human_handoff blocks other intents",
            tool_name="escalate_to_human" if "escalate_to_human" in intent_rule.allowed_tools else None,
            response_policy="cascual_required",
        )
    if frame.confidence < policy.confidence.medium:
        msg = intent_rule.clarify_message or "could you share a bit more about what you need?"
        return _make_decision(
            action="ask_clarification",
            intent=effective_intent,
            matched_priority_id=matched_priority_id,
            confidence=frame.confidence,
            reasoning=f"confidence {frame.confidence:.2f} < medium={policy.confidence.medium}",
            clarification_message=msg,
            response_policy="missing_critical_data",
        )
    missing_ctx=_check_required_context(intent_rule, soft_state)
    if missing_ctx:
        msg = intent_rule.clarify_message or f"I need {missing_ctx} first."
        return _make_decision(
            action="ask_clarification",
            intent=effective_intent,
            matched_priority_id=matched_priority_id,
            confidence=frame.confidence,
            reasoning=f"missing context: {missing_ctx}",
            clarification_message=msg,
            response_policy="missing_critical_data",
        )

    if effective_intent in ("booking_continuation", "booking_confirmation"):
        booking_state, booking_missing = _compute_booking_missing(frame, soft_state)
        
        if booking_missing:
            return _make_decision(
                action="execute_tool",
                intent=effective_intent,
                matched_priority_id=matched_priority_id,
                confidence=frame.confidence,
                reasoning=f"booking missing: {booking_missing}",
                tool_name="request_booking_details",
                tool_args={
                    k: v for k, v in frame.entities.items()
                    if k in (booking_required_fields() + booking_numeric_fields())
                },
                response_policy="gathering_info",
            )
        if effective_intent == "booking_confirmation":
            return _make_decision(
                action="execute_tool",
                intent=effective_intent,
                matched_priority_id=matched_priority_id,
                confidence=frame.confidence,
                reasoning="All booking fields present + explicit user confirmation.",
                tool_name="process_v2_booking",
                tool_args=dict(booking_state),
                response_policy="booking_confirmed",
            )
        return _make_decision(
            action="execute_tool",
            intent=effective_intent,
            matched_priority_id=matched_priority_id,
            confidence=frame.confidence,
            reasoning="All booking fields present, ready for review",
            tool_name="review_booking_details",
            tool_args=dict(booking_state),
            response_policy="review_pending",
        )
    missing_fields = _check_required_context(intent_rule, soft_state)
    if missing_fields:
        msg=intent_rule.clarify_message or f"I need: {', '.join(missing_fields)}."
        return _make_decision(
            action="ask_clarification",
            intent=effective_intent,
                matched_priority_id=matched_priority_id,
                confidence=frame.confidence,
                reasoning=f"Missing required fields: {missing_fields}",
                clarification_message=msg,
                response_policy="missing_critical_data",
            )

    if not intent_rule.allowed_tools:
        return _make_decision(
            action="fallback",
            intent=effective_intent,
            matched_priority_id=matched_priority_id,
            confidence=frame.confidence,
            reasoning=f"No allowed tools for intent '{effective_intent}'",
            response_policy="casual_interaction",
        )
    primary_tool = intent_rule.allowed_tools[0]
    tool_args = _build_tool_args(primary_tool, frame, soft_state)
    return _make_decision(
        action="execute_tool",
        intent=effective_intent,
        matched_priority_id=matched_priority_id,
        confidence=frame.confidence,
        reasoning=f"confidence {frame.confidence:.2f}' ≥ medium → {primary_tool}",
        tool_name=primary_tool,
        tool_args=tool_args,
        response_policy=intent_rule.response_policy or "found",
    )

def compute_override(
    decision: RouterDecision,
    actual_tool_called: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return an override summary dict if decision and actual disagree, else None."""
    policy_action = decision.get("action")
    policy_tool = decision.get("tool_name")

    if policy_action != "execute_tool":
        if actual_tool_called:
            return {
                "actual_tool": actual_tool_called,
                "policy_action": policy_action,
                "policy_tool": None,
                "effective_intent": decision.get("effective_intent"),
                "matched_priority_id": decision.get("matched_priority_id"),
                "confidence": decision.get("confidence"),
                "reasoning": decision.get("reasoning"),
            }
        return None

    if not actual_tool_called:
        return None

    if actual_tool_called == policy_tool:
        return None

    return {
        "actual_tool": actual_tool_called,
        "policy_action": "execute_tool",
        "policy_tool": policy_tool,
        "effective_intent": decision.get("effective_intent"),
        "matched_priority_id": decision.get("matched_priority_id"),
        "confidence": decision.get("confidence"),
        "reasoning": decision.get("reasoning"),
    }

def synthesize_router_output(decision: RouterDecision) -> Dict[str, Any]:
    """Build a router_output dict from a non-execute_tool decision.
    
    Used in enforce mode for ask_clarification / escalate / block / fallback.
    The result is consumed by concierge_voice (via _render_voice_from_router_output)
    and rendered as a normal reply.
    """
    action = decision.get("action")
    base = {
        "policy_overridden": True,
        "effective_intent": decision.get("effective_intent"),
        "response_policy": decision.get("response_policy"),
    }
    if action == "ask_clarification":
        base.update({
            "status": "missing_critical_date",
            "context_message": decision.get("clarification_message") or "Could you tell me more so I can help?",

        })
    elif action == "escalate":
        base.update({
            "status": "handoff_required",
            "context_message": "Connecting you with a human agent,",
        })
    elif action == "block":
        base.update({
            "status": "blocked",
            "context_message":"I can't help with that request.",
        })
    else:
        base.update({
            "status": "cascual_interaction",
            "context_message": "I'm here to help - could you share a bit more?.",
        })

    if not base.get("response_policy"):
        base["response_policy"] = base["status"]
    
    return base