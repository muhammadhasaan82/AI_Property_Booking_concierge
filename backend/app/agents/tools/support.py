"""
Tools: handle_small_talk, check_faq, check_booking_status, escalate_to_human
"""
from __future__ import annotations

import logging
from typing import Optional

from ..status_codes import SMALL_TALK_TYPES, Source, Status
from .helpers import _finalize_payload, _is_blank, _missing_critical_data
from app.config.agent_config_loader import cfg

logger = logging.getLogger(__name__)

def handle_small_talk(
    message_type: Optional[str] = None,
    user_message: Optional[str] = "",
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Handle greetings, thanks, casual conversation, and acknowledgements.

    Use this tool ONLY for non-actionable social messages such as:
    - Greetings: "hi", "hello", "hey", "good morning"
    - Acknowledgements: "ok", "thanks", "thank you", "got it", "sure", "alright"
    - Goodbyes: "bye", "goodbye", "see you"
    - Affirmations with no booking context: "great", "perfect", "cool"

    Do NOT use this for booking intent, property questions, or policy questions.

    Args:
        message_type: One of 'greeting', 'thanks', 'goodbye', 'acknowledgement'.
        user_message: The user's raw message text.
        action_intent: Optional context flag for state acknowledgements.
        context_flag: Optional secondary context flag.
    """
    normalized_type = (message_type or "").strip().lower()
    if normalized_type not in SMALL_TALK_TYPES:
        normalized_type = cfg.small_talk_default_type
    return _finalize_payload(
        {
            "status": Status.CASUAL_INTERACTION,
            "message_type": normalized_type,
            "user_input": user_message or "",
        },
        action_intent, context_flag,
    )

async def check_faq(
    question: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Look up a policy or FAQ question about the booking platform.

    Use this tool ONLY when the user asks a genuine question about rules,
    policies, check-in/check-out times, cancellation, refunds, wifi, pets,
    smoking, parking, payment methods, or security deposits.

    DO NOT call this for greetings, thanks, or casual chat — use handle_small_talk.
    Args:
        question: The user's specific policy or FAQ question (optional).
        action_intent: Optional context flag for routing.
        context_flag: Optional secondary context flag.
    """
    if not question or len(question.strip()) < 4:
        return _missing_critical_data(
            ["question"],
            "User asked about policies but did not provide a specific question.",
            action_intent, context_flag,
        )

    from ..tools.rust_client import execute_tool


    try:
        result = await execute_tool(data={"intent": "faq", "question": question})
        if result is not None and not result.get("fallback"):
            answer = result.get("answer") or (result.get("result") or {}).get("answer")
            if answer:
                return _finalize_payload(
                    {"status": Status.ANSWERED, "answer": answer, "source": Source.POLICY_DB},
                    action_intent, context_flag,
                )
    except Exception as e:
        logger.warning("Rust FAQ lookup failed: %s, using Python fallback", e)


    try:
        from ...components.faq_enhanced import enhanced_faq_agent
        faq_result = enhanced_faq_agent(question, {})
        reply = faq_result.get("reply", "")
        if reply:
            return _finalize_payload(
                {"status": Status.ANSWERED, "answer": reply, "source": Source.RAG},
                action_intent, context_flag,
            )
    except Exception as e:
        logger.warning("FAQ enhanced agent failed: %s", e)


    try:
        from ...services.faq import faq_lookup
        ans = faq_lookup(question)
        if ans:
            return _finalize_payload(
                {"status": Status.ANSWERED, "answer": ans, "source": Source.BASIC_FAQ},
                action_intent, context_flag,
            )
    except Exception as e:
        logger.warning("Basic FAQ fallback failed: %s", e)

    return _finalize_payload(
        {"status": Status.FAQ_NOT_FOUND, "question": question},
        action_intent, context_flag,
    )

async def check_booking_status(
    booking_id: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Check the status of an existing booking.

    Use this tool when the user asks about a booking status, wants to check
    their reservation, or provides a booking ID.

    Args:
        booking_id: The booking ID (UUID format).
        action_intent: Optional context flag.
        context_flag: Optional secondary context flag.
    """
    if _is_blank(booking_id):
        return _missing_critical_data(
            ["booking_id"],
            "User asked about booking status but no booking ID was provided.",
            action_intent, context_flag,
        )

    from ...services.booking import get_booking_status
    from ...observability.db_logging import get_successful_booking_status

    try:
        r = await get_booking_status(booking_id)
        if r.get("ok"):
            return _finalize_payload(
                {
                    "status": Status.FOUND,
                    "booking_id": booking_id,
                    "booking_status": str(r.get("status", "unknown")).replace("_", " "),
                    "check_in": r.get("check_in", "?"),
                    "check_out": r.get("check_out", "?"),
                },
                action_intent, context_flag,
            )
    except Exception:
        pass

    try:
        db_row = await get_successful_booking_status(str(booking_id))
        if db_row:
            return _finalize_payload(
                {
                    "status": Status.FOUND,
                    "booking_id": booking_id,
                    "booking_status": str(db_row.get("status", "confirmed")).replace("_", " "),
                    "check_in": db_row.get("check_in", "?"),
                    "check_out": db_row.get("check_out", "?"),
                    "source": "successful_bookings",
                },
                action_intent, context_flag,
            )
    except Exception:
        pass

    return _finalize_payload(
        {"status": Status.BOOKING_NOT_FOUND, "booking_id": booking_id},
        action_intent, context_flag,
    )

async def escalate_to_human(
    reason: Optional[str] = None,
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Transfer the conversation to a human support agent.

    Use this tool when:
    - The user explicitly asks to speak with a human or agent.
    - You cannot resolve the user's issue with the available tools.
    - The user seems frustrated and needs personal assistance.

    Args:
        reason: Brief description of why the handoff is needed.
        action_intent: Optional context flag.
        context_flag: Optional secondary context flag.
    """
    reason_value = (
        reason.strip()
        if isinstance(reason, str) and reason.strip()
        else cfg.msg_escalation_default
    )
    return _finalize_payload(
        {"status": Status.HANDOFF_REQUIRED, "reason": reason_value},
        action_intent, context_flag,
    )
