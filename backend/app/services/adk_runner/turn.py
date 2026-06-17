from __future__ import annotations
"""ADK runner submodule."""

import asyncio
import hashlib
import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from google.adk.runners import Runner
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.genai.types import Content, Part

from app.config.agent_config_loader import cfg as _cfg
from app.services.redis_store import (
    clear_session_snapshot,
    get_redis_client,
    get_session_snapshot,
    save_session_snapshot,
)

logger = logging.getLogger(__name__)
ADK_TURN_TIMEOUT = float(getattr(_cfg, "runtime_turn_timeout_seconds", 45))
MEM0_ENABLED = os.getenv("MEM0_ENABLED", "1").strip().lower() in ("1", "true", "yes")
APP_NAME = "ai_concierge"


from app.services.adk_runner.events import _extract_tool_call, _extract_tool_response
from app.services.adk_runner.handlers import (
    _maybe_handle_active_booking_turn,
    _maybe_handle_booking_amendment_turn,
    _maybe_handle_booking_cancellation_turn,
    _maybe_handle_booking_status_check,
    _maybe_handle_faq_resume_turn,
    _maybe_handle_search_state_shortcut,
    _maybe_handle_property_refinement_followup,
    _maybe_handle_service_coverage_followup,
    _maybe_record_unsupported_region,
)
from app.services.adk_runner.rendering import (
    _deterministic_reply_from_router_output,
    _render_no_results_from_search_payload,
    _render_property_details_from_router_output,
    _render_property_results_from_router_output,
    _render_voice_from_router_output,
)
from app.services.adk_runner.session_service import _get_runner, _get_session_service
from app.services.adk_runner.state_helpers import (
    _jsonable,
    _merge_soft_state,
    _normalize_cognitive_context,
    _soft_state_from_router_output,
    _state_with_persisted_soft_state,
)
from app.services.direct_property_search import maybe_handle_direct_property_search

from app.agents.schemas.understanding_frame import UnderstandingFrame
from app.config.agent_config_loader import cfg as _cfg
from app.observability import telemetry
from app.security import anomaly, policy_router
from app.security.guardrails import sanitize_input, sanitize_output
from app.services.config import ADK_SESSION_MAX_EVENTS
from app.services.faq_interruption import detect_policy_question
from app.services.observability.langfuse_observer import get_observer, sanitize_for_observability, summarize_soft_state
from app.services.pre_router import route_pre_adk

async def run_adk_turn(
    user_id: str,
    session_id: str,
    message: str,
) -> AsyncGenerator[str, None]:
    """
    Run a single ADK conversation turn and stream the assistant's reply as text chunks.
    
    Performs input sanitization, optional pre-routing or search-shortcut handling, invokes the ADK runner for one turn (including tool calls and routing), applies deterministic property rendering or policy overrides when applicable, persists soft-state, and records telemetry. Yields partial or final text segments produced for the user.
    
    Parameters:
        user_id (str): Unique identifier for the user.
        session_id (str): Conversation/session identifier.
        message (str): The raw user message.
    
    Yields:
        str: Partial or final text segments from the assistant (concierge_voice) or a deterministic render; each yielded value is a chunk of the reply.
    """
    observer = get_observer()
    trace = observer.trace(
        name="chat_turn",
        user_id=user_id,
        session_id=session_id,
        metadata={
            "environment": os.getenv("LANGFUSE_ENVIRONMENT", "dev"),
            "release": os.getenv("LANGFUSE_RELEASE", ""),
            "input_length": len(message),
            "sanitized_input_preview": sanitize_for_observability(message[:100]),
            "dispatcher_model": getattr(_cfg, "dispatcher_model", "unknown"),
            "voice_model": getattr(_cfg, "voice_model", "unknown"),
            "pre_router_fast_model": getattr(_cfg, "pre_router_fast_model", "unknown"),
            "mem0_llm_model": os.getenv("MEM0_LLM_MODEL", "unknown"),
        }
    )
    
    initial_state: Dict[str, Any] = {}
    initial_soft_state: Dict[str, Any] = {}
    initial_history: List[Any] = []
    initial_metadata: Dict[str, Any] = {}
    try:
        snapshot = await get_session_snapshot(session_id)
        if isinstance(snapshot, dict):
            state = snapshot.get("state") or {}
            if isinstance(state, dict):
                initial_state = state
                soft_state = state.get("soft_state")
                if isinstance(soft_state, dict):
                    initial_soft_state = soft_state
            initial_history = snapshot.get("history", [])
            meta = snapshot.get("meta") or {}
            initial_metadata = {
                key: meta[key]
                for key in ("app_name", "user_id", "last_update_time")
                if key in meta
            }
    except Exception:
        pass
        
    trace.update(metadata={
        "soft_state_summary": summarize_soft_state(initial_soft_state)
    })

    with trace.span(name="input_sanitization"):
        cleaned_message, is_safe = sanitize_input(message)
    
    if not is_safe:
        yield "I'm sorry, I can't process that request. Could you rephrase?"
        trace.end()
        return

    with trace.span(name="booking_cancellation_priority"):
        cancellation_payload = await _maybe_handle_booking_cancellation_turn(
            session_id=session_id,
            message=cleaned_message,
        )
    if cancellation_payload:
        deterministic_reply = cancellation_payload.get("deterministic_reply")
        if not deterministic_reply:
            deterministic_reply = "I'm sorry, I couldn't process your request. Could you try again?"
        trace.end()
        yield str(deterministic_reply)
        return

    with trace.span(name="booking_amendment_priority"):
        amendment_payload = await _maybe_handle_booking_amendment_turn(
            session_id=session_id,
            message=cleaned_message,
        )
    if amendment_payload:
        deterministic_reply = (
            str(amendment_payload.get("deterministic_reply") or "").strip()
            or _deterministic_reply_from_router_output(amendment_payload)
        )
        if not deterministic_reply:
            deterministic_reply = "I'm sorry, I couldn't process your request. Could you try again?"
        trace.end()
        yield deterministic_reply
        return

    with trace.span(name="service_coverage_followup"):
        service_coverage_payload = await _maybe_handle_service_coverage_followup(
            session_id=session_id,
            message=cleaned_message,
        )
    if service_coverage_payload:
        deterministic_reply = str(service_coverage_payload.get("deterministic_reply") or "").strip()
        if deterministic_reply:
            trace.end()
            yield deterministic_reply
            return

    with trace.span(name="service_coverage_guard"):
        coverage_decision = evaluate_message_coverage(cleaned_message)
    if coverage_decision.blocked and coverage_decision.message:
        await _maybe_record_unsupported_region(
            session_id=session_id,
            country=coverage_decision.country,
        )
        trace.end()
        yield coverage_decision.message
        return

    with trace.span(name="faq_resume_guard"):
        faq_resume_payload = await _maybe_handle_faq_resume_turn(
            session_id=session_id,
            message=cleaned_message,
        )
    if faq_resume_payload:
        deterministic_reply = str(faq_resume_payload.get("deterministic_reply") or "").strip()
        if deterministic_reply:
            trace.end()
            yield deterministic_reply
            return

    with trace.span(name="semantic_shortcut"):
        shortcut_payload = await _maybe_handle_search_state_shortcut(
            session_id = session_id,
            message=cleaned_message
        )
    if shortcut_payload:
        deterministic_reply = shortcut_payload.get("deterministic_reply")
        if deterministic_reply:
            trace.end()
            yield str(deterministic_reply)
            return
        if shortcut_payload.get("status") == "properties_found" and shortcut_payload.get("properties"):
            deterministic = _render_property_results_from_router_output(shortcut_payload)
            if deterministic:
                trace.end()
                yield deterministic
                return
        if str(shortcut_payload.get("status") or "").lower() == "property_details":
            details = _render_property_details_from_router_output(shortcut_payload)
            if details:
                trace.end()
                yield details
                return

        shortcut_text = await _render_voice_from_router_output(
            json.dumps(_jsonable(shortcut_payload), ensure_ascii = False),
            "",
            "",
        )
        if shortcut_text:
            trace.end()
            yield shortcut_text
            return
        trace.end()
        yield "I'm sorry, I couldn't process your request. Could you try again?"
        return

    with trace.span(name="property_refinement_followup"):
        property_refinement_payload = await _maybe_handle_property_refinement_followup(
            session_id=session_id,
            message=cleaned_message,
        )
    if property_refinement_payload:
        deterministic_reply = str(property_refinement_payload.get("deterministic_reply") or "").strip()
        if deterministic_reply:
            trace.end()
            yield deterministic_reply
            return

    with trace.span(name="booking_flow"):
        booking_payload = await _maybe_handle_active_booking_turn(
            session_id=session_id,
            message=cleaned_message,
        )
    if booking_payload:
        deterministic_reply = (
            str(booking_payload.get("deterministic_reply") or "").strip()
            or _deterministic_reply_from_router_output(booking_payload)
        )
        if not deterministic_reply:
            deterministic_reply = "I'm sorry, I couldn't process your request. Could you try again?"
        trace.end()
        yield str(deterministic_reply)
        return

    with trace.span(name="booking_status_check"):
        status_payload = await _maybe_handle_booking_status_check(
            session_id=session_id,
            message=cleaned_message,
        )
    if status_payload:
        deterministic_reply = status_payload.get("deterministic_reply")
        if deterministic_reply:
            trace.end()
            yield str(deterministic_reply)
            return

    with trace.span(name="direct_property_search"):
        direct_search_payload = await maybe_handle_direct_property_search(
            cleaned_message,
            session_id,
            get_snapshot=get_session_snapshot,
            save_snapshot=save_session_snapshot,
        )
    if direct_search_payload:
        deterministic_reply = str(direct_search_payload.get("deterministic_reply") or "").strip()
        if deterministic_reply:
            trace.end()
            yield deterministic_reply
            return
        if (
            direct_search_payload.get("status") == "properties_found"
            and direct_search_payload.get("properties")
        ):
            deterministic = _render_property_results_from_router_output(direct_search_payload)
            if deterministic:
                trace.end()
                yield deterministic
                return
        status = str(direct_search_payload.get("status") or "").lower()
        if status == "no_results":
            trace.end()
            yield _render_no_results_from_search_payload(direct_search_payload)
            return
    if detect_policy_question(cleaned_message):
        from app.agents.tools.support import check_faq

        soft_state = initial_soft_state if isinstance(initial_soft_state, dict) else {}
        tool_context = SimpleNamespace(state={"soft_state": soft_state})
        faq_payload = await check_faq(question=cleaned_message, tool_context=tool_context)
        if isinstance(faq_payload, dict):
            deterministic_reply = (
                str(faq_payload.get("deterministic_reply") or "").strip()
                or str(faq_payload.get("answer") or "").strip()
            )
            if deterministic_reply:
                await save_session_snapshot(
                    session_id=session_id,
                    history=initial_history,
                    state=_state_with_persisted_soft_state(initial_state, soft_state),
                    metadata=initial_metadata,
                )
                trace.end()
                yield deterministic_reply
                return

    with trace.span(name="faq_retrieval"):
        pre_routed = await route_pre_adk(
            message=cleaned_message,
            user_id=user_id,
            session_id=session_id,
        )
    if pre_routed and pre_routed.get("reply"):
        trace.end()
        yield str(pre_routed["reply"])
        return
    if not cleaned_message.strip():
        trace.end()
        yield "I didn't catch that. Could you repeat your question?"
        return

    runner = _get_runner()
    session_service = _get_session_service()
    state_delta = await _build_invocation_state_delta(user_id=user_id, current_query=cleaned_message, session_id=session_id)
    user_cognitive_context = _normalize_cognitive_context(state_delta.get("user_cognitive_context"))

    user_content = Content(parts=[Part(text=cleaned_message)])
    t0 = time.monotonic()

    streamed_parts: List[str] = []
    tool_calls_log: List[Dict[str, Any]] = []
    anomaly_triggered = False
    router_output = ""
    router_output_dict: Optional[Dict[str, Any]] = None
    _deterministic_render: Optional[str] = None
    pipeline_failed_reply = ""

    pending_soft_state_updates: Dict[str, Any] = {}
    event_count = 0
    max_adk_events = int(getattr(_cfg, "runtime_max_adk_events_per_turn", 8))
    try:
        async with asyncio.timeout(ADK_TURN_TIMEOUT):
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content,
                state_delta=state_delta,
            ):
                event_count += 1
                if event_count > max_adk_events:
                    logger.warning("[ADK] Event limit exceeded: %s/%s", event_count, max_adk_events)
                    pipeline_failed_reply = str(getattr(_cfg, "runtime_routing_limit_fallback", ""))
                    break
                tool_name, tool_params = _extract_tool_call(event)
                if tool_name:
                    tool_span = trace.span(name="search_tool" if "search" in tool_name.lower() else "tool_call")
                    if await anomaly.check_tool_loop(session_id, tool_name, tool_params):
                        anomaly_triggered = True
                        tool_calls_log.append({
                            "tool": tool_name,
                            "params_hash": hashlib.md5(
                                json.dumps(tool_params, sort_keys=True, default=str).encode()
                            ).hexdigest()[:12],
                            "result_status": "anomaly_blocked",
                        })
                        tool_span.end()
                        break
                    await anomaly.record_tool_call(session_id, tool_name, tool_params)
                    tool_calls_log.append({
                        "tool": tool_name,
                        "params_hash": hashlib.md5(
                            json.dumps(tool_params, sort_keys=True, default=str).encode()
                        ).hexdigest()[:12],
                        "result_status": "ok",
                    })
                    tool_span.end()
                    continue

                author = getattr(event, "author", None)
                event_text = _extract_text_parts(event)
                tool_response = _extract_tool_response(event)
                if author == "triage_router" and tool_response is not None:
                    try:
                        jsonable_response = _jsonable(tool_response)
                        router_output = json.dumps(jsonable_response, ensure_ascii=False)
                        router_output_dict = jsonable_response if isinstance(jsonable_response, dict) else None
                    except Exception:
                        router_output = str(tool_response)
                        router_output_dict = None

                    if isinstance(router_output_dict, dict):
                        _status = router_output_dict.get("status") or ""
                        _props = router_output_dict.get("properties") or []
                        logger.info(
                            "[ADK] router_output preview: status=%s total_found=%s shown_count=%s "
                            "first_title=%r",
                            _status,
                            router_output_dict.get("total_found"),
                            router_output_dict.get("shown_count"),
                            (_props[0].get("title") if _props else None),
                        )
                        if _status == "properties_found" and _props:
                            _det = _render_property_results_from_router_output(router_output_dict)
                            if _det:
                                _deterministic_render = _det
                                trace.update(metadata={"bypassed_voice_llm": True})
                                with trace.span(name="deterministic_render"):
                                    pass
                                logger.info(
                                    "[ADK] Deterministic property render active — "
                                    "concierge_voice LLM output will be suppressed (%d properties)",
                                    len(_props),
                                )

                        if _deterministic_render is None:
                            _det = _deterministic_reply_from_router_output(router_output_dict)
                            if _det:
                                _deterministic_render = _det
                                trace.update(metadata={"bypassed_voice_llm": True})

                        pending_soft_state_updates = _merge_soft_state(
                            pending_soft_state_updates,
                            _soft_state_from_router_output(router_output_dict),
                        )

                if author == "triage_router" and event_text:
                    router_output = event_text

                if author == "concierge_voice":
                    if _deterministic_render is not None:
                        pass
                    elif event.content and event.content.parts:
                        with trace.span(name="llm_generation"):
                            for part in event.content.parts:
                                if hasattr(part, "text") and part.text:
                                    streamed_parts.append(part.text)
                                    yield part.text

                if event.is_final_response():
                    if author == "triage_router":
                        continue
                    if author == "concierge_voice":
                        if _deterministic_render is None and not streamed_parts and event_text:
                            streamed_parts.append(event_text)
                            yield event_text
                        break
                    if streamed_parts or _deterministic_render is not None:
                        break

    except asyncio.TimeoutError:
        logger.error("[ADK] Turn timed out after %ss - check providers keys/networks", ADK_TURN_TIMEOUT)

    except Exception as exc:
        logger.error("[ADK] Pipeline execution error: %s", exc, exc_info=True)
        pipeline_failed_reply = "I'm sorry, something went wrong. Please try again."

    latency_ms = (time.monotonic() - t0) * 1000.0

    if _deterministic_render is not None:
        logger.info(
            "[ADK] Yielding deterministic property render (%d chars) — "
            "no concierge_voice LLM text used",
            len(_deterministic_render),
        )
        yield _deterministic_render
        final_reply = _deterministic_render
    else:
        final_reply = "".join(streamed_parts)

    try:
        updated_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as exc:
        logger.error("[ADK] Could not retrieve updated session %s: %s", session_id, exc, exc_info=True)
        updated_session = None

    try:
        current_snapshot = await get_session_snapshot(session_id)
        current_state = current_snapshot.get("state", {})
        merged_state = dict(current_state) if isinstance(current_state, dict) else {}


        if pending_soft_state_updates:

            _SEARCH_NAV_KEYS = frozenset({
                "visible_results", "all_search_results", "option_map",
                "current_page", "page_size", "active_flow",
            })
            existing_soft = merged_state.get("soft_state") or {}
            _pss_status = (router_output_dict or {}).get("status", "").lower()
            if _pss_status == "property_details":

                safe_updates = {
                    k: v for k, v in pending_soft_state_updates.items()
                    if k not in _SEARCH_NAV_KEYS
                }
                merged_state["soft_state"] = _merge_soft_state(existing_soft, safe_updates)
            else:
                merged_state["soft_state"] = _merge_soft_state(
                    existing_soft, pending_soft_state_updates
                )
            logger.debug(
                "[ADK] Pending soft_state from router_output merged: status=%s keys=%s",
                _pss_status, list(pending_soft_state_updates.keys()),
            )


        if updated_session and updated_session.state:
            fresh_soft_state = updated_session.state.get("soft_state")
            if isinstance(fresh_soft_state, dict) and fresh_soft_state:
                merged_state["soft_state"] = _merge_soft_state(
                    merged_state.get("soft_state"),
                    fresh_soft_state,
                )
                logger.debug("[ADK] ADK session soft_state merged for %s", session_id)


        if merged_state.get("soft_state"):
            meta = current_snapshot.get("meta") or {}
            await save_session_snapshot(
                session_id=session_id,
                history=current_snapshot.get("history", []),
                state=merged_state,
                metadata={
                    key: meta[key]
                    for key in ("app_name", "user_id", "last_update_time")
                    if key in meta
                },
            )
            logger.debug("[ADK] soft_state persisted for session %s", session_id)
    except Exception as exc:
        logger.warning("[ADK] Could not persist soft_state to Redis: %s", exc)
    frame_obj = None
    try:
        if updated_session and updated_session.state:
            router_output = router_output or str(updated_session.state.get("router_output", "") or "")
            user_cognitive_context = _normalize_cognitive_context(
                updated_session.state.get("user_cognitive_context", user_cognitive_context)
            )
            understanding_frame_json = ""
            if updated_session and updated_session.state and _cfg.feature_understanding_frame:
                raw_frame = updated_session.state.get("understanding")
                if raw_frame is not None:
                    try:
                        if isinstance(raw_frame, dict):
                            frame_obj = UnderstandingFrame(**raw_frame)
                        elif isinstance(raw_frame, UnderstandingFrame):
                            frame_obj = raw_frame
                        elif isinstance(raw_frame, str):
                            import json as _json
                            frame_obj = UnderstandingFrame(**_json.loads(raw_frame))
                        else:
                            frame_obj = None

                        if frame_obj is not None:
                            understanding_frame_json = frame_obj.to_compact_json()
                            logger.debug(
                                "[ADK] Understanding frame captured: intent=%s confidence=%.2f mood=%s",
                                frame_obj.primary_intent, frame_obj.confidence, frame_obj.user_mood,
                            )
                    except Exception as _frame_exc:
                        logger.warning("[ADK] Failed to parse UnderstandingFrame: %s", _frame_exc)

    except Exception:
        pass

    policy_override_json = None
    policy_override_applied = False
    _mode = (_cfg.feature_policy_router_mode or "off").lower()

    if _mode in ("shadow", "enforce") and frame_obj is not None:
        try:
            soft_state_for_policy = (
                updated_session.state.get("soft_state", {})
                if updated_session and updated_session.state else {}
            )
            decision = policy_router.decide(frame_obj, soft_state_for_policy)
            actual_tool = tool_calls_log[-1]["tool"] if tool_calls_log else None
            override = policy_router.compute_override(decision, actual_tool)

            if override:
                override_record = {
                    "mode": _mode,
                    "applied": False,
                    "decision":{
                        "action": decision.get("action"),
                        "tool_name": decision.get("tool_name"),
                        "effective_intent": decision.get("effective_intent"),
                        "matched_priority_id": decision.get("matched_priority_id"),
                        "reasoning": decision.get("reasoning"),
                    },
                    "actual_tool": override.get("actual_tool"),
                }
                logger.info(
                    "[POLICY_OVERRIDE] mode=%s actual=%s policy=%s action=%s reason=%s",
                    _mode,
                    override.get("actual_tool"),
                    override.get("tool_name"),
                    override.get("action"),
                    override.get("reasoning"),
                )

                if _mode == "enforce" and decision.get("action") != "execute_tool":
                    synthetic = policy_router.synthesize_router_output(decision)
                    router_output = json.dumps(synthetic, ensure_ascii=False)
                    final_reply = ""
                    policy_override_applied = True
                    override_record["applied"] = True
                    logger.info(
                        "[POLICY_OVERRIDE] enforced action=%s status=%s",
                        decision.get("action"), synthetic.get("status"),
                    )
                policy_override_json = json.dumps(override_record, ensure_ascii=False)
        except Exception as _policy_exc:
            logger.warning("[ADK] policy_router failed: %s", _policy_exc)
            
    if anomaly_triggered:
        final_reply = anomaly.GRACEFUL_FALLBACK_REPLY
        yield final_reply

    if not final_reply:
        try:
            if updated_session and updated_session.state:
                final_reply = str(updated_session.state.get("final_reply", "") or "")
        except Exception:
            pass

    if not final_reply and router_output:
        if updated_session and updated_session.state:
            final_reply = str(updated_session.state.get("final_reply", "") or "")


    if not final_reply and pipeline_failed_reply:
        final_reply = pipeline_failed_reply
        yield final_reply
    if not final_reply:
        final_reply = str(
            getattr(_cfg, "messages_pipeline_timeout_fallback", None)
            or "I'm sorry, I couldn't process your request. Could you try again?"
        )
        yield final_reply

    logged_reply = sanitize_output(final_reply)

    try:
        asyncio.create_task(
            telemetry.log_trajectory(
                session_id=session_id,
                user_id=user_id,
                user_message=cleaned_message,
                tool_calls=tool_calls_log,
                final_reply=logged_reply,
                latency_ms=latency_ms,
                cognitive_context=user_cognitive_context or None,
                understanding_frame_json=understanding_frame_json or None,
                policy_override_json=policy_override_json,
                turn_count=turn_count,
            )
        )
    except Exception:
        pass

    try:
        from ..observability.db_logging import log_chat

        asyncio.create_task(log_chat(cleaned_message, logged_reply))
    except Exception:
        pass
    
    trace.end()

from app.config.service_coverage_loader import evaluate_message_coverage 

from app.services.adk_runner.invocation import _build_invocation_state_delta 

def _extract_text_parts(event) -> str:
    """Extract text from ADK/Event-like objects without assuming one shape."""
    if event is None:
        return ""

    if isinstance(event, str):
        return event

    direct_text = (
        event.get("text") if isinstance(event, dict) else getattr(event, "text", None)
    )
    if direct_text:
        return str(direct_text)

    if isinstance(event, dict):
        content = event.get("content") or {}
    else:
        content = getattr(event, "content", None) or {}

    if isinstance(content, dict):
        parts = content.get("parts") or []
    else:
        parts = getattr(content, "parts", None) or []

    texts = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
        else:
            text = getattr(part, "text", None)
        if text:
            texts.append(str(text))

    return "\n".join(texts).strip()
