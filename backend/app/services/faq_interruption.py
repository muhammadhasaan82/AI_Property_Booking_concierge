from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Optional

from app.config.agent_config_loader import cfg

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")
_ACTIVE_BOOKING_STAGES = {
    "collecting_details",
    "awaiting_confirmation",
    "awaiting_modification_choice",
    "modifying_details",
    "awaiting_property_reselection",
}


def _config_block() -> Dict[str, Any]:
    raw = getattr(cfg, "_raw", {})
    if not isinstance(raw, dict):
        return {}
    block = raw.get("faq_interruption")
    return dict(block) if isinstance(block, dict) else {}


def _normalize(text: str) -> str:
    if not text:
        return ""
    lowered = str(text).lower()
    for src in ("\u2019", "\u2018", "\u02bc", "`"):
        lowered = lowered.replace(src, "'")
    cleaned = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", cleaned).strip()


def _tokenize(text: str) -> List[str]:
    return [token for token in _normalize(text).split(" ") if token]


def _group_matches(group: Iterable[str], token_set: Iterable[str]) -> bool:
    tokens = [str(token).strip() for token in group if str(token).strip()]
    if not tokens:
        return False
    available = set(token_set)
    return all(token in available for token in tokens)


def _matches_semantic_cues(message: str, cues: Any) -> bool:
    if not isinstance(cues, dict):
        return False
    token_set = set(_tokenize(message))
    any_groups = [group for group in (cues.get("any") or []) if isinstance(group, list) and group]
    if not any_groups:
        return False
    for group in cues.get("none") or []:
        if isinstance(group, list) and _group_matches(group, token_set):
            return False
    return any(_group_matches(group, token_set) for group in any_groups)


def get_state_key() -> str:
    block = _config_block()
    return str(block.get("state_key") or "faq_interruption")


def get_faq_interruption(soft_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(soft_state, dict):
        return {}
    value = soft_state.get(get_state_key())
    return value if isinstance(value, dict) else {}


def is_active(soft_state: Optional[Dict[str, Any]]) -> bool:
    return bool(get_faq_interruption(soft_state).get("active"))


def clear_faq_interruption(soft_state: Optional[Dict[str, Any]]) -> None:
    if isinstance(soft_state, dict):
        soft_state.pop(get_state_key(), None)


def detect_policy_question(message: str) -> bool:
    if not message or not message.strip():
        return False
    from app.components.faq_enhanced import detect_faq_intent

    return bool(detect_faq_intent(message))


def detect_resume_cue(message: str) -> bool:
    block = _config_block()
    cues = block.get("resume_cues") or {}
    normalized = _normalize(message)
    if not normalized:
        return False

    exact = {
        _normalize(item)
        for item in (cues.get("exact") or [])
        if isinstance(item, str) and _normalize(item)
    }
    if normalized in exact:
        return True

    return _matches_semantic_cues(normalized, cues.get("semantic_cues"))


def _selected_property_from_state(soft_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ("selected_property", "booking_selected_property"):
        value = soft_state.get(key)
        if isinstance(value, dict) and value:
            return copy.deepcopy(value)

    selected_id = soft_state.get("last_selected_property_id") or soft_state.get("selected_property_id")
    if selected_id in (None, ""):
        return None

    for key in ("last_visible_results", "visible_results", "all_search_results"):
        values = soft_state.get(key) or []
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict) and str(item.get("id")) == str(selected_id):
                return copy.deepcopy(item)
    return None


def sync_alias_keys(soft_state: Optional[Dict[str, Any]]) -> None:
    if not isinstance(soft_state, dict):
        return

    visible_results = soft_state.get("visible_results")
    if isinstance(visible_results, list) and visible_results:
        soft_state["last_visible_results"] = copy.deepcopy(visible_results)

    last_filters = soft_state.get("last_filters")
    if isinstance(last_filters, dict) and last_filters:
        soft_state["last_search_filters"] = copy.deepcopy(last_filters)

    selected_property = _selected_property_from_state(soft_state)
    if selected_property:
        soft_state["selected_property"] = copy.deepcopy(selected_property)


def resolve_resume_target(soft_state: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(soft_state, dict):
        return None

    booking_stage = str(soft_state.get("booking_stage") or "").strip()
    if (
        booking_stage in _ACTIVE_BOOKING_STAGES
        or soft_state.get("awaiting_field")
        or soft_state.get("booking_state")
        or soft_state.get("pending_booking")
    ):
        return "booking_flow"

    last_view = str(soft_state.get("last_presented_view") or "").strip()
    if last_view == "property_details" and _selected_property_from_state(soft_state):
        return "selected_property"

    if any(
        soft_state.get(key)
        for key in ("last_visible_results", "visible_results", "all_search_results", "option_map", "last_search")
    ):
        return "property_menu"

    if _selected_property_from_state(soft_state):
        return "selected_property"
    return None


def build_resume_payload(
    soft_state: Optional[Dict[str, Any]],
    resume_target: Optional[str],
) -> Dict[str, Any]:
    if not isinstance(soft_state, dict) or not resume_target:
        return {}

    if resume_target == "property_menu":
        visible_results = soft_state.get("last_visible_results") or soft_state.get("visible_results") or []
        all_results = soft_state.get("all_search_results") or visible_results
        last_search = soft_state.get("last_search") if isinstance(soft_state.get("last_search"), dict) else {}
        pagination = last_search.get("pagination") if isinstance(last_search, dict) else None
        if not isinstance(pagination, dict):
            shown_count = len(visible_results) if isinstance(visible_results, list) else 0
            total_found = soft_state.get("active_property_options_total_found") or len(all_results) or shown_count
            current_page = soft_state.get("current_page") or 1
            page_size = soft_state.get("page_size") or shown_count or 1
            pagination = {
                "current_page": current_page,
                "page_size": page_size,
                "page_start": 1 if shown_count else 0,
                "page_end": shown_count,
                "total_pages": 1,
                "has_more": False,
                "has_next": False,
                "has_prev": False,
                "pagination_enabled": False,
            }

        return {
            "status": getattr(cfg.status, "properties_found", "properties_found"),
            "properties": copy.deepcopy(visible_results) if isinstance(visible_results, list) else [],
            "all_search_results": copy.deepcopy(all_results) if isinstance(all_results, list) else [],
            "shown_count": len(visible_results) if isinstance(visible_results, list) else 0,
            "total_found": soft_state.get("active_property_options_total_found") or len(all_results) if isinstance(all_results, list) else 0,
            "query_context": copy.deepcopy(
                soft_state.get("last_search_filters")
                or soft_state.get("last_filters")
                or (last_search.get("query_context") if isinstance(last_search, dict) else {})
                or {}
            ),
            "pagination": copy.deepcopy(pagination),
            "summary_mode": bool(last_search.get("summary_mode", False)) if isinstance(last_search, dict) else False,
        }

    if resume_target == "selected_property":
        selected_property = _selected_property_from_state(soft_state)
        return {
            "status": getattr(cfg.status, "property_details", "property_details"),
            "property": selected_property or {},
        }

    if resume_target == "booking_flow":
        return {
            "booking_stage": soft_state.get("booking_stage"),
            "awaiting_field": soft_state.get("awaiting_field"),
            "booking_state": copy.deepcopy(soft_state.get("booking_state") or {}),
            "pending_booking": copy.deepcopy(soft_state.get("pending_booking") or {}),
            "booking_review": copy.deepcopy(soft_state.get("booking_review") or {}),
            "booking_receipt": copy.deepcopy(soft_state.get("booking_receipt") or {}),
            "selected_property": _selected_property_from_state(soft_state) or {},
        }

    return {}


def capture_faq_interruption(
    soft_state: Optional[Dict[str, Any]],
    *,
    last_faq_intent: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(soft_state, dict):
        return {}

    sync_alias_keys(soft_state)
    resume_target = resolve_resume_target(soft_state)
    payload = build_resume_payload(soft_state, resume_target)
    interruption = {
        "active": bool(resume_target),
        "resume_target": resume_target,
        "resume_payload": payload,
        "source": str(_config_block().get("source") or "faq"),
        "last_faq_intent": str(last_faq_intent or "").strip() or None,
    }
    soft_state[get_state_key()] = interruption
    return interruption


def get_resume_prompt(resume_target: Optional[str]) -> str:
    prompts = _config_block().get("prompts") or {}
    if resume_target and isinstance(prompts.get(resume_target), str):
        return str(prompts.get(resume_target)).strip()
    fallback = prompts.get("fallback")
    return str(fallback).strip() if isinstance(fallback, str) else ""


def build_answer_with_resume_prompt(
    answer: str,
    soft_state: Optional[Dict[str, Any]],
) -> str:
    base_answer = str(answer or "").strip()
    if not base_answer:
        return ""

    interruption = get_faq_interruption(soft_state)
    if not interruption.get("active"):
        return base_answer
    prompt = get_resume_prompt(interruption.get("resume_target"))
    if not prompt:
        return base_answer

    joiner = str(_config_block().get("reply_joiner") or "\n\n")
    return f"{base_answer}{joiner}{prompt}"
