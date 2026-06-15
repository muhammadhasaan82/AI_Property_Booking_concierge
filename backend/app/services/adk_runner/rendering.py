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


from app.config.agent_config_loader import cfg as _cfg
from app.config.response_policies_loader import render_policy_snippet
from app.services.booking_flow import (
    resume_booking_flow as _resume_booking_flow,
    start_booking_for_selected_property as _start_booking_for_selected_property_flow,
)

async def _render_voice_from_router_output(
    router_output: str,
    user_cognitive_context: str,
    understanding_frame_json: str = "",
) -> str:
    """
    Constructs a voice-style concierge response from router output and contextual information.
    
    If `router_output` contains routable JSON or text, this function synthesizes a final conversational reply that incorporates the provided user cognitive context and optional understanding frame. Returns an empty string when `router_output` is blank, when synthesis produces no usable text, or when an internal failure occurs.
    
    Parameters:
        router_output (str | dict): Router output as a JSON string or dict containing routing results.
        user_cognitive_context (str): Normalized, truncated user cognitive context to include in the prompt.
        understanding_frame_json (str): Optional compact JSON representation of an UnderstandingFrame to inform generation.
    
    Returns:
        str: The synthesized concierge reply, or an empty string if no reply could be generated.
    """
    if not router_output or not router_output.strip():
        return ""

    try:
        import litellm
        from ..agents.adk_agents import VOICE_CONFIG, VOICE_INSTRUCTION, VOICE_MODEL
        status = None
        if isinstance(router_output, dict):
            status = router_output.get("status")
        response_policy_snippet = ""
        if _cfg.feature_response_policies and status:
            response_policy_snippet = render_policy_snippet(status)

        system_prompt = (
            VOICE_INSTRUCTION
            .replace("{router_output}", router_output)
            .replace("{user_cognitive_context}", _normalize_cognitive_context(user_cognitive_context))
            .replace("{understanding}", understanding_frame_json or "{}")
        )
        temperature = getattr(VOICE_CONFIG, "temperature", 0.6)

        def _generate() -> str:
            response = litellm.completion(
                model=VOICE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "Generate the final concierge response for this turn.",
                    },
                ],
                temperature=temperature,
            )
            if response_policy_snippet:
                system_prompt = system_prompt + "\n\n" + response_policy_snippet
                
            if getattr(response, "choices", None):
                choice0 = response.choices[0]
                message = getattr(choice0, "message", None)
                content = getattr(message, "content", "") if message else ""
                if isinstance(content, str):
                    return content.strip()

            if isinstance(response, dict):
                return (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )

            return ""

        return await asyncio.to_thread(_generate)
    except Exception as exc:
        logger.warning("[ADK] Voice handoff fallback failed: %s", exc)
        return ""

def _configured_booking_details_fields() -> List[str]:
    fields = getattr(_cfg, "booking_details_request_fields", None) or []
    cleaned = [str(field).strip() for field in fields if str(field).strip()]
    if cleaned:
        return cleaned
    return list(_cfg.booking_required_fields + _cfg.booking_required_numeric_fields)

def _property_id_matches(item: Any, selected_id: str) -> bool:
    if not isinstance(item, dict):
        return False
    for key in ("id", "property_id"):
        value = item.get(key)
        if value is not None and str(value) == selected_id:
            return True
    return False

def _resolve_selected_property_from_soft_state(
    soft_state: Dict[str, Any],
    selected_id: str,
) -> Optional[Dict[str, Any]]:
    for state_key in ("visible_results", "all_search_results"):
        candidates = soft_state.get(state_key) or []
        if not isinstance(candidates, list):
            continue
        for item in candidates:
            if _property_id_matches(item, selected_id):
                return dict(item)

    last_search = soft_state.get("last_search")
    if isinstance(last_search, dict):
        for item in last_search.get("properties", []) or []:
            if _property_id_matches(item, selected_id):
                return dict(item)

    active_map = soft_state.get("active_property_options_map")
    if isinstance(active_map, dict):
        for option in active_map.values():
            if not _property_id_matches(option, selected_id):
                continue
            selected = dict(option)
            selected.setdefault("id", selected.get("property_id"))
            return selected

    return None

def _render_booking_details_request(
    *,
    selected_property: Optional[Dict[str, Any]],
    selected_id: str,
    required_fields: List[str],
) -> str:
    title = ""
    if isinstance(selected_property, dict):
        title = str(selected_property.get("title") or "").strip()
    property_title = title or "this property"
    template = str(
        getattr(_cfg, "booking_details_request_prompt_template", "") or ""
    ).strip()
    field_list = "\n".join(f"- {field}" for field in required_fields)
    if not template:
        return ""
    try:
        return template.format(
            property_title=property_title,
            property_id=selected_id,
            required_fields=field_list,
        )
    except Exception as exc:
        logger.warning("[ADK] Booking details prompt template failed: %s", exc)
        return template

def _apply_booking_collection_top_level_compat(
    soft_state: Dict[str, Any],
    *,
    selected_id: str,
    selected_property: Optional[Dict[str, Any]],
    required_fields: List[str],
) -> None:
    """Re-apply legacy top-level booking keys after canonical booking_state helpers.

    Nested ``booking_state`` updates must not erase compatibility fields used by
    shortcuts, debug endpoints, and follow-up collection turns.
    """
    soft_state["booking_property_id"] = selected_id
    soft_state["booking_stage"] = "collecting_details"
    soft_state["last_presented_view"] = "booking_details_request"
    soft_state["booking_required_fields"] = list(required_fields)
    if selected_property:
        soft_state["booking_selected_property"] = selected_property

def _start_booking_for_selected_property(
    soft_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    return _start_booking_for_selected_property_flow(soft_state)

def _render_property_results_from_router_output(router_output: Dict[str, Any]) -> str:
    """Build a deterministic property-list reply directly from router_output.

    Called when status == 'properties_found' and properties array is non-empty.
    NEVER invents titles, prices, ratings, bedrooms, bathrooms, ids or counts —
    every value comes from the tool payload.
    """
    props: List[Dict[str, Any]] = router_output.get("properties") or []
    if not props:
        return ""

    total: int = int(router_output.get("total_found") or len(props))
    shown: int = int(router_output.get("shown_count") or len(props))
    qctx: Dict[str, Any] = router_output.get("query_context") or {}
    city: str = (qctx.get("city") or "").strip().title()
    prop_type: str = (qctx.get("property_type") or "").strip().lower()
    budget = qctx.get("budget")
    bedrooms = qctx.get("bedrooms")
    bedrooms_operator = str(qctx.get("bedrooms_operator") or "").strip().lower()
    pagination: Dict[str, Any] = router_output.get("pagination") or {}
    summary_mode: bool = bool(router_output.get("summary_mode", False))

    if prop_type:
        type_label = f"{prop_type}s"
    else:
        type_label = "properties"
    if bedrooms is not None and bedrooms_operator == "exact" and prop_type:
        type_label = f"{int(bedrooms)}-bedroom {type_label}"
    city_part = f" in {city}" if city else ""
    budget_part = f" under ${int(budget)}" if budget else ""

    total_pages: int = int(pagination.get("total_pages") or 1)
    if total_pages > 1:
        page = int(pagination.get("current_page") or 1)
        page_start = int(pagination.get("page_start") or 1)
        page_end = int(pagination.get("page_end") or shown)
        header = (
            f"I found {total} {type_label}{city_part}{budget_part}. "
            f"Showing {page_start}\u2013{page_end} (page {page} of {total_pages}):"
        )
    else:
        header = f"I found {shown} {type_label}{city_part}{budget_part}:"

    lines: List[str] = [header, ""]

    for item in props:
        num = item.get("number", "")
        title = (item.get("title") or "").strip()
        if not title:
            continue
        price = item.get("price_per_night")
        beds = item.get("bedrooms")
        baths = item.get("bathrooms")
        rating = item.get("rating")
        ptype = (item.get("property_type") or "").strip().title()

        price_str = f"${int(price)}/night" if price is not None else ""
        beds_str = (
            f"{beds} bed{'s' if beds != 1 else ''}" if beds is not None else ""
        )
        baths_str = (
            f"{baths} bath{'s' if baths != 1 else ''}" if baths is not None else ""
        )
        rating_str = f"\u2605 {float(rating):.1f}" if rating is not None else ""

        if summary_mode:
            details = " | ".join(filter(None, [price_str, beds_str, rating_str]))
        else:
            details = " | ".join(
                filter(None, [ptype, price_str, beds_str, baths_str, rating_str])
            )
        lines.append(
            f"{num}. **{title}**" + (f" \u2014 {details}" if details else "")
        )

    lines.append("")

    has_next: bool = bool(pagination.get("has_next", False))
    has_prev: bool = bool(pagination.get("has_prev", False))
    if has_next and has_prev:
        lines.append(
            "Say \u2018next\u2019 for more, \u2018previous\u2019 to go back, "
            "or pick an option number for details."
        )
    elif has_next:
        lines.append("Say \u2018next\u2019 for more results, or pick an option number for details.")
    elif has_prev:
        lines.append("Say \u2018previous\u2019 to go back, or pick an option number for details.")
    else:
        lines.append(
            "Reply with an option number to see full details, "
            "or tell me your preferences to refine the search."
        )

    return "\n".join(lines)


def _render_no_results_from_search_payload(payload: Dict[str, Any]) -> str:
    qctx: Dict[str, Any] = payload.get("query_context") or payload.get("filters_applied") or {}
    city = str(qctx.get("city") or payload.get("city") or "").strip().title()
    property_type = str(
        qctx.get("property_type") or payload.get("property_type") or ""
    ).strip().lower()
    bedrooms = qctx.get("bedrooms")
    bedrooms_operator = str(qctx.get("bedrooms_operator") or "").strip().lower()
    bathrooms = qctx.get("bathrooms")
    amenities = [
        str(item).strip().replace("_", " ")
        for item in (qctx.get("amenities") or payload.get("amenities") or [])
        if str(item).strip()
    ]

    city_part = f" in {city}" if city else ""
    amenity_part = ""
    if amenities:
        amenity_part = f" with {' and '.join(amenities)}"

    if property_type and bedrooms is not None and bedrooms_operator == "exact":
        subject = f"{int(bedrooms)}-bedroom {property_type}s"
        if bathrooms is not None:
            subject = f"{subject} with {int(bathrooms)} bathrooms"
        return (
            f"I couldn't find any exact matches for {subject}{city_part}{amenity_part}. "
            f"I can show other {property_type} options{city_part} or help you relax a filter."
        )

    subject = "matching properties"
    if property_type:
        subject = f"{property_type}s"
    if property_type and amenities:
        subject = f"{property_type}s{amenity_part}"

    return (
        f"I couldn't find any exact matches for {subject}{city_part}. "
        "I can help you relax one of the filters if you'd like."
    )


def _render_property_details_from_router_output(router_output: Dict[str, Any]) -> str:
    """Build a deterministic property-details reply from tool payload fields only."""
    prop: Dict[str, Any] = router_output.get("property") or {}
    if not isinstance(prop, dict) or not prop:
        return ""

    title = (prop.get("title") or "").strip()
    if not title:
        return ""

    city = (prop.get("city") or "").strip().title()
    price = prop.get("price_per_night")
    beds = prop.get("bedrooms")
    baths = prop.get("bathrooms")
    rating = prop.get("rating")
    amenities = prop.get("amenities") or []
    description = (prop.get("description") or "").strip()

    price_str = f"${int(price)}/night" if price is not None else ""
    beds_str = f"{beds} bed{'s' if beds != 1 else ''}" if beds is not None else ""
    baths_str = f"{baths} bath{'s' if baths != 1 else ''}" if baths is not None else ""
    rating_str = f"\u2605 {float(rating):.1f}" if rating is not None else ""
    location = city if city else ""
    amenity_str = ", ".join(str(a) for a in amenities if a) if isinstance(amenities, list) else ""

    lines = [f"**{title}**"]
    meta = " | ".join(filter(None, [location, price_str, beds_str, baths_str, rating_str]))
    if meta:
        lines.append(meta)
    if amenity_str:
        lines.append(f"Amenities: {amenity_str}")
    if description:
        lines.append(description)
    lines.append("")
    lines.append("Want to book this one?")
    return "\n".join(lines)


def _deterministic_reply_from_router_output(router_output: Optional[Dict[str, Any]]) -> str:
    if not isinstance(router_output, dict):
        return ""

    status = str(router_output.get("status") or "").lower()
    if status == str(getattr(_cfg.status, "answered", "answered")).lower():
        return str(router_output.get("deterministic_reply") or router_output.get("answer") or "").strip()
    if status == str(getattr(_cfg.status, "properties_found", "properties_found")).lower():
        return _render_property_results_from_router_output(router_output)
    if status == str(getattr(_cfg.status, "property_details", "property_details")).lower():
        return _render_property_details_from_router_output(router_output)
    return ""


