"""Acceptance tests generated from specs/adk_tool_contract.md."""
from __future__ import annotations

import inspect

from app.agents.tools import booking as booking_tools
from app.agents.tools import search as search_tools
from app.agents.tools import support as support_tools


_BANNED_PARAM_NAMES = {"tool_context", "search_plan"}
_BANNED_TYPE_TOKENS = {"SearchPlan", "DynamicConstraints", "ToolContext"}


def _signature_tokens(fn) -> tuple[list[str], str]:
    signature = inspect.signature(fn)
    return list(signature.parameters.keys()), str(signature)


def test_search_tool_public_signatures_are_json_safe():
    for fn in (
        search_tools.search_properties,
        search_tools.select_property,
        search_tools.get_property_details,
    ):
        param_names, rendered = _signature_tokens(fn)
        assert not (_BANNED_PARAM_NAMES & set(param_names)), rendered
        assert not any(token in rendered for token in _BANNED_TYPE_TOKENS), rendered


def test_booking_tool_public_signatures_are_json_safe():
    for fn in (
        booking_tools.request_booking_details,
        booking_tools.review_booking_details,
        booking_tools.process_v2_booking,
    ):
        param_names, rendered = _signature_tokens(fn)
        assert not (_BANNED_PARAM_NAMES & set(param_names)), rendered
        assert not any(token in rendered for token in _BANNED_TYPE_TOKENS), rendered


def test_support_tool_public_signatures_are_json_safe():
    for fn in (
        support_tools.check_faq,
        support_tools.check_booking_status,
    ):
        param_names, rendered = _signature_tokens(fn)
        assert not (_BANNED_PARAM_NAMES & set(param_names)), rendered
        assert not any(token in rendered for token in _BANNED_TYPE_TOKENS), rendered
