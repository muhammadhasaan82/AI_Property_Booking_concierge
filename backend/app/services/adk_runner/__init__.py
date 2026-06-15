from __future__ import annotations
"""
ADK 2.0 Runner - compatibility facade for the adk_runner package.
"""

from app.services.adk_runner.events import _extract_tool_call, _extract_tool_response
from app.services.adk_runner.handlers import (
    _maybe_handle_active_booking_turn,
    _maybe_handle_booking_amendment_turn,
    _maybe_handle_booking_cancellation_turn,
    _maybe_handle_booking_status_check,
    _maybe_handle_faq_resume_turn,
    _maybe_handle_search_state_shortcut,
    _maybe_record_unsupported_region,
)
from app.services.adk_runner.rendering import (
    _deterministic_reply_from_router_output,
    _render_no_results_from_search_payload,
    _render_property_details_from_router_output,
    _render_property_results_from_router_output,
)
from app.services.adk_runner.session_service import (
    RedisSessionService,
    _get_runner,
    _get_session_service,
)
from app.services.adk_runner.state_helpers import (
    _merge_soft_state,
    _soft_state_from_router_output,
    _state_with_persisted_soft_state,
)
from app.services.adk_runner.turn import run_adk_turn

__all__ = [
    "RedisSessionService",
    "run_adk_turn",
    "_extract_tool_call",
    "_extract_tool_response",
    "_merge_soft_state",
    "_render_property_results_from_router_output",
    "_state_with_persisted_soft_state",
]

# Compatibility exports for legacy adk_runner.py imports/tests.
from app.services.adk_runner.invocation import APP_NAME  # noqa: F401
from app.services.redis_store import get_session_snapshot  # noqa: F401
from app.services.redis_store import save_session_snapshot  # noqa: F401
from app.services.redis_store import clear_session_snapshot  # noqa: F401
from app.services.pre_router import route_pre_adk  # noqa: F401
from app.security.guardrails import sanitize_input  # noqa: F401
from app.security.guardrails import sanitize_output  # noqa: F401
from app.services.property_query_constraints import get_observer  # noqa: F401
from app.services.booking.creation import start_booking_for_selected_property as _start_booking_for_selected_property  # noqa: F401

from app.services.adk_runner.rendering import _render_voice_from_router_output  # noqa: F401
from app.services.adk_runner.invocation import _build_invocation_state_delta  # noqa: F401
from app.services.direct_property_search import maybe_handle_direct_property_search  # noqa: F401

# Compatibility wrapper: legacy tests monkeypatch app.services.adk_runner.*.
# The split implementation lives in app.services.adk_runner.turn, so copy
# facade-level monkeypatches into that module before each turn.
from app.services.adk_runner.turn import run_adk_turn as _run_adk_turn_impl

def _sync_facade_monkeypatches_to_turn_module() -> None:
    import sys
    import app.services.adk_runner.turn as _turn

    facade = sys.modules[__name__]
    names = [
        "get_session_snapshot",
        "save_session_snapshot",
        "clear_session_snapshot",
        "route_pre_adk",
        "sanitize_input",
        "sanitize_output",
        "get_observer",
        "maybe_handle_direct_property_search",
        "_render_voice_from_router_output",
        "_render_property_results_from_router_output",
        "_build_invocation_state_delta",
    ]

    for name in names:
        if hasattr(facade, name):
            setattr(_turn, name, getattr(facade, name))

async def run_adk_turn(*args, **kwargs):
    _sync_facade_monkeypatches_to_turn_module()
    async for chunk in _run_adk_turn_impl(*args, **kwargs):
        yield chunk

# Compatibility wrapper v2: sync facade monkeypatches into all split ADK modules.
from app.services.adk_runner.turn import run_adk_turn as _run_adk_turn_impl_v2

def _sync_facade_monkeypatches_to_adk_modules() -> None:
    import sys
    import app.services.adk_runner.turn as _turn
    import app.services.adk_runner.handlers as _handlers
    import app.services.adk_runner.rendering as _rendering
    import app.services.adk_runner.invocation as _invocation
    import app.services.adk_runner.session_service as _session_service
    import app.services.adk_runner.state_helpers as _state_helpers

    facade = sys.modules[__name__]
    modules = (_turn, _handlers, _rendering, _invocation, _session_service, _state_helpers)

    names = [
        "get_session_snapshot",
        "save_session_snapshot",
        "clear_session_snapshot",
        "route_pre_adk",
        "sanitize_input",
        "sanitize_output",
        "get_observer",
        "maybe_handle_direct_property_search",
        "_render_voice_from_router_output",
        "_render_property_results_from_router_output",
        "_build_invocation_state_delta",
        "_state_with_persisted_soft_state",
    ]

    for name in names:
        if hasattr(facade, name):
            value = getattr(facade, name)
            for module in modules:
                if hasattr(module, name):
                    setattr(module, name, value)

async def run_adk_turn(*args, **kwargs):
    _sync_facade_monkeypatches_to_adk_modules()
    async for chunk in _run_adk_turn_impl_v2(*args, **kwargs):
        yield chunk

# Compatibility wrapper v3: include runner/session-service monkeypatches.
from app.services.adk_runner.turn import run_adk_turn as _run_adk_turn_impl_v3

def _sync_facade_monkeypatches_to_adk_modules_v3() -> None:
    import sys
    import app.services.adk_runner.turn as _turn
    import app.services.adk_runner.handlers as _handlers
    import app.services.adk_runner.rendering as _rendering
    import app.services.adk_runner.invocation as _invocation
    import app.services.adk_runner.session_service as _session_service
    import app.services.adk_runner.state_helpers as _state_helpers

    facade = sys.modules[__name__]
    modules = (_turn, _handlers, _rendering, _invocation, _session_service, _state_helpers)

    names = [
        "get_session_snapshot",
        "save_session_snapshot",
        "clear_session_snapshot",
        "route_pre_adk",
        "sanitize_input",
        "sanitize_output",
        "get_observer",
        "maybe_handle_direct_property_search",
        "_get_runner",
        "_get_session_service",
        "_render_voice_from_router_output",
        "_render_property_results_from_router_output",
        "_build_invocation_state_delta",
        "_state_with_persisted_soft_state",
        "_extract_text_parts",
    ]

    for name in names:
        if hasattr(facade, name):
            value = getattr(facade, name)
            for module in modules:
                # Set unconditionally; split modules may reference names dynamically.
                setattr(module, name, value)

async def run_adk_turn(*args, **kwargs):
    _sync_facade_monkeypatches_to_adk_modules_v3()
    async for chunk in _run_adk_turn_impl_v3(*args, **kwargs):
        yield chunk

# Compatibility wrapper v4: sync booking_flow.check_faq monkeypatch into split booking modules.
from app.services.adk_runner.turn import run_adk_turn as _run_adk_turn_impl_v4

def _sync_booking_flow_facade_to_booking_modules_v4() -> None:
    try:
        import app.services.booking_flow as _booking_flow
        import app.services.booking.creation as _booking_creation
        import app.services.booking.faq as _booking_faq
    except Exception:
        return

    if hasattr(_booking_flow, "check_faq"):
        patched = getattr(_booking_flow, "check_faq")
        setattr(_booking_creation, "check_faq", patched)
        setattr(_booking_faq, "check_faq", patched)

async def run_adk_turn(*args, **kwargs):
    _sync_facade_monkeypatches_to_adk_modules_v3()
    _sync_booking_flow_facade_to_booking_modules_v4()
    async for chunk in _run_adk_turn_impl_v4(*args, **kwargs):
        yield chunk

# Compatibility wrapper v5: sync booking cancellation monkeypatches.
from app.services.adk_runner.turn import run_adk_turn as _run_adk_turn_impl_v5

def _sync_booking_cancellation_monkeypatches_v5() -> None:
    try:
        import app.services.booking as _booking_pkg
        import app.services.booking.cancellation as _booking_cancellation
        import app.observability.db_logging as _db_logging
    except Exception:
        return

    for name in ("get_booking_status", "update_booking_status"):
        if hasattr(_booking_pkg, name):
            setattr(_booking_cancellation, name, getattr(_booking_pkg, name))

    for name in ("get_successful_booking_status", "update_successful_booking"):
        if hasattr(_db_logging, name):
            setattr(_booking_cancellation, name, getattr(_db_logging, name))

async def run_adk_turn(*args, **kwargs):
    _sync_facade_monkeypatches_to_adk_modules_v3()
    _sync_booking_flow_facade_to_booking_modules_v4()
    _sync_booking_cancellation_monkeypatches_v5()
    async for chunk in _run_adk_turn_impl_v5(*args, **kwargs):
        yield chunk

# Compatibility wrapper v6: direct helper calls must also sync facade monkeypatches.
from app.services.adk_runner.handlers import (
    _maybe_handle_faq_resume_turn as _maybe_handle_faq_resume_turn_impl_v6,
    _maybe_handle_search_state_shortcut as _maybe_handle_search_state_shortcut_impl_v6,
    _maybe_handle_booking_status_check as _maybe_handle_booking_status_check_impl_v6,
)

def _sync_all_facade_compat_v6() -> None:
    _sync_facade_monkeypatches_to_adk_modules_v3()
    _sync_booking_flow_facade_to_booking_modules_v4()
    _sync_booking_cancellation_monkeypatches_v5()

async def _maybe_handle_faq_resume_turn(*args, **kwargs):
    _sync_all_facade_compat_v6()
    return await _maybe_handle_faq_resume_turn_impl_v6(*args, **kwargs)

async def _maybe_handle_search_state_shortcut(*args, **kwargs):
    _sync_all_facade_compat_v6()
    return await _maybe_handle_search_state_shortcut_impl_v6(*args, **kwargs)

async def _maybe_handle_booking_status_check(*args, **kwargs):
    _sync_all_facade_compat_v6()
    return await _maybe_handle_booking_status_check_impl_v6(*args, **kwargs)
