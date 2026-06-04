"""
Langfuse Observability Wrapper — SDK 4.7.1 Compatible

Implementation strategy
-----------------------
We wrap every trace emission in a tiny helper function decorated at call-time
with ``@observe(name=<trace_name>)``.  This causes Langfuse to register the
call as a named observation.  Inside that helper we additionally call
``client.update_current_span`` (if the client supports it) so that metadata
is attached to the active span.

Call-site interface (unchanged):
    with observer.trace("chat_turn", metadata={…}) as trace:
        trace.update(metadata={…})
        with trace.span("step"):
            …

    observer.trace("booking_flow").update(metadata={…})

    t = observer.trace("chat_turn", user_id=uid)
    t.update(metadata={…})
    t.end()

Design constraints:
  - Redis behaviour unchanged.
  - Langfuse optional and non-blocking; every SDK call wrapped in try/except.
  - ``langfuse`` module symbol exposed at module level (required by tests that
    patch ``app.services.observability.langfuse_observer.langfuse``).
  - ``_langfuse`` attribute on LangfuseObserver keeps working for prompt_registry.
  - If ``observe`` is None (import failed), all traces silently become NoOp.
"""
from __future__ import annotations

import logging
import os
import random
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import langfuse                            
    from langfuse import Langfuse
    from langfuse import observe
except Exception:
    langfuse = None                              
    Langfuse = None                            
    observe = None                               

LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in ("true", "1", "yes")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "http://127.0.0.1:3000").strip()
LANGFUSE_ENVIRONMENT = os.getenv("LANGFUSE_ENVIRONMENT", "dev").strip()
LANGFUSE_RELEASE = os.getenv("LANGFUSE_RELEASE", "").strip()
LANGFUSE_PROMPTS_ENABLED = (
    os.getenv("LANGFUSE_PROMPTS_ENABLED", "false").strip().lower() in ("true", "1", "yes")
)
LANGFUSE_SAMPLE_RATE = float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
LANGFUSE_REDACT_INPUTS = (
    os.getenv("LANGFUSE_REDACT_INPUTS", "true").strip().lower() in ("true", "1", "yes")
)
LANGFUSE_MAX_TEXT_CHARS = int(os.getenv("LANGFUSE_MAX_TEXT_CHARS", "2000"))

_observer: Optional["LangfuseObserver"] = None


def _is_configured() -> bool:
    """Return True when both API credentials are present."""
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def _should_sample() -> bool:
    """Honour LANGFUSE_SAMPLE_RATE (0.0–1.0).  1.0 always traces."""
    if LANGFUSE_SAMPLE_RATE >= 1.0:
        return True
    return random.random() < LANGFUSE_SAMPLE_RATE


def sanitize_for_observability(value: Any) -> Any:
    """
    Redact PII and secrets from a value for safe observability logging.
    Preserves safe keys while omitting or redacting sensitive ones.
    """
    if value is None:
        return None

    if isinstance(value, str):
        if len(value) > LANGFUSE_MAX_TEXT_CHARS:
            value = value[:LANGFUSE_MAX_TEXT_CHARS] + "..."
        value = re.sub(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "[REDACTED_EMAIL]",
            value,
        )
        value = re.sub(r"\+?\d[\d -]{8,12}\d", "[REDACTED_PHONE]", value)
        value = re.sub(
            r"(?i)(postgres|mysql|mongodb|redis)://[^\s]+",
            "[REDACTED_DB_URL]",
            value,
        )
        return value

    if isinstance(value, dict):
        sensitive_keys = {
            "password", "secret", "token", "api_key", "authorization", "cookie",
            "database_url", "db_url", "connection_string", "dsn", "email", "phone",
        }
        sanitized: Dict[str, Any] = {}
        for k, v in value.items():
            if any(sens in str(k).lower() for sens in sensitive_keys):
                continue
            sanitized[k] = sanitize_for_observability(v)
        return sanitized

    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_observability(item) for item in value]

    if isinstance(value, (int, float, bool)):
        return value

    return str(value)


def summarize_soft_state(soft_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(soft_state, dict):
        return {}

    summary: Dict[str, Any] = {
        "keys": list(soft_state.keys()),
        "last_presented_view": soft_state.get("last_presented_view"),
        "booking_stage": soft_state.get("booking_stage"),
    }

    visible_results = soft_state.get("visible_results")
    summary["visible_results_count"] = (
        len(visible_results) if isinstance(visible_results, list) else 0
    )

    option_map = soft_state.get("option_map")
    summary["option_map_count"] = len(option_map) if isinstance(option_map, dict) else 0

    all_search_results = soft_state.get("all_search_results")
    summary["all_search_results_count"] = (
        len(all_search_results) if isinstance(all_search_results, list) else 0
    )

    summary["booking_property_id_present"] = bool(soft_state.get("booking_property_id"))
    summary["booking_review_present"] = bool(soft_state.get("booking_review"))
    summary["booking_receipt_present"] = bool(soft_state.get("booking_receipt"))

    return summary


def summarize_property_results(results: Any) -> Dict[str, Any]:
    if not isinstance(results, list):
        return {"count": 0, "has_results": False}

    return {
        "count": len(results),
        "has_results": len(results) > 0,
        "first_property_id": (
            results[0].get("id") if results and isinstance(results[0], dict) else None
        ),
    }


def summarize_booking_state(soft_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(soft_state, dict):
        return {}

    booking_state = soft_state.get("booking_state", {})
    if not isinstance(booking_state, dict):
        booking_state = {}

    summary: Dict[str, Any] = {
        "stage": soft_state.get("booking_stage"),
        "property_id_present": bool(booking_state.get("property_id")),
        "guest_name_present": bool(booking_state.get("guest_name")),
        "guest_email_present": bool(booking_state.get("guest_email")),
        "guest_phone_present": bool(booking_state.get("guest_phone")),
        "check_in_present": bool(booking_state.get("check_in")),
        "check_out_present": bool(booking_state.get("check_out")),
        "guests": booking_state.get("guests"),
    }

    if LANGFUSE_REDACT_INPUTS:
        if summary["guest_name_present"]:
            summary["guest_name"] = "[REDACTED]"
        if summary["guest_email_present"]:
            summary["guest_email"] = "[REDACTED]"
        if summary["guest_phone_present"]:
            summary["guest_phone"] = "[REDACTED]"
    else:
        summary["guest_name"] = booking_state.get("guest_name")
        summary["guest_email"] = booking_state.get("guest_email")
        summary["guest_phone"] = booking_state.get("guest_phone")

    return summary

class NoOpSpan:
    """Silent no-op span — returned when Langfuse is disabled or unavailable."""

    def update(self, **kwargs: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class NoOpTrace:
    """
    Silent no-op trace — returned when Langfuse is disabled or unavailable.

    Supports every call pattern used across the codebase:
      - ``trace.update(...)``           — fire-and-forget metadata update
      - ``trace.end()``                 — explicit end
      - ``with trace.span(name): …``    — nested span context manager
      - ``with observer.trace() as t:`` — trace as context manager
    """

    def span(self, name: str, **kwargs: Any) -> NoOpSpan:
        return NoOpSpan()

    def update(self, **kwargs: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "NoOpTrace":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class NoOpObserver:
    """Observer returned when Langfuse is fully disabled."""

    def trace(self, name: str, **kwargs: Any) -> NoOpTrace:
        return NoOpTrace()

    def flush(self) -> None:
        pass

def _emit_langfuse_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Target function wrapped by ``@observe`` at call-time.

    Langfuse intercepts this call and records a named observation.
    We also try to attach the payload as span metadata via the client
    helpers, but every call is guarded so no exception can escape.

    Returns a minimal summary dict (becomes the span ``output`` in Langfuse).
    """
    try:
        from langfuse import get_client as _get_client 
        _client = _get_client()
        if _client is not None:
            if hasattr(_client, "update_current_span"):
                try:
                    _client.update_current_span(metadata=payload)
                except Exception:
                    pass
            if hasattr(_client, "set_current_trace_io"):
                try:
                    _client.set_current_trace_io(
                        input=payload.get("input"),
                        output=payload.get("output"),
                    )
                except Exception:
                    pass
    except Exception:
        pass

    return {"status": "ok", "metadata_keys": list(payload.keys())}

class _ObservedSpan:
    """
    Lightweight child-span object returned by ``_ObservedTrace.span()``.

    In phase-1 we record child-span metadata locally into the parent trace
    payload (under ``child_spans``).  This avoids needing real nested SDK
    spans while still capturing the structured information.
    """

    def __init__(self, parent: "_ObservedTrace", name: str, metadata: Any = None) -> None:
        self._parent = parent
        self._name = name
        self._metadata = metadata

    def update(self, **kwargs: Any) -> None:
        """Merge span metadata into the parent trace payload."""
        try:
            meta = sanitize_for_observability(kwargs.get("metadata", {}))
            self._parent._child_spans.append(
                {"span_name": self._name, "metadata": meta}
            )
        except Exception as exc:
            logger.debug("[Langfuse] Span update failed (non-fatal): %s", exc)

    def end(self) -> None:
        pass

    def __enter__(self) -> "_ObservedSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

class _ObservedTrace:
    """
    Wraps a Langfuse v4 observation created via ``@observe(name=…)``.

    Lifecycle
    ---------
    Metadata is accumulated in ``self._payload`` during the trace lifetime.
    The actual SDK call (``_emit_langfuse_event``) happens exactly once,
    driven by ``_emit()``:

    * **Context manager** (``with observer.trace(…) as t:``) — ``_emit``
      fires on ``__exit__``.
    * **Fire-and-forget** (``observer.trace(…).update(…)``) — ``_emit``
      fires immediately inside ``update()`` because ``_in_context`` is False.
    * **Explicit lifecycle** (``t = observer.trace(…); t.end()``) — ``_emit``
      fires on ``end()``.
    """

    def __init__(
        self,
        name: str,
        initial_payload: Dict[str, Any],
        client: Any,
    ) -> None:
        self._name = name
        self._payload: Dict[str, Any] = sanitize_for_observability(initial_payload) or {}
        self._client = client
        self._child_spans: List[Dict[str, Any]] = []
        self._in_context: bool = False
        self._sent: bool = False


    def span(self, name: str, **kwargs: Any) -> _ObservedSpan:
        """Return a child span that records locally into this trace."""
        meta = kwargs.get("metadata")
        return _ObservedSpan(parent=self, name=name, metadata=meta)

    def update(self, **kwargs: Any) -> None:
        """
        Merge sanitized kwargs into the accumulated payload.

        If we are NOT inside a ``with`` block (fire-and-forget pattern),
        emit immediately so the trace reaches Langfuse.
        """
        try:
            for k, v in kwargs.items():
                if k in ("metadata", "input", "output"):
                    existing = self._payload.get(k)
                    sanitized_v = sanitize_for_observability(v)
                    if isinstance(existing, dict) and isinstance(sanitized_v, dict):
                        existing.update(sanitized_v)
                    else:
                        self._payload[k] = sanitized_v
                else:
                    self._payload[k] = v
        except Exception as exc:
            logger.debug("[Langfuse] Trace update merge failed: %s", exc)

        if not self._in_context:
            self._emit()

    def end(self) -> None:
        """Explicitly end the trace (called by adk_runner on early returns)."""
        self._emit()

    def __enter__(self) -> "_ObservedTrace":
        self._in_context = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._in_context = False
        self._emit()

    def _emit(self) -> None:
        """
        Send exactly one observation to Langfuse.

        Idempotent — subsequent calls after the first are silently ignored.
        Uses ``@observe(name=…)`` applied dynamically to ``_emit_langfuse_event``
        so that Langfuse registers a named trace/span without needing
        ``start_as_current_span`` or the old ``client.trace()`` API.
        """
        if self._sent:
            return

        self._sent = True

        _observe_fn = observe
        if _observe_fn is None:
            return

        try:
            final_payload: Dict[str, Any] = dict(self._payload)
            if self._child_spans:
                final_payload["child_spans"] = self._child_spans
            observed_fn = _observe_fn(name=self._name)(_emit_langfuse_event)
            observed_fn(final_payload)

            if self._client is not None and hasattr(self._client, "flush"):
                try:
                    self._client.flush()
                except Exception as flush_exc:
                    logger.debug("[Langfuse] Flush after emit failed: %s", flush_exc)

        except Exception as exc:
            logger.debug(
                "[Langfuse] Emit failed for trace '%s' (non-fatal): %s",
                self._name,
                exc,
            )


class LangfuseObserver:
    """
    Failure-tolerant Langfuse observer for SDK 4.7.1.

    Exposes ``trace()`` which returns an ``_ObservedTrace`` (live) or
    ``NoOpTrace`` (disabled/unavailable).  Both share the same interface so
    call-sites never need to branch.
    """

    def __init__(self) -> None:
        self._langfuse: Any = None
        self._is_active: bool = False

        if not LANGFUSE_ENABLED:
            logger.debug("[Langfuse] Disabled via environment variable.")
            return

        if not _is_configured():
            logger.warning(
                "[Langfuse] Enabled but missing PUBLIC_KEY or SECRET_KEY. "
                "Falling back to no-op."
            )
            return

        if Langfuse is None:
            logger.warning("[Langfuse] SDK import failed. Falling back to no-op.")
            return

        if observe is None:
            logger.warning(
                "[Langfuse] 'observe' not available from langfuse package. "
                "Falling back to no-op."
            )
            return

        try:
            self._langfuse = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_BASE_URL,
            )
            self._is_active = True
            logger.info(
                "[Langfuse] Initialized (SDK %s). Environment=%s.",
                getattr(langfuse, "__version__", "?") if langfuse is not None else "?",
                LANGFUSE_ENVIRONMENT,
            )
        except Exception as exc:
            logger.error(
                "[Langfuse] Failed to initialize: %s. Falling back to no-op.", exc
            )

    def is_active(self) -> bool:
        return self._is_active

    def trace(self, name: str, **kwargs: Any) -> "_ObservedTrace | NoOpTrace":
        """
        Start a new trace.  Returns an ``_ObservedTrace`` when active or a
        ``NoOpTrace`` when disabled/misconfigured.

        Supported call patterns
        -----------------------
        ::

            with observer.trace("chat_turn", metadata={…}) as t:
                t.update(metadata={…})
                with t.span("step"):
                    …

            observer.trace("booking_flow").update(metadata={…})

            t = observer.trace("chat_turn", user_id=uid)
            t.update(metadata={…})
            t.end()
        """
        if not self._is_active or self._langfuse is None:
            return NoOpTrace()

        if not _should_sample():
            return NoOpTrace()

        initial_payload: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in ("metadata", "input", "output"):
                initial_payload[k] = sanitize_for_observability(v)
            else:
                initial_payload[k] = v

        return _ObservedTrace(
            name=name,
            initial_payload=initial_payload,
            client=self._langfuse,
        )

    def flush(self) -> None:
        """Flush pending events to the Langfuse server (best-effort)."""
        if self._is_active and self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as exc:
                logger.error("[Langfuse] Failed to flush: %s", exc)


def get_observer() -> "LangfuseObserver | NoOpObserver":
    """
    Return the singleton ``LangfuseObserver``.

    Initialises the observer on first call.  Always returns an object with
    ``trace()`` and ``flush()`` — never raises.
    """
    global _observer
    if _observer is None:
        _observer = LangfuseObserver()
    return _observer
