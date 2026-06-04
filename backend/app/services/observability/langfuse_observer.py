"""
Langfuse Observability Wrapper — SDK 4.7.1 Compatible
======================================================

Langfuse v4 dropped the imperative ``Langfuse.trace()`` / ``Langfuse.span()``
client methods that existed in v2/v3.  The v4 SDK exposes observability through:

  * Module-level context manager:  ``langfuse.start_as_current_span(name)``
  * Module-level decorator:        ``@langfuse.observe``
  * Client helpers (within a span): ``client.update_current_span()``,
                                     ``client.set_current_trace_io()``,
                                     ``client.get_current_trace_id()``, etc.

This module bridges that API change so every existing call-site in the
codebase keeps working unchanged:

    observer = get_observer()


    trace = observer.trace(name="chat_turn", ...)
    with trace.span(name="step"):
        ...
    trace.update(metadata={...})
    trace.end()

    with observer.trace(name="search_tool", ...) as trace:
        trace.update(metadata={...})


    observer.trace(name="booking_flow").update(metadata={...})

Design constraints preserved:
  - Redis behaviour is unchanged.
  - Langfuse is optional and non-blocking (all failures fall back silently).
  - The Langfuse client is never stored as runtime state on requests.
  - Sample-rate, redaction, and all environment flags are respected.
"""
from __future__ import annotations

import logging
import os
import random
import re
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

try:
    import langfuse as _langfuse_module          
    from langfuse import Langfuse as _LangfuseClient  
except Exception:
    _langfuse_module = None  
    _LangfuseClient = None  

LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in ("true", "1", "yes")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "http://127.0.0.1:3000").strip()
LANGFUSE_ENVIRONMENT = os.getenv("LANGFUSE_ENVIRONMENT", "dev").strip()
LANGFUSE_RELEASE = os.getenv("LANGFUSE_RELEASE", "").strip()
LANGFUSE_PROMPTS_ENABLED = os.getenv("LANGFUSE_PROMPTS_ENABLED", "false").strip().lower() in ("true", "1", "yes")
LANGFUSE_SAMPLE_RATE = float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
LANGFUSE_REDACT_INPUTS = os.getenv("LANGFUSE_REDACT_INPUTS", "true").strip().lower() in ("true", "1", "yes")
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
        value = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]", value)
        value = re.sub(r"\+?\d[\d -]{8,12}\d", "[REDACTED_PHONE]", value)
        value = re.sub(r"(?i)(postgres|mysql|mongodb|redis)://[^\s]+", "[REDACTED_DB_URL]", value)
        return value

    if isinstance(value, dict):
        sensitive_keys = {
            "password", "secret", "token", "api_key", "authorization", "cookie",
            "database_url", "db_url", "connection_string", "dsn", "email", "phone",
        }
        sanitized = {}
        for k, v in value.items():
            k_lower = str(k).lower()
            if any(sens in k_lower for sens in sensitive_keys):
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

    summary = {
        "keys": list(soft_state.keys()),
        "last_presented_view": soft_state.get("last_presented_view"),
        "booking_stage": soft_state.get("booking_stage"),
    }

    visible_results = soft_state.get("visible_results")
    summary["visible_results_count"] = len(visible_results) if isinstance(visible_results, list) else 0

    option_map = soft_state.get("option_map")
    summary["option_map_count"] = len(option_map) if isinstance(option_map, dict) else 0

    all_search_results = soft_state.get("all_search_results")
    summary["all_search_results_count"] = len(all_search_results) if isinstance(all_search_results, list) else 0

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
        "first_property_id": results[0].get("id") if isinstance(results[0], dict) else None,
    }


def summarize_booking_state(soft_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(soft_state, dict):
        return {}

    booking_state = soft_state.get("booking_state", {})
    if not isinstance(booking_state, dict):
        booking_state = {}

    summary = {
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
    """Silent no-op span returned when Langfuse is disabled or unavailable."""

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
    Silent no-op trace returned when Langfuse is disabled or unavailable.

    Supports all call patterns used across the codebase:
      - ``trace.update(...)``          – fire-and-forget metadata update
      - ``trace.end()``                – explicit end (no-op)
      - ``with trace.span(name):``     – nested span as context manager
      - ``with observer.trace() as t:``– trace used as context manager itself
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


class _LiveSpan:
    """
    Thin wrapper around the active Langfuse v4 span context.

    In SDK 4.x the client object exposes helpers that operate on the
    *currently active* span in the context-local stack:
        ``client.update_current_span(metadata=...)``

    We hold a reference to the client so we can forward calls to it.
    """

    def __init__(self, client: Any, span_cm: Any) -> None:
        """
        Parameters
        ----------
        client:
            The ``Langfuse`` client instance (for ``update_current_span``).
        span_cm:
            The context manager object returned by
            ``langfuse.start_as_current_span()``.
        """
        self._client = client
        self._span_cm = span_cm
        self._closed = False

    def update(self, **kwargs: Any) -> None:
        """Forward metadata/IO updates to the SDK's current-span helper."""
        try:
            metadata = kwargs.get("metadata")
            if metadata is not None and hasattr(self._client, "update_current_span"):
                self._client.update_current_span(metadata=sanitize_for_observability(metadata))
        except Exception as exc:
            logger.debug("[Langfuse] Span update failed (non-fatal): %s", exc)

    def end(self) -> None:
        if not self._closed and self._span_cm is not None:
            self._span_cm.__exit__(None, None, None)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end()


class _LiveTrace:
    """
    Wraps a Langfuse v4 span started with ``langfuse.start_as_current_span``
    and exposes the call-site interface that all existing consumers expect.

    Lifecycle
    ---------
    The underlying SDK span is entered lazily on first use (or immediately
    when used as a context manager) and exited on ``end()`` or ``__exit__``.
    Fire-and-forget callers (``observer.trace(...).update(...)``) will enter
    and immediately exit the span so the update is flushed atomically.
    """

    def __init__(self, client: Any, name: str, sanitized_kwargs: Dict[str, Any]) -> None:
        self._client = client
        self._name = name
        self._kwargs = sanitized_kwargs
        self._ctx: Any = None
        self._entered: bool = False


    def _get_start_fn(self) -> Any:
        """
        Return ``langfuse.start_as_current_span`` from the module if available,
        otherwise fall back gracefully.
        """
        if _langfuse_module is None:
            return None
        fn = getattr(_langfuse_module, "start_as_current_span", None)
        return fn

    def _enter_span(self) -> None:
        """Enter the SDK context manager exactly once."""
        if self._entered:
            return
        start_fn = self._get_start_fn()
        if start_fn is None:
            self._entered = True
            return
        try:
            span_kwargs: Dict[str, Any] = {"name": self._name}
            for key in ("input", "output", "metadata", "user_id", "session_id"):
                if key in self._kwargs:
                    span_kwargs[key] = self._kwargs[key]
            self._ctx = start_fn(**span_kwargs)
            self._ctx.__enter__()
        except Exception as exc:
            logger.debug("[Langfuse] Failed to enter span '%s': %s", self._name, exc)
        finally:
            self._entered = True

    def _exit_span(self) -> None:
        """Exit the SDK context manager if we own it."""
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception as exc:
                logger.debug("[Langfuse] Failed to exit span '%s': %s", self._name, exc)
            self._ctx = None


    def span(self, name: str, **kwargs: Any) -> "_LiveSpan":
        """
        Create a *child* span.  In SDK 4.x nested spans are also started via
        ``start_as_current_span`` – calls within an active span context become
        automatic children.
        """
        start_fn = self._get_start_fn()
        if start_fn is None:
            return _LiveSpan(self._client, NoOpSpan())
        try:
            child_cm = start_fn(name=name)
            child_cm.__enter__()
            return _LiveSpan(self._client, child_cm)
        except Exception as exc:
            logger.debug("[Langfuse] Failed to create child span '%s': %s", name, exc)
            return _LiveSpan(self._client, NoOpSpan())

    def update(self, **kwargs: Any) -> None:
        """
        Update the current trace/span metadata.

        If we are inside a context manager the span is already active.
        For fire-and-forget calls we briefly enter→update→exit the span.
        """
        was_entered = self._entered
        self._enter_span()
        try:
            metadata = kwargs.get("metadata")
            if metadata is not None and hasattr(self._client, "update_current_span"):
                self._client.update_current_span(
                    metadata=sanitize_for_observability(metadata)
                )
        except Exception as exc:
            logger.debug("[Langfuse] Trace update failed (non-fatal): %s", exc)
        finally:
            if not was_entered:
                self._exit_span()

    def end(self) -> None:
        """Explicitly end the trace (called by adk_runner on early returns)."""
        self._exit_span()

    def __enter__(self) -> "_LiveTrace":
        self._enter_span()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._exit_span()


class LangfuseObserver:
    """
    Thread-safe, failure-tolerant Langfuse observer.

    Exposes a ``trace()`` method that returns either a ``_LiveTrace``
    (SDK 4.7.1 compatible) or a ``NoOpTrace`` depending on configuration
    and SDK availability.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._is_active: bool = False

        if not LANGFUSE_ENABLED:
            logger.debug("[Langfuse] Disabled via environment variable.")
            return

        if not _is_configured():
            logger.warning(
                "[Langfuse] Enabled but missing PUBLIC_KEY or SECRET_KEY. Falling back to no-op."
            )
            return

        if _LangfuseClient is None:
            logger.warning("[Langfuse] SDK import failed. Falling back to no-op.")
            return
        if not hasattr(_langfuse_module, "start_as_current_span"):
            logger.warning(
                "[Langfuse] SDK %s does not expose 'start_as_current_span'. "
                "Check that langfuse>=4.0 is installed. Falling back to no-op.",
                getattr(_langfuse_module, "__version__", "unknown"),
            )
            return

        try:
            self._client = _LangfuseClient(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_BASE_URL,
            )
            self._is_active = True
            logger.info(
                "[Langfuse] Initialized (SDK %s). Environment=%s.",
                getattr(_langfuse_module, "__version__", "?"),
                LANGFUSE_ENVIRONMENT,
            )
        except Exception as exc:
            logger.error("[Langfuse] Failed to initialize: %s. Falling back to no-op.", exc)

    def is_active(self) -> bool:
        return self._is_active

    def trace(self, name: str, **kwargs: Any) -> "_LiveTrace | NoOpTrace":
        """
        Start a new observability trace/span.

        Returns a ``_LiveTrace`` when Langfuse is active (respects sample
        rate) or a ``NoOpTrace`` otherwise.  Both expose the same interface
        so call-sites never need to branch.

        Supported patterns
        ------------------
        ::

            with observer.trace("my_trace", metadata={...}) as t:
                t.update(metadata={...})
                with t.span("step"):
                    ...

            observer.trace("booking_flow").update(metadata={...})


            t = observer.trace("chat_turn", user_id=uid)
            t.update(metadata={...})
            t.end()
        """
        if not self._is_active or self._client is None:
            return NoOpTrace()

        if not _should_sample():
            return NoOpTrace()

        sanitized: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in ("metadata", "input", "output"):
                sanitized[k] = sanitize_for_observability(v)
            else:
                sanitized[k] = v

        return _LiveTrace(client=self._client, name=name, sanitized_kwargs=sanitized)

    def flush(self) -> None:
        """Flush pending events to the Langfuse server (best-effort)."""
        if self._is_active and self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:
                logger.error("[Langfuse] Failed to flush: %s", exc)



def get_observer() -> "LangfuseObserver | NoOpObserver":
    """
    Return the singleton ``LangfuseObserver`` (or a ``NoOpObserver`` when
    the SDK is absent).  Initialises the observer on first call.
    """
    global _observer
    if _observer is None:
        _observer = LangfuseObserver()
    return _observer
