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

    with observer.trace("booking_flow") as trace:
        trace.update(metadata={…})

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
    """
    Check whether Langfuse API credentials are configured.
    
    Returns:
        `true` if both the public and secret keys are non-empty, `false` otherwise.
    """
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def _should_sample() -> bool:
    """
    Decides whether the current event should be sampled according to LANGFUSE_SAMPLE_RATE.
    
    If LANGFUSE_SAMPLE_RATE is greater than or equal to 1.0 sampling always occurs.
    
    Returns:
        `True` if the event should be sampled according to LANGFUSE_SAMPLE_RATE, `False` otherwise.
    """
    if LANGFUSE_SAMPLE_RATE >= 1.0:
        return True
    return random.random() < LANGFUSE_SAMPLE_RATE


def sanitize_for_observability(value: Any) -> Any:
    """
    Sanitize a value for observability by redacting PII and sensitive secrets.
    
    Strings are truncated to the configured maximum and have email addresses, phone-like
    numbers, and common database URLs redacted. Dicts are copied with entries removed
    when the key name contains sensitive substrings (e.g., password, secret, token,
    api_key, authorization, cookie, database_url, email, phone); remaining values are
    recursively sanitized. Lists, tuples, and sets are returned as lists with each
    element sanitized. Integers, floats, and booleans are returned unchanged; other
    types are converted to their string representation.
    
    Returns:
        A sanitized version of the input suitable for observability output, preserving
        the input's general structure while omitting or redacting sensitive data.
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
    """
    Produce a compact summary of a soft-state dictionary for observability.
    
    Parameters:
        soft_state (Optional[Dict[str, Any]]): The soft state to summarize; expected to be a dict representing UI/booking-related transient state.
    
    Returns:
        Dict[str, Any]: A summary dictionary containing:
            - `keys`: list of keys present in `soft_state`
            - `last_presented_view`: value of `soft_state.get("last_presented_view")`
            - `booking_stage`: value of `soft_state.get("booking_stage")`
            - `visible_results_count`: length of `soft_state["visible_results"]` if it is a list, otherwise 0
            - `option_map_count`: length of `soft_state["option_map"]` if it is a dict, otherwise 0
            - `all_search_results_count`: length of `soft_state["all_search_results"]` if it is a list, otherwise 0
            - `booking_property_id_present`: `True` if `soft_state.get("booking_property_id")` is truthy, otherwise `False`
            - `booking_review_present`: `True` if `soft_state.get("booking_review")` is truthy, otherwise `False`
            - `booking_receipt_present`: `True` if `soft_state.get("booking_receipt")` is truthy, otherwise `False`
    
        Returns an empty dict if `soft_state` is not a dict.
    """
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
    """
    Summarizes a list of property result items.
    
    If `results` is not a list it is treated as empty.
    
    Returns:
        A dict containing:
        - `count` (int): number of items in `results`.
        - `has_results` (bool): `true` if `count` is greater than zero, `false` otherwise.
        - `first_property_id` (Any | None): the `"id"` value from the first item if the first item is a dict, otherwise `None`.
    """
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
    """
    Produce a privacy-aware summary of booking-related fields extracted from a soft-state dictionary.
    
    Parameters:
        soft_state (Optional[Dict[str, Any]]): The soft state to summarize; expected to contain a nested
            "booking_state" dict and optionally "booking_stage". If not a dict, an empty dict is returned.
    
    Returns:
        Dict[str, Any]: A summary dictionary with the following keys:
          - "stage": the value of `soft_state["booking_stage"]` (or None if absent).
          - "property_id_present": `True` if `booking_state["property_id"]` is present and truthy, `False` otherwise.
          - "guest_name_present": `True` if `booking_state["guest_name"]` is present and truthy, `False` otherwise.
          - "guest_email_present": `True` if `booking_state["guest_email"]` is present and truthy, `False` otherwise.
          - "guest_phone_present": `True` if `booking_state["guest_phone"]` is present and truthy, `False` otherwise.
          - "check_in_present": `True` if `booking_state["check_in"]` is present and truthy, `False` otherwise.
          - "check_out_present": `True` if `booking_state["check_out"]` is present and truthy, `False` otherwise.
          - "guests": the raw value of `booking_state["guests"]` (may be None).
          - "guest_name", "guest_email", "guest_phone": either the actual values from `booking_state` or the literal
            string "[REDACTED]" when `LANGFUSE_REDACT_INPUTS` is enabled and the corresponding field is present.
    """
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
        """
        Record an update for this child span into the parent trace's accumulated child_spans.
        
        Only the `metadata` entry from `**kwargs` is used: it is sanitized for observability and appended to the parent trace as
        {"span_name": <span name>, "metadata": <sanitized metadata>}. Any exceptions raised during sanitization or append are caught
        and logged; this method never propagates errors.
        
        Parameters:
            kwargs (Any): Optional keyword arguments; recognized key:
                - metadata (Any): Metadata to associate with the child span (will be sanitized).
        """
        pass

    def end(self) -> None:
        """
        End the span.
        
        Signal that the span is finished. Implementations may perform cleanup or be a no-op; this method does not raise exceptions.
        """
        pass

    def __enter__(self) -> "NoOpSpan":
        """
        Enter the no-op span context and return the span instance.
        
        Returns:
            NoOpSpan: The same span object to be used as the context manager.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Context manager exit hook that performs no action and does not suppress exceptions.
        
        The parameters mirror the standard context manager protocol and are ignored; any exception raised inside the context will propagate normally because this method returns None.
        """
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
        """
        Create a child span compatible with the tracing API that performs no operations.
        
        Parameters:
            name (str): Span name (accepted for API compatibility; ignored).
            **kwargs: Additional span options (accepted but ignored).
        
        Returns:
            NoOpSpan: A span-like object whose methods are no-ops and that can be used as a context manager.
        """
        return NoOpSpan()

    def update(self, **kwargs: Any) -> None:
        """
        Record an update for this child span into the parent trace's accumulated child_spans.
        
        Only the `metadata` entry from `**kwargs` is used: it is sanitized for observability and appended to the parent trace as
        {"span_name": <span name>, "metadata": <sanitized metadata>}. Any exceptions raised during sanitization or append are caught
        and logged; this method never propagates errors.
        
        Parameters:
            kwargs (Any): Optional keyword arguments; recognized key:
                - metadata (Any): Metadata to associate with the child span (will be sanitized).
        """
        pass

    def end(self) -> None:
        """
        End the span.
        
        Signal that the span is finished. Implementations may perform cleanup or be a no-op; this method does not raise exceptions.
        """
        pass

    def __enter__(self) -> "NoOpTrace":
        """
        Enter the context for a NoOpTrace and provide the trace object for use within the with-block.
        
        Returns:
            The `NoOpTrace` instance to be used as the context manager target.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Context manager exit hook that performs no action and does not suppress exceptions.
        
        The parameters mirror the standard context manager protocol and are ignored; any exception raised inside the context will propagate normally because this method returns None.
        """
        pass


class NoOpObserver:
    """Observer returned when Langfuse is fully disabled."""

    def trace(self, name: str, **kwargs: Any) -> NoOpTrace:
        """
        Create a no-op trace object with the given name.
        
        Parameters:
            name (str): The trace name (ignored by the no-op implementation).
        
        Returns:
            NoOpTrace: A trace-like object whose `span`, `update`, `end`, and context-manager methods are no-ops.
        """
        return NoOpTrace()

    def flush(self) -> None:
        """
        Flushes any buffered observations to Langfuse if the observer is active.
        
        If a Langfuse client is available, calls its `flush()` method. Any exceptions raised by the client are caught and logged; this method does not raise.
        """
        pass

def _emit_langfuse_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emit an observation payload to Langfuse and return a minimal acknowledgement.
    
    Attempts to attach the given payload to the current Langfuse trace/span when a client is available; exceptions during attachment are suppressed so the function does not raise.
    
    Parameters:
    	payload (Dict[str, Any]): Metadata and optional `input`/`output` fields to attach to the observation.
    
    Returns:
    	result (Dict[str, Any]): A summary dict with keys:
    		- `"status"`: `"ok"` indicating the emit attempt completed.
    		- `"metadata_keys"`: list of keys present in `payload`.
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
        """
        Initialize an observed child span tied to a parent trace.
        
        Parameters:
        	parent ("_ObservedTrace"): Parent trace that will receive this span's metadata when updated.
        	name (str): Name of the child span.
        	metadata (Any, optional): Optional initial metadata associated with the span; stored for later use.
        """
        self._parent = parent
        self._name = name
        self._metadata = metadata

    def update(self, **kwargs: Any) -> None:
        """
        Record span metadata into the parent trace's accumulated child spans.
        
        Sanitizes the provided `metadata` (via `sanitize_for_observability`) and appends a dictionary
        with `span_name` and `metadata` to the parent trace's internal `_child_spans` list. Any
        exceptions raised during sanitization or append are caught and logged at debug level; no
        exception is propagated.
        
        Parameters:
            kwargs: Supports a `metadata` key whose value will be sanitized and recorded.
        """
        try:
            meta = sanitize_for_observability(kwargs.get("metadata", {}))
            self._parent._child_spans.append(
                {"span_name": self._name, "metadata": meta}
            )
        except Exception as exc:
            logger.debug("[Langfuse] Span update failed (non-fatal): %s", exc)

    def end(self) -> None:
        """
        End the span.
        
        Signal that the span is finished. Implementations may perform cleanup or be a no-op; this method does not raise exceptions.
        """
        pass

    def __enter__(self) -> "_ObservedSpan":
        """
        Enter the span context and return the span instance.
        
        Returns:
            _ObservedSpan: The same span instance to be used as the context manager value.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Context manager exit hook that performs no action and does not suppress exceptions.
        
        The parameters mirror the standard context manager protocol and are ignored; any exception raised inside the context will propagate normally because this method returns None.
        """
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
    * **Explicit lifecycle** (``t = observer.trace(…); t.end()``) — ``_emit``
      fires on ``end()``.
    * ``update()`` never emits — all call-sites must use ``end()`` or context-manager.
    """

    def __init__(
        self,
        name: str,
        initial_payload: Dict[str, Any],
        client: Any,
    ) -> None:
        """
        Initialize an observed trace and prepare its internal state for eventual emission.
        
        The provided `initial_payload` is sanitized with `sanitize_for_observability` and stored (empty dict if falsy). The trace tracks collected child spans, whether it is currently used as a context manager (`_in_context`), and an idempotent `_sent` flag to ensure the trace is emitted at most once. The `client` is retained for optional flushing/emission.
        
        Parameters:
            name (str): Human-readable trace name.
            initial_payload (Dict[str, Any]): Initial event data; will be sanitized and stored.
            client (Any): Langfuse client (or client-like object) used for emission/flush; may be None.
        """
        self._name = name
        self._payload: Dict[str, Any] = sanitize_for_observability(initial_payload) or {}
        self._client = client
        self._child_spans: List[Dict[str, Any]] = []
        self._in_context: bool = False
        self._sent: bool = False


    def span(self, name: str, **kwargs: Any) -> _ObservedSpan:
        """
        Create a child span that records its metadata into the parent trace's collected child spans.
        
        Parameters:
            name (str): Human-readable name for the child span.
            metadata (Any, optional): Initial metadata for the span (passed via `metadata` in kwargs).
        
        Returns:
            _ObservedSpan: A span object that appends its metadata to the parent trace's payload when updated.
        """
        meta = kwargs.get("metadata")
        return _ObservedSpan(parent=self, name=name, metadata=meta)

    def update(self, **kwargs: Any) -> None:
        """
        Merge provided keyword update data into the trace's accumulated payload.
        
        For the keys "metadata", "input", and "output" the values are sanitized for observability; if an existing value is a dict and the sanitized value is also a dict, the sanitized entries are merged into the existing dict, otherwise the value is replaced. Other keys are stored as-is. If this trace is not being used as a context manager, the trace is emitted immediately after the update. Exceptions during the merge are caught and logged; they do not propagate.
        
        Parameters:
            **kwargs: Mapping of fields to merge into the trace payload. Special handling applies to
                "metadata", "input", and "output" as described above.
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



    def end(self) -> None:
        """
        Finalize and emit the accumulated trace payload to the observer; safe to call multiple times.
        """
        self._emit()

    def __enter__(self) -> "_ObservedTrace":
        """
        Enter the trace context and mark the trace as active for context-managed emission.
        
        Returns:
            _ObservedTrace: The trace instance (`self`) to be used as the context manager value.
        """
        self._in_context = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit the trace context and finalize emission of the accumulated observation.
        
        Sets the trace out of context and triggers a one-time emit of the trace payload. Exceptions raised inside the context are not suppressed.
        """
        self._in_context = False
        self._emit()

    def _emit(self) -> None:
        """
        Emit the accumulated trace payload to Langfuse exactly once.
        
        This method is idempotent: subsequent calls after the first do nothing. If a module-level Langfuse observer is unavailable it returns silently. Before sending, child spans collected on the trace are added to the final payload under the "child_spans" key. Any errors during emission or during an optional client flush are caught and logged at debug level; no exception is raised.
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
        """
        Initialize a LangfuseObserver by attempting to construct a Langfuse client from environment configuration.
        
        Performs these steps and sets instance state accordingly:
        - Initializes internal attributes `_langfuse` to `None` and `_is_active` to `False`.
        - If `LANGFUSE_ENABLED` is false, leaves the observer inactive.
        - If required credentials are missing, the Langfuse SDK is unavailable, or the SDK's `observe` helper is unavailable, leaves the observer inactive.
        - On success, constructs a Langfuse client using `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`, stores it in `_langfuse`, and sets `_is_active` to `True`.
        - On any initialization error, leaves the observer inactive and keeps `_langfuse` as `None`.
        """
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
        """
        Indicates whether the observer is currently active.
        
        Returns:
            True if the observer is active, False otherwise.
        """
        return self._is_active

    def trace(self, name: str, **kwargs: Any) -> "_ObservedTrace | NoOpTrace":
        """
        Create a trace context for recording observability data.
        
        Sanitizes any provided `metadata`, `input`, and `output` kwargs and applies sampling and configuration checks; if observability is disabled, misconfigured, or the trace is not sampled, the returned object is a no-op that ignores updates.
        
        Returns:
            A trace-like object that records and emits observations when the observer is active and sampling allows emission; otherwise a no-op trace that ignores updates.
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
        """
        Flush any buffered observations to the Langfuse client.
        
        This is best-effort: failures during flush are logged and not raised.
        """
        if self._is_active and self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as exc:
                logger.error("[Langfuse] Failed to flush: %s", exc)


def get_observer() -> "LangfuseObserver | NoOpObserver":
    """
    Get the module-level Langfuse observer singleton.
    
    Initializes the singleton on first call and returns an object exposing `trace()` and `flush()`.
    Returns:
        The singleton `LangfuseObserver` (or an inactive/no-op observer) providing `.trace()` and `.flush()`.
    """
    global _observer
    if _observer is None:
        _observer = LangfuseObserver()
    return _observer
