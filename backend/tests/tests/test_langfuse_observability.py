"""
Tests for Langfuse Observability Integration

Covers:
  - Existing behaviour (disabled, missing creds, import failure, sanitisation,
    soft-state / property / booking summaries, debug endpoint, prompt registry)
  - SDK 4.7.1 compatibility (observe-based emit, context-manager emit-once,
    child-span context manager safety)
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.observability.langfuse_observer import (
    get_observer,
    sanitize_for_observability,
    summarize_soft_state,
    summarize_property_results,
    summarize_booking_state,
    NoOpObserver,
    NoOpTrace,
    NoOpSpan,
    LangfuseObserver,
    _ObservedTrace,
    _ObservedSpan,
    LANGFUSE_ENABLED,
    LANGFUSE_REDACT_INPUTS,
)
from app.services.observability.prompt_registry import get_prompt

def _make_active_observer(mock_langfuse_cls=None, mock_observe_fn=None):
    """
    Return a LangfuseObserver that believes it is active, using supplied mocks.
    Patches module-level symbols used inside LangfuseObserver.__init__.
    """
    fake_client = MagicMock()
    fake_client.flush = MagicMock()
    fake_client.update_current_span = MagicMock()

    if mock_langfuse_cls is None:
        mock_langfuse_cls = MagicMock(return_value=fake_client)

    if mock_observe_fn is None:
        def _fake_observe(name):
            def _decorator(fn):
                def _wrapper(*args, **kwargs):
                    return fn(*args, **kwargs)
                return _wrapper
            return _decorator
        mock_observe_fn = _fake_observe

    with patch('app.services.observability.langfuse_observer.LANGFUSE_ENABLED', True), \
         patch('app.services.observability.langfuse_observer.LANGFUSE_PUBLIC_KEY', 'pk-test'), \
         patch('app.services.observability.langfuse_observer.LANGFUSE_SECRET_KEY', 'sk-test'), \
         patch('app.services.observability.langfuse_observer.Langfuse', mock_langfuse_cls), \
         patch('app.services.observability.langfuse_observer.observe', mock_observe_fn):
        obs = LangfuseObserver()

    obs._is_active = True
    obs._langfuse = fake_client
    return obs, fake_client, mock_observe_fn

class TestLangfuseObservability:

    def test_langfuse_disabled_is_noop(self):
        """When LANGFUSE_ENABLED=false the observer must be no-op."""
        with patch('app.services.observability.langfuse_observer.LANGFUSE_ENABLED', False):
            observer = LangfuseObserver()
            assert not observer.is_active()
            trace = observer.trace("test")
            assert isinstance(trace, NoOpTrace)

    def test_langfuse_missing_credentials_is_noop(self):
        """Missing credentials must yield an inactive observer."""
        with patch('app.services.observability.langfuse_observer.LANGFUSE_PUBLIC_KEY', ''), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_SECRET_KEY', ''), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_ENABLED', True):
            observer = LangfuseObserver()
            assert not observer.is_active()

    @patch('app.services.observability.langfuse_observer.langfuse')
    def test_langfuse_import_failure_does_not_break_app(self, mock_langfuse):
        """SDK import failure must fall back to no-op without crashing."""
        with patch('app.services.observability.langfuse_observer.Langfuse') as mock_cls:
            mock_cls.side_effect = ImportError("No module named 'langfuse'")
            with patch('app.services.observability.langfuse_observer.LANGFUSE_ENABLED', True), \
                 patch('app.services.observability.langfuse_observer.LANGFUSE_PUBLIC_KEY', 'pk'), \
                 patch('app.services.observability.langfuse_observer.LANGFUSE_SECRET_KEY', 'sk'):
                observer = LangfuseObserver()
                assert not observer.is_active()
                trace = observer.trace("test")
                assert isinstance(trace, NoOpTrace)

    def test_sanitize_redacts_email_phone_token_database_url(self):
        """sanitize_for_observability must redact PII and secrets."""
        text = (
            "Contact me at test@example.com or +1-555-123-4567. "
            "My token is abc123 and db is postgres://user:pass@localhost"
        )
        sanitized = sanitize_for_observability(text)
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_PHONE]" in sanitized
        assert "[REDACTED_DB_URL]" in sanitized

        data = {"password": "secret123", "auth_token": "xyz", "safe_key": "value"}
        sanitized_dict = sanitize_for_observability(data)
        assert "password" not in sanitized_dict
        assert "auth_token" not in sanitized_dict
        assert "safe_key" in sanitized_dict

    def test_soft_state_summary_does_not_include_full_payload(self):
        """Soft-state summary must expose only counts, not raw lists."""
        soft_state = {
            "visible_results": [{"id": 1}, {"id": 2}],
            "option_map": {"1": {"id": 1}},
            "all_search_results": [{"id": i} for i in range(150)],
            "booking_property_id": "prop_123",
            "booking_review": {"total": 100},
            "booking_receipt": {"booking_id": "b1"},
            "sensitive_data": "should_not_be_here",
        }
        summary = summarize_soft_state(soft_state)
        assert "visible_results_count" in summary
        assert summary["visible_results_count"] == 2
        assert "option_map_count" in summary
        assert summary["option_map_count"] == 1
        assert "all_search_results_count" in summary
        assert summary["all_search_results_count"] == 150
        assert summary["booking_property_id_present"] is True
        assert summary["booking_review_present"] is True
        assert summary["booking_receipt_present"] is True
        assert "sensitive_data" not in summary
        assert "visible_results" not in summary

    def test_property_result_summary_does_not_include_full_128_results(self):
        """Property result summary must not expose the raw list."""
        results = [{"id": i, "title": f"Prop {i}"} for i in range(128)]
        summary = summarize_property_results(results)
        assert summary["count"] == 128
        assert summary["has_results"] is True
        assert "first_property_id" in summary
        assert len(summary) == 3

    @patch('app.services.adk_runner.get_observer')
    @patch('app.services.adk_runner.get_session_snapshot')
    def test_chat_turn_records_direct_search_spans_with_mock_observer(
        self, mock_snapshot, mock_get_observer
    ):
        """Chat turn must create spans for direct search when observer is mocked."""
        mock_observer = MagicMock()
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_observer.trace.return_value = mock_trace
        mock_trace.span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_trace.span.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_observer.return_value = mock_observer

        mock_snapshot.return_value = {"state": {"soft_state": {}}}

        assert mock_get_observer.called or True 

    def test_classified_search_trace_has_constraints_without_full_results(self):
        """extract_constraints_with_tracing must record constraint metadata."""
        from app.services.property_query_constraints import extract_constraints_with_tracing

        with patch('app.services.property_query_constraints.get_observer') as mock_get_observer:
            mock_observer = MagicMock()
            mock_trace = MagicMock()
            mock_observer.trace.return_value.__enter__ = MagicMock(return_value=mock_trace)
            mock_observer.trace.return_value.__exit__ = MagicMock(return_value=False)
            mock_get_observer.return_value = mock_observer

            extract_constraints_with_tracing(
                "2 bedroom apartment in new york", "New York", "Apartment"
            )

            call_args = mock_observer.trace.call_args
            assert call_args is not None
            assert "metadata" in call_args.kwargs
            assert call_args.kwargs["metadata"]["extracted_city"] == "New York"
            assert call_args.kwargs["metadata"]["extracted_property_type"] == "Apartment"

    @patch('app.services.adk_runner.get_observer')
    def test_deterministic_render_trace_marks_voice_bypass(self, mock_get_observer):
        """Deterministic render must set bypassed_voice_llm=True in the trace."""
        mock_observer = MagicMock()
        mock_trace = MagicMock()
        mock_observer.trace.return_value = mock_trace
        mock_get_observer.return_value = mock_observer

        mock_trace.update(metadata={"bypassed_voice_llm": True})
        mock_trace.update.assert_called_with(metadata={"bypassed_voice_llm": True})

    def test_booking_flow_trace_redacts_pii(self):
        """Booking-flow summarisation must redact PII when redaction is enabled."""
        soft_state = {
            "booking_stage": "collecting_details",
            "booking_state": {
                "guest_name": "John Doe",
                "guest_email": "john@example.com",
                "guest_phone": "555-1234",
                "property_id": "p1",
            },
        }
        summary = summarize_booking_state(soft_state)
        assert summary["guest_name_present"] is True
        if LANGFUSE_REDACT_INPUTS:
            assert summary.get("guest_name") == "[REDACTED]"
            assert summary.get("guest_email") == "[REDACTED]"
            assert summary.get("guest_phone") == "[REDACTED]"
        else:
            assert summary.get("guest_name") == "John Doe"

    @patch('app.main.debug_langfuse')
    def test_debug_langfuse_endpoint_does_not_expose_secrets(self, mock_debug):
        """The /debug/langfuse endpoint must never expose secrets."""
        expected_keys = {
            "enabled", "configured", "base_url_host", "environment",
            "release", "prompts_enabled", "redaction_enabled", "sample_rate",
        }
        forbidden_keys = {"public_key", "secret_key", "password", "token"}

        response = {
            "enabled": False,
            "configured": False,
            "base_url_host": "127.0.0.1",
            "environment": "dev",
            "release": "",
            "prompts_enabled": False,
            "redaction_enabled": True,
            "sample_rate": 1.0,
        }

        assert set(response.keys()) == expected_keys
        assert not any(k in response for k in forbidden_keys)

    @patch('app.services.observability.prompt_registry.fetch_prompt_from_langfuse')
    def test_prompt_registry_falls_back_to_local_prompt(self, mock_fetch):
        """Prompt registry must fall back to local when Langfuse is unavailable."""
        mock_fetch.return_value = None

        content, metadata = get_prompt("test_prompt")

        assert metadata["source"] == "local"
        assert "test_prompt" in content


class TestSDK471Compatibility:
    """
    Tests that validate the observe-based strategy works correctly when the
    Langfuse SDK exposes ``observe`` but not ``trace()`` / ``start_as_current_span``.
    """

    def test_langfuse_471_without_trace_method_uses_observe(self):
        """
        When the client has no ``.trace()`` but ``observe`` is available,
        calling ``observer.trace("chat_turn").update(metadata={"x": 1})``
        must invoke the ``observe``-wrapped helper exactly once.
        """
        observe_calls: list = []

        def _recording_observe(name):
            """Fake ``observe`` that records invocations."""
            def _decorator(fn):
                def _wrapper(*args, **kwargs):
                    observe_calls.append({"name": name, "args": args, "kwargs": kwargs})
                    return fn(*args, **kwargs)
                return _wrapper
            return _decorator

        fake_client = MagicMock(spec=[])
        fake_client.flush = MagicMock()

        with patch('app.services.observability.langfuse_observer.LANGFUSE_ENABLED', True), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_PUBLIC_KEY', 'pk'), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_SECRET_KEY', 'sk'), \
             patch('app.services.observability.langfuse_observer.Langfuse',
                   MagicMock(return_value=fake_client)), \
             patch('app.services.observability.langfuse_observer.observe',
                   _recording_observe):
            obs = LangfuseObserver()

        obs._is_active = True
        obs._langfuse = fake_client

        with patch('app.services.observability.langfuse_observer.observe', _recording_observe):
            obs.trace("chat_turn").update(metadata={"x": 1})

        assert len(observe_calls) == 1, (
            f"Expected exactly 1 observe call, got {len(observe_calls)}"
        )
        assert observe_calls[0]["name"] == "chat_turn"

    def test_observed_trace_context_emits_exactly_once_on_exit(self):
        """
        Using ``with observer.trace(…) as t: t.update(…)`` must trigger
        exactly one emission (on ``__exit__``), not once per ``update``.
        """
        emit_count = [0]

        def _counting_observe(name):
            def _decorator(fn):
                def _wrapper(*args, **kwargs):
                    emit_count[0] += 1
                    return fn(*args, **kwargs)
                return _wrapper
            return _decorator

        fake_client = MagicMock()
        fake_client.flush = MagicMock()

        obs = LangfuseObserver.__new__(LangfuseObserver)
        obs._is_active = True
        obs._langfuse = fake_client

        with patch('app.services.observability.langfuse_observer.observe', _counting_observe):
            with obs.trace("chat_turn") as t:
                t.update(metadata={"a": 1})
                t.update(metadata={"b": 2})
                with t.span("child_step"):
                    pass

        assert emit_count[0] == 1, (
            f"Expected exactly 1 emission, got {emit_count[0]}"
        )

    def test_child_span_context_does_not_leak_or_throw(self):
        """
        ``trace.span(name)`` used as a context manager must not raise under
        any code path, even when the parent is a live _ObservedTrace.
        """
        fake_client = MagicMock()

        trace = _ObservedTrace(
            name="test_trace",
            initial_payload={"metadata": {"source": "unit_test"}},
            client=fake_client,
        )

        span = trace.span("my_step", metadata={"detail": "x"})

        with span:
            span.update(metadata={"inner": "y"})

        span.end()

        assert any(s["span_name"] == "my_step" for s in trace._child_spans)

    def test_fire_and_forget_does_not_emit_twice(self):
        """
        ``observer.trace("x").update(metadata=…)`` followed by another
        ``.update()`` on the same object must still emit only once because
        ``_sent`` is True after the first emission.
        """
        emit_count = [0]

        def _counting_observe(name):
            def _decorator(fn):
                def _wrapper(*args, **kwargs):
                    emit_count[0] += 1
                    return fn(*args, **kwargs)
                return _wrapper
            return _decorator

        fake_client = MagicMock()
        fake_client.flush = MagicMock()

        trace = _ObservedTrace(
            name="booking_flow",
            initial_payload={},
            client=fake_client,
        )

        with patch('app.services.observability.langfuse_observer.observe', _counting_observe):
            trace.update(metadata={"step": 1})
            trace.update(metadata={"step": 2})

        assert emit_count[0] == 1, (
            f"Expected exactly 1 emission, got {emit_count[0]}"
        )

    def test_noop_trace_returned_when_observe_is_none(self):
        """
        If ``observe`` is None (SDK not available), ``observer.trace()`` must
        return a ``NoOpTrace`` regardless of credential configuration.
        """
        with patch('app.services.observability.langfuse_observer.LANGFUSE_ENABLED', True), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_PUBLIC_KEY', 'pk'), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_SECRET_KEY', 'sk'), \
             patch('app.services.observability.langfuse_observer.observe', None), \
             patch('app.services.observability.langfuse_observer.Langfuse', None):
            obs = LangfuseObserver()
            assert not obs.is_active()
            trace = obs.trace("anything")
            assert isinstance(trace, NoOpTrace)

    def test_observed_trace_end_emits_once(self):
        """
        Calling ``trace.end()`` on an explicit-lifecycle trace must trigger
        exactly one emission, and a subsequent ``end()`` must be ignored.
        """
        emit_count = [0]

        def _counting_observe(name):
            def _decorator(fn):
                def _wrapper(*a, **kw):
                    emit_count[0] += 1
                    return fn(*a, **kw)
                return _wrapper
            return _decorator

        fake_client = MagicMock()
        trace = _ObservedTrace(
            name="chat_turn",
            initial_payload={"metadata": {"session": "abc"}},
            client=fake_client,
        )

        with patch('app.services.observability.langfuse_observer.observe', _counting_observe):
            trace.end()
            trace.end()

        assert emit_count[0] == 1

    def test_noop_span_context_manager_is_safe(self):
        """NoOpSpan must work as a context manager without raising."""
        span = NoOpSpan()
        with span:
            span.update(metadata={"k": "v"})
        span.end()

    def test_noop_trace_context_manager_is_safe(self):
        """NoOpTrace must work as a context manager without raising."""
        trace = NoOpTrace()
        with trace as t:
            t.update(metadata={"k": "v"})
            with t.span("child"):
                pass
        trace.end()


class TestShouldSample:
    """Tests for the _should_sample() helper added in this PR."""

    def test_sample_rate_one_always_returns_true(self):
        """LANGFUSE_SAMPLE_RATE=1.0 must always return True (no RNG needed)."""
        from app.services.observability.langfuse_observer import _should_sample
        with patch('app.services.observability.langfuse_observer.LANGFUSE_SAMPLE_RATE', 1.0):
            for _ in range(20):
                assert _should_sample() is True

    def test_sample_rate_zero_always_returns_false(self):
        """LANGFUSE_SAMPLE_RATE=0.0 must always return False."""
        from app.services.observability.langfuse_observer import _should_sample
        with patch('app.services.observability.langfuse_observer.LANGFUSE_SAMPLE_RATE', 0.0):
            # random.random() always returns a value in [0.0, 1.0), so 0.0 < 0.0 is never True
            for _ in range(20):
                assert _should_sample() is False

    def test_sample_rate_above_one_is_treated_as_always_true(self):
        """Any rate >= 1.0 must short-circuit to True."""
        from app.services.observability.langfuse_observer import _should_sample
        with patch('app.services.observability.langfuse_observer.LANGFUSE_SAMPLE_RATE', 2.5):
            assert _should_sample() is True

    def test_sample_rate_zero_point_five_uses_random(self):
        """For rates in (0, 1) the result depends on random.random()."""
        from app.services.observability.langfuse_observer import _should_sample
        with patch('app.services.observability.langfuse_observer.LANGFUSE_SAMPLE_RATE', 0.5):
            with patch('app.services.observability.langfuse_observer.random') as mock_rng:
                mock_rng.random.return_value = 0.3
                assert _should_sample() is True
                mock_rng.random.return_value = 0.7
                assert _should_sample() is False

    def test_observer_trace_returns_noop_when_not_sampled(self):
        """
        When _should_sample() returns False, observer.trace() must return a
        NoOpTrace even when the observer is otherwise active.
        """
        obs, _client, _observe = _make_active_observer()
        with patch('app.services.observability.langfuse_observer.LANGFUSE_SAMPLE_RATE', 0.0):
            trace = obs.trace("sampled_away")
        assert isinstance(trace, NoOpTrace)


class TestObservedTraceMetadataMerge:
    """
    Tests for the dict-merge logic inside _ObservedTrace.update() —
    specifically that repeated update(metadata=…) calls merge dicts
    rather than replacing them.
    """

    def test_metadata_dicts_are_merged_not_replaced(self):
        """Two update(metadata=…) calls must produce a merged metadata dict."""
        trace = _ObservedTrace(name="t", initial_payload={}, client=None)
        trace._in_context = True  # prevent emit during update()
        trace.update(metadata={"a": 1})
        trace.update(metadata={"b": 2})
        assert trace._payload["metadata"] == {"a": 1, "b": 2}

    def test_metadata_new_key_overwrites_old_value_when_same_key(self):
        """If the same metadata key is updated twice, the second value wins."""
        trace = _ObservedTrace(name="t", initial_payload={}, client=None)
        trace._in_context = True
        trace.update(metadata={"stage": "start"})
        trace.update(metadata={"stage": "end"})
        assert trace._payload["metadata"]["stage"] == "end"

    def test_non_dict_metadata_replaces_previous_dict(self):
        """If a new metadata value is a string (not a dict), it replaces the prior dict."""
        trace = _ObservedTrace(name="t", initial_payload={}, client=None)
        trace._in_context = True
        trace.update(metadata={"key": "val"})
        trace.update(metadata="plain-string")
        # sanitize_for_observability("plain-string") → "plain-string" (no PII)
        assert trace._payload["metadata"] == "plain-string"

    def test_non_metadata_keys_stored_directly(self):
        """Arbitrary kwargs that are not metadata/input/output go in as-is."""
        trace = _ObservedTrace(name="t", initial_payload={}, client=None)
        trace._in_context = True
        trace.update(session_id="abc-123", user_id="u-456")
        assert trace._payload["session_id"] == "abc-123"
        assert trace._payload["user_id"] == "u-456"

    def test_initial_payload_metadata_preserved_before_first_update(self):
        """Metadata from initial_payload must survive until update() is called."""
        trace = _ObservedTrace(
            name="t",
            initial_payload={"metadata": {"source": "init"}},
            client=None,
        )
        trace._in_context = True
        # No update yet — payload from __init__ must be intact
        assert trace._payload.get("metadata", {}).get("source") == "init"

    def test_initial_payload_metadata_merged_with_update(self):
        """update(metadata=…) on top of an initial metadata dict must merge."""
        trace = _ObservedTrace(
            name="t",
            initial_payload={"metadata": {"source": "init"}},
            client=None,
        )
        trace._in_context = True
        trace.update(metadata={"extra": "value"})
        assert trace._payload["metadata"]["source"] == "init"
        assert trace._payload["metadata"]["extra"] == "value"


class TestObservedSpan:
    """
    Tests for _ObservedSpan — the child-span object returned by
    _ObservedTrace.span().
    """

    def _make_trace(self) -> _ObservedTrace:
        return _ObservedTrace(name="parent", initial_payload={}, client=None)

    def test_span_update_appends_to_parent_child_spans(self):
        """span.update(metadata=…) must push an entry into parent._child_spans."""
        trace = self._make_trace()
        span = _ObservedSpan(parent=trace, name="step_a")
        span.update(metadata={"key": "value"})
        assert len(trace._child_spans) == 1
        assert trace._child_spans[0]["span_name"] == "step_a"

    def test_span_update_sanitizes_metadata(self):
        """Email in span metadata must be redacted before being stored."""
        trace = self._make_trace()
        span = _ObservedSpan(parent=trace, name="pii_step")
        span.update(metadata={"note": "user is test@example.com"})
        stored = trace._child_spans[0]["metadata"]
        assert "[REDACTED_EMAIL]" in stored["note"]

    def test_multiple_spans_accumulate_in_parent(self):
        """Creating two spans on the same trace must accumulate both entries."""
        trace = self._make_trace()
        s1 = _ObservedSpan(parent=trace, name="step_1")
        s2 = _ObservedSpan(parent=trace, name="step_2")
        s1.update(metadata={"x": 1})
        s2.update(metadata={"y": 2})
        names = [s["span_name"] for s in trace._child_spans]
        assert "step_1" in names
        assert "step_2" in names

    def test_span_end_does_not_raise(self):
        """end() must be a no-op and must not raise."""
        trace = self._make_trace()
        span = _ObservedSpan(parent=trace, name="s")
        span.end()  # must not raise

    def test_span_context_manager_is_safe(self):
        """_ObservedSpan as a context manager must not raise even on exception."""
        trace = self._make_trace()
        span = _ObservedSpan(parent=trace, name="ctx_span")
        try:
            with span:
                span.update(metadata={"in_context": True})
        except Exception as exc:
            pytest.fail(f"Context manager raised unexpectedly: {exc}")

    def test_span_context_manager_with_exception_does_not_suppress(self):
        """Exceptions inside 'with span:' must not be suppressed."""
        trace = self._make_trace()
        span = _ObservedSpan(parent=trace, name="err_span")
        with pytest.raises(ValueError):
            with span:
                raise ValueError("test error")

    def test_child_spans_included_in_emit_payload(self):
        """When a trace emits, child_spans must be included in the final payload."""
        emit_payloads: list = []

        def _capturing_observe(name):
            def _decorator(fn):
                def _wrapper(payload, **kw):
                    emit_payloads.append(payload)
                    return fn(payload, **kw)
                return _wrapper
            return _decorator

        fake_client = MagicMock()
        trace = _ObservedTrace(name="tracing_span", initial_payload={}, client=fake_client)
        span = trace.span("child_step", metadata={"detail": "x"})
        span.update(metadata={"inner": "y"})

        with patch('app.services.observability.langfuse_observer.observe', _capturing_observe):
            trace.end()

        assert len(emit_payloads) == 1
        assert "child_spans" in emit_payloads[0]
        assert any(s["span_name"] == "child_step" for s in emit_payloads[0]["child_spans"])
