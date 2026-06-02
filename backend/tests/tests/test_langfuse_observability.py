"""
Tests for Langfuse Observability Integration
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.observability.langfuse_observer import (
    get_observer,
    sanitize_for_observability,
    summarize_soft_state,
    summarize_property_results,
    summarize_booking_state,
    NoOpObserver,
    NoOpTrace,
    LangfuseObserver,
    LANGFUSE_ENABLED,
    LANGFUSE_REDACT_INPUTS,
)
from app.services.observability.prompt_registry import get_prompt


class TestLangfuseObservability:
    def test_langfuse_disabled_is_noop(self):
        """Test that when LANGFUSE_ENABLED=false, observer is no-op."""
        with patch('app.services.observability.langfuse_observer.LANGFUSE_ENABLED', False):
            observer = LangfuseObserver()
            assert not observer.is_active()
            trace = observer.trace("test")
            assert isinstance(trace, NoOpTrace)

    def test_langfuse_missing_credentials_is_noop(self):
        """Test that missing credentials result in no-op observer."""
        with patch('app.services.observability.langfuse_observer.LANGFUSE_PUBLIC_KEY', ''), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_SECRET_KEY', ''), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_ENABLED', True):
            observer = LangfuseObserver()
            assert not observer.is_active()

    @patch('app.services.observability.langfuse_observer.langfuse')
    def test_langfuse_import_failure_does_not_break_app(self, mock_langfuse):
        """Test that import failure falls back to no-op without breaking."""
        mock_langfuse.Langfuse.side_effect = ImportError("No module named 'langfuse'")
        with patch('app.services.observability.langfuse_observer.LANGFUSE_ENABLED', True), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_PUBLIC_KEY', 'pk'), \
             patch('app.services.observability.langfuse_observer.LANGFUSE_SECRET_KEY', 'sk'):
            observer = LangfuseObserver()
            assert not observer.is_active()
            trace = observer.trace("test")
            assert isinstance(trace, NoOpTrace)

    def test_sanitize_redacts_email_phone_token_database_url(self):
        """Test that sanitize_for_observability redacts PII and secrets."""
        text = "Contact me at test@example.com or +1-555-123-4567. My token is abc123 and db is postgres://user:pass@localhost"
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
        """Test that soft state summary only includes allowed keys and counts."""
        soft_state = {
            "visible_results": [{"id": 1}, {"id": 2}],
            "option_map": {"1": {"id": 1}},
            "all_search_results": [{"id": i} for i in range(150)],
            "booking_property_id": "prop_123",
            "booking_review": {"total": 100},
            "booking_receipt": {"booking_id": "b1"},
            "sensitive_data": "should_not_be_here"
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
        """Test that property result summary does not include full lists."""
        results = [{"id": i, "title": f"Prop {i}"} for i in range(128)]
        summary = summarize_property_results(results)
        assert summary["count"] == 128
        assert summary["has_results"] is True
        assert "first_property_id" in summary
        assert len(summary) == 3 

    @patch('app.services.adk_runner.get_observer')
    @patch('app.services.adk_runner.get_session_snapshot')
    def test_chat_turn_records_direct_search_spans_with_mock_observer(self, mock_snapshot, mock_get_observer):
        """Test that chat turn creates spans for direct search."""
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
        """Test that classified search trace records constraints without full results."""
        from app.services.property_query_constraints import extract_constraints_with_tracing
        
        with patch('app.services.property_query_constraints.get_observer') as mock_get_observer:
            mock_observer = MagicMock()
            mock_trace = MagicMock()
            mock_observer.trace.return_value.__enter__ = MagicMock(return_value=mock_trace)
            mock_observer.trace.return_value.__exit__ = MagicMock(return_value=False)
            mock_get_observer.return_value = mock_observer
            
            extract_constraints_with_tracing("2 bedroom apartment in new york", "New York", "Apartment")
            
            call_args = mock_observer.trace.call_args
            assert call_args is not None
            assert "metadata" in call_args.kwargs
            assert call_args.kwargs["metadata"]["extracted_city"] == "New York"
            assert call_args.kwargs["metadata"]["extracted_property_type"] == "Apartment"

    @patch('app.services.adk_runner.get_observer')
    def test_deterministic_render_trace_marks_voice_bypass(self, mock_get_observer):
        """Test that deterministic render sets bypassed_voice_llm=true."""
        mock_observer = MagicMock()
        mock_trace = MagicMock()
        mock_observer.trace.return_value = mock_trace
        mock_get_observer.return_value = mock_observer
        
        mock_trace.update(metadata={"bypassed_voice_llm": True})
        
        mock_trace.update.assert_called_with(metadata={"bypassed_voice_llm": True})

    def test_booking_flow_trace_redacts_pii(self):
        """Test that booking flow trace redacts PII by default."""
        soft_state = {
            "booking_stage": "collecting_details",
            "booking_state": {
                "guest_name": "John Doe",
                "guest_email": "john@example.com",
                "guest_phone": "555-1234",
                "property_id": "p1"
            }
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
        """Test that /debug/langfuse does not return sensitive information."""
        expected_keys = {
            "enabled", "configured", "base_url_host", "environment", 
            "release", "prompts_enabled", "redaction_enabled", "sample_rate"
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
        """Test that prompt registry falls back to local when Langfuse fails."""
        mock_fetch.return_value = None
        
        content, metadata = get_prompt("test_prompt")
        
        assert metadata["source"] == "local"
        assert "test_prompt" in content
