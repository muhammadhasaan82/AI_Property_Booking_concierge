"""
Langfuse Observability Wrapper

Provides a safe, no-op fallback observer when Langfuse is disabled, misconfigured,
or fails to import. Ensures the chat app never breaks due to observability issues.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
    """Check if Langfuse is properly configured with required credentials."""
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def sanitize_for_observability(value: Any) -> Any:
    """
    Redact PII and secrets from a value for safe observability logging.
    Truncates long text based on LANGFUSE_MAX_TEXT_CHARS.
    """
    if not LANGFUSE_REDACT_INPUTS:
        if isinstance(value, str) and len(value) > LANGFUSE_MAX_TEXT_CHARS:
            return value[:LANGFUSE_MAX_TEXT_CHARS] + "..."
        return value

    if value is None:
        return None

    if isinstance(value, str):
        if len(value) > LANGFUSE_MAX_TEXT_CHARS:
            value = value[:LANGFUSE_MAX_TEXT_CHARS] + "..."
        
        value = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]", value)
        value = re.sub(r"\+?\d[\d -]{8,12}\d", "[REDACTED_PHONE]", value)
        value = re.sub(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[\w\-]+['\"]?", r"\1: [REDACTED]", value)
        value = re.sub(r"(?i)(postgres|mysql|mongodb|redis)://[^\s]+", "[REDACTED_DB_URL]", value)
        
        return value

    if isinstance(value, dict):
        return {
            k: sanitize_for_observability(v) 
            for k, v in value.items() 
            if not any(secret in k.lower() for secret in ["password", "secret", "token", "key", "auth", "cookie", "authorization"])
        }

    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_observability(item) for item in value]

    if isinstance(value, (int, float, bool)):
        return value

    return str(value)


def summarize_soft_state(soft_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize soft_state for observability without exposing full payloads.
    """
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
    """
    Summarize property search results without sending full lists.
    """
    if not isinstance(results, list):
        return {"count": 0, "has_results": False}

    return {
        "count": len(results),
        "has_results": len(results) > 0,
        "first_property_id": results[0].get("id") if isinstance(results[0], dict) else None,
    }


def summarize_booking_state(soft_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize booking state for observability, redacting PII.
    """
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


class NoOpObserver:
    """A no-op observer that safely ignores all tracing calls."""
    
    def trace(self, name: str, **kwargs) -> "NoOpTrace":
        return NoOpTrace()

    def flush(self) -> None:
        pass


class NoOpTrace:
    """A no-op trace that safely ignores all span creation and updates."""
    
    def span(self, name: str, **kwargs) -> "NoOpSpan":
        return NoOpSpan()

    def update(self, **kwargs) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "NoOpTrace":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class NoOpSpan:
    """A no-op span that safely ignores all updates."""
    
    def update(self, **kwargs) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class LangfuseObserver:
    """Wrapper around the Langfuse SDK with safe error handling."""
    
    def __init__(self):
        self._langfuse = None
        self._is_active = False
        
        if not LANGFUSE_ENABLED:
            logger.debug("[Langfuse] Disabled via environment variable.")
            return
            
        if not _is_configured():
            logger.warning("[Langfuse] Enabled but missing PUBLIC_KEY or SECRET_KEY. Falling back to no-op.")
            return
            
        try:
            import langfuse
            self._langfuse = langfuse.Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_BASE_URL,
            )
            self._is_active = True
            logger.info("[Langfuse] Initialized successfully.")
        except ImportError:
            logger.warning("[Langfuse] 'langfuse' package not installed. Falling back to no-op.")
        except Exception as exc:
            logger.error("[Langfuse] Failed to initialize: %s. Falling back to no-op.", exc)

    def is_active(self) -> bool:
        return self._is_active

    def trace(self, name: str, **kwargs) -> Any:
        if not self._is_active or self._langfuse is None:
            return NoOpTrace()
        
        try:
            sanitized_kwargs = {}
            for k, v in kwargs.items():
                if k in ("metadata", "input", "output"):
                    sanitized_kwargs[k] = sanitize_for_observability(v)
                else:
                    sanitized_kwargs[k] = v
                    
            return self._langfuse.trace(name=name, **sanitized_kwargs)
        except Exception as exc:
            logger.error("[Langfuse] Failed to create trace '%s': %s", name, exc)
            return NoOpTrace()

    def flush(self) -> None:
        if self._is_active and self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as exc:
                logger.error("[Langfuse] Failed to flush: %s", exc)


def get_observer() -> LangfuseObserver | NoOpObserver:
    """
    Get the global Langfuse observer instance.
    Returns a NoOpObserver if Langfuse is disabled, misconfigured, or failed to import.
    """
    global _observer
    if _observer is None:
        _observer = LangfuseObserver()
    return _observer
