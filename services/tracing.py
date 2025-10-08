# services/tracing.py
import time
import threading
from contextlib import ContextDecorator
from typing import Any, Dict, Optional


class _Span(ContextDecorator):
    """Simple span that logs start/end with duration and optional attributes.

    - Thread-safe ID counter to correlate nested spans
    - Minimal, stdout-based for now (no external deps)
    """

    _counter = 0
    _lock = threading.Lock()

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
        with _Span._lock:
            _Span._counter += 1
            self.span_id = _Span._counter

    def __enter__(self):
        self.start_time = time.perf_counter()
        kv = " ".join(f"{k}={v}" for k, v in self.attributes.items())
        print(f"[TRACE] > span={self.name} id={self.span_id} {kv}".strip())
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        status = "ok" if exc is None else f"error={exc}"
        print(f"[TRACE] < span={self.name} id={self.span_id} duration_ms={self.duration_ms:.1f} {status}")
        # Do not suppress exceptions
        return False

    # Allow async usage by sharing the same underlying logic
    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self.__exit__(exc_type, exc, tb)


def span(name: str, attributes: Optional[Dict[str, Any]] = None) -> _Span:
    """Create a span context manager.

    Usage:
        with span("node_property", {"intent": state.get("intent") } ):
            ...
    """
    return _Span(name=name, attributes=attributes)


def annotate(extra: Dict[str, Any]) -> None:
    """Placeholder for richer attribute updates mid-span. Currently logs."""
    kv = " ".join(f"{k}={v}" for k, v in (extra or {}).items())
    if kv:
        print(f"[TRACE] • {kv}")
