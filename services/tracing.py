# services/tracing.py
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import ContextDecorator
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency at import time
    trace = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False

_TRACER = None
_TRACER_LOCK = threading.Lock()
_TRACER_INIT_DONE = False


def _safe_attr_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return ""
    return str(value)


def _init_tracer() -> None:
    global _TRACER_INIT_DONE, _TRACER
    if _TRACER_INIT_DONE:
        return
    with _TRACER_LOCK:
        if _TRACER_INIT_DONE:
            return

        if not _OTEL_AVAILABLE:
            logger.warning("OpenTelemetry packages not installed; using log-only spans")
            _TRACER_INIT_DONE = True
            return

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        provider = TracerProvider(resource=Resource.create({"service.name": "ai-concierge"}))

        if endpoint:
            try:
                exporter = OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"))
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except Exception as exc:  # pragma: no cover - exporter setup failures are runtime-specific
                logger.warning("Failed to configure OTLP exporter (%s); spans will stay local", exc)

        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("ai-concierge")
        _TRACER_INIT_DONE = True


class _Span(ContextDecorator):
    _counter = 0
    _lock = threading.Lock()

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
        self._otel_cm = None
        self._otel_span = None
        with _Span._lock:
            _Span._counter += 1
            self.span_id = _Span._counter

    def __enter__(self):
        _init_tracer()
        self.start_time = time.perf_counter()

        if _OTEL_AVAILABLE and _TRACER is not None:
            self._otel_cm = _TRACER.start_as_current_span(self.name)
            self._otel_span = self._otel_cm.__enter__()
            self._otel_span.set_attribute("span_id", self.span_id)
            for key, value in self.attributes.items():
                self._otel_span.set_attribute(key, _safe_attr_value(value))
        else:
            kv = " ".join(f"{k}={v}" for k, v in self.attributes.items())
            logger.debug("span=%s id=%d %s", self.name, self.span_id, kv)

        return self

    def __exit__(self, exc_type, exc, tb):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0

        if self._otel_span is not None:
            self._otel_span.set_attribute("duration_ms", self.duration_ms)
            self._otel_span.set_attribute("status", "ok" if exc is None else "error")
            if exc is not None:
                self._otel_span.record_exception(exc)
            self._otel_cm.__exit__(exc_type, exc, tb)
        else:
            status = "ok" if exc is None else f"error={exc}"
            logger.debug("span=%s id=%d duration_ms=%.1f %s", self.name, self.span_id, self.duration_ms, status)

        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self.__exit__(exc_type, exc, tb)


def span(name: str, attributes: Optional[Dict[str, Any]] = None) -> _Span:
    return _Span(name=name, attributes=attributes)


def annotate(extra: Dict[str, Any]) -> None:
    _init_tracer()
    if _OTEL_AVAILABLE and trace is not None:
        span_obj = trace.get_current_span()
        if span_obj is not None:
            for key, value in (extra or {}).items():
                span_obj.set_attribute(key, _safe_attr_value(value))
            return
    kv = " ".join(f"{k}={v}" for k, v in (extra or {}).items())
    if kv:
        logger.debug("%s", kv)
