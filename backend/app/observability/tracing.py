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
except Exception:
    trace = None  
    OTLPSpanExporter = None  
    TracerProvider = None  
    BatchSpanProcessor = None 
    Resource = None
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
            except Exception as exc:
                logger.warning("Failed to configure OTLP exporter (%s); spans will stay local", exc)

        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("ai-concierge")
        _TRACER_INIT_DONE = True


tracer = trace.get_tracer(__name__)

class _Span(ContextDecorator):
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self.span_ctx = None
        self.current_span = None

    def __enter__(self):
        self.span_ctx = tracer.start_as_current_span(self.name)
        self.current_span = self.span_ctx.__enter__()
        for k, v in self.attributes.items():
            self.current_span.set_attribute(k, _safe_attr_value(v))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.current_span.record_exception(exc_value)
        self.span_ctx.__exit__(exc_type, exc_value, traceback)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        return self.__exit__(exc_type, exc_value, traceback)

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
