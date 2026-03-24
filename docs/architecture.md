# Architecture

## System Overview

The AI Property Booking Concierge uses a **nested hybrid multi-agent graph** composed of three phases of evolution built on top of one another.

```
User (Chainlit UI)
        │
        ▼
  frontend/chainlit_app.py
        │
        ├── ADK_ENABLED=1  ──▶  app/services/adk_runner.py
        │                               │
        │              ┌────────────────▼────────────────┐
        │              │      SequentialAgent (ADK 2.0)  │
        │              │  triage_router ──▶ concierge_voice│
        │              │  (GPT-4o-mini)    (Llama-3.3-70B) │
        │              └────────────┬────────────────────┘
        │                           │ tools
        │              ┌────────────▼────────────────────┐
        │              │         Tool Registry            │
        │              │  search_properties               │
        │              │  check_faq                       │
        │              │  check_booking_status            │
        │              │  trigger_checkout_flow           │
        │              └──┬──────────┬──────────┬────────┘
        │                 │          │          │
        │         Rust CAG     Python Search  LangGraph
        │         (port 3001)  (components/)  Checkout Vault
        │
        └── ADK_ENABLED=0  ──▶  app/services/graph.py (V1 LangGraph)
```

## Layer Descriptions

### ADK V2 Pipeline (`app/agents/`)
- **`adk_agents.py`** — Defines `triage_router` (LiteLLM/GPT-4o-mini) and `concierge_voice` (LiteLLM/Groq) as `LlmAgent` instances, composed into a `SequentialAgent`.
- **`agents.py`** — V1 agent functions used by both V1 graph and V2 tools.
- **`tools/rust_client.py`** — Async HTTP client bridging Python → Rust gateway (CAG layer).

### Rust Gateway (`rust_gateway/`)
- Phase 1: Cache-Augmented Generation intercepts FAQ queries before they hit the LLM.
- Implements sliding-window rate limiting middleware (`src/rate_limiter.rs`).
- Falls back gracefully when unavailable.

### LangGraph Pipelines (`app/services/`)
- **`graph.py`** — V1 chat graph with intent routing, property search, booking, payment nodes.
- **`checkout_graph.py`** — Deterministic Vault: isolated booking state machine for the ADK tool.

### Observability (`app/observability/`)
- **`tracing.py`** — OpenTelemetry spans with log-only fallback.
- **`telemetry.py`** — DPO trajectory logging to SQLite + optional Supabase mirror.
- **`db_logging.py`** — Async chat / booking / feedback logging.

### Security (`app/security/`)
- **`guardrails.py`** — Prompt injection, script injection, and leak detection.
- **`anomaly.py`** — In-memory tool-loop detection per session.
