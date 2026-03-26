# AGENTS.md — AI Concierge & Calling Agent

## Architecture Overview

This project is a **production-grade AI Property Booking Concierge** using a nested hybrid multi-agent graph.

### Agent Layers

| Layer | File | Model | Role |
|-------|------|-------|------|
| **Triage Router** | `app/agents/adk_agents.py` | GPT-4o-mini via LiteLLM | Intent classification + tool dispatch |
| **Concierge Voice** | `app/agents/adk_agents.py` | Groq Llama-3.3-70B | Conversational response synthesis |
| **V1 LangGraph** | `app/services/graph.py` | OpenAI GPT | Legacy fallback pipeline |
| **Checkout Vault** | `app/services/checkout_graph.py` | Deterministic | Booking state machine (no LLM) |

### Tool Registry (Triage Router)

| Tool | Module | Description |
|------|--------|-------------|
| `search_properties` | `app/agents/adk_agents.py` | Property search via Rust CAG + Python fallback |
| `check_faq` | `app/agents/adk_agents.py` | Policy lookup via Rust CAG + RAG |
| `check_booking_status` | `app/agents/adk_agents.py` | Booking status from DB |
| `trigger_checkout_flow` | `app/agents/adk_agents.py` | LangGraph booking state machine |
| `get_available_cities` | `app/agents/agents.py` | City list from dataset |

## Module Map

```
app/
├── agents/          — ADK agents + V1 LangGraph agents
│   └── tools/       — Rust gateway client wrappers
├── components/      — Retrieval, NLP, search, FAQ enhanced
├── observability/   — Tracing, DPO telemetry, DB logging
├── prompts/         — Agent instruction templates
├── security/        — Input/output guardrails, anomaly detection
├── services/        — Core business logic (booking, checkout, graph, DB)
└── route/           — FastAPI route handlers
```

## Running Agents Locally

```bash
# Start FastAPI backend (from project root)
uvicorn app.main:app --reload --port 8000

# Start Chainlit frontend
chainlit run frontend/chainlit_app.py --port 8501

# Start Rust gateway
cd rust_gateway && cargo run --release

# Export DPO dataset
python scripts/export_dpo_dataset.py --output dpo_dataset.jsonl
```

## Environment Flags

| Variable | Default | Effect |
|----------|---------|--------|
| `ADK_ENABLED` | `1` | Enable ADK V2 pipeline (set `0` for V1 fallback) |
| `DPO_TELEMETRY_ENABLED` | `1` | Enable DPO trajectory logging |
| `ANOMALY_DETECTION_ENABLED` | `1` | Enable tool-loop anomaly guard |
| `RATE_LIMIT_ENABLED` | `1` | Enable Rust rate limiter middleware |

## Phase History

- **Phase 1**: Rust Cache-Augmented Generation (CAG) for FAQ intercepts
- **Phase 2**: Nested Hybrid Graph — ADK 2.0 dual-LLM SequentialAgent
- **Phase 3**: Continuous Learning — DPO telemetry, anomaly detection, RLHF export
