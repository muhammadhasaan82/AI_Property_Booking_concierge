# AGENTS.md — AI Concierge & Calling Agent

## Architecture Overview

This project is a **production-grade AI Property Booking Concierge** using a 100% Native V2 Agentic Architecture with dual-LLM tool-calling.

### Agent Layers

| Layer | File | Model | Role |
|-------|------|-------|------|
| **Triage Router** | `app/agents/adk_agents.py` | GPT-5 Nano via LiteLLM | Intent classification + native tool dispatch |
| **Concierge Voice** | `app/agents/adk_agents.py` | Groq Llama-3.3-70B | Conversational response synthesis + streaming |

### Tool Registry (Triage Router)

| Tool | Module | Description |
|------|--------|-------------|
| `search_properties` | `app/agents/adk_agents.py` | Property search via Rust CAG + Python fallback |
| `check_faq` | `app/agents/adk_agents.py` | Policy lookup via Rust CAG + RAG |
| `check_booking_status` | `app/agents/adk_agents.py` | Booking status from DB |
| `request_booking_details` | `app/agents/adk_agents.py` | V2 Off-Switch: prompts user for missing booking fields |
| `process_v2_booking` | `app/agents/adk_agents.py` | V2 Native: processes final booking with receipt generation |
| `escalate_to_human` | `app/agents/adk_agents.py` | Human handoff escalation |
| `get_all_available_cities` | `app/agents/adk_agents.py` | City list from dataset |

## Module Map

```
app/
├── agents/          — ADK V2 agents (SequentialAgent pipeline)
│   └── tools/       — Rust gateway client wrappers
├── components/      — Retrieval, NLP, search, FAQ enhanced
├── observability/   — Tracing, DPO telemetry, DB logging
├── prompts/         — Agent instruction templates
├── security/        — Input/output guardrails, anomaly detection
├── services/        — Core business logic (booking, DB, config)
└── route/           — FastAPI route handlers
```

## Running Agents Locally

```bash
# Start FastAPI backend (from backend/)
uvicorn app.main:app --reload --port 8000

# Start AI Booking frontend
chainlit run frontend/chainlit_app.py --port 8501

# Start Rust gateway
cd rust_gateway && cargo run --release

# Export DPO dataset
python scripts/export_dpo_dataset.py --output dpo_dataset.jsonl
```

## Environment Flags

| Variable | Default | Effect |
|----------|---------|--------|
| `ADK_DISPATCHER_MODEL` | `openai/gpt-5-nano` | LiteLLM model for triage router |
| `ADK_VOICE_MODEL` | `groq/llama-3.3-70b-versatile` | LiteLLM model for concierge voice |
| `DPO_TELEMETRY_ENABLED` | `1` | Enable DPO trajectory logging |
| `ANOMALY_DETECTION_ENABLED` | `1` | Enable tool-loop anomaly guard |
| `RATE_LIMIT_ENABLED` | `1` | Enable Rust rate limiter middleware |

## Phase History

- **Phase 1**: Rust Cache-Augmented Generation (CAG) for FAQ intercepts
- **Phase 2**: Nested Hybrid Graph — ADK 2.0 dual-LLM SequentialAgent
- **Phase 3**: Continuous Learning — DPO telemetry, anomaly detection, RLHF export
- **Phase 4**: 100% Native V2 — LangGraph deleted, native LLM tool-calling for bookings
