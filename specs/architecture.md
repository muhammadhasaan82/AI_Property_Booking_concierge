# Architecture

## High-Level Flow

1. Users interact through Chainlit or direct HTTP clients.
2. FastAPI receives API requests and delegates to route modules.
3. Chat traffic flows through `run_adk_turn` in `backend/app/services/adk_runner.py`.
4. The deterministic pre-router may answer obvious low-value turns before ADK runs.
5. The ADK `SequentialAgent` runs the configured agent nodes:
   - `understanding_agent`: emits a typed `UnderstandingFrame`.
   - `triage_router`: selects a registered tool.
   - `concierge_voice`: synthesizes the final user-facing response.
6. Tools call Python services and may fall back from Rust gateway/RAG/database failures.
7. Redis stores ADK session snapshots and soft state.
8. Postgres/Supabase stores users, bookings, chat logs, and successful booking records.

## Ownership Map

- Agent wiring belongs in `backend/app/agents/adk_agents.py`.
- Agent execution, state persistence, guardrails, anomaly checks, and telemetry belong in `backend/app/services/adk_runner.py`.
- Tool functions belong under `backend/app/agents/tools/`.
- Prompt text belongs under `backend/app/prompts/`.
- Configurable routing and behavior values belong under `backend/app/config/`.
- API request/response validation belongs in route-local Pydantic models.
- Database connection behavior belongs in `backend/app/services/db_client.py`.

## Configuration Boundary

`backend/app/services/config.py` is the central runtime environment loader for service-level values. YAML loaders under `backend/app/config/` remain the central source for agent behavior, routing, tool registry, response policy, and booking schema metadata.

## External Dependency Assumptions

- Rust gateway is optional for Python fallback paths, except where callers explicitly require gateway-only tools.
- Redis is preferred for session state, but chat should degrade gracefully where possible.
- Postgres/Supabase failures should not crash booking capture when mock or transient fallback behavior exists.
- LLM provider failures should result in concise fallback copy, not raw tracebacks.