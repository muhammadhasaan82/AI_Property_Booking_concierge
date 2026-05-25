# Configuration

## Sources

Environment variables are loaded by `backend/app/services/config.py` from:

1. repository root `.env`
2. `backend/app/services/.env`

The real `.env` must not be committed or modified by automation.

## Runtime Environment Contract

`.env.example` is the public template. It must contain placeholders only.

Critical groups:

- LLM providers: `OPENAI_API_KEY`, `GROQ_API_KEY`, `ADK_DISPATCHER_MODEL`, `ADK_VOICE_MODEL`
- Database: `DATABASE_URL`, `SUPABASE_DB_*`, `DB_*`
- Redis/session: `REDIS_URL`, `REDIS_SESSION_TTL_SECONDS`, `ADK_SESSION_*`
- Rust gateway: `RUST_GATEWAY_URL`, `RUST_TIMEOUT`, `RUST_RATE_LIMIT_*`
- Payments/webhooks: `PAYMENT_BASE_URL`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`
- Security/admin: `ADMIN_TOKEN`, `JWT_SECRET`, `CHAINLIT_AUTH_SECRET`
- Features: `UNDERSTANDING_FRAME_ENABLED`, `TOOL_REGISTRY_ENABLED`, `RESPONSE_POLICIES_ENABLED`, `POLICY_ROUTER_MODE`

## YAML Behaviour Configuration

Agent behavior belongs in:

- `backend/app/config/agent_config.yaml`
- `backend/app/config/tool_registry.yaml`
- `backend/app/config/agent_policy.yaml`
- `backend/app/config/booking_schema.yaml`
- `backend/app/config/response_policies.yaml`
- `backend/app/config/routing_policies.yaml`
- `backend/app/config/vocabulary.yaml`
- `backend/app/config/guardrails.yaml`
- `backend/app/config/thresholds.yaml`
- `backend/app/config/retrieval.yaml`

## Precedence

Environment variables override YAML scalar defaults where a loader explicitly supports an override. YAML remains the source for lists, routing metadata, and prompt-adjacent policy blocks unless a loader states otherwise.