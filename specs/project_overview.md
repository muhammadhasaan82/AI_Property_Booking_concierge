# Project Overview

## Purpose

AI Property Booking Concierge is a FastAPI, Chainlit, Rust, Redis, and Postgres/Supabase application for property discovery, FAQ lookup, booking capture, booking status lookup, and human escalation.

The product boundary is a property booking concierge. The assistant should help users search listings, answer hotel/property policy questions, collect booking details, confirm reservations, and hand off to a human when needed.

## Current Runtime Surfaces

- Backend API: `backend/app/main.py` with API routes under `/api/v1`.
- Agent pipeline: `backend/app/services/adk_runner.py` and `backend/app/agents/adk_agents.py`.
- Direct property search: `backend/app/route/properties.py` and `backend/app/components/search.py`.
- Booking service: `backend/app/route/booking.py` and `backend/app/services/booking.py`.
- Mobile compatibility API: `backend/app/route/mobile.py`.
- Rust gateway: `backend/rust_gateway/`.
- Configuration: `backend/app/config/*.yaml` and environment variables loaded by `backend/app/services/config.py`.
- Database schema assumptions: `backend/infrastructure/database/migrations/001_initial_schema.sql`, `backend/infrastructure/supabase/migrations/20240101000000_init_tables.sql`, and `backend/app/services/db_setup.py`.

## Spec-Driven Development Rule

Behavior changes should start by updating the relevant file in `/specs`. Code changes must then point back to the spec or to an explicit gap in `specs/gap_analysis.md`.

## Stability Constraints

- Do not break public API paths or response keys without a migration plan.
- Do not commit real secrets or modify a real `.env`.
- Keep agent behavior declarative where possible: YAML, prompts, typed schemas, and small helper functions.
- Prefer narrow fixes to large rewrites.
- Preserve fallback behavior when external systems are unavailable.