# Testing Strategy

## Test Layers

1. Contract tests for config/schema loaders.
2. API validation tests for direct FastAPI route models and compatibility paths.
3. Agent unit tests that do not call live LLMs, Redis, Rust, or Supabase.
4. Tool tests for booking state, property search, FAQ fallback, and anomaly handling.
5. Integration smoke tests for local Docker Compose services.
6. Optional eval suite for LLM routing quality.

## Required Safe Tests

Any small refactor in this pass should include tests for:

- Tool registry `intents` metadata.
- Understanding frame validation/normalization.
- API request validation for critical booking/search inputs.
- Mobile booking async service behavior.

## Non-Goals For This Pass

- No live provider tests.
- No real Stripe webhook calls.
- No destructive database tests.
- No broad end-to-end LLM routing assertions unless fully stubbed.

## Verification Commands

Preferred local commands:

```bash
cd backend
python -m pytest tests/test_spec_driven_contracts.py
python -m pytest tests/tests/test_booking_memory_fallback.py
```

If dependencies are unavailable, run syntax checks:

```bash
python -m compileall backend/app backend/tests
```