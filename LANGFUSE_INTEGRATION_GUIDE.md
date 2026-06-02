# Langfuse Integration Guide

This document provides instructions for validating the Langfuse observability integration in your GCP environment.

## Part 10: Running Tests

Run these commands from the `backend/` directory of your project.

### 1. Syntax Check
First, verify that all modified files compile without syntax errors:
```bash
PYTHONPATH=. python -m py_compile \
  app/services/observability/langfuse_observer.py \
  app/services/observability/prompt_registry.py \
  app/services/adk_runner.py \
  app/services/direct_property_search.py \
  app/services/property_query_constraints.py \
  app/agents/tools/search.py \
  app/services/booking_flow.py \
  app/main.py \
  tests/tests/test_langfuse_observability.py
```

### 2. Targeted Observability Tests
Run the new Langfuse-specific tests to ensure the observer, sanitization, and tracing logic work correctly:
```bash
PYTHONPATH=. pytest \
  tests/tests/test_langfuse_observability.py \
  -v --tb=short
```

### 3. Full Regression Suite
Ensure that the instrumentation did not break any existing functionality:
```bash
PYTHONPATH=. pytest \
  tests/tests/test_langfuse_observability.py \
  tests/tests/test_property_query_constraints.py \
  tests/tests/test_booking_details_flow.py \
  tests/tests/test_booking_confirmation_flow.py \
  tests/tests/test_direct_property_search.py \
  tests/tests/test_search_all_matching_results.py \
  tests/tests/test_property_rejection_flow.py \
  tests/tests/test_service_coverage_guard.py \
  tests/tests/test_config_driven_shortcuts.py \
  tests/tests/test_booking_search_state_phase1.py \
  tests/tests/test_property_listing_flow.py \
  tests/tests/test_v25_smoke_user_flows.py \
  tests/tests/test_model_config_resolution.py \
  tests/test_env_precedence.py \
  tests/tests/test_e2e_booking_to_receipt_flow.py \
  tests/tests/test_booking_memory_fallback.py \
  -v --tb=short
```

## Part 11: Live Smoke Testing

### Phase 1: Verify No-Op Behavior (Langfuse Disabled)
1. Ensure your `.env` file has `LANGFUSE_ENABLED=false`.
2. Start the backend: `uvicorn app.main:app --reload --port 8000`
3. Run the following queries in your frontend/Chainlit:
   - "show me some 2 bedrooms apartment in new york city"
   - "show me properties in Los Angeles"
   - Initiate a booking flow: search -> option -> "yeah sure" -> details -> review -> "yes"
4. **Expected Result**: App behavior is completely unchanged. All existing tests pass. No errors in logs.

### Phase 2: Connect to Self-Hosted Langfuse
1. Ensure your self-hosted Langfuse is running on your GCP instance (see `infrastructure/langfuse/README.md`).
2. In the Langfuse UI (`http://<server-ip>:3000`), create a new project.
3. Copy the **Public Key** and **Secret Key** for that project.
4. Update your `.env` file:
   ```env
   LANGFUSE_ENABLED=true
   LANGFUSE_BASE_URL=http://127.0.0.1:3000  # Or your internal GCP IP
   LANGFUSE_PUBLIC_KEY=<your_project_public_key>
   LANGFUSE_SECRET_KEY=<your_project_secret_key>
   LANGFUSE_ENVIRONMENT=dev
   LANGFUSE_REDACT_INPUTS=true
   ```
5. Restart the backend.

### Phase 3: Validate Traces in Langfuse UI
Run the following queries and verify the traces in the Langfuse dashboard:

1. **Query**: "show me some 2 bedrooms apartment in new york city"
   - **Expected**: Same deterministic app response.
   - **Langfuse Trace**: Should show `classified_search` and `direct_property_search` spans. Metadata should show `extracted_city="New York"`, `extracted_property_type="Apartment"`, and `result_count`.

2. **Query**: "show me properties in Los Angeles"
   - **Expected**: Deterministic property render, no Groq 413 errors.
   - **Langfuse Trace**: The `chat_turn` trace metadata must include `"bypassed_voice_llm": true`. There should be a `deterministic_render` span, and **no** `llm_generation` span.

3. **Booking Flow**: Search -> Select Option -> "yeah sure" -> Provide Details -> Review -> "yes"
   - **Expected**: Receipt generated successfully.
   - **Langfuse Trace**: Should show `booking_flow` spans with `previous_booking_stage` and `next_booking_stage` transitions. 
   - **Security Check**: Verify that `guest_name`, `guest_email`, and `guest_phone` are either absent from the trace metadata or explicitly marked as `"[REDACTED]"` (since `LANGFUSE_REDACT_INPUTS=true`).

### Debug Endpoint Verification
You can verify the configuration status without exposing secrets by calling:
```bash
curl http://127.0.0.1:8000/debug/langfuse
```
**Expected Response**:
```json
{
  "enabled": true,
  "configured": true,
  "base_url_host": "127.0.0.1",
  "environment": "dev",
  "release": "",
  "prompts_enabled": false,
  "redaction_enabled": true,
  "sample_rate": 1.0
}
```
*Note: It will never return `public_key`, `secret_key`, or any connection strings.*
