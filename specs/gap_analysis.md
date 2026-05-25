# Gap Analysis

## Confirmed Gaps Found During Spec Pass

1. `docs/api-reference.md` references endpoints that do not match current route code, such as `/api/v1/chat` instead of `/api/v1/chat/message`, and `/api/v1/booking/{booking_id}` instead of `/api/v1/booking/status/{booking_id}`.
2. `.github/workflows/ci.yml` appears to be a fragment rather than a complete GitHub Actions workflow.
3. Database schema sources are not fully aligned. `db_setup.py` uses a `booking_status` enum and defaults `bookings.status` to `confirmed`; the SQL migration uses text status and defaults to `pending`.
4. The repository has both `env.example` and no canonical `.env.example` before this pass.
5. Existing tests include assertions that appear inconsistent with current config, such as expecting non-empty `intent_catalog.yaml#intents` while the file is intentionally minimal.
6. Several docs still refer to older V1/LangGraph or GPT-4o-mini architecture details while current root README and AGENTS instructions describe Native V2 ADK.
7. Some route modules expose compatibility/mock mobile endpoints whose production status is unclear.
8. CI/runtime dependency installation is not pinned by a lock file, and the workspace may not have test dependencies installed.

## Uncertainties To Resolve Before Larger Refactors

1. Which database schema is canonical: `db_setup.py`, `backend/infrastructure/database/migrations/001_initial_schema.sql`, or `backend/infrastructure/supabase/migrations/20240101000000_init_tables.sql`.
2. Whether the mobile API should be treated as production API or demo compatibility API.
3. Whether direct booking API should enforce date ordering at request validation time or leave it to the agent/booking schema.
4. Whether CORS should remain wildcard by default in deployed environments.
5. Whether the eval suite is part of CI or a manual quality gate.