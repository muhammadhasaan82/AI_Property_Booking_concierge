# Error Handling

## Public API Rules

- Validation errors should be returned by FastAPI/Pydantic as structured 422 responses.
- Internal errors should not leak stack traces or provider credentials.
- Webhook verification failures should return 400.
- Missing webhook secret in non-development environments should fail closed.

## Agent Runtime Rules

- Prompt injection or unsafe input returns safe fallback copy.
- Empty or whitespace input returns a clarification prompt.
- ADK turn timeout returns configured fallback copy.
- Tool-loop detection returns the anomaly fallback.
- Telemetry failures must not fail the user-facing response.

## Service Rules

- Database failures in booking creation may use transient mock fallback where existing behavior allows it.
- Database status/update failures return `{"ok": false, "error": "..."}`.
- Rust gateway failures fall back to Python/RAG/basic services when available.

## Logging Rules

- Log enough context to debug tool/provider failure paths.
- Do not log secrets, full authorization headers, or raw payment credentials.
- User-facing responses should be sanitized before telemetry or chat logging.