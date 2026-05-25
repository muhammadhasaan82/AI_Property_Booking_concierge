# API Contracts

## Base Application

The canonical FastAPI app is `backend/app/main.py`.

Root endpoints are compatibility/debug helpers. Primary application endpoints are mounted under `/api/v1`.

## Health

`GET /api/v1/health`

Response:

```json
{"ok": true}
```

## Chat

`POST /api/v1/chat/message`

Request:

```json
{
  "message": "Find a 2 bed in New York under 200",
  "user_id": "optional-user-id",
  "session_id": "optional-session-id"
}
```

Response:

```json
{
  "reply": "string",
  "user_id": "resolved-user-id",
  "session_id": "resolved-session-id"
}
```

Contract:

- `message` is required and must be a string.
- If `user_id` is omitted, the API uses `api_user`.
- If `session_id` is omitted, the API generates one.
- The route must not expose provider tracebacks to clients.

## Properties

`POST /api/v1/properties/search`

Request:

```json
{
  "query_text": "optional text",
  "budget": 200.0,
  "amenities": ["wifi"],
  "location": "New York",
  "beds": 2
}
```

Response:

```json
{"results": []}
```

Contract:

- `budget`, when provided, is non-negative.
- `beds`, when provided, is at least 1.
- Response always contains `results`.

## Booking

`POST /api/v1/booking/create`

Request:

```json
{
  "user_id": "string",
  "property_id": "string",
  "check_in": "YYYY-MM-DD",
  "check_out": "YYYY-MM-DD",
  "guests": 1,
  "phone": "optional string"
}
```

Response comes from `backend/app/services/booking.py` and currently uses `ok` as the success key.

`POST /api/v1/booking/update-status`

Request:

```json
{
  "booking_id": "string",
  "current_status": "string",
  "new_status": "string"
}
```

`GET /api/v1/booking/status/{booking_id}`

Response comes from `get_booking_status`.

## FAQ

`GET /api/v1/faq?question=...`

Response:

```json
{
  "answer": "string or null",
  "source": "db or llm_fallback"
}
```

## Webhooks

`POST /api/v1/webhooks/stripe`

Contract:

- Production mode requires `STRIPE_WEBHOOK_SECRET`.
- Development mode may parse unsigned JSON.
- The endpoint returns `{"received": true}` when an event is accepted.

## Mobile Compatibility API

Routes under `/api/v1/mobile/*` are compatibility/mobile-facing endpoints. They should preserve existing response keys such as `success`, `message`, `booking`, `response`, and `session_id`.

Important compatibility contracts:

- `POST /api/v1/mobile/booking/create` calls the same async booking service as the direct booking API and treats `ok=true` as success.
- `GET /api/v1/mobile/booking/{booking_id}` awaits the booking status service before returning.
- `GET /api/v1/mobile/faq` accepts query parameters and may also tolerate a JSON body for older clients.