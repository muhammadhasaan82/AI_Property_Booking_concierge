# API Reference

Base URL: `http://localhost:8000`

## Health

### `GET /health`
Returns service liveness status.

**Response**
```json
{"status": "ok", "timestamp": "2025-01-01T00:00:00Z"}
```

---

## Chat

### `POST /api/v1/chat`
Send a message to the AI concierge.

**Request**
```json
{
  "message": "Find a 2-bedroom apartment in Miami under $200/night",
  "session_id": "optional-session-uuid"
}
```

**Response**
```json
{
  "reply": "I found 8 properties in Miami...",
  "session_id": "uuid",
  "intent": "property_search"
}
```

---

## Properties

### `POST /api/v1/properties/search`
Direct property search (bypasses agent pipeline).

**Request**
```json
{
  "query_text": "beachfront villa",
  "budget": 300.0,
  "amenities": ["pool", "wifi"],
  "location": "Miami",
  "beds": 3
}
```

---

## Booking

### `POST /api/v1/booking/create`
Create a new booking.

### `GET /api/v1/booking/{booking_id}`
Get booking status.

### `DELETE /api/v1/booking/{booking_id}`
Cancel a booking.

---

## Webhooks

### `POST /api/v1/webhooks/stripe`
Stripe payment webhook endpoint.

---

## Debug

### `GET /debug/config`
Inspect dynamic config (intent catalog, routing policies, guardrails, vocabulary).

---

## Rust Gateway

Base URL: `http://localhost:3001`

### `POST /tool`
Execute a tool via the CAG layer (search, FAQ, booking validation).

### `GET /health`
Rust gateway liveness check.
