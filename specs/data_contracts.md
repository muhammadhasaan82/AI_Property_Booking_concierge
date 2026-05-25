# Data Contracts

## Property Dataset

The CSV dataset is expected at `backend/data/dataset.csv` by default.

Property rows are normalized by `backend/app/components/search.py` into fields:

- `id`
- `title`
- `city`
- `price_per_night`
- `property_type`
- `bedrooms`
- `bathrooms`
- `beds`
- `amenities`
- `rating`
- `description`

The direct property search API returns a list of these dictionaries.

## Booking State

Canonical in-session booking state is managed by `backend/app/agents/state/booking_state.py`.

Booking fields:

- Required string fields come from `agent_config.yaml#booking.required_fields`.
- Required numeric fields come from `agent_config.yaml#booking.required_numeric_fields`.
- Rich prompts and validators come from `booking_schema.yaml#booking`.

Current required fields:

- `property_id`
- `property_title`
- `guest_email`
- `check_in`
- `check_out`
- `guests`
- `price_per_night`

## Booking Service Payload

`create_booking` requires:

- `user_id`
- `property_id`
- `check_in`
- `check_out`

Optional:

- `guests`
- `phone`

The service returns:

```json
{
  "ok": true,
  "booking_id": "string",
  "status": "pending",
  "payment_url": "string"
}
```

On validation failure:

```json
{"ok": false, "error": "string"}
```

## Database Tables

Expected tables:

- `public.users`
- `public.bookings`
- `public.chat_history`
- `public.booking_details`
- `public.successful_bookings`

Schema sources currently differ between `db_setup.py` and SQL migrations. Any future schema change must update the canonical migration and note compatibility in `specs/gap_analysis.md`.