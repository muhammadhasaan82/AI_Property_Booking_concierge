# Booking Workflow Spec

## Scope

This document defines the deterministic backend contract for:

1. Booking workflow
2. Cancellation workflow
3. Booking-status workflow

Intent detection may be probabilistic. Once a workflow is entered, state progression, validation, and persistence are deterministic.

## Booking Workflow

### Allowed States

- `idle`
- `collecting_details`
- `awaiting_confirmation`
- `awaiting_modification_choice`
- `modifying_details`
- `awaiting_property_reselection`
- `confirmed`

### State Transitions

- `property_details -> collecting_details`
  When the user explicitly starts booking the selected property.
- `collecting_details -> collecting_details`
  While gathering or validating missing fields.
- `collecting_details -> awaiting_confirmation`
  When all required fields are valid and the review summary is ready.
- `awaiting_confirmation -> confirmed`
  When the user explicitly confirms the booking.
- `awaiting_confirmation -> awaiting_modification_choice`
  When the user rejects the review summary without specifying a field.
- `awaiting_confirmation -> modifying_details`
  When the user requests a specific field change.
- `awaiting_confirmation -> awaiting_property_reselection`
  When the user asks to change the property.
- `awaiting_property_reselection -> collecting_details`
  When the user selects a replacement property.

### Inputs

Public booking inputs must be JSON-schema-compatible:

- `property_id: string | null`
- `property_title: string | null`
- `guest_name: string | null`
- `guest_email: string | null`
- `guest_phone: string | null`
- `check_in: string | null`
- `check_out: string | null`
- `guests: integer | null`
- `price_per_night: number | null`
- `missing_fields: array[string] | null`
- `action_intent: string | null`
- `context_flag: string | null`

### Outputs

- `booking_details_required`
- `gathering_info`
- `review_pending`
- `amendment_acknowledged`
- `booking_confirmed`
- `missing_critical_data`

### Invariants

- Booking is workflow-state driven, not keyword-patch driven.
- Booking may start only from an explicit property-selection context.
- Final confirmation may occur only from `awaiting_confirmation`.
- `check_in` must be on or after the configured reference date.
- `check_out` must be after `check_in`.
- `guests` must be validated against the selected property's capacity before confirmation.
- Search and pagination shortcuts are blocked during active booking except in `awaiting_property_reselection`.
- FAQ or policy questions during booking preserve booking state and resume target.

### Failure Behavior

- Missing fields return `gathering_info` and set `awaiting_field`.
- Invalid `check_in` returns a deterministic validation message and keeps the flow in `collecting_details`.
- Invalid `check_out` returns a deterministic validation message and keeps the flow in `collecting_details`.
- Missing property context prevents booking start.
- Confirmation without a review summary is rejected.

### Acceptance Tests

- Saying `book this property` from `property_details` enters `collecting_details`.
- Providing all booking fields in one message moves the flow to `awaiting_confirmation`.
- An invalid `check_out` preserves valid collected fields and asks only for `check_out`.
- Saying `change property` from review moves the flow to `awaiting_property_reselection`.
- Confirming from review generates a receipt and moves the flow to `confirmed`.

## Cancellation Workflow

### Allowed States

Current backend scope is informational and state-preserving only:

- `idle`
- `booking_context_preserved`
- `cancellation_info_presented`
- `cancellation_handoff_required`

No destructive booking-cancel execution is part of the current contract unless a dedicated cancellation tool is added.

### State Transitions

- `idle -> cancellation_info_presented`
  When the user asks about cancellation or refund policy.
- `collecting_details|awaiting_confirmation|confirmed -> booking_context_preserved`
  When a cancellation-policy question interrupts an active booking flow.
- `booking_context_preserved -> collecting_details|awaiting_confirmation|confirmed`
  When the user resumes the prior workflow.
- `idle|confirmed -> cancellation_handoff_required`
  When the user asks to actually cancel a booking but no cancellation-execution tool is available.

### Inputs

- `question: string`
- Optional booking context from soft state
- Optional booking ID if future cancellation execution is added

### Outputs

- `answered`
- `gathering_info`
- `handoff_required`

### Invariants

- Cancellation-policy questions are FAQ-like informational intents, not property-search intents.
- Informational cancellation turns must not destroy booking or search state.
- Actual cancellation execution is out of scope until a dedicated deterministic tool exists.

### Failure Behavior

- If the request is informational, answer from policy sources and preserve context.
- If the user asks to cancel a booking transactionally and no cancel tool exists, return a deterministic handoff or unsupported-action response.

### Acceptance Tests

- `What is the cancellation policy?` routes to FAQ/policy handling, not property search.
- `Can I cancel before check-in?` during booking preserves booking state and exposes a resume cue.
- `Cancel my booking` does not invoke property search.

## Booking-Status Workflow

### Allowed States

- `idle`
- `status_lookup`
- `status_found`
- `status_missing_id`
- `status_not_found`

### State Transitions

- `idle -> status_lookup`
  When a booking-status intent is detected.
- `status_lookup -> status_found`
  When a matching receipt or persisted record is found.
- `status_lookup -> status_missing_id`
  When no booking ID is present and no session receipt exists.
- `status_lookup -> status_not_found`
  When a booking ID is provided but no record exists.

### Inputs

- `booking_id: string | null`
- Session receipt state when available

### Outputs

- `found`
- `gathering_info`
- `booking_not_found`

### Invariants

- Booking-status intent is handled before general property search execution.
- A booking ID token is sufficient to trigger status lookup.
- Session receipts are preferred before DB lookup.
- Booking-status messages must not be interpreted as search refinements or new property searches.

### Failure Behavior

- Missing ID with no session receipt prompts the user for the registration ID.
- Unknown ID returns deterministic not-found messaging.
- DB lookup failures degrade to session-only behavior and not-found responses.

### Acceptance Tests

- `My booking ID is BK-...` returns deterministic status from the current session receipt when present.
- `I want to check my booking status` uses the latest session receipt if present.
- `I want to check my booking status` asks for booking ID when no receipt exists.
- Booking-status messages are not intercepted by direct property search.
