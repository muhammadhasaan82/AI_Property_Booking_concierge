# ADK Tool Contract

## Scope

This document defines the public ADK-facing contract for exposed tools.

## Allowed Public Parameter Types

Public tool signatures may expose only JSON-schema-compatible shapes:

- `string`
- `number`
- `integer`
- `boolean`
- `null`
- `array[...]` composed of JSON-safe items
- `object` composed of JSON-safe fields

## Disallowed Public Parameter Types

Public tool signatures must not expose internal Python runtime types, including:

- `SearchPlan`
- `SearchTrace`
- `DynamicConstraints`
- `ToolContext`
- dataclass instances
- `SimpleNamespace`
- arbitrary Python classes

These may exist internally behind wrappers or runtime adapters, but they are not part of the public ADK contract.

## Public Tool Surface

### Search Tools

- `search_properties`
- `select_property`
- `get_property_details`
- `get_all_available_cities`

### Booking Tools

- `request_booking_details`
- `review_booking_details`
- `process_v2_booking`
- `check_booking_status`

### Support Tools

- `check_faq`
- `handle_small_talk`
- `escalate_to_human`

## Inputs

Each public tool must accept only JSON-safe parameters appropriate to its workflow.

Examples:

- `search_properties(city, budget, beds, bathrooms, guests, property_type, amenities, free_text, max_results, sort_preferences, action_intent, context_flag)`
- `select_property(option_number, action_intent, context_flag)`
- `request_booking_details(property_id, property_title, guest_name, guest_email, guest_phone, check_in, check_out, guests, price_per_night, missing_fields, action_intent, context_flag)`

## Outputs

- Tool returns must be JSON-safe objects.
- Returned payloads must include deterministic `status`.
- Returned payloads may include `deterministic_reply`, structured payload fields, and compatibility routing metadata.

## Invariants

- Public tool contracts remain stable and schema-driven.
- Public tool signatures do not reveal planner, schema, context, or persistence implementation types.
- Internal runtime context injection must not leak into the public schema.
- Soft state persistence is a backend concern, not a public tool argument.

## Failure Behavior

- Tools with missing required business inputs return structured missing-data payloads.
- Tool schema generation must not fail because of non-JSON-safe annotations.
- If an internal helper requires runtime context, that requirement must be satisfied behind the public tool surface.

## Acceptance Tests

- Inspecting each ADK-exposed callable yields only JSON-safe public parameters.
- No public ADK tool signature includes `SearchPlan`, `DynamicConstraints`, or `ToolContext`.
- Search and booking tools still remain callable by internal deterministic code paths after public-schema adaptation.
