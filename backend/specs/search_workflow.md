# Search Workflow Spec

## Scope

This document defines the deterministic backend contract for:

1. Search workflow
2. Property selection workflow
3. Pagination and refinement workflow

Intent understanding may be probabilistic upstream, but workflow execution, filtering, pagination, and state mutation are deterministic.

## Search Workflow

### Allowed States

- `idle`
- `needs_clarification`
- `property_list`
- `property_details`
- `no_results`
- `blocked_by_active_booking`

### State Transitions

- `idle -> needs_clarification`
  When a fresh search request is missing required deterministic inputs such as city.
- `idle -> property_list`
  When a valid search request resolves to one or more results.
- `idle -> no_results`
  When a valid search request resolves to zero results.
- `property_list -> property_list`
  When a refinement or pagination request updates the visible result window.
- `property_list -> property_details`
  When the user selects a property from the active option map.
- `property_details -> property_list`
  When the user rejects the selected property and returns to the previous shortlist.
- `any -> blocked_by_active_booking`
  When a new search or pagination shortcut is attempted during an active booking flow, unless the booking stage is `awaiting_property_reselection`.

### Inputs

Public search inputs must be JSON-schema-compatible:

- `city: string | null`
- `budget: number | null`
- `beds: integer | null`
- `beds_operator: "exact" | "min" | "max"`
- `bathrooms: integer | null`
- `bathrooms_operator: "exact" | "min" | "max"`
- `guests: integer | null`
- `guests_operator: "min" | "exact" | "max"`
- `property_type: string | null`
- `amenities: array[string] | string | null`
- `free_text: string | null`
- `max_results: integer | null`
- `sort_preferences: array[object] | null`

Internal planner objects are not part of the public contract.

### Outputs

- `properties_found`
- `property_details`
- `no_results`
- `missing_critical_data`
- `needs_clarification`

Required output fields for `properties_found`:

- `properties`
- `total_found`
- `shown_count`
- `query_context`
- `pagination.current_page`
- `pagination.page_size`
- `pagination.total_pages`
- `pagination.has_next`
- `pagination.has_prev`

### Invariants

- Search filtering is schema-driven and soft-coded from config and dataset schema.
- A fresh search requires deterministic city resolution before execution.
- Budget phrases normalize to `price_per_night` with operator `max`.
  Accepted examples include `under $300`, `budget $250`, `less than $400`.
- Guest-capacity phrases normalize to `occupancy_max` with operator `min`.
  Example: `for 4 guests` means `occupancy_max >= 4`.
- Known amenities are hard filters only when they exist in the configured dataset-backed amenity vocabulary.
- Unknown or vibe-style amenity terms must never become hard filters.
  They may be preserved as ranking or free-text terms.
- Comma-delimited and semicolon-delimited amenity input must normalize identically.
- Search execution must not broaden hard filters when zero exact matches are found.
  Example: a `villa` search may return `no_results`, but must not silently return apartments.
- Active booking blocks search and pagination shortcuts unless `booking_stage == "awaiting_property_reselection"`.
- Search must not intercept booking-status or cancellation-policy intents.

### Failure Behavior

- Missing city on a fresh search returns `missing_critical_data` or `needs_clarification`.
- Ambiguous city resolution returns a clarification response with supported suggestions.
- Zero exact matches return `no_results` with the exact hard filters echoed back.
- Invalid pagination requests clamp to valid page bounds and never corrupt stored page state.
- Search while booking is active returns no search mutation and leaves booking state authoritative.

### Acceptance Tests

- Search for `apartment in Seattle under $300` applies `price_per_night <= 300`.
- Search for `villa in Seattle budget $250` applies `price_per_night <= 250`.
- Search for `condo in Seattle less than $400` applies `price_per_night <= 400`.
- Search for `apartment in New York for 4 guests` only returns properties with `occupancy_max >= 4`.
- Search for `villa in Seattle with wifi, parking` and `villa in Seattle with wifi; parking` produce the same hard amenity filters.
- Search for `villa in Seattle with wifi, parking, cozy` hard-filters only `wifi` and `parking`; `cozy` remains non-blocking.
- An active booking session prevents `show more` from paginating search results unless the booking stage is `awaiting_property_reselection`.
- A booking-status or cancellation-policy message is not consumed by direct property search even if it contains city or property words.

## Property Selection Workflow

### Allowed States

- `property_list`
- `property_details`
- `selection_invalid`
- `blocked_by_active_booking`

### State Transitions

- `property_list -> property_details`
  When a valid option number resolves through the active option map.
- `property_list -> selection_invalid`
  When the requested selection is not present in the current option map.
- `property_details -> property_list`
  When the user rejects the property and requests alternatives.
- `property_list -> blocked_by_active_booking`
  When selection is attempted during active booking outside property reselection.

### Inputs

- `option_number: integer`
- Active state:
  `option_map`, `active_property_options_map`, `visible_results`, `all_search_results`

### Outputs

- `property_details`
- `missing_critical_data`
- `properties_found`

### Invariants

- The active `option_map` is the primary selection authority.
- `active_property_options_map` is a compatibility alias and must represent the same visible window.
- Selection must prefer exact property ID resolution over duplicate title or city matching.
- A selection number is scoped to the current visible page, not the historical full result set.

### Failure Behavior

- Invalid selections return a deterministic error or clarification and do not mutate selected property state.
- Selection during non-reselection booking stages is rejected without changing search state.

### Acceptance Tests

- Selecting option `2` resolves through `option_map["2"]` when present.
- Legacy `active_property_options_map` still works when `option_map` is absent.
- Duplicate property titles do not cause the wrong property to be selected.
- A selection request during `collecting_details` is blocked.

## Pagination And Refinement Workflow

### Allowed States

- `property_list`
- `needs_clarification`
- `blocked_by_active_booking`

### State Transitions

- `property_list -> property_list`
  For `show more`, `previous`, and search refinements.
- `property_list -> blocked_by_active_booking`
  For pagination or refinement attempted during active booking, except property reselection stage.

### Inputs

- Pagination shortcut input:
  `direction: "next" | "previous"`
- Refinement input:
  JSON-safe search inputs merged onto stored search filters

### Outputs

- `properties_found`
- `no_results`
- `missing_critical_data`

### Invariants

- `page_size`, `current_page`, `visible_results`, `option_map`, and `active_property_options_map` must describe the same page window.
- `shown_count == len(properties) == len(visible_results) == len(option_map)`.
- `active_property_options_shown_count == shown_count`.
- `active_property_options_total_found == total_found`.
- Refinements merge onto existing stored constraints in a schema-driven way.
- Hard constraints remain hard after refinement.
- Sort preferences persist across refinements until replaced.

### Failure Behavior

- `next` on the final page returns a deterministic "already shown" response and leaves page state unchanged.
- `previous` on page 1 clamps to page 1.
- Invalid or missing stored results return no pagination payload.

### Acceptance Tests

- Paginating from page 1 to page 2 updates `current_page`, `visible_results`, `option_map`, and `active_property_options_map` consistently.
- Refining `villa in Seattle` with `with pool`, then `cheaper`, then `3 bedrooms` preserves prior applicable filters and sort order.
- Replacing `villa` with `apartments only` swaps property type without dropping city, amenities, or persistent sort preferences.
