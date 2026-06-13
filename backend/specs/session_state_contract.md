# Session Soft State Contract

## Scope

This document defines the persisted `soft_state` contract used for deterministic backend workflow enforcement.

## Allowed State Shapes

`soft_state` must be a JSON-safe object. Non-dict values normalize to `{}`.

### Search Keys

- `active_flow`
- `last_presented_view`
- `last_filters`
- `last_search_filters`
- `last_dynamic_constraints`
- `last_sort_preferences`
- `last_search`
- `last_search_at`
- `all_search_results`
- `visible_results`
- `last_visible_results`
- `option_map`
- `active_property_options_map`
- `active_property_options_shown_count`
- `active_property_options_total_found`
- `active_property_options_generated_at`
- `current_page`
- `page_size`
- `last_selected_property_id`
- `selected_property_id`
- `selected_property`
- `last_rejected_property_id`

### Booking Keys

- `booking_stage`
- `booking_property_id`
- `booking_selected_property`
- `booking_required_fields`
- `booking_state`
- `booking_state_updated_at`
- `awaiting_field`
- `booking_review`
- `pending_booking`
- `pending_booking_updated_at`
- `booking_receipt`
- `booking_receipts`
- `booking_registration_id`
- `booking_status`

### Interruption And Meta Keys

- `faq_interruption`
- `unresolved_turns`
- `last_unsupported_region`

## State Transitions

- Search writes or replaces active shortlist keys atomically for the visible page.
- Pagination updates page window keys without changing stored full results.
- Property selection updates selection keys without mutating search filters.
- Booking start seeds booking keys while preserving search context.
- Booking confirmation clears transient pending booking state but preserves receipt state.
- FAQ interruption captures a resumable snapshot without invalidating the active workflow.

## Inputs

- Persisted runner state
- Tool output payloads
- Deterministic workflow handlers

## Outputs

- JSON-safe persisted session snapshot
- Compatibility aliases for legacy readers where still required

## Invariants

- `soft_state` must always be persisted as a dictionary.
- Persisted values must remain JSON-serializable.
- `option_map` and `active_property_options_map` must describe the same visible options.
- `current_page`, `page_size`, and `visible_results` must correspond to the same page window.
- `last_filters` is the canonical search refinement memory.
- `last_search_filters` is a compatibility alias and must mirror `last_filters`.
- `selected_property_id` and `last_selected_property_id` must be consistent after selection.
- `booking_stage` is the canonical booking workflow state.
- `awaiting_field` must be either `null`/missing or one currently missing or being modified.
- Workflow enforcement reads only soft state and deterministic tool payloads, not model-only hidden memory.

## Failure Behavior

- Invalid persisted state is normalized rather than raising.
- Missing compatibility aliases are rebuilt from canonical keys when possible.
- Empty or stale search state prevents pagination and selection shortcuts from firing.
- Unknown extra keys are tolerated but ignored by deterministic workflow guards.

## Acceptance Tests

- Persisting state always yields a top-level `soft_state` dict.
- Search pagination keeps `current_page`, `page_size`, `visible_results`, `option_map`, and `active_property_options_map` synchronized.
- Booking start preserves search keys while adding booking keys.
- FAQ interruption preserves enough state to resume the interrupted workflow.
