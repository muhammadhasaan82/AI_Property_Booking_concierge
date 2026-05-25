# Property Listing Flow Spec

## A. Filter Pipeline (Hard Rules — Must Not Be Violated)

1. **Normalize property_type** — resolve any user-supplied alias or plural
   to the canonical key using `specs/property_type_taxonomy.yaml`.
   - "apartments" → "apartment"
   - "flat"       → "apartment"
   - "town house" → "townhouse"
   - If alias not found and `unknown_type_policy=pass_through`, skip type filter.

2. **Apply all hard filters first** — before any semantic or vector ranking:
   - city: exact normalized match (trim + lowercase, no substring)
   - property_type: exact canonical match
   - beds: row.beds >= requested
   - budget: row.price_per_night <= budget
   - amenities: all requested amenities present (set subset)
   - Empty/null property_type in dataset row: excluded when type filter is active.

3. **Semantic re-ranking** — only on the already-filtered candidate set.
   Never run re-ranking across unfiltered data and then filter afterward.

4. **Paginate** — the filtered + ranked set.

5. **Show** — only filtered, ranked, paged results.

## B. Retrieval Policy (from specs/retrieval_policies.yaml)
- Single source of truth: `specs/retrieval_policies.yaml`.
- Hard cap max_results: 50 (env: PROPERTY_SEARCH_MAX_RESULTS).
- Page size: 5 (env: PROPERTY_LIST_PAGE_SIZE).
- No silent truncation.

## C. Listing Policy
- Every rendered item MUST have stable `number` (1-based, page-local) and
  `id` (property_id from dataset).
- Selection is resolved against `soft_state.active_property_options_map`.
- Voice agent renders exactly `shown_count` items.

## D. Conversation State After Listing
- `soft_state.last_search` (full payload incl. ALL filtered results, not paged)
- `soft_state.last_rendered_menu_type = "property_list"`
- `soft_state.current_page`
- `soft_state.page_size`
- `soft_state.active_property_options_map`
- `soft_state.previous_state` (snapshot before last transition)
- `soft_state.active_filters.property_type` = canonical key (not alias)

## E. What the Voice Agent Must Show
- The `property_type` field is in the payload — render it.
- If filtering was applied, confirm it once: "Here are apartments in New York:"
- Never invent or show types that differ from the canonical filter applied.

## F. Navigation (from prior spec — unchanged)
- Same back/next/previous/reset behavior as documented in prior version.