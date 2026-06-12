# Acceptance Tests

## Search And Selection

1. `apartment in Seattle under $300` applies `price_per_night <= 300`.
2. `villa in Seattle budget $250` applies `price_per_night <= 250`.
3. `condo in Seattle less than $400` applies `price_per_night <= 400`.
4. `apartment in New York for 4 guests` filters by `occupancy_max >= 4`.
5. `villa in Seattle with wifi, parking` and `villa in Seattle with wifi; parking` normalize to the same hard amenity filter set.
6. `villa in Seattle with wifi, parking, cozy` treats `cozy` as a soft ranking term, not a hard amenity filter.
7. Selection by option number resolves only from the active visible option map.
8. Duplicate titles do not cause the wrong property to be selected.

## Pagination And Refinement

9. Pagination keeps `page_size`, `current_page`, `visible_results`, `option_map`, and `active_property_options_map` consistent.
10. Refinements preserve prior hard filters unless explicitly replaced.
11. Replacing property type during refinement does not drop city, amenity, or sort state.

## Booking Priority And Workflow

12. An active booking session blocks search and pagination shortcuts unless `booking_stage == "awaiting_property_reselection"`.
13. Booking start is allowed only from selected-property context.
14. Invalid checkout preserves valid collected booking fields and asks only for checkout correction.
15. Booking confirmation is allowed only from review state and produces a deterministic receipt.

## Booking Status And Cancellation

16. Booking-status messages are handled deterministically before property search fallthrough.
17. Cancellation-policy questions route to FAQ/policy handling and preserve active booking state.
18. `Cancel my booking` is not intercepted by property search.

## ADK Tool Contract

19. Public ADK tool signatures expose only JSON-safe parameters.
20. Public ADK tool signatures do not expose `SearchPlan`, `DynamicConstraints`, or `ToolContext`.
