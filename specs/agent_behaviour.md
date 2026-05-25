# Agent Behaviour

## Agent Pipeline

The ADK pipeline is defined in `backend/app/agents/adk_agents.py` and executed by `backend/app/services/adk_runner.py`.

Current intended sequence:

1. `understanding_agent` emits `UnderstandingFrame`.
2. `triage_router` selects and calls exactly the relevant tool.
3. `concierge_voice` converts tool output into user-facing text.

If `UNDERSTANDING_FRAME_ENABLED=0`, the pipeline may run without the understanding node.

## Understanding Frame Contract

`backend/app/agents/schemas/understanding_frame.py` owns the typed agent-understanding payload.

Allowed primary intents:

- `search_property`
- `select_property`
- `property_details_request`
- `faq`
- `booking_continuation`
- `booking_confirmation`
- `booking_status`
- `small_talk`
- `human_handoff`
- `city_list`
- `unclear`

Unknown primary intents should normalize to `unclear`. Unknown moods should normalize to `neutral`.

## Tool Registry Contract

`backend/app/config/tool_registry.yaml` is the source of truth for tool metadata. `backend/app/config/tool_registry_loader.py` must preserve:

- tool name
- module/function import target
- intent list
- required/optional input metadata
- context requirements
- response policy
- explicit authorization requirements

Adding or removing a tool should not require editing `adk_agents.py` unless the Python implementation itself changes.

## Off-Switch Booking Rule

The agent must not call `process_v2_booking` until all booking fields are present and the user has explicitly confirmed the reviewed booking. Missing data must route to `request_booking_details` or `review_booking_details`.

## Fallback Behaviour

- Unsafe input returns a safe refusal/redirect.
- Empty input returns a clarification prompt.
- Tool-loop anomalies return the configured anomaly fallback.
- Provider timeouts return the configured pipeline timeout fallback.
- External lookup failures should use lower-cost/local fallbacks where available.