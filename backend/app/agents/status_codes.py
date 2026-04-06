"""
app/agents/status_codes.py
--------------------------
Single source of truth for all status string constants used in tool payloads.

The triage_router returns structured dicts with a "status" key.
The concierge_voice reads that key to decide how to respond.
Using typed constants here prevents silent routing breaks from typos.
"""


class Status:
    # General
    MISSING_CRITICAL_DATA = "missing_critical_data"
    ERROR = "error"

    # City / Search
    CITIES_FOUND = "cities_found"
    PROPERTIES_FOUND = "properties_found"
    NO_RESULTS = "no_results"

    # Property details / selection
    PROPERTY_DETAILS = "property_details"
    PROPERTY_SELECTION_UNRESOLVED = "property_selection_unresolved"
    NOT_FOUND = "not_found"

    # Small talk
    CASUAL_INTERACTION = "casual_interaction"

    # FAQ
    ANSWERED = "answered"
    FAQ_NOT_FOUND = "faq_not_found"

    # Booking flow
    GATHERING_INFO = "gathering_info"
    REVIEW_PENDING = "review_pending"
    BOOKING_CONFIRMED = "booking_confirmed"

    # Booking status lookup
    FOUND = "found"
    BOOKING_NOT_FOUND = "booking_not_found"

    # Escalation
    HANDOFF_REQUIRED = "handoff_required"


class Source:
    """Source identifiers for tool payload provenance tracking."""
    MEMORY = "memory"
    POLICY_DB = "policy_database"
    RAG = "rag_pipeline"
    BASIC_FAQ = "basic_faq"
    V2_ADK = "v2_adk"


# Valid engagement state labels produced by _classify_user_engagement_state
ENGAGEMENT_STATES: frozenset[str] = frozenset({
    "engaged",
    "fatigued",
    "exhausted_or_frustrated",
})

# Valid intent classes produced by _resolve_property_reference_with_model
INTENT_CLASSES: frozenset[str] = frozenset({
    "select_property",
    "general_inquiry",
    "modify_search",
    "confirm_booking",
    "escalate",
})

# Valid message types for handle_small_talk
SMALL_TALK_TYPES: frozenset[str] = frozenset({
    "greeting",
    "thanks",
    "goodbye",
    "acknowledgement",
})

# Required booking fields (in validation order)
BOOKING_REQUIRED_FIELDS: tuple[tuple[str, str], ...] = (
    ("property_id", "property_id"),
    ("property_title", "property_title"),
    ("guest_name", "guest_name"),
    ("guest_email", "guest_email"),
    ("guest_phone", "guest_phone"),
    ("check_in", "check_in"),
    ("check_out", "check_out"),
)
