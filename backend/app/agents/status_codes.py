"""
app/agents/status_codes.py
--------------------------
All status strings, source tags, and valid-value sets are loaded from
app/config/agent_config.yaml via the cfg singleton.

NO VALUE IN THIS FILE IS HARDCODED.
To change any string, threshold, or set — edit agent_config.yaml only.
"""
from app.config.agent_config_loader import cfg


class Status:
    MISSING_CRITICAL_DATA      = cfg.status.missing_critical_data
    ERROR                      = cfg.status.error
    CITIES_FOUND               = cfg.status.cities_found
    PROPERTIES_FOUND           = cfg.status.properties_found
    NO_RESULTS                 = cfg.status.no_results
    PROPERTY_DETAILS           = cfg.status.property_details
    PROPERTY_SELECTION_UNRESOLVED = cfg.status.property_selection_unresolved
    NOT_FOUND                  = cfg.status.not_found
    CASUAL_INTERACTION         = cfg.status.casual_interaction
    ANSWERED                   = cfg.status.answered
    FAQ_NOT_FOUND              = cfg.status.faq_not_found
    GATHERING_INFO             = cfg.status.gathering_info
    REVIEW_PENDING             = cfg.status.review_pending
    AMENDMENT_ACKNOWLEDGED     = cfg.status.amendment_acknowledged
    BOOKING_CONFIRMED          = cfg.status.booking_confirmed
    FOUND                      = cfg.status.found
    BOOKING_NOT_FOUND          = cfg.status.booking_not_found
    HANDOFF_REQUIRED           = cfg.status.handoff_required

class Source:
    MEMORY     = cfg.source.memory
    POLICY_DB  = cfg.source.policy_db
    RAG        = cfg.source.rag
    BASIC_FAQ  = cfg.source.basic_faq
    V2_ADK     = cfg.source.v2_adk

ENGAGEMENT_STATES: frozenset = cfg.engagement_valid_states
INTENT_CLASSES: frozenset    = cfg.resolution_valid_intents
SMALL_TALK_TYPES: frozenset  = cfg.small_talk_valid_types

BOOKING_REQUIRED_FIELDS: tuple = tuple(
    (field, field) for field in cfg.booking_required_fields
)
