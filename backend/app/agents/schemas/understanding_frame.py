"""
Structured understanding frame emitted by understanding_agent (Phase 2).
 
This is the single typed surface the LLM uses to express what it thinks the
user means. The frame is consumed by:
  - triage_router  : as additional context for tool selection
  - concierge_voice: to align response style with detected mood / clarification
  - policy_router  : (Phase 3) deterministic routing on top of the frame
  - telemetry      : DPO training signal (was the frame correct?)
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

ALLOWED_PRIMARY_INTENTS = (
    "search_property",
    "select_property",
    "faq",
    "booking_continuation",
    "booking_confirmation",
    "booking_status",
    "small_talk",
    "human_handoff",
    "city_list",
    "unclear",
)

ALLOWED_USER_MOODS = ("neutral", "engaged", "fatigued", "frustrated", "confused")

class UnderstandingFrame(BaseModel):
    """LLM-emitted structured understanding of a single user turn.

    Output contract- the understanding_agent MUST emit JSON conforming to this model. No prose, no code fences. ADK enforces the schema via output_schema.
    """

    primary_intent: str = Field(
        description=(
            "One of: search_property, select_property, property_details_request,"
            "faq, booking_continuation, booking_confirmation, booking_status,"
            "small_talk, human_handoff, city_list, unclear"
        )
    )
    secondary_intents: List[str] = Field(default_factory=list)

    confidence: float = Field(
        
        ge=0.0,
        le=1.0,
        description="Confidence in primary_intent, 0..1 (calibrate honestly).",
    )

    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extracted slots: city, budget, beds, property_type, anemities,"
            "check_in, check_out, guests, booking_id, etc. Only include"
            "fields you are confident about."
        ),
    )

    reference_previous_results: bool = Field(
        default=False,
        description="True if the user is referring to a prior shortlist.",
    )
    selection_number: Optional[int] = Field(
        default=None,
        description="If user picked an option by number/ordinal, that number.",
    )
    is_booking_continuation: bool = Field(
        default=False,
        description="True if continuing/modifying an active booking flow.",
    )
    user_mood: str = Field(
        default="neutral",
        description="One of: neutral, engaged, fatigued, frustrated",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if confidence is too low to act safely.",
    )
    clarification_field: Optional[str] = Field(
        default=None,
        description="If needs_clarification, the missing piece (e.g 'city')",
    )
    rationale: str = Field(
        default="",
        description="One short sentence explaining the classification.",
    )

    def is_high_confidence(self, threshold: float = 0.80) -> bool:
        return self.confidence >= threshold

    def is_actionable(self, medium_threshold: float = 0.55) -> bool:
        return (
            self.confidence >= medium_threshold
            and not self.needs_clarification
            and self.primary_intent != "unclear"
        )

    def to_compact_json(self) -> str:
        """Serialize to single-line JSON suitable for prompt injection."""
        import json
        return json.dump(self.model_dump(mode='json'), ensure_ascii=False)
