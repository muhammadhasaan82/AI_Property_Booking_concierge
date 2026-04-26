"""
Loads agent_policy.yaml and exposes typed accessors for the Phase 3 policy
router. Pure data layer — no LLM calls, no business logic.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_POLICY_PATH = Path(__file__).parent / "agent_policy.yaml"

class CofidenceConfig(BaseModel):
    high: float = 0.80
    medium: float = 0.55
    low: float = 0.30

class PriorityCondition(BaseModel):
    primary_intent_in: List[str] = Field(default_factory=list)
    has_active_shortlist: Optioanl[bool] = None
    has_pending_booking: Optional[bool] = None
    awaiting_field_present: Optional[bool] = None
    selection_number_present: Optional[bool] = None
    explicit_status_present: Optional[bool] = None
    faq_question_form: Optional[bool] = None

class PriorityRule(BaseModel):
    id: str
    when: PriorityCondition = Field(default_factory=PriorityCondition)
    prefer_intent: str
    priority: int = 0

class IntentRule(BaseModel):
    allowed_tools: List[str] = Field(default_factory=list)
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(defualt_factory=list)
    requires_context: List[str] = Field(default_factory=list)
    source_priority: List[str] = Field(default_factory=list)
    on_missing: Optional[str] = None
    on_missing_context: Optional[str] = None
    clarify_message: Optional[str] = None
    fallback_intent: Optional[str] = None
    schema_ref: Optional[str] = None
    response_policy: Optional[str] = None
    requires_explicit_user_authorization: bool = None
    block_other_intents: bool = False

class _AgentPolicyRoot(BaseModel):
    version: str = "1.0"
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    priorities: List[PriorityRule] = Field(default_factory=list)
    intents: Dict[str, IntentRule] = Field(default_factory=dict)

def _load() -> _AgentPolicyRoot:
    if not _POLICY_PATH.exists():
        logger.warning("[agent_policy] %s missing, using defaults", _POLICY_PATH)
        return _AgentPolicyRoot()
    with open(_POLICY_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _AgentPolicyRoot(**data)

policy: _AgentPolicyRoot = _load()

def is_high_confidence(score: float) -> bool:
    return score >= policy.confidence.high

def is_medium_confidence(score: float) -> bool:
    return score >= policy.confidence.medium

def get_intent(name: str) -> Optional[IntentRule]:
    return policy.intents.get(name)

def sorted_priorities() -> List[PriorityRule]:
    return sorted(policy.priorities, key=lambda r: r.priority, reverse=True)

def reload() -> None:
    global policy
    policy = _load()
    logger.info("[agent_policy] reloaded version=%s, %d intents, %d priorities", policy.version, len(policy.intents), len(policy.priorities))


