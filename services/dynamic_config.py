# services/dynamic_config.py
"""
Dynamic configuration loader — typed, validated, hot-reloadable.

All business constants (intent prototypes, phrase lists, routing policies,
guardrail patterns) are loaded from YAML files in the config/ directory.
Source code never needs editing for normal behavior tuning.

Usage:
    from services.dynamic_config import get_intent_catalog, get_vocabulary, ...

Set LEGACY_RULES=true in .env to fall back to hardcoded in-source values.
"""
from __future__ import annotations

import os
import re
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_INTENT_CATALOG_PATH = _CONFIG_DIR / "intent_catalog.yaml"
_VOCABULARY_PATH = _CONFIG_DIR / "vocabulary.yaml"
_GUARDRAILS_PATH = _CONFIG_DIR / "guardrails.yaml"
_ROUTING_POLICIES_PATH = _CONFIG_DIR / "routing_policies.yaml"
_THRESHOLDS_PATH = _CONFIG_DIR / "thresholds.yaml"
_RETRIEVAL_PATH = _CONFIG_DIR / "retrieval.yaml"

# ---------------------------------------------------------------------------
# Legacy mode flag
# ---------------------------------------------------------------------------
LEGACY_RULES: bool = os.getenv("LEGACY_RULES", "false").lower() in ("1", "true", "yes")


# ═══════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════

class IntentConfig(BaseModel):
    """Single intent entry with threshold and prototype sentences."""
    threshold: float = 0.55
    prototypes: List[str] = Field(default_factory=list)


class PreviousResultsConfig(BaseModel):
    """Configuration for previous-results detection."""
    threshold: float = 0.55
    prototypes: List[str] = Field(default_factory=list)
    fallback_keywords: List[str] = Field(default_factory=list)


class IntentCatalogConfig(BaseModel):
    """Full intent catalog loaded from intent_catalog.yaml."""
    version: str = "1.0"
    default_threshold: float = 0.55
    intents: Dict[str, IntentConfig] = Field(default_factory=dict)
    field_prototypes: Dict[str, List[str]] = Field(default_factory=dict)
    modification_prototypes: List[str] = Field(default_factory=list)
    property_search_request_prototypes: List[str] = Field(default_factory=list)
    receipt_request_prototypes: List[str] = Field(default_factory=list)
    resume_request_prototypes: List[str] = Field(default_factory=list)
    affirm_yes_prototypes: List[str] = Field(default_factory=list)
    affirm_no_prototypes: List[str] = Field(default_factory=list)
    previous_results_prototypes: PreviousResultsConfig = Field(
        default_factory=PreviousResultsConfig
    )
    keyword_fallback_map: Dict[str, List[str]] = Field(default_factory=dict)
    vader_fallback: Dict[str, List[str]] = Field(default_factory=dict)


class SlotFillingConfig(BaseModel):
    """Slot-filling schema."""
    required_fields: List[str] = Field(default_factory=list)
    field_prompts: Dict[str, str] = Field(default_factory=dict)
    modification_prompts: Dict[str, str] = Field(default_factory=dict)


class NlpFallbackConfig(BaseModel):
    """Fallback lexical configuration for NLP helpers."""
    greeting_seeds: List[str] = Field(default_factory=list)
    greeting_phrases: List[str] = Field(default_factory=list)
    identity_phrases: List[str] = Field(default_factory=list)
    acknowledgment_tokens: List[str] = Field(default_factory=list)
    acknowledgment_phrases: List[str] = Field(default_factory=list)
    affirm_yes_tokens: List[str] = Field(default_factory=list)
    affirm_no_tokens: List[str] = Field(default_factory=list)
    handoff_seeds: List[str] = Field(default_factory=list)
    handoff_phrases: List[str] = Field(default_factory=list)
    availability_phrases: List[str] = Field(default_factory=list)
    end_exact: List[str] = Field(default_factory=list)
    end_phrases: List[str] = Field(default_factory=list)
    status_seeds: List[str] = Field(default_factory=list)
    status_explicit_keywords: List[str] = Field(default_factory=list)
    status_resume_phrases: List[str] = Field(default_factory=list)
    status_query_actions: List[str] = Field(default_factory=list)
    status_check_in_actions: List[str] = Field(default_factory=list)
    status_check_out_actions: List[str] = Field(default_factory=list)
    status_booking_id_markers: List[str] = Field(default_factory=list)
    search_signals: List[str] = Field(default_factory=list)
    search_phrases: List[str] = Field(default_factory=list)
    money_intent_pattern: str = ""
    modification_seeds: List[str] = Field(default_factory=list)
    property_search_request_seeds: List[str] = Field(default_factory=list)
    receipt_seeds: List[str] = Field(default_factory=list)
    receipt_phrases: List[str] = Field(default_factory=list)
    receipt_quantity_terms: List[str] = Field(default_factory=list)
    receipt_amount_terms: List[str] = Field(default_factory=list)
    resume_exact_phrases: List[str] = Field(default_factory=list)
    resume_phrases: List[str] = Field(default_factory=list)
    faq_strong_keywords: List[str] = Field(default_factory=list)
    faq_seeds: List[str] = Field(default_factory=list)
    faq_question_starts: List[str] = Field(default_factory=list)
    faq_question_cues: List[str] = Field(default_factory=list)
    selection_faq_blocklist: List[str] = Field(default_factory=list)
    name_search_guards: List[str] = Field(default_factory=list)
    name_conversational_guards: List[str] = Field(default_factory=list)
    name_explicit_patterns: List[str] = Field(default_factory=list)
    email_username_common: List[str] = Field(default_factory=list)
    parse_name_reject_exact: List[str] = Field(default_factory=list)
    parse_name_invalid_chars: List[str] = Field(default_factory=list)
    parse_name_disallowed_words: List[str] = Field(default_factory=list)
    parse_name_disallowed_phrases: List[str] = Field(default_factory=list)
    phrase_fillers: List[str] = Field(default_factory=list)
    selection_patterns: List[str] = Field(default_factory=list)
    selection_context_patterns: List[str] = Field(default_factory=list)
    selection_ordinal_templates: List[str] = Field(default_factory=list)
    selection_referential_patterns: List[str] = Field(default_factory=list)
    selection_cardinal_context_pattern: str = ""
    selection_entity_labels: List[str] = Field(default_factory=list)
    selection_ordinals: Dict[str, int] = Field(default_factory=dict)
    selection_cardinals: Dict[str, int] = Field(default_factory=dict)
    guest_unit_terms: List[str] = Field(default_factory=list)
    guest_context_terms: List[str] = Field(default_factory=list)
    unavailable_city_decline_phrases: List[str] = Field(default_factory=list)
    property_type_any_phrases: List[str] = Field(default_factory=list)
    city_candidate_block_words: List[str] = Field(default_factory=list)
    city_candidate_prefix_pattern: str = ""
    city_candidate_split_pattern: str = ""


class VocabularyConfig(BaseModel):
    """Vocabulary config loaded from vocabulary.yaml."""
    version: str = "1.0"
    slot_filling: SlotFillingConfig = Field(default_factory=SlotFillingConfig)
    proceed_phrases: List[str] = Field(default_factory=list)
    modify_phrases: List[str] = Field(default_factory=list)
    faq_fallback_keywords: List[str] = Field(default_factory=list)
    booking_triggers: List[str] = Field(default_factory=list)
    status_keywords: List[str] = Field(default_factory=list)
    payment_keywords: List[str] = Field(default_factory=list)
    fallback_cities: List[str] = Field(default_factory=list)
    city_aliases: Dict[str, str] = Field(default_factory=dict)
    seed_property_types: List[str] = Field(default_factory=list)
    amenity_synonyms: Dict[str, List[str]] = Field(default_factory=dict)
    nlp_fallback: NlpFallbackConfig = Field(default_factory=NlpFallbackConfig)

    # Convenience properties
    @property
    def fallback_cities_set(self) -> Set[str]:
        return set(self.fallback_cities)

    @property
    def seed_property_types_set(self) -> Set[str]:
        return set(self.seed_property_types)


class GuardrailPatternConfig(BaseModel):
    """A single guardrail regex pattern."""
    pattern: str
    severity: str = "medium"
    action: str = "block"
    description: str = ""


class GuardrailsConfig(BaseModel):
    """Guardrails config loaded from guardrails.yaml."""
    version: str = "1.0"
    max_input_length: int = 2000
    injection_patterns: List[GuardrailPatternConfig] = Field(default_factory=list)
    script_patterns: List[GuardrailPatternConfig] = Field(default_factory=list)
    leak_patterns: List[GuardrailPatternConfig] = Field(default_factory=list)


class RoutingCondition(BaseModel):
    """Condition for a routing policy."""
    intent: Optional[str] = None
    intent_in: Optional[List[str]] = None
    filter_key: Optional[str] = None
    any_filter_key: Optional[List[str]] = None
    has_context_key: Optional[str] = None
    lacks_context_key: Optional[str] = None
    has_booking_context: Optional[bool] = None
    has_field_data: Optional[bool] = None
    awaiting_field_in: Optional[List[str]] = None
    requires_cardinal_extraction: Optional[bool] = None
    lacks_explicit_status_keywords: Optional[bool] = None
    no_selected_property: Optional[bool] = None
    no_awaiting_field: Optional[bool] = None
    no_active_selection: Optional[bool] = None
    is_safe: Optional[bool] = None
    always: Optional[bool] = None


class RoutingPolicy(BaseModel):
    """A single routing policy rule."""
    id: str
    description: str = ""
    priority: int = 0
    condition: RoutingCondition = Field(default_factory=RoutingCondition)
    route: str
    reply: Optional[str] = None
    extract: Optional[str] = None
    note: Optional[str] = None


class RoutingPoliciesConfig(BaseModel):
    """Routing policies loaded from routing_policies.yaml."""
    version: str = "1.0"
    policies: List[RoutingPolicy] = Field(default_factory=list)

    @property
    def sorted_policies(self) -> List[RoutingPolicy]:
        """Return policies sorted by priority (highest first)."""
        return sorted(self.policies, key=lambda p: p.priority, reverse=True)


class NlpThresholdsConfig(BaseModel):
    """NLP fuzzy-match thresholds."""
    fuzzy_match_strict: float = 0.93
    fuzzy_match_high: float = 0.90
    fuzzy_match_medium: float = 0.88
    fuzzy_match_low: float = 0.78


class FaqThresholdsConfig(BaseModel):
    """FAQ confidence thresholds."""
    high_confidence: float = 0.7
    low_confidence: float = 0.4


class RagThresholdsConfig(BaseModel):
    """RAG pipeline hyperparameters."""
    rrf_constant: int = 60
    vector_k: int = 6
    bm25_k: int = 6
    max_context_chars: int = 1500
    grounding_threshold: float = 0.4


class ThresholdsConfig(BaseModel):
    """NLP thresholds loaded from thresholds.yaml."""
    nlp: NlpThresholdsConfig = Field(default_factory=NlpThresholdsConfig)
    faq: FaqThresholdsConfig = Field(default_factory=FaqThresholdsConfig)


class RetrievalScoringWeightsConfig(BaseModel):
    """Heuristic fallback scoring weights for retrieval.py JSON mode."""
    exact_doc: float = 1.0
    exact_title: float = 2.0
    token_doc: float = 0.5
    token_title: float = 1.0


class RetrievalRuntimeConfig(BaseModel):
    """Retrieval runtime settings."""
    top_k: int = 10
    result_limit: int = 5
    scoring_weights: RetrievalScoringWeightsConfig = Field(default_factory=RetrievalScoringWeightsConfig)


class RetrievalConfig(BaseModel):
    """RAG + retrieval settings loaded from retrieval.yaml."""
    rag: RagThresholdsConfig = Field(default_factory=RagThresholdsConfig)
    retrieval: RetrievalRuntimeConfig = Field(default_factory=RetrievalRuntimeConfig)


# ═══════════════════════════════════════════════════════════════════════
# Cache + Thread-safe loading
# ═══════════════════════════════════════════════════════════════════════

_lock = threading.Lock()
_cache: Dict[str, Any] = {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file. Returns empty dict on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        logger.warning(f"[dynamic_config] Config file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"[dynamic_config] YAML parse error in {path}: {e}")
        return {}


def _get_or_load(key: str, path: Path, model_cls: type) -> Any:
    """Load config from cache, or read from disk and validate."""
    if key in _cache:
        return _cache[key]
    with _lock:
        if key in _cache:
            return _cache[key]
        raw = _load_yaml(path)
        try:
            obj = model_cls(**raw) if raw else model_cls()
            _cache[key] = obj
            logger.info(f"[dynamic_config] Loaded {key} (version={getattr(obj, 'version', '?')})")
            return obj
        except Exception as e:
            logger.error(f"[dynamic_config] Validation error for {key}: {e}")
            obj = model_cls()
            _cache[key] = obj
            return obj


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def get_intent_catalog() -> IntentCatalogConfig:
    """Get the intent catalog configuration."""
    return _get_or_load("intent_catalog", _INTENT_CATALOG_PATH, IntentCatalogConfig)


def get_vocabulary() -> VocabularyConfig:
    """Get the vocabulary configuration."""
    return _get_or_load("vocabulary", _VOCABULARY_PATH, VocabularyConfig)


def get_guardrails() -> GuardrailsConfig:
    """Get the guardrails configuration."""
    return _get_or_load("guardrails", _GUARDRAILS_PATH, GuardrailsConfig)


def get_routing_policies() -> RoutingPoliciesConfig:
    """Get the routing policies configuration."""
    return _get_or_load("routing_policies", _ROUTING_POLICIES_PATH, RoutingPoliciesConfig)


def get_thresholds() -> ThresholdsConfig:
    """Get NLP thresholds configuration."""
    return _get_or_load("thresholds", _THRESHOLDS_PATH, ThresholdsConfig)


def get_retrieval_config() -> RetrievalConfig:
    """Get RAG/retrieval runtime configuration."""
    return _get_or_load("retrieval", _RETRIEVAL_PATH, RetrievalConfig)


def reload_all() -> None:
    """Clear cache and reload all config files from disk."""
    with _lock:
        _cache.clear()
    clear_compiled_guardrails()
    # Touch each to force re-load
    get_intent_catalog()
    get_vocabulary()
    get_guardrails()
    get_routing_policies()
    get_thresholds()
    get_retrieval_config()
    logger.info("[dynamic_config] All configuration reloaded")


def load_all() -> None:
    """Load all configs at startup (convenience for validation check)."""
    reload_all()


def get_config_summary() -> Dict[str, Any]:
    """Return a summary of loaded configuration for debug endpoints."""
    ic = get_intent_catalog()
    vc = get_vocabulary()
    gc = get_guardrails()
    rp = get_routing_policies()
    rt = get_retrieval_config()
    return {
        "intent_catalog": {
            "version": ic.version,
            "intent_count": len(ic.intents),
            "intents": list(ic.intents.keys()),
            "field_prototype_count": len(ic.field_prototypes),
        },
        "vocabulary": {
            "version": vc.version,
            "proceed_phrases_count": len(vc.proceed_phrases),
            "modify_phrases_count": len(vc.modify_phrases),
            "fallback_cities_count": len(vc.fallback_cities),
            "property_types_count": len(vc.seed_property_types),
            "amenity_count": len(vc.amenity_synonyms),
        },
        "guardrails": {
            "version": gc.version,
            "injection_patterns_count": len(gc.injection_patterns),
            "script_patterns_count": len(gc.script_patterns),
            "leak_patterns_count": len(gc.leak_patterns),
            "max_input_length": gc.max_input_length,
        },
        "routing_policies": {
            "version": rp.version,
            "policy_count": len(rp.policies),
            "policy_ids": [p.id for p in rp.sorted_policies],
        },
        "retrieval": {
            "rag": rt.rag.model_dump(),
            "retrieval": rt.retrieval.model_dump(),
        },
        "legacy_rules": LEGACY_RULES,
    }


# ═══════════════════════════════════════════════════════════════════════
# Compiled guardrail regex cache
# ═══════════════════════════════════════════════════════════════════════

_compiled_guardrails: Dict[str, list] = {}


def get_compiled_injection_patterns() -> list:
    """Return compiled injection regex patterns from config."""
    if "injection" not in _compiled_guardrails:
        gc = get_guardrails()
        _compiled_guardrails["injection"] = [
            re.compile(p.pattern, re.I) for p in gc.injection_patterns
        ]
    return _compiled_guardrails["injection"]


def get_compiled_script_patterns() -> list:
    """Return compiled script injection regex patterns from config."""
    if "script" not in _compiled_guardrails:
        gc = get_guardrails()
        _compiled_guardrails["script"] = [
            re.compile(p.pattern, re.I) for p in gc.script_patterns
        ]
    return _compiled_guardrails["script"]


def get_compiled_leak_patterns() -> list:
    """Return compiled output leak regex patterns from config."""
    if "leak" not in _compiled_guardrails:
        gc = get_guardrails()
        _compiled_guardrails["leak"] = [
            re.compile(p.pattern, re.I) for p in gc.leak_patterns
        ]
    return _compiled_guardrails["leak"]


def clear_compiled_guardrails() -> None:
    """Clear compiled regex cache (e.g. after reload)."""
    _compiled_guardrails.clear()
