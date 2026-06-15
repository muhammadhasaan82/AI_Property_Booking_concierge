from __future__ import annotations
"""Loads booking flow routing phrases and patterns from booking_schema.yaml."""

import logging
import re
from functools import lru_cache
from typing import Dict, FrozenSet, List, Pattern, Tuple

from app.config.booking_schema_loader import booking_schema

logger = logging.getLogger(__name__)

_DEFAULT_ACTIVE_STAGES = frozenset({
    "collecting_details",
    "awaiting_confirmation",
    "awaiting_modification_choice",
    "modifying_details",
    "awaiting_property_reselection",
    "confirmed",
    "awaiting_amendment_choice",
    "awaiting_amendment_values",
    "awaiting_amendment_confirmation",
})

_DEFAULT_POST_CONFIRMATION_STAGES = frozenset({
    "confirmed",
    "awaiting_amendment_choice",
    "awaiting_amendment_values",
    "awaiting_amendment_confirmation",
})

_DEFAULT_FAQ_KEYWORDS = (
    "pet", "pets", "parking", "cancellation", "refund", "policy", "policies",
    "check-in time", "check in time", "check-out time", "check out time",
    "smoking", "wifi", "amenities", "payment", "accessibility", "allowed",
)

_DEFAULT_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

_DEFAULT_RECEIPT_COLUMNS = {
    "guest_name": "user_name",
    "guest_email": "user_email",
    "guest_phone": "user_phone",
    "check_in": "check_in",
    "check_out": "check_out",
    "guests": "guests",
    "nights": "nights",
    "price_per_night": "price_per_night",
    "total_amount": "total_amount",
    "property_title": "property_title",
    "city": "city",
    "status": "status",
}

_DEFAULT_STATUS_PATTERNS = [
    r"\bmy\s+booking\s+id\s+is\b",
    r"\bbooking\s+id\b",
    r"\bcheck\s+my\s+booking\s+status\b",
    r"\bcheck\s+booking\s+status\b",
    r"\bmy\s+booking\s+status\b",
    r"\bregistration\s+id\b",
    r"\bmy\s+registration\s+id\s+is\b",
    r"\bi\s+want\s+to\s+check\s+my\s+booking\s+status\b",
    r"\bstatus\s+of\s+my\s+booking\b",
    r"\bbooking\s+status\b",
    r"\bmy\s+boking\s+id\s+is\b",
    r"\bboking\s+id\b",
    r"\bcheck\s+my\s+boking\s+status\b",
    r"\bcheck\s+boking\s+status\b",
    r"\bmy\s+boking\s+status\b",
    r"\bi\s+want\s+to\s+check\s+my\s+boking\s+status\b",
    r"\bstatus\s+of\s+my\s+boking\b",
    r"\bboking\s+status\b",
]

_DEFAULT_AMENDMENT_NAME_PATTERNS = [
    r"\b(?:change|update|modify|set|edit)\s+(?:my\s+)?(?:full\s+)?name\s+(?:to|as)\s+"
    r"([a-z][a-z .'-]+?)(?=\s+\band\b|\s+for\b|\s+in\b|[,.;]|$)",
    r"\b(?:change|update|modify|set|edit)\s+full\s+name\s+(?:to|as)\s+"
    r"([a-z][a-z .'-]+?)(?=\s+\band\b|\s+for\b|\s+in\b|[,.;]|$)",
    r"\b(?:full\s+)?name\s+(?:is|as)\s+"
    r"([a-z][a-z .'-]+?)(?=\s+\band\b|\s+for\b|\s+in\b|[,.;]|$)",
    r"\b(?:full\s+)?name\s*:\s*"
    r"([a-z][a-z .'-]+?)(?=\s+\band\b|\s+for\b|\s+in\b|[,.;]|$)",
    r"\bset\s+full\s+name\s+as\s+"
    r"([a-z][a-z .'-]+?)(?=\s+\band\b|\s+for\b|\s+in\b|[,.;]|$)",
    r"\b(?:my\s+)?(?:full\s+)?name\s+is\s+"
    r"([a-z][a-z .'-]+?)(?=\s+\band\b|\s+for\b|\s+in\b|[,.;]|$)",
]

_DEFAULT_CANCELLATION_PATTERNS = [
    r"\b(delete|cancel|remove)\s+(?:this\s+|my\s+|the\s+)?booking\b",
    r"\b(delete|cancel|remove)\s+(?:this\s+|my\s+|the\s+)?registration\b",
]

_DEFAULT_NEGATED_CANCELLATION = [
    r"\b(?:do\s+not|don'?t|dont)\s+(?:cancel|delete|remove)\b",
    r"\bkeep\s+it\b",
    r"\bnever\s*mind\b",
    r"\bnevermind\b",
    r"\bchange\s+my\s+mind\b",
]

_DEFAULT_YES = frozenset({"yes", "yeah", "sure", "confirm", "proceed", "correct", "ok", "okay"})
_DEFAULT_NO = frozenset({"no", "nope", "nah", "reject"})


def _flow_block() -> dict:
    raw = getattr(booking_schema.booking, "flow", None)
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=1)
def get_active_booking_stages() -> FrozenSet[str]:
    stages = _flow_block().get("active_stages") or list(_DEFAULT_ACTIVE_STAGES)
    return frozenset(str(s).strip() for s in stages if str(s).strip())


@lru_cache(maxsize=1)
def get_post_confirmation_amendment_stages() -> FrozenSet[str]:
    stages = _flow_block().get("post_confirmation_amendment_stages") or list(_DEFAULT_POST_CONFIRMATION_STAGES)
    return frozenset(str(s).strip() for s in stages if str(s).strip())


@lru_cache(maxsize=1)
def get_faq_keywords() -> Tuple[str, ...]:
    keywords = _flow_block().get("faq_keywords") or list(_DEFAULT_FAQ_KEYWORDS)
    return tuple(str(k).strip().lower() for k in keywords if str(k).strip())


@lru_cache(maxsize=1)
def get_check_in_date_pattern() -> str:
    return str(
        _flow_block().get("check_in_date_pattern")
        or r"\b(check[- ]?in(?: date)?|arrival(?: date)?)\b"
    )


@lru_cache(maxsize=1)
def get_check_out_date_pattern() -> str:
    return str(
        _flow_block().get("check_out_date_pattern")
        or r"\b(check[- ]?out(?: date)?|checkout(?: date)?|departure(?: date)?)\b"
    )


@lru_cache(maxsize=1)
def get_month_name_map() -> Dict[str, int]:
    raw = _flow_block().get("months") or _DEFAULT_MONTHS
    if not isinstance(raw, dict):
        return dict(_DEFAULT_MONTHS)
    return {str(k).lower(): int(v) for k, v in raw.items()}


@lru_cache(maxsize=1)
def get_receipt_column_map() -> Dict[str, str]:
    raw = _flow_block().get("receipt_to_db_columns") or _DEFAULT_RECEIPT_COLUMNS
    return dict(raw) if isinstance(raw, dict) else dict(_DEFAULT_RECEIPT_COLUMNS)


@lru_cache(maxsize=1)
def get_booking_status_intent_patterns() -> Tuple[Pattern[str], ...]:
    raw = _flow_block().get("status_intent_patterns") or _DEFAULT_STATUS_PATTERNS
    return tuple(re.compile(str(p), re.I) for p in raw if str(p).strip())


@lru_cache(maxsize=1)
def get_amendment_name_patterns() -> Tuple[Pattern[str], ...]:
    raw = _flow_block().get("amendment_name_patterns") or _DEFAULT_AMENDMENT_NAME_PATTERNS
    return tuple(re.compile(str(p), re.I) for p in raw if str(p).strip())


@lru_cache(maxsize=1)
def get_cancellation_patterns() -> Tuple[Pattern[str], ...]:
    raw = _flow_block().get("cancellation_patterns") or _DEFAULT_CANCELLATION_PATTERNS
    return tuple(re.compile(str(p), re.I) for p in raw if str(p).strip())


@lru_cache(maxsize=1)
def get_negated_cancellation_patterns() -> Tuple[Pattern[str], ...]:
    raw = _flow_block().get("negated_cancellation_patterns") or _DEFAULT_NEGATED_CANCELLATION
    return tuple(re.compile(str(p), re.I) for p in raw if str(p).strip())


@lru_cache(maxsize=1)
def get_yes_tokens() -> FrozenSet[str]:
    raw = _flow_block().get("confirmation_yes_tokens") or list(_DEFAULT_YES)
    return frozenset(str(t).strip().lower() for t in raw if str(t).strip())


@lru_cache(maxsize=1)
def get_no_tokens() -> FrozenSet[str]:
    raw = _flow_block().get("confirmation_no_tokens") or list(_DEFAULT_NO)
    return frozenset(str(t).strip().lower() for t in raw if str(t).strip())


def reload_flow_config() -> None:
    """Clear cached flow config (called by admin reload)."""
    for fn in (
        get_active_booking_stages,
        get_post_confirmation_amendment_stages,
        get_faq_keywords,
        get_check_in_date_pattern,
        get_check_out_date_pattern,
        get_month_name_map,
        get_receipt_column_map,
        get_booking_status_intent_patterns,
        get_amendment_name_patterns,
        get_cancellation_patterns,
        get_negated_cancellation_patterns,
        get_yes_tokens,
        get_no_tokens,
    ):
        fn.cache_clear()
