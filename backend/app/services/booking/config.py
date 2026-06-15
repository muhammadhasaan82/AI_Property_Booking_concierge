from __future__ import annotations
"""Config-driven booking flow phrases, stages, and compiled regex catalogs."""

import logging
import re
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Pattern, Tuple

from app.config.booking_flow_loader import (
    get_active_booking_stages,
    get_amendment_name_patterns,
    get_booking_status_intent_patterns,
    get_cancellation_patterns,
    get_check_in_date_pattern,
    get_check_out_date_pattern,
    get_faq_keywords,
    get_month_name_map,
    get_negated_cancellation_patterns,
    get_post_confirmation_amendment_stages,
    get_receipt_column_map,
    get_yes_tokens,
    get_no_tokens,
)

logger = logging.getLogger(__name__)

_ACTIVE_BOOKING_STAGES = get_active_booking_stages()
_POST_CONFIRMATION_AMENDMENT_STAGES = get_post_confirmation_amendment_stages()
_FAQ_KEYWORDS = get_faq_keywords()
_CHECK_IN_DATE_PATTERN = get_check_in_date_pattern()
_CHECK_OUT_DATE_PATTERN = get_check_out_date_pattern()
_MONTHS = get_month_name_map()
RECEIPT_TO_SUCCESSFUL_BOOKING_COLUMNS = get_receipt_column_map()
SUCCESSFUL_BOOKING_TO_RECEIPT_KEYS = {
    db_key: receipt_key
    for receipt_key, db_key in RECEIPT_TO_SUCCESSFUL_BOOKING_COLUMNS.items()
}
_BOOKING_STATUS_INTENT_PATTERNS = get_booking_status_intent_patterns()
_AMENDMENT_NAME_VALUE_PATTERNS = get_amendment_name_patterns()
_CANCELLATION_PATTERNS = get_cancellation_patterns()
_NEGATED_CANCELLATION_PATTERNS = get_negated_cancellation_patterns()
_YES_TOKENS = get_yes_tokens()
_NO_TOKENS = get_no_tokens()

BOOKING_ID_PATTERN = re.compile(r"BK-\d{8}-[A-Z0-9]+", re.I)


def _extract_booking_id(message: str) -> str | None:
    if not message:
        return None
    match = BOOKING_ID_PATTERN.search(message)
    return match.group(0).upper() if match else None
