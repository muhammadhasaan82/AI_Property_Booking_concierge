# services/config.py
"""
Central configuration module — thin wrapper over dynamic_config.

When LEGACY_RULES=true, returns the original hardcoded values.
When LEGACY_RULES=false (default), delegates to YAML-loaded config.

Consumers import from here for backward compatibility:
    from services.config import REQUIRED_FIELDS, PROCEED_PHRASES, ...
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# --- Environment Loading ---
_env_root = Path(__file__).parent.parent / ".env"
_env_services = Path(__file__).parent / ".env"
if _env_root.exists():
    load_dotenv(_env_root)
elif _env_services.exists():
    load_dotenv(_env_services)

# ---------------------------------------------------------------------------
# Environment-driven settings (always from env, not YAML)
# ---------------------------------------------------------------------------

DATASET_PATH: str = os.getenv("DATASET_PATH", "./services/dataset.csv")
MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() in ("1", "true", "yes")
PAYMENT_BASE_URL: str = os.getenv("PAYMENT_BASE_URL", "https://example.com/pay")


# ---------------------------------------------------------------------------
# Legacy fallback values (only used when LEGACY_RULES=true)
# ---------------------------------------------------------------------------
_LEGACY = {
    "REQUIRED_FIELDS": ["name", "phone", "email", "check_in", "check_out", "guests"],
    "FIELD_PROMPTS": {
        "name": "Please share your full name.",
        "phone": "Please share your phone number.",
        "email": "Please share your email address.",
        "check_in": "What is your check-in date (YYYY-MM-DD)?",
        "check_out": "What is your check-out date (YYYY-MM-DD)?",
        "guests": "How many guests?",
    },
    "FIELD_MODIFICATION_PROMPTS": {
        "name": "What's the correct full name?",
        "phone": "What's the correct phone number?",
        "email": "What's the correct email address?",
        "check_in": "What's the new check-in date (YYYY-MM-DD)?",
        "check_out": "Please share the new check-out date (YYYY-MM-DD).",
        "guests": "How many guests now?",
    },
    "PROCEED_PHRASES": [
        "proceed", "continue", "receipt", "total", "total bill", "bill",
        "show", "final", "summary", "confirm", "no", "see", "see receipt",
        "see total", "show total", "show receipt", "go ahead", "next", "done",
        "updated total", "yes i want to proceed", "yes proceed",
        "proceed to total", "proceed to payment", "show me total bill",
        "please show me my total bill", "i want to proceed",
        "i want total bill", "total bill for payment",
        "proceed to total bill", "payment", "pay",
    ],
    "MODIFY_PHRASES": [
        "modify", "change", "edit", "update", "adjust", "more", "another",
        "again", "i want to make more changes", "i want make more changes",
        "make more changes", "want to change", "want to modify",
        "i want to change", "i want to modify", "tweak", "fix", "yes",
    ],
    "FAQ_FALLBACK_KEYWORDS": [
        "wifi", "faq", "policy", "check-in time", "rules", "password",
        "refund", "cancel", "terms", "internet",
    ],
    "BOOKING_TRIGGERS": [" book ", "reserve", "hold", "lock it", "go ahead", "confirm it"],
    "STATUS_KEYWORDS": ["check in", "check-in", "check out", "check-out", "status"],
    "PAYMENT_KEYWORDS": ["pay", "payment", "link", "invoice"],
    "FALLBACK_CITIES": {
        "new york", "los angeles", "san francisco", "miami", "boston",
        "chicago", "seattle", "austin", "dallas", "houston", "washington",
        "washington dc", "san diego", "san jose", "orlando", "las vegas",
        "philadelphia", "phoenix", "atlanta",
    },
    "FALLBACK_CITY_ALIASES": {
        "nyc": "new york", "ny": "new york", "newyork": "new york",
        "la": "los angeles", "losangeles": "los angeles",
        "sf": "san francisco", "sanfrancisco": "san francisco",
        "dc": "washington", "washingtondc": "washington",
    },
    "SEED_PROPERTY_TYPES": {
        "condo", "loft", "apartment", "house", "studio", "villa",
        "townhouse", "flat", "cottage", "bungalow", "penthouse", "duplex",
    },
    "BASE_AMENITY_SYNONYMS": {
        "wifi": ["wifi", "wi-fi", "internet", "wireless"],
        "pool": ["pool", "swimming"],
        "parking": ["parking", "garage", "car"],
        "gym": ["gym", "fitness", "workout"],
        "kitchen": ["kitchen", "cooking"],
        "ac": ["ac", "air conditioning", "air-conditioning", "cooling"],
        "heating": ["heating", "heater", "warm"],
        "washer": ["washer", "washing machine", "laundry"],
        "dryer": ["dryer", "drying"],
        "dishwasher": ["dishwasher", "dishes"],
        "balcony": ["balcony", "terrace", "patio"],
        "view": ["view", "scenic", "ocean view", "city view"],
    },
}


# ---------------------------------------------------------------------------
# Lazy module-level attribute access via __getattr__
# ---------------------------------------------------------------------------
_DYNAMIC_MAP = {
    "REQUIRED_FIELDS":           lambda v: v.slot_filling.required_fields,
    "FIELD_PROMPTS":             lambda v: v.slot_filling.field_prompts,
    "FIELD_MODIFICATION_PROMPTS":lambda v: v.slot_filling.modification_prompts,
    "PROCEED_PHRASES":           lambda v: v.proceed_phrases,
    "MODIFY_PHRASES":            lambda v: v.modify_phrases,
    "FAQ_FALLBACK_KEYWORDS":     lambda v: v.faq_fallback_keywords,
    "BOOKING_TRIGGERS":          lambda v: v.booking_triggers,
    "STATUS_KEYWORDS":           lambda v: v.status_keywords,
    "PAYMENT_KEYWORDS":          lambda v: v.payment_keywords,
    "FALLBACK_CITIES":           lambda v: v.fallback_cities_set,
    "FALLBACK_CITY_ALIASES":     lambda v: v.city_aliases,
    "SEED_PROPERTY_TYPES":       lambda v: v.seed_property_types_set,
    "BASE_AMENITY_SYNONYMS":     lambda v: v.amenity_synonyms,
}


def __getattr__(name: str):
    """Lazy attribute access — loads from dynamic config on first use."""
    if name in _DYNAMIC_MAP:
        from services.dynamic_config import LEGACY_RULES as _lr
        if _lr:
            return _LEGACY[name]
        from services.dynamic_config import get_vocabulary
        return _DYNAMIC_MAP[name](get_vocabulary())
    raise AttributeError(f"module 'services.config' has no attribute {name!r}")
