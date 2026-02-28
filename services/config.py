# services/config.py
"""
Central configuration module — single source of truth for all
prompts, phrase lists, field definitions, cities, amenities, and
property types. Consumers import from here; no more copy-paste.
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
# Environment-driven settings
# ---------------------------------------------------------------------------

DATASET_PATH: str = os.getenv("DATASET_PATH", "./services/dataset.csv")

MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() in ("1", "true", "yes")

PAYMENT_BASE_URL: str = os.getenv("PAYMENT_BASE_URL", "https://example.com/pay")

# ---------------------------------------------------------------------------
# Slot-filling schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: List[str] = [
    "name", "phone", "email", "check_in", "check_out", "guests",
]

FIELD_PROMPTS: Dict[str, str] = {
    "name":      "Please share your full name.",
    "phone":     "Please share your phone number.",
    "email":     "Please share your email address.",
    "check_in":  "What is your check-in date (YYYY-MM-DD)?",
    "check_out": "What is your check-out date (YYYY-MM-DD)?",
    "guests":    "How many guests?",
}

FIELD_MODIFICATION_PROMPTS: Dict[str, str] = {
    "name":      "What's the correct full name?",
    "phone":     "What's the correct phone number?",
    "email":     "What's the correct email address?",
    "check_in":  "What's the new check-in date (YYYY-MM-DD)?",
    "check_out": "Please share the new check-out date (YYYY-MM-DD).",
    "guests":    "How many guests now?",
}

# ---------------------------------------------------------------------------
# Proceed / Modify phrase lists  (union of all existing copies)
# ---------------------------------------------------------------------------

PROCEED_PHRASES: List[str] = [
    "proceed", "continue", "receipt", "total", "total bill", "bill",
    "show", "final", "summary", "confirm", "no", "see", "see receipt",
    "see total", "show total", "show receipt", "go ahead", "next", "done",
    "updated total", "yes i want to proceed", "yes proceed",
    "proceed to total", "proceed to payment", "show me total bill",
    "please show me my total bill", "i want to proceed",
    "i want total bill", "total bill for payment",
    "proceed to total bill", "payment", "pay",
]

MODIFY_PHRASES: List[str] = [
    "modify", "change", "edit", "update", "adjust", "more", "another",
    "again", "i want to make more changes", "i want make more changes",
    "make more changes", "want to change", "want to modify",
    "i want to change", "i want to modify", "tweak", "fix", "yes",
]

# ---------------------------------------------------------------------------
# Intent routing keyword lists
# ---------------------------------------------------------------------------

FAQ_FALLBACK_KEYWORDS: List[str] = [
    "wifi", "faq", "policy", "check-in time", "rules", "password",
    "refund", "cancel", "terms", "internet",
]

BOOKING_TRIGGERS: List[str] = [
    " book ", "reserve", "hold", "lock it", "go ahead", "confirm it",
]

STATUS_KEYWORDS: List[str] = [
    "check in", "check-in", "check out", "check-out", "status",
]

PAYMENT_KEYWORDS: List[str] = [
    "pay", "payment", "link", "invoice",
]

# ---------------------------------------------------------------------------
# Fallback cities & aliases
# ---------------------------------------------------------------------------

FALLBACK_CITIES: Set[str] = {
    "new york", "los angeles", "san francisco", "miami", "boston",
    "chicago", "seattle", "austin", "dallas", "houston", "washington",
    "washington dc", "san diego", "san jose", "orlando", "las vegas",
    "philadelphia", "phoenix", "atlanta",
}

FALLBACK_CITY_ALIASES: Dict[str, str] = {
    "nyc": "new york",
    "ny": "new york",
    "newyork": "new york",
    "la": "los angeles",
    "losangeles": "los angeles",
    "sf": "san francisco",
    "sanfrancisco": "san francisco",
    "dc": "washington",
    "washingtondc": "washington",
}

# ---------------------------------------------------------------------------
# Property types
# ---------------------------------------------------------------------------

SEED_PROPERTY_TYPES: Set[str] = {
    "condo", "loft", "apartment", "house", "studio", "villa",
    "townhouse", "flat", "cottage", "bungalow", "penthouse", "duplex",
}

# ---------------------------------------------------------------------------
# Amenity synonyms
# ---------------------------------------------------------------------------

BASE_AMENITY_SYNONYMS: Dict[str, List[str]] = {
    "wifi":       ["wifi", "wi-fi", "internet", "wireless"],
    "pool":       ["pool", "swimming"],
    "parking":    ["parking", "garage", "car"],
    "gym":        ["gym", "fitness", "workout"],
    "kitchen":    ["kitchen", "cooking"],
    "ac":         ["ac", "air conditioning", "air-conditioning", "cooling"],
    "heating":    ["heating", "heater", "warm"],
    "washer":     ["washer", "washing machine", "laundry"],
    "dryer":      ["dryer", "drying"],
    "dishwasher": ["dishwasher", "dishes"],
    "balcony":    ["balcony", "terrace", "patio"],
    "view":       ["view", "scenic", "ocean view", "city view"],
}
