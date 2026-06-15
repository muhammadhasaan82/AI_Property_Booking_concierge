from __future__ import annotations
"""Search tool submodule."""

from collections.abc import Iterable
import csv
import logging
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from app.agents.status_codes import Source, Status
from app.agents.tools.helpers import (
    _build_active_options,
    _classify_engagement_state,
    _coerce_float,
    _coerce_int,
    _finalize_payload,
    _get_cached_last_search,
    _get_soft_state,
    _get_unresolved_turns,
    _is_blank,
    _missing_critical_data,
    _normalize_action_intent,
    _set_cached_last_search,
    _set_unresolved_turns,
    HISTORY_ACTION_INTENTS,
    NEW_SEARCH_ACTION_INTENTS,
)
from app.config.agent_config_loader import cfg
from app.services.dynamic_constraints import DynamicConstraints
from app.services.faq_interruption import clear_faq_interruption, sync_alias_keys
from app.services.observability.langfuse_observer import (
    get_observer,
    sanitize_for_observability,
    summarize_property_results,
)
from app.services.property_type_normalizer import normalize_property_type as _normalize_property_type
from app.services.search_planner import SearchPlan, SearchTrace, get_search_planner
from app.agents.tools.search.constants import (
    CITY_COLUMN_CANDIDATES,
    DATASET_PATH,
    PROPERTY_RERANK_LIMIT,
    PROPERTY_RERANK_TIMEOUT_SECONDS,
    PROPERTY_RESULT_LIMIT_DEFAULT,
    PROPERTY_RESULT_LIMIT_MAX,
    PROPERTY_SUMMARY_THRESHOLD,
)

logger = logging.getLogger(__name__)
import csv

def get_all_available_cities(
    action_intent: Optional[str] = None,
    context_flag: Optional[str] = None,
) -> dict:
    """Use this tool when the user asks for a list of available cities or locations."""
    try:
        cities: set[str] = set()
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col_name = next(
                (c for c in CITY_COLUMN_CANDIDATES if c in (reader.fieldnames or [])),
                "city",
            )
            for row in reader:
                val = row.get(col_name)
                if val:
                    cities.add(val.strip())
        city_list = sorted(cities)
        payload = {
            "status": Status.CITIES_FOUND,
            "total_cities": len(city_list),
            "cities": city_list,
        }
        return _finalize_payload(payload, action_intent, context_flag)
    except Exception as e:
        return {"status": Status.ERROR, "error": str(e)}

