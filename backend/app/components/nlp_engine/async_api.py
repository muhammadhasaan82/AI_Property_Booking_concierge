from __future__ import annotations
import logging
import os
import re
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from app.services.dynamic_config import get_intent_catalog as _get_catalog
from app.services.dynamic_config import get_retrieval_config as _get_retrieval_config
from app.services.dynamic_config import get_thresholds as _get_thresholds
from app.services.dynamic_config import get_vocabulary as _get_vocabulary

logger = logging.getLogger(__name__)

import asyncio
from app.components.nlp_engine.classification import (
    classify_affirmation,
    classify_intent_sync,
    detect_faq_intent,
    is_greeting,
    is_greeting_sync,
    is_property_search,
    is_receipt_request,
    is_resume_request,
    is_status_query,
    wants_previous_results_sync,
)
from app.components.nlp_engine.extraction import (
    detect_requested_fields,
    extract_cardinal,
    extract_dates,
    extract_person_name,
    has_cardinal_extraction,
    is_low_semantic_density,
)

async def classify_affirmation_async(text: str) -> str:
    return await asyncio.to_thread(classify_affirmation, text)

async def is_greeting_async(text: str) -> bool:
    return await asyncio.to_thread(is_greeting, text)

async def extract_person_name_async(text: str) -> Optional[str]:
    return await asyncio.to_thread(extract_person_name, text)

async def extract_dates_async(text: str) -> List[str]:
    return await asyncio.to_thread(extract_dates, text)

async def extract_cardinal_async(text: str) -> Optional[int]:
    return await asyncio.to_thread(extract_cardinal, text)

async def has_cardinal_extraction_async(text: str) -> bool:
    return await asyncio.to_thread(has_cardinal_extraction, text)

async def is_low_semantic_density_async(text: str) -> bool:
    return await asyncio.to_thread(is_low_semantic_density, text)

async def classify_intent_async(text: str, candidates: List[str]) -> str:
    return await asyncio.to_thread(classify_intent_sync, text, candidates)

async def detect_faq_intent_async(text: str) -> bool:
    return await asyncio.to_thread(detect_faq_intent, text)

async def detect_requested_fields_async(text: str) -> List[str]:
    return await asyncio.to_thread(detect_requested_fields, text)

async def is_property_search_async(text: str) -> bool:
    return await asyncio.to_thread(is_property_search, text)

async def is_status_query_async(text: str) -> bool:
    return await asyncio.to_thread(is_status_query, text)

async def is_receipt_request_async(text: str) -> bool:
    return await asyncio.to_thread(is_receipt_request, text)

async def is_resume_request_async(text: str) -> bool:
    return await asyncio.to_thread(is_resume_request, text)

async def wants_previous_results_async(text: str) -> bool:
    return await asyncio.to_thread(wants_previous_results_sync, text)

