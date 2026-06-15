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

def _local_model_load(enabled: bool):
    if not enabled:
        yield
        return

    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def _is_local_model_reference(model_name: str) -> bool:
    try:
        return os.path.exists(os.path.expanduser(model_name))
    except OSError:
        return False

def _get_intent_prototypes() -> Dict[str, List[str]]:
    catalog = _get_catalog()
    return {name: cfg.prototypes for name, cfg in catalog.intents.items() if cfg.prototypes}

def _get_field_prototypes() -> Dict[str, List[str]]:
    return _get_catalog().field_prototypes

def _get_modification_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().modification_prototypes)

def _get_property_search_request_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().property_search_request_prototypes)

def _get_receipt_request_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().receipt_request_prototypes)

def _get_resume_request_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().resume_request_prototypes)

def _get_affirm_yes_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().affirm_yes_prototypes)

def _get_affirm_no_prototypes() -> Tuple[str, ...]:
    return tuple(_get_catalog().affirm_no_prototypes)

def _get_vocab():
    return _get_vocabulary().nlp_fallback

def _get_nlp_thresholds():
    return _get_thresholds().nlp

def _get_intent_threshold(intent: str) -> float:
    catalog = _get_catalog()
    if intent in catalog.intents:
        threshold = float(catalog.intents[intent].threshold or 0.0)
        if threshold > 0:
            return threshold
    catalog_default = float(catalog.default_threshold or 0.0)
    if catalog_default > 0:
        return catalog_default
    return float(_get_nlp_thresholds().intent_threshold_default)

def _name_full_pattern(max_chars: int):
    return re.compile(rf"^([A-Za-z][A-Za-z .'-]{{1,{max_chars}}})$", re.I)

