from __future__ import annotations
"""
Loads booking_schema.yaml at startup. Provides typed validators and prompt
lookup helpers consumed by the policy router (Phase 3) and downstream tools.
 
Keys that overlap with agent_config.yaml
migration — see Phase 3 of the soft-coding roadmap.
"""
import logging 
import os
import re
from datetime import date, datetime 
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent / "booking_schema.yaml"

class _ValidatorSpec(BaseModel):
    type: str
    pattern: Optional[str] = None
    min: Optional[Any] = None
    max: Optional[Any] = None
    format: Optional[str] = None
    not_before: Optional[str] = None
    after_field: Optional[str] = None
    message: Optional[str] = None

class _BookingBlock(BaseModel):
    required_fields: List[str] = Field(default_factory=list)
    required_numeric_fields: List[str] = Field(default_factory=list)
    ask_order: List[str] = Field(default_factory=list)
    field_display_names: Dict[str, str] = Field(default_factory=dict)
    field_prompts: Dict[str, str] = Field(default_factory=dict)
    field_aliases: Dict[str, List[str]] = Field(default_factory=dict)
    amendable_fields: List[str] = Field(default_factory=list)
    amendment_display_names: Dict[str, str] = Field(default_factory=dict)
    amendment_intent_verbs: List[str] = Field(default_factory=list)
    amendment_context_markers: List[str] = Field(default_factory=list)
    amendment_field_groups: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    validators: Dict[str, _ValidatorSpec] = Field(default_factory=dict)
    date_format: str = "%Y-%m-%d"
    source_tag: str = "v2_adk"
    confirmed_status: str = "confirmed"
    flow: Dict[str, Any] = Field(default_factory=dict)

class _BookingSchemaRoot(BaseModel):
    version: str = "1.0"
    booking: _BookingBlock = Field(default_factory=_BookingBlock)

def _load() -> _BookingSchemaRoot:
    if not _SCHEMA_PATH.exists():
        logger.warning("[booking_schema] %s missing, using defaults", _SCHEMA_PATH)
        return _BookingSchemaRoot()
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
        return _BookingSchemaRoot(**raw)

booking_schema: _BookingSchemaRoot = _load()

def get_required_fields() -> List[str]:
    return list(booking_schema.booking.required_fields)

def get_required_numeric_fields() -> List[str]:
    return list(booking_schema.booking.required_numeric_fields)

def get_ask_order() -> List[str]:
    return list(booking_schema.booking.ask_order)

def get_field_prompt(field: str) -> Optional[str]:
    return booking_schema.booking.field_prompts.get(field)


def get_field_display_name(field: str) -> str:
    return str(booking_schema.booking.field_display_names.get(field) or field)


def get_field_aliases() -> Dict[str, List[str]]:
    return {
        str(field): [str(alias).strip() for alias in aliases if str(alias).strip()]
        for field, aliases in (booking_schema.booking.field_aliases or {}).items()
    }


def get_amendable_fields() -> List[str]:
    fields = booking_schema.booking.amendable_fields or []
    if fields:
        return [str(field) for field in fields if str(field).strip()]
    return list(get_ask_order())


def get_amendment_display_name(field: str) -> str:
    return str(
        booking_schema.booking.amendment_display_names.get(field)
        or get_field_display_name(field).lower()
    )


def get_amendment_intent_verbs() -> List[str]:
    verbs = booking_schema.booking.amendment_intent_verbs or []
    return [str(verb).strip().lower() for verb in verbs if str(verb).strip()]


def get_amendment_context_markers() -> List[str]:
    markers = booking_schema.booking.amendment_context_markers or []
    return [str(marker).strip().lower() for marker in markers if str(marker).strip()]


def get_amendment_field_groups() -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for key, raw in (booking_schema.booking.amendment_field_groups or {}).items():
        if not isinstance(raw, dict):
            continue
        groups[str(key)] = {
            "display_name": str(raw.get("display_name") or key),
            "fields": [str(field) for field in raw.get("fields", []) if str(field).strip()],
            "aliases": [str(alias) for alias in raw.get("aliases", []) if str(alias).strip()],
        }
    return groups

def next_field_to_ask(missing_fields: List[str]) -> Optional[str]:
    """
    Selects which missing booking field should be asked next.
    
    Parameters:
    	missing_fields (List[str]): Sequence of field names that are currently missing.
    
    Returns:
    	selected_field (Optional[str]): The first field from the schema's ask order that appears in `missing_fields`; if none of the ask-order fields are missing, returns the first entry of `missing_fields`. Returns `None` when `missing_fields` is empty.
    """
    if not missing_fields:
        return None
    missing_set = set(missing_fields)
    for field in get_ask_order():
        if field in missing_set:
            return field
    return missing_fields[0]
def _reference_today() -> date:
    """
    Provide the reference date used by booking date validators.
    
    If the environment variable BOOKING_REFERENCE_DATE is set to a valid YYYY-MM-DD string, that date is returned; on parse failure or if the variable is unset/empty, the current local date is returned (and a warning is logged on parse failure).
    
    Returns:
        date: The parsed reference date from BOOKING_REFERENCE_DATE when valid, otherwise date.today().
    """
    raw = os.getenv("BOOKING_REFERENCE_DATE", "").strip()
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("[booking_schema] Invalid BOOKING_REFERENCE_DATE=%r; using date.today()", raw)
    return date.today()
def validate_field(field: str, value: Any, current_state: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate a single booking field against the schema-defined validator for that field.
    
    Parameters:
        field (str): The name of the booking field to validate.
        value (Any): The candidate value for the field.
        current_state (Optional[Dict[str, Any]]): Optional mapping of other field values used for cross-field checks (for example, comparing dates).
    
    Returns:
        tuple: A two-element tuple where the first element is `True` if the value satisfies the field's validator (or no validator exists), `False` otherwise; the second element is an error message string when invalid, or `None` when valid.
    """
    spec = booking_schema.booking.validators.get(field)
    if spec is None:
        return True, None

    if value is None or (isinstance(value, str) and not value.strip()):
        return False, spec.message or f"{field} is required"

    try:
        if spec.type == "regex" and spec.pattern:
            if not re.match(spec.pattern, str(value).strip()):
                return False, spec.message or f"{field} format is invalid"
        elif spec.type == "length":
            length = len(str(value).strip())
            if spec.min is not None and length < spec.min:
                return False, spec.message or f"{field} is too short"
            if spec.max is not None and length > spec.max:
                return False, spec.message or f"{field} is too long"
        elif spec.type == "integer":
            invalue = int(value)
            if spec.min is not None and invalue < spec.min:
                return False, spec.message or f"{field} must be ≥ {int(spec.min)}"
            if spec.max is not None and invalue > spec.max:
                return False, spec.message or f"{field} must be ≤ {int(spec.max)}"

        elif spec.type == "float":
            fvalue = float(value)
            if spec.min is not None and fvalue < spec.min:
                return False, spec.message or f"{field} must be ≥ {float(spec.min)}"
            if spec.max is not None and fvalue > spec.max:
                return False, spec.message or f"{field} must be ≤ {float(spec.max)}"

        elif spec.type == "date":
            fmt = spec.format or booking_schema.booking.date_format
            parsed = datetime.strptime(str(value).strip(), fmt).date()
            if spec.not_before == "today" and parsed < _reference_today():
                return False, spec.message or f"{field} must be today or later"
            if spec.after_field and current_state:
                other_raw = current_state.get(spec.after_field)
                if other_raw:
                    other = datetime.strptime(str(other_raw).strip(), fmt).date()
                    if parsed <= other:
                        return False, spec.message or f"{field} must be after {spec.after_field}"
    except (ValueError, TypeError) as exc:
        return False, spec.message or f"{field} is invalid: {exc}"

    return True, None

def validate_full_booking(state: Dict[str, Any]) -> tuple[list[str],Dict[str, str]]:
    """
    Validate a full booking State.
    Returns (missing_fields, validation_errors_by_field).
    """
    missing: List[str] = []
    errors: Dict[str, str] = {}

    for field in get_required_fields() + get_required_numeric_fields():
        value = state.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
            continue
        ok, err= validate_field(field, value, current_state=state)
        if not ok and err:
            errors[field] = err

    return list(dict.fromkeys(missing)), errors

def reload() -> None:
    """Hot-reload helper used by phase 4's / admin/reload-config endpoint."""
    global booking_schema
    booking_schema = _load()
    try:
        from app.config.booking_flow_loader import reload_flow_config
        reload_flow_config()
    except Exception as exc:
        logger.warning("[booking_schema] flow config reload failed: %s", exc)
    logger.info("[booking_schema] reloaded version=%s", booking_schema.version)
