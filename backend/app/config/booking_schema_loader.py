"""
app/config/booking_schema_loader.py
------------------------------------
Loads booking_schema.yaml at startup. Provides typed validators and prompt
lookup helpers consumed by the policy router (Phase 3) and downstream tools.
 
Keys that overlap with agent_config.yaml#booking remain in sync until full
migration — see Phase 3 of the soft-coding roadmap.
"""
from __future__ import annotations
import logging 
import re 
from datetime import date, datetime 
from pathlib import Path
from torch.optim.optimizer import required
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent / "booking_schema.yaml"

class _ValidatorSpec(BaseModel):
    type: str
    pattern: Optional[str] = None
    min: Optional[str] = None
    max: Optional[str] = None
    format: Optional[str] = None
    not_before: Optional[str] = None
    after_field: Optional[str] = None
    message: Optional[str] = None

class _BookingBlock(BaseModel):
    required_fields: List[str] = Field(default_factory=list)
    required_numeric_fields: List[str] = Field(default_factory=list)
    ask_order: List[str] = Field(default_factory=list)
    field_prompts: Dict[str, str] = Field(default_factory=dict)
    validators: Dict[str, _ValidatorSpec] = Field(default_factory=dict)
    date_format: str = "%Y-%m-%d"
    source_tag: str = "v2_adk"
    confirmed_status: str = "confirmed"

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
    return List(booking_schema.booking.required_numeric_fields)

def get_ask_order() -> List[str]:
    return list(booking_schema.booking.ask_order)

def get_field_prompt(field: str) -> Optional[str]:
    return booking_schema.booking.field_prompts.get(field)

def next_field_to_ask(missing_fields: List[str]) -> Optional[str]:
    """Return the first field in ask_order that is also in missing_fields."""
    if not missing_fields:
        return None
    missing_set = set(missing_fields)
    for field in get_ask_order():
        if field in missing_set:
            return field
    return missing_fields[0]

def validate_field(field: str, value:Any, current_state: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
    """Validate a single field. Returns (ok, error_message)."""
    spec = booking_schema.booking.validators.get(field)
    if spec is None:
        return True, None

    if value is None or (isinstance(value, str)and not value.strip()):
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
            if spec.not_before == "today" and parsed < date.today():
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

    for field in get_required_fields() + get_required_fields():
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
    logger.info("[booking_schema] reloaded version=%s", booking_schema.version)