from __future__ import annotations
"""
Schema-aware deterministic extraction of property search constraints.

This module keeps extraction deterministic and independent from Langfuse runtime state.
Langfuse tracing is optional and non-blocking through extract_constraints_with_tracing().
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from app.services.property_schema import get_property_schema
from app.services.search_planner import get_search_planner

try:
    from app.services.observability.langfuse_observer import get_observer
except Exception:

    def get_observer() -> Any:
        return None


Operator = Literal["exact", "min", "max", "contains"]


@dataclass
class SearchConstraint:
    field: str
    column: str
    operator: Operator
    value: Any
    source_text: str = ""


@dataclass
class PropertySearchQuery:
    city: Optional[str] = None
    property_type: Optional[str] = None
    constraints: List[SearchConstraint] = field(default_factory=list)
    confidence: float = 1.0
    unresolved: List[str] = field(default_factory=list)


Constraint = SearchConstraint


def _norm(text: str) -> str:
    """
    Normalize a text string by trimming, lowercasing, and collapsing consecutive whitespace.
    
    Parameters:
    	text (str): Input text; falsy values (e.g., None or empty string) are treated as empty.
    
    Returns:
    	str: Normalized string with leading/trailing whitespace removed, converted to lowercase, and consecutive internal whitespace collapsed to single spaces. Returns an empty string for falsy input.
    """
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _canonical_city(value: Optional[str]) -> Optional[str]:
    """Normalize a city string for display; blank values return None."""
    if value is None:
        return None

    normalized = " ".join(str(value).strip().split())
    if not normalized:
        return None

    return normalized.title()

def _existing_city(message: str) -> Optional[str]:
    """
    Extract the city name from a user message using the project's direct search extractor.
    
    Parameters:
        message (str): Free-form user message to analyze for a city name.
    
    Returns:
        Optional[str]: The extracted city string, or `None` if extraction fails or no city could be determined.
    """
    try:
        from app.services.direct_property_search import extract_city_from_message

        return extract_city_from_message(message)
    except Exception:
        return None


def _existing_property_type(message: str) -> Optional[str]:
    try:
        from app.services.direct_property_search import extract_property_type_from_message

        value = extract_property_type_from_message(message)
        if value:
            return str(value).title()
    except Exception:
        pass
    return None


def _has_constraint(constraints: List[SearchConstraint], field_name: str) -> bool:
    return any(c.field == field_name for c in constraints)


def _add(
    constraints: List[SearchConstraint],
    *,
    field_name: str,
    column: str,
    operator: Operator,
    value: Any,
    source_text: str,
) -> None:
    constraints.append(
        SearchConstraint(
            field=field_name,
            column=column,
            operator=operator,
            value=value,
            source_text=source_text,
        )
    )


def extract_property_search_query(message: str) -> PropertySearchQuery:
    """
    Compatibility wrapper over the schema-driven search planner.

    Natural-language extraction is delegated to `SearchPlanner`, which discovers
    searchable fields from dataset/config metadata. This function only converts
    the planner's dynamic constraints back into the legacy `PropertySearchQuery`
    shape expected by older callers.
    """
    planner = get_search_planner()
    schema = get_property_schema()
    plan = planner.plan_search(message=message)

    city = _canonical_city(plan.constraints.get("city"))
    property_type = plan.constraints.get("property_type")
    if property_type:
        property_type = str(property_type).title()

    constraints: List[SearchConstraint] = []
    for field_name in sorted(plan.constraints.fields()):
        if field_name in {"city", "property_type"}:
            continue
        value = plan.constraints.get(field_name)
        operator = plan.constraints.get_operator(field_name, "exact")
        field_schema = schema.get_field(field_name)
        column = field_schema.name if field_schema else field_name

        if isinstance(value, list):
            for item in value:
                constraints.append(
                    SearchConstraint(
                        field=field_name,
                        column=column,
                        operator=operator,
                        value=item,
                        source_text=str(item),
                    )
                )
            continue

        constraints.append(
            SearchConstraint(
                field=field_name,
                column=column,
                operator=operator,
                value=value,
                source_text=str(value),
            )
        )

    return PropertySearchQuery(
        city=city,
        property_type=property_type,
        constraints=constraints,
        confidence=1.0,
        unresolved=[],
    )


def extract_constraints_with_tracing(
    message: str,
    city: Optional[str] = None,
    property_type: Optional[str] = None,
    *,
    observer: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PropertySearchQuery:
    """
    Compatibility wrapper around extract_property_search_query with optional tracing.

    Tracing is best-effort only and must never affect runtime behavior.
    """
    result = extract_property_search_query(message)

    if city and not result.city:
        result.city = city
    if property_type and not result.property_type:
        result.property_type = property_type

    try:
        obs = observer
        if obs is None:
            obs = get_observer()

        trace_metadata = {
            "extracted_city": city or result.city,
            "extracted_property_type": property_type or result.property_type,
            "city": result.city,
            "property_type": result.property_type,
            "constraints": [
                {"field": c.field, "operator": c.operator, "value": c.value}
                for c in result.constraints
            ],
            "confidence": result.confidence,
            "unresolved": result.unresolved,
        }
        if metadata:
            trace_metadata.update(metadata)

        with obs.trace(name="property_query_constraints", metadata=trace_metadata):
            pass
    except Exception:
        pass

    return result


__all__ = [
    "SearchConstraint",
    "Constraint",
    "PropertySearchQuery",
    "extract_property_search_query",
    "extract_constraints_with_tracing",
]
