"""
Schema-aware deterministic extraction of property search constraints.

This module keeps extraction deterministic and independent from Langfuse runtime state.
Langfuse tracing is optional and non-blocking through extract_constraints_with_tracing().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


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


# Backward compatibility for any older code that imported Constraint.
Constraint = SearchConstraint


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _existing_city(message: str) -> Optional[str]:
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
    Extract structured property search constraints from natural language.

    Examples:
    - "show me some 2 bedrooms apartment in new york city"
    - "2br apartment in New York under 300"
    - "apartment in Seattle for 4 guests"
    """
    text = _norm(message)
    constraints: List[SearchConstraint] = []

    city = _existing_city(message)
    property_type = _existing_property_type(message)

    # Bedrooms: min forms first so "at least 3 bedrooms" does not become exact.
    min_bed_match = re.search(
        r"\b(?:at\s+least|minimum|min)\s+(\d+)\s*(?:bedrooms?|beds?|br)\b",
        text,
    )
    if min_bed_match:
        _add(
            constraints,
            field_name="bedrooms",
            column="bedrooms",
            operator="min",
            value=int(min_bed_match.group(1)),
            source_text=min_bed_match.group(0),
        )
    else:
        exact_bed_match = re.search(r"\b(\d+)\s*(?:bedrooms?|beds?|br)\b", text)
        if exact_bed_match:
            _add(
                constraints,
                field_name="bedrooms",
                column="bedrooms",
                operator="exact",
                value=int(exact_bed_match.group(1)),
                source_text=exact_bed_match.group(0),
            )

    max_bed_match = re.search(
        r"\b(?:up\s+to|maximum|max)\s+(\d+)\s*(?:bedrooms?|beds?|br)\b",
        text,
    )
    if max_bed_match and not _has_constraint(constraints, "bedrooms"):
        _add(
            constraints,
            field_name="bedrooms",
            column="bedrooms",
            operator="max",
            value=int(max_bed_match.group(1)),
            source_text=max_bed_match.group(0),
        )

    # Bathrooms exact.
    bath_match = re.search(r"\b(\d+)\s*(?:bathrooms?|baths?|ba)\b", text)
    if bath_match:
        _add(
            constraints,
            field_name="bathrooms",
            column="bathrooms",
            operator="exact",
            value=int(bath_match.group(1)),
            source_text=bath_match.group(0),
        )

    # Price max/min.
    price_max_match = re.search(
        r"\b(?:under|below|less\s+than|budget|within)\s+\$?\s*(\d+(?:\.\d+)?)\b",
        text,
    )
    if price_max_match:
        _add(
            constraints,
            field_name="price_per_night",
            column="price_per_night",
            operator="max",
            value=float(price_max_match.group(1)),
            source_text=price_max_match.group(0),
        )

    price_min_match = re.search(
        r"\b(?:above|over|more\s+than)\s+\$?\s*(\d+(?:\.\d+)?)\b",
        text,
    )
    if price_min_match:
        _add(
            constraints,
            field_name="price_per_night",
            column="price_per_night",
            operator="min",
            value=float(price_min_match.group(1)),
            source_text=price_min_match.group(0),
        )

    # Occupancy/capacity.
    guest_match = re.search(r"\b(?:for\s+)?(\d+)\s*(?:guests?|people|persons?)\b", text)
    if guest_match:
        _add(
            constraints,
            field_name="occupancy_max",
            column="occupancy_max",
            operator="min",
            value=int(guest_match.group(1)),
            source_text=guest_match.group(0),
        )

    # Amenities.
    amenity_aliases = {
        "pet friendly": "pet_friendly",
        "pets allowed": "pet_friendly",
        "wifi": "wifi",
        "wi fi": "wifi",
        "parking": "parking",
        "pool": "pool",
        "gym": "gym",
    }
    for phrase, amenity in amenity_aliases.items():
        if phrase in text:
            _add(
                constraints,
                field_name="amenities",
                column="amenities",
                operator="contains",
                value=amenity,
                source_text=phrase,
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
            from app.services.observability.langfuse_observer import get_observer

            obs = get_observer()

        trace_metadata = {
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
