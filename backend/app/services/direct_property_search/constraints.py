from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from app.services.direct_property_search.extraction import _canonical_property_type_key
from app.services.property_query_constraints import PropertySearchQuery

logger = logging.getLogger(__name__)

def _constraint_values_from_query(
    query: Optional[PropertySearchQuery],
    *,
    property_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract structured constraint values from a PropertySearchQuery."""
    values: Dict[str, Any] = {
        "property_type": _canonical_property_type_key(property_type),
        "bedroom_exact": None,
        "bedroom_min": None,
        "bedroom_max": None,
        "bathroom_exact": None,
        "price_max": None,
        "occupancy_min": None,
        "amenities": [],
    }
    if query is not None and query.property_type:
        values["property_type"] = _canonical_property_type_key(query.property_type)
    if query is None:
        return values
    for c in query.constraints or []:
        if c.field == "bedrooms":
            try:
                num = int(c.value)
            except (TypeError, ValueError):
                continue
            if c.operator == "exact":
                values["bedroom_exact"] = num
            elif c.operator == "min":
                values["bedroom_min"] = num
            elif c.operator == "max":
                values["bedroom_max"] = num
        elif c.field == "bathrooms" and c.operator == "exact":
            try:
                values["bathroom_exact"] = int(c.value)
            except (TypeError, ValueError):
                continue
        elif c.field == "price_per_night" and c.operator == "max":
            try:
                values["price_max"] = float(c.value)
            except (TypeError, ValueError):
                continue
        elif c.field == "occupancy_max" and c.operator == "min":
            try:
                values["occupancy_min"] = int(c.value)
            except (TypeError, ValueError):
                continue
        elif c.field == "amenities":
            values["amenities"].append(str(c.value))
    return values

def _property_matches_constraints(prop: Any, constraints: Dict[str, Any]) -> bool:
    if not isinstance(prop, dict):
        return False
    property_type = constraints.get("property_type")
    bedroom_exact = constraints.get("bedroom_exact")
    bedroom_min = constraints.get("bedroom_min")
    bedroom_max = constraints.get("bedroom_max")
    bathroom_exact = constraints.get("bathroom_exact")
    price_max = constraints.get("price_max")
    occupancy_min = constraints.get("occupancy_min")
    amenities = constraints.get("amenities") or []

    if property_type:
        row_type = _canonical_property_type_key(prop.get("property_type"))
        if row_type != property_type:
            return False
    if bedroom_exact is not None:
        try:
            if int(prop.get("bedrooms") or 0) != int(bedroom_exact):
                return False
        except (TypeError, ValueError):
            return False
    if bedroom_min is not None:
        try:
            if int(prop.get("bedrooms") or 0) < int(bedroom_min):
                return False
        except (TypeError, ValueError):
            return False
    if bedroom_max is not None:
        try:
            if int(prop.get("bedrooms") or 0) > int(bedroom_max):
                return False
        except (TypeError, ValueError):
            return False
    if bathroom_exact is not None:
        try:
            if int(prop.get("bathrooms") or 0) != int(bathroom_exact):
                return False
        except (TypeError, ValueError):
            return False
    if price_max is not None:
        try:
            if float(prop.get("price_per_night") or 0) > float(price_max):
                return False
        except (TypeError, ValueError):
            return False
    if occupancy_min is not None:
        try:
            occupancy_value = prop.get("occupancy_max")
            if occupancy_value is None:
                bedrooms_value = prop.get("bedrooms")
                occupancy_value = int(bedrooms_value) * 2 if bedrooms_value is not None else 0
            if int(occupancy_value) < int(occupancy_min):
                return False
        except (TypeError, ValueError):
            return False
    if amenities:
        prop_amenities = {
            str(a).strip().lower()
            for a in (prop.get("amenities") or [])
            if a is not None
        }
        for required in amenities:
            token = str(required).strip().lower()
            if token and token not in prop_amenities:
                return False
    return True

def _has_active_constraints(constraints: Dict[str, Any]) -> bool:
    if not constraints:
        return False
    for key in (
        "property_type",
        "bedroom_exact",
        "bedroom_min",
        "bedroom_max",
        "bathroom_exact",
        "price_max",
        "occupancy_min",
    ):
        if constraints.get(key) is not None:
            return True
    if constraints.get("amenities"):
        return True
    return False

def _build_query_context(
    *,
    city: str,
    property_type: Optional[str],
    constraints: Dict[str, Any],
    price_max: Optional[float],
) -> Dict[str, Any]:
    qctx: Dict[str, Any] = {"city": city}
    if property_type:
        qctx["property_type"] = property_type
    bedrooms_exact = constraints.get("bedroom_exact")
    bedrooms_min = constraints.get("bedroom_min")
    bedrooms_max = constraints.get("bedroom_max")
    if bedrooms_exact is not None:
        qctx["bedrooms"] = bedrooms_exact
        qctx["bedrooms_operator"] = "exact"
    elif bedrooms_min is not None:
        qctx["bedrooms"] = bedrooms_min
        qctx["bedrooms_operator"] = "min"
    elif bedrooms_max is not None:
        qctx["bedrooms"] = bedrooms_max
        qctx["bedrooms_operator"] = "max"
    if constraints.get("bathroom_exact") is not None:
        qctx["bathrooms"] = constraints["bathroom_exact"]
        qctx["bathrooms_operator"] = "exact"
    if constraints.get("occupancy_min") is not None:
        qctx["guests"] = constraints["occupancy_min"]
    if constraints.get("amenities"):
        qctx["amenities"] = list(constraints["amenities"])
    if price_max is not None:
        qctx["budget"] = price_max
    return qctx

def _build_no_results_payload(
    *,
    city: str,
    property_type: Optional[str],
    constraints: Dict[str, Any],
    price_max: Optional[float],
) -> Dict[str, Any]:
    qctx = _build_query_context(
        city=city,
        property_type=property_type,
        constraints=constraints,
        price_max=price_max,
    )
    return {
        "status": "no_results",
        "city": city,
        "property_type": property_type,
        "bedrooms": constraints.get("bedroom_exact"),
        "bedrooms_operator": "exact" if constraints.get("bedroom_exact") is not None else None,
        "bathrooms": constraints.get("bathroom_exact"),
        "amenities": list(constraints.get("amenities") or []),
        "query_context": qctx,
        "filters_applied": dict(qctx),
    }

def _filter_properties_by_constraints(
    properties: Optional[List[Any]],
    constraints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not properties:
        return []
    if not _has_active_constraints(constraints):
        return [p for p in properties if isinstance(p, dict)]
    return [
        p
        for p in properties
        if isinstance(p, dict) and _property_matches_constraints(p, constraints)
    ]

def _build_option_map(properties: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    option_map: Dict[str, Dict[str, Any]] = {}
    for prop in properties:
        if not isinstance(prop, dict):
            continue
        number = prop.get("number")
        if number is None:
            continue
        prop_id = prop.get("id")
        if prop_id is None:
            continue
        option_map[str(number)] = {
            "property_id": str(prop_id),
            "title": prop.get("title"),
        }
    return option_map

def _renumber_properties(properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for idx, prop in enumerate(properties, start=1):
        if isinstance(prop, dict):
            prop["number"] = idx
    return properties

def _sort_properties_for_display(properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        properties,
        key=lambda p: (
            float(p.get("rating") or 0),
            int(p.get("reviews_count") or 0),
        ),
        reverse=True,
    )

def _has_search_phrase(message: str) -> bool:
    normalized = _normalize(message)
    if not normalized:
        return False
    nlp = get_vocabulary().nlp_fallback
    for phrase in nlp.search_phrases:
        if phrase and phrase.lower() in normalized:
            return True
    for signal in nlp.search_signals:
        if signal and _contains_term(normalized, signal):
            return True
    return False

from app.services.direct_property_search.city import _normalize  # noqa: F401

from app.services.dynamic_config import get_vocabulary  # noqa: F401

def _contains_term(text: str, term: str) -> bool:
    normalized_text = _normalize(text or "")
    normalized_term = _normalize(term or "")
    return bool(normalized_term and normalized_term in normalized_text)
