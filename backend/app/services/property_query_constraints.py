"""
Property Query Constraints Extraction and Tracing

This module handles the extraction of structured constraints from user messages
and instruments the process with Langfuse observability.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.services.observability.langfuse_observer import get_observer, sanitize_for_observability
from app.services.direct_property_search import extract_city_from_message, extract_property_type_from_message

logger = logging.getLogger(__name__)


@dataclass
class Constraint:
    field: str
    operator: str
    value: Any


@dataclass
class PropertySearchQuery:
    city: Optional[str]
    property_type: Optional[str]
    constraints: List[Constraint]
    confidence: float
    unresolved: List[str]


def extract_constraints_with_tracing(
    message: str, 
    city: Optional[str] = None,
    property_type: Optional[str] = None,
    *, 
    observer: Any = None, 
    metadata: Optional[Dict[str, Any]] = None
) -> PropertySearchQuery:
    """
    Extract structured constraints and trace the process.
    Compatibility wrapper that calls existing extractors and safely traces.
    """
    if city is None:
        city = extract_city_from_message(message)
    if property_type is None:
        property_type = extract_property_type_from_message(message)
    
    constraints = []
    if city:
        constraints.append(Constraint(field="city", operator="exact", value=city))
    if property_type:
        constraints.append(Constraint(field="property_type", operator="exact", value=property_type))
        
    normalized = " ".join((message or "").strip().lower().split())
    beds_match = re.search(r'\b(\d+)\s+(?:bedroom|bed|bd|br)\b', normalized)
    if beds_match:
        beds_val = int(beds_match.group(1))
        constraints.append(Constraint(field="bedrooms", operator="exact", value=beds_val))
        
    unresolved = []
    confidence = 0.9 if constraints else 0.0
    
    result = PropertySearchQuery(
        city=city,
        property_type=property_type,
        constraints=constraints,
        confidence=confidence,
        unresolved=unresolved
    )
    
    try:
        obs = observer or get_observer()
        with obs.trace(name="property_query_constraints", metadata={
            "city": result.city,
            "property_type": result.property_type,
            "constraints": [
                {"field": c.field, "operator": c.operator, "value": c.value}
                for c in result.constraints
            ],
            "confidence": result.confidence,
            "unresolved": result.unresolved,
            **(metadata or {}),
        }):
            pass
    except Exception:
        pass
        
    return result
