"""
Schema-driven constraint extraction using dynamic property schema.

This module provides a schema-driven interface for extracting and merging
property search constraints from user messages. It uses DynamicConstraints
and SearchPlanner to avoid hardcoded field names and adapt to schema changes.

Backward Compatibility:
    The ExtractedConstraints dataclass is maintained for backward compatibility
    with existing code. New code should use DynamicConstraints directly.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

from app.services.dynamic_constraints import DynamicConstraints, create_constraints_from_dict
from app.services.search_planner import SearchPlanner, SearchPlan, get_search_planner
from app.services.property_schema import get_property_schema

# Backward compatibility imports
from app.services.property_type_normalizer import normalize_property_type
from app.services.direct_property_search import (
    resolve_supported_city_from_message,
    extract_property_type_from_message,
)
from app.services.property_query_constraints import extract_property_search_query

logger = logging.getLogger(__name__)


@dataclass
class ExtractedConstraints:
    """
    Structured constraints extracted from user message.
    
    DEPRECATED: Use DynamicConstraints for new code.
    This class is maintained for backward compatibility.
    """
    city: Optional[str] = None
    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    bedrooms_operator: str = "exact"  # exact, min, max
    bathrooms: Optional[int] = None
    price_max: Optional[float] = None
    price_min: Optional[float] = None
    guests: Optional[int] = None
    amenities: List[str] = field(default_factory=list)
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    rating_min: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and empty lists."""
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None and (not isinstance(v, list) or len(v) > 0)
        }
    
    def to_dynamic_constraints(self) -> DynamicConstraints:
        """Convert to DynamicConstraints for schema-driven processing."""
        return create_constraints_from_dict(self.to_dict())
    
    def merge_with(self, other: 'ExtractedConstraints', override: bool = True) -> 'ExtractedConstraints':
        """
        Merge with another constraint set.
        
        Args:
            other: New constraints to merge
            override: If True, new non-None values override old. If False, only fill missing.
        
        Returns:
            Merged constraints
        """
        merged = ExtractedConstraints()
        
        for field_name in self.__dataclass_fields__:
            old_value = getattr(self, field_name)
            new_value = getattr(other, field_name)
            
            # Special handling for lists (amenities)
            if field_name == "amenities":
                # Always merge amenities (union) and remove duplicates
                if old_value or new_value:
                    merged_amenities = list(set((old_value or []) + (new_value or [])))
                    setattr(merged, field_name, merged_amenities)
                else:
                    setattr(merged, field_name, [])
            else:
                # Standard field handling
                if override:
                    # New non-None values override old
                    if new_value is not None:
                        setattr(merged, field_name, new_value)
                    else:
                        setattr(merged, field_name, old_value)
                else:
                    # Only fill missing values
                    if old_value is None:
                        setattr(merged, field_name, new_value)
                    else:
                        setattr(merged, field_name, old_value)
        
        return merged
    
    def has_any_constraints(self) -> bool:
        """Check if any constraints are set."""
        # Metadata fields that don't count as constraints
        metadata_fields = {"bedrooms_operator"}
        
        for field_name in self.__dataclass_fields__:
            if field_name in metadata_fields:
                continue
            if field_name == "amenities":
                if self.amenities:
                    return True
            else:
                if getattr(self, field_name) is not None:
                    return True
        return False
    
    def is_empty(self) -> bool:
        """Check if no constraints are set (inverse of has_any_constraints)."""
        return not self.has_any_constraints()


def _overlay_legacy_query_constraints(
    message: str,
    constraints: ExtractedConstraints,
) -> ExtractedConstraints:
    """
    Compatibility bridge for old ExtractedConstraints callers.

    Extraction stays schema/planner-driven. This only maps the newer
    PropertySearchQuery shape back to the legacy dataclass fields used by
    existing tests and older code paths.
    """
    try:
        legacy_query = extract_property_search_query(message)
    except Exception as exc:
        logger.debug("[constraint_extractor] legacy overlay skipped: %s", exc)
        return constraints

    if legacy_query.city and not constraints.city:
        constraints.city = legacy_query.city

    if legacy_query.property_type and not constraints.property_type:
        constraints.property_type = str(legacy_query.property_type).strip().lower()

    for item in legacy_query.constraints:
        field = item.field
        operator = item.operator
        value = item.value

        if field == "price_per_night":
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if operator == "min":
                constraints.price_min = numeric_value
            else:
                constraints.price_max = numeric_value

        elif field == "occupancy_max":
            try:
                constraints.guests = int(float(value))
            except (TypeError, ValueError):
                continue

        elif field == "amenities":
            if value not in constraints.amenities:
                constraints.amenities.append(value)

    return constraints


def extract_constraints_from_message(
    message: str,
    previous_constraints: Optional[ExtractedConstraints] = None,
) -> ExtractedConstraints:
    """
    Extract constraints from user message using schema-driven approach.
    
    This function now uses SearchPlanner for dynamic extraction while
    maintaining backward compatibility with ExtractedConstraints.
    
    Args:
        message: User's natural language message
        previous_constraints: Constraints from previous turns to merge with
    
    Returns:
        ExtractedConstraints with all detected constraints
    """
    # Use schema-driven extraction via SearchPlanner
    planner = get_search_planner()
    
    # Convert previous constraints to DynamicConstraints if provided
    previous_dynamic = None
    if previous_constraints:
        previous_dynamic = previous_constraints.to_dynamic_constraints()
    
    # Create search plan (this extracts and merges constraints)
    plan = planner.plan_search(
        message=message,
        session_constraints=previous_dynamic,
        session_id="extraction"
    )
    
    # Log the extraction trace
    plan.trace.log(f"Extracted constraints from message: '{message}'")
    logger.info(
        "[constraint_extractor] schema-driven extraction: %s",
        plan.trace.extracted_constraints
    )
    
    # Convert DynamicConstraints back to ExtractedConstraints for backward compatibility
    constraints = ExtractedConstraints()
    dynamic_dict = plan.constraints.to_dict()
    
    # Map dynamic constraints to ExtractedConstraints fields
    if "city" in dynamic_dict:
        constraints.city = dynamic_dict["city"]
    if "property_type" in dynamic_dict:
        constraints.property_type = dynamic_dict["property_type"]
    if "bedrooms" in dynamic_dict:
        constraints.bedrooms = dynamic_dict["bedrooms"]
        # Get operator from dynamic constraints
        operator = plan.constraints.get_operator("bedrooms", "exact")
        constraints.bedrooms_operator = operator
    if "bathrooms" in dynamic_dict:
        constraints.bathrooms = dynamic_dict["bathrooms"]
    if "price_per_night" in dynamic_dict:
        # Determine if max or min based on operator
        operator = plan.constraints.get_operator("price_per_night", "exact")
        if operator == "max":
            constraints.price_max = dynamic_dict["price_per_night"]
        elif operator == "min":
            constraints.price_min = dynamic_dict["price_per_night"]
    if "guests" in dynamic_dict:
        constraints.guests = dynamic_dict["guests"]
    if "amenities" in dynamic_dict:
        amenities = dynamic_dict["amenities"]
        if isinstance(amenities, list):
            constraints.amenities = amenities
        else:
            constraints.amenities = [amenities]
    
    constraints = _overlay_legacy_query_constraints(message, constraints)

    logger.info(
        "[constraint_extractor] final constraints: %s",
        constraints.to_dict()
    )
    
    if previous_constraints:
        logger.info(
            "[constraint_extractor] merged: previous=%s, new=%s, merged=%s",
            previous_constraints.to_dict(),
            plan.trace.extracted_constraints,
            constraints.to_dict()
        )
    
    return constraints


def extract_dynamic_constraints(
    message: str,
    session_constraints: Optional[DynamicConstraints] = None,
    session_id: Optional[str] = None
) -> tuple[DynamicConstraints, SearchPlan]:
    """
    Extract constraints using schema-driven approach (no hardcoded fields).
    
    This is the preferred method for new code. Returns DynamicConstraints
    that adapt to schema changes automatically.
    
    Args:
        message: User's natural language message
        session_constraints: Previous constraints from session
        session_id: Session identifier for logging
    
    Returns:
        Tuple of (DynamicConstraints, SearchPlan with trace)
    """
    planner = get_search_planner()
    plan = planner.plan_search(
        message=message,
        session_constraints=session_constraints,
        session_id=session_id
    )
    
    logger.info(
        "[constraint_extractor] dynamic extraction: message='%s', constraints=%s",
        message,
        plan.constraints.to_dict()
    )
    
    return plan.constraints, plan


def constraints_to_search_kwargs(constraints: ExtractedConstraints) -> Dict[str, Any]:
    """
    Convert constraints to search_properties() kwargs.
    
    DEPRECATED: Use dynamic_constraints_to_search_kwargs() for schema-driven approach.
    
    Args:
        constraints: Extracted constraints
    
    Returns:
        Dictionary of kwargs for search_properties()
    """
    kwargs = {}
    
    if constraints.city:
        kwargs["city"] = constraints.city
    if constraints.property_type:
        kwargs["property_type"] = constraints.property_type
    if constraints.bedrooms is not None:
        # Map operator to appropriate parameter
        if constraints.bedrooms_operator == "min":
            kwargs["beds"] = constraints.bedrooms
        elif constraints.bedrooms_operator == "exact":
            kwargs["beds"] = constraints.bedrooms
        # Note: max operator not currently supported by search_properties
    if constraints.price_max is not None:
        kwargs["budget"] = constraints.price_max
    if constraints.amenities:
        kwargs["amenities"] = ",".join(constraints.amenities)
    
    return kwargs


def dynamic_constraints_to_search_kwargs(constraints: DynamicConstraints) -> Dict[str, Any]:
    """
    Convert DynamicConstraints to search_properties() kwargs (schema-driven).
    
    This function dynamically maps constraint fields to search parameters
    based on the property schema, without hardcoded field names.
    
    Args:
        constraints: DynamicConstraints instance
    
    Returns:
        Dictionary of kwargs for search_properties()
    """
    kwargs = {}
    
    # Map all constraints dynamically
    for field_name in constraints.fields():
        value = constraints.get(field_name)
        operator = constraints.get_operator(field_name)
        
        if value is None:
            continue
        
        # Map field names to search parameter names
        # This mapping can be extended based on schema metadata
        param_mapping = {
            "city": "city",
            "property_type": "property_type",
            "bedrooms": "beds",
            "bathrooms": "baths",
            "price_per_night": "budget" if operator == "max" else "min_price",
            "guests": "guests",
            "amenities": "amenities",
            "rating": "min_rating" if operator == "min" else "rating",
            "occupancy_max": "guests",
        }
        
        param_name = param_mapping.get(field_name, field_name)
        
        # Handle list values (e.g., amenities)
        if isinstance(value, list):
            if param_name == "amenities":
                kwargs[param_name] = ",".join(value)
            else:
                kwargs[param_name] = value
        else:
            kwargs[param_name] = value
    
    return kwargs


def constraints_from_soft_state(soft_state: Dict[str, Any]) -> ExtractedConstraints:
    """
    Extract constraints from session soft_state.
    
    DEPRECATED: Use dynamic_constraints_from_soft_state() for schema-driven approach.
    
    Args:
        soft_state: Session soft_state dictionary
    
    Returns:
        ExtractedConstraints from stored filters
    """
    last_filters = soft_state.get("last_filters", {})
    
    return ExtractedConstraints(
        city=last_filters.get("city"),
        property_type=last_filters.get("property_type"),
        bedrooms=last_filters.get("bedrooms"),
        bathrooms=last_filters.get("bathrooms"),
        price_max=last_filters.get("budget"),
        guests=last_filters.get("guests"),
        amenities=last_filters.get("amenities", []),
    )


def dynamic_constraints_from_soft_state(soft_state: Dict[str, Any]) -> DynamicConstraints:
    """
    Extract constraints from session soft_state (schema-driven).
    
    This function dynamically extracts all available constraint fields
    from the soft_state without hardcoded field names.
    
    Args:
        soft_state: Session soft_state dictionary
    
    Returns:
        DynamicConstraints from stored filters
    """
    stored_dynamic = soft_state.get("last_dynamic_constraints")
    if isinstance(stored_dynamic, dict) and stored_dynamic:
        constraints = create_constraints_from_dict(stored_dynamic)
        logger.debug(
            "[constraint_extractor] extracted dynamic constraints from soft_state: %s",
            constraints.to_full_dict()
        )
        return constraints

    last_filters = soft_state.get("last_filters", {})
    normalized_filters: Dict[str, Any] = {}
    if isinstance(last_filters, dict):
        if last_filters.get("city"):
            normalized_filters["city"] = last_filters["city"]
        if last_filters.get("property_type"):
            normalized_filters["property_type"] = last_filters["property_type"]
        if last_filters.get("bedrooms") is not None:
            normalized_filters["bedrooms"] = {
                "value": last_filters.get("bedrooms"),
                "operator": last_filters.get("bedrooms_operator") or "exact",
            }
        if last_filters.get("bathrooms") is not None:
            normalized_filters["bathrooms"] = {
                "value": last_filters.get("bathrooms"),
                "operator": last_filters.get("bathrooms_operator") or "exact",
            }
        if last_filters.get("budget") is not None:
            normalized_filters["price_per_night"] = {
                "value": last_filters.get("budget"),
                "operator": "max",
            }
        if last_filters.get("guests") is not None:
            normalized_filters["occupancy_max"] = {
                "value": last_filters.get("guests"),
                "operator": last_filters.get("guests_operator") or "min",
            }
        if last_filters.get("amenities"):
            normalized_filters["amenities"] = {
                "value": list(last_filters.get("amenities") or []),
                "operator": "contains",
            }

    constraints = create_constraints_from_dict(normalized_filters)
    
    logger.debug(
        "[constraint_extractor] extracted from soft_state: %s",
        constraints.to_full_dict()
    )
    
    return constraints
