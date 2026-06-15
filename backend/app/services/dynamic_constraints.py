"""
Schema-driven dynamic constraint container.

Instead of hardcoding constraint fields (city, property_type, bedrooms, etc.),
this module stores constraints as a dynamic dictionary validated against
the property schema. Constraints are discovered from user messages and
dataset metadata, not predefined in code.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Set

from app.services.property_schema import FieldSchema, FilterType, PropertySchema, get_property_schema

logger = logging.getLogger(__name__)


class DynamicConstraints:
    """
    Schema-driven constraint container.
    
    Stores constraints as {field_name: (value, operator)} pairs where:
    - field_name: Dataset column name (e.g., "city", "property_type", "bedrooms")
    - value: The constraint value (string, int, float, list)
    - operator: How to apply the constraint (exact, contains, min, max, etc.)
    
    Constraints are validated against the property schema dynamically.
    """
    
    def __init__(self, schema: Optional[PropertySchema] = None):
        """
        Initialize dynamic constraints.
        
        Args:
            schema: Property schema for validation. If None, uses global schema.
        """
        self._schema = schema or get_property_schema()
        self._constraints: Dict[str, Dict[str, Any]] = {}
    
    def set(self, field_name: str, value: Any, operator: str = "exact") -> None:
        """
        Set a constraint for a field.
        
        Args:
            field_name: Dataset column name
            value: Constraint value
            operator: Filter operator (exact, contains, min, max, range)
        
        Raises:
            ValueError: If field is not in schema or value is invalid
        """
        if not self._schema.is_searchable(field_name):
            logger.warning(f"Field '{field_name}' is not searchable in schema")
        
        field_schema = self._schema.get_field(field_name)
        if field_schema and not field_schema.validate_value(value):
            raise ValueError(
                f"Invalid value '{value}' for field '{field_name}' "
                f"(expected {field_schema.field_type.value})"
            )
        
        self._constraints[field_name] = {
            "value": value,
            "operator": operator
        }
        logger.debug(f"Set constraint: {field_name}={value} (operator={operator})")
    
    def get(self, field_name: str, default: Any = None) -> Any:
        """
        Get constraint value for a field.
        
        Args:
            field_name: Dataset column name
            default: Default value if field not set
        
        Returns:
            Constraint value or default
        """
        constraint = self._constraints.get(field_name)
        return constraint["value"] if constraint else default
    
    def get_operator(self, field_name: str, default: str = "exact") -> str:
        """
        Get operator for a field constraint.
        
        Args:
            field_name: Dataset column name
            default: Default operator if field not set
        
        Returns:
            Operator string or default
        """
        constraint = self._constraints.get(field_name)
        return constraint["operator"] if constraint else default
    
    def has(self, field_name: str) -> bool:
        """Check if a field has a constraint."""
        return field_name in self._constraints
    
    def remove(self, field_name: str) -> None:
        """Remove a constraint for a field."""
        if field_name in self._constraints:
            del self._constraints[field_name]
            logger.debug(f"Removed constraint: {field_name}")
    
    def clear(self) -> None:
        """Clear all constraints."""
        self._constraints.clear()
        logger.debug("Cleared all constraints")
    
    def is_empty(self) -> bool:
        """Check if no constraints are set."""
        return len(self._constraints) == 0
    
    def fields(self) -> Set[str]:
        """Get all field names with constraints."""
        return set(self._constraints.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with just values (no operators).
        
        Returns:
            Dict mapping field names to constraint values
        """
        return {
            field: constraint["value"]
            for field, constraint in self._constraints.items()
        }
    
    def to_full_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert to dictionary with values and operators.
        
        Returns:
            Dict mapping field names to {value, operator} dicts
        """
        return dict(self._constraints)
    
    def merge_with(
        self,
        other: 'DynamicConstraints',
        override: bool = True,
        accumulate_lists: bool = True
    ) -> 'DynamicConstraints':
        """
        Merge with another constraint set.
        
        Args:
            other: Other constraints to merge
            override: If True, new values override old. If False, only fill missing.
            accumulate_lists: If True, merge list values (e.g., amenities).
        
        Returns:
            New DynamicConstraints with merged values
        """
        merged = DynamicConstraints(schema=self._schema)
        
        for field_name, constraint in self._constraints.items():
            merged._constraints[field_name] = dict(constraint)
        
        for field_name, other_constraint in other._constraints.items():
            other_value = other_constraint["value"]
            other_operator = other_constraint["operator"]
            
            if field_name in merged._constraints:
                existing = merged._constraints[field_name]
                existing_value = existing["value"]
                
                if override:
                    if accumulate_lists and isinstance(existing_value, list) and isinstance(other_value, list):
                        merged_list = list(set(existing_value + other_value))
                        merged.set(field_name, merged_list, other_operator)
                    else:
                        merged.set(field_name, other_value, other_operator)
                else:
                    if existing_value is None or existing_value == [] or existing_value == "":
                        merged.set(field_name, other_value, other_operator)
            else:
                merged.set(field_name, other_value, other_operator)
        
        return merged
    
    def validate(self) -> Dict[str, str]:
        """
        Validate all constraints against schema.
        
        Returns:
            Dict mapping invalid field names to error messages
        """
        return self._schema.validate_constraints(self.to_dict())
    
    def get_hard_filters(self) -> Dict[str, Any]:
        """
        Get constraints that should be enforced as hard filters.
        
        Hard filters are exact matches that must be satisfied before ranking.
        Examples: city, property_type, bedrooms (exact)
        
        Returns:
            Dict of field_name -> value for hard filters
        """
        hard_filters = {}
        for field_name, constraint in self._constraints.items():
            operator = constraint["operator"]
            value = constraint["value"]
            
            if operator in ("exact", "contains", "min", "max"):
                hard_filters[field_name] = {
                    "value": value,
                    "operator": operator
                }
        
        return hard_filters
    
    def get_soft_filters(self) -> Dict[str, Any]:
        """
        Get constraints that should be used for ranking, not filtering.
        
        Soft filters influence ranking but don't exclude results.
        Examples: fuzzy text matches, preferences
        
        Returns:
            Dict of field_name -> value for soft filters
        """
        soft_filters = {}
        for field_name, constraint in self._constraints.items():
            operator = constraint["operator"]
            value = constraint["value"]
            
            if operator == "fuzzy":
                soft_filters[field_name] = {
                    "value": value,
                    "operator": operator
                }
        
        return soft_filters
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self._constraints)
    
    def __len__(self) -> int:
        """Number of constraints."""
        return len(self._constraints)
    
    def __repr__(self) -> str:
        """String representation."""
        constraints_str = ", ".join(
            f"{field}={c['value']}({c['operator']})"
            for field, c in self._constraints.items()
        )
        return f"DynamicConstraints({constraints_str})"
    
    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, DynamicConstraints):
            return False
        return self._constraints == other._constraints


def create_constraints_from_dict(
    data: Dict[str, Any],
    schema: Optional[PropertySchema] = None
) -> DynamicConstraints:
    """
    Create DynamicConstraints from a plain dictionary.
    
    Args:
        data: Dict mapping field names to values
        schema: Optional property schema
    
    Returns:
        DynamicConstraints instance
    """
    constraints = DynamicConstraints(schema=schema)
    for field_name, value in data.items():
        if value is None:
            continue

        if isinstance(value, dict) and "value" in value:
            operator = str(value.get("operator") or "exact")
            constraints.set(field_name, value.get("value"), operator)
            continue

        field_schema = constraints._schema.get_field(field_name)
        if field_schema:
            if field_schema.filter_type == FilterType.RANGE:
                operator = "max" if "price" in field_name else "exact"
            elif field_schema.filter_type == FilterType.CONTAINS:
                operator = "contains"
            else:
                operator = "exact"
        else:
            operator = "exact"

        constraints.set(field_name, value, operator)
    
    return constraints
