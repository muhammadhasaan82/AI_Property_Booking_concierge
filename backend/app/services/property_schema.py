"""
Dynamic property schema discovery from dataset metadata.

Instead of hardcoding searchable fields, this module discovers them from
the actual dataset columns and their types. This makes the system adaptable
to schema changes without code modifications.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Dataset field types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    LIST = "list"  # semicolon/comma-separated values
    DATE = "date"
    BOOLEAN = "boolean"


class FilterType(Enum):
    """How a field can be filtered."""
    EXACT = "exact"  # exact match
    CONTAINS = "contains"  # substring or list membership
    RANGE = "range"  # min/max comparisons
    FUZZY = "fuzzy"  # fuzzy text matching


@dataclass
class FieldSchema:
    """Schema for a single dataset field."""
    name: str
    field_type: FieldType
    filter_type: FilterType
    searchable: bool = True
    rankable: bool = False
    aliases: List[str] = field(default_factory=list)
    sample_values: List[Any] = field(default_factory=list)
    
    def validate_value(self, value: Any) -> bool:
        """Validate a value against this field's schema."""
        if value is None:
            return True
        
        try:
            if self.field_type == FieldType.INTEGER:
                int(value)
                return True
            elif self.field_type == FieldType.FLOAT:
                float(value)
                return True
            elif self.field_type == FieldType.LIST:
                return isinstance(value, (list, str))
            elif self.field_type == FieldType.STRING:
                return isinstance(value, str)
            elif self.field_type == FieldType.BOOLEAN:
                return isinstance(value, bool) or value in ("true", "false", "0", "1")
            return True
        except (ValueError, TypeError):
            return False


class PropertySchema:
    """
    Dynamic property schema derived from dataset metadata.
    
    Discovers searchable fields, their types, and filter capabilities
    directly from the dataset rather than hardcoding them.
    """
    
    def __init__(self, dataset_path: Optional[Path] = None):
        self._fields: Dict[str, FieldSchema] = {}
        self._searchable_fields: List[str] = []
        self._rankable_fields: List[str] = []
        self._loaded = False
        
        if dataset_path:
            self.load_from_dataset(dataset_path)
    
    def load_from_dataset(self, dataset_path: Path) -> None:
        """
        Load schema by analyzing dataset columns.
        
        Args:
            dataset_path: Path to the CSV dataset file
        """
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            return
        
        logger.info(f"Loading property schema from {dataset_path}")
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            
            # Sample rows for type inference
            sample_rows = []
            for i, row in enumerate(reader):
                if i >= 100:  # Sample first 100 rows
                    break
                sample_rows.append(row)
        
        # Infer schema for each column
        for column in columns:
            field_schema = self._infer_field_schema(column, sample_rows)
            if field_schema and field_schema.searchable:
                self._fields[column] = field_schema
                self._searchable_fields.append(column)
                if field_schema.rankable:
                    self._rankable_fields.append(column)
        
        self._loaded = True
        logger.info(
            f"Loaded {len(self._searchable_fields)} searchable fields: "
            f"{', '.join(self._searchable_fields)}"
        )
    
    def _infer_field_schema(self, column: str, sample_rows: List[Dict[str, str]]) -> Optional[FieldSchema]:
        """Infer field schema from sample data."""
        
        # Extract sample values
        values = [row.get(column, "") for row in sample_rows if row.get(column)]
        if not values:
            return None
        
        # Take first 10 non-empty values for analysis
        sample_values = values[:10]
        
        # Skip internal IDs and metadata
        skip_columns = {"id", "property_id", "host_id", "created_at", "updated_at", "image_url", "images"}
        if column.lower() in skip_columns:
            return FieldSchema(
                name=column,
                field_type=FieldType.STRING,
                filter_type=FilterType.EXACT,
                searchable=False
            )
        
        # Infer type
        field_type = self._infer_type(sample_values)
        filter_type = self._infer_filter_type(column, field_type, sample_values)
        
        # Determine if searchable
        searchable = self._is_searchable(column, field_type)
        
        # Determine if rankable (numeric fields used for sorting)
        rankable = field_type in (FieldType.INTEGER, FieldType.FLOAT) and column in {
            "price_per_night", "rating", "reviews_count", "bedrooms", "bathrooms"
        }
        
        return FieldSchema(
            name=column,
            field_type=field_type,
            filter_type=filter_type,
            searchable=searchable,
            rankable=rankable,
            sample_values=sample_values[:5]
        )
    
    def _infer_type(self, values: List[str]) -> FieldType:
        """Infer field type from sample values."""
        
        # Check for list (semicolon or comma separated)
        if any(";" in v or ("," in v and " " not in v) for v in values):
            return FieldType.LIST
        
        # Check for integer
        try:
            [int(v) for v in values if v]
            return FieldType.INTEGER
        except ValueError:
            pass
        
        # Check for float
        try:
            [float(v) for v in values if v]
            return FieldType.FLOAT
        except ValueError:
            pass
        
        # Check for date patterns
        date_patterns = ["20", "19", "-"]
        if all(any(p in v for p in date_patterns) and len(v) <= 10 for v in values):
            return FieldType.DATE
        
        # Check for boolean
        bool_values = {"true", "false", "yes", "no", "0", "1"}
        if all(v.lower() in bool_values for v in values):
            return FieldType.BOOLEAN
        
        # Default to string
        return FieldType.STRING
    
    def _infer_filter_type(self, column: str, field_type: FieldType, values: List[str]) -> FilterType:
        """Infer filter type based on column name and type."""
        
        # Range filters for numeric fields
        if field_type in (FieldType.INTEGER, FieldType.FLOAT):
            range_columns = {"price_per_night", "rating", "reviews_count", "bedrooms", "bathrooms", "guests", "occupancy"}
            if any(col in column.lower() for col in range_columns):
                return FilterType.RANGE
        
        # Contains filter for list fields
        if field_type == FieldType.LIST:
            return FilterType.CONTAINS
        
        # Fuzzy filter for text fields
        if field_type == FieldType.STRING:
            text_columns = {"title", "description", "city", "neighborhood", "property_type"}
            if any(col in column.lower() for col in text_columns):
                return FilterType.FUZZY
        
        # Default to exact match
        return FilterType.EXACT
    
    def _is_searchable(self, column: str, field_type: FieldType) -> bool:
        """Determine if a field should be searchable."""
        
        # Always searchable fields
        searchable_keywords = {
            "city", "location", "property_type", "title", "description",
            "bedrooms", "bathrooms", "price", "amenities", "features",
            "rating", "guests", "occupancy", "neighborhood"
        }
        
        if any(keyword in column.lower() for keyword in searchable_keywords):
            return True
        
        # Numeric and list fields are usually searchable
        if field_type in (FieldType.INTEGER, FieldType.FLOAT, FieldType.LIST):
            return True
        
        return False
    
    def get_field(self, name: str) -> Optional[FieldSchema]:
        """Get schema for a specific field."""
        return self._fields.get(name)
    
    def get_searchable_fields(self) -> List[str]:
        """Get all searchable field names."""
        return list(self._searchable_fields)
    
    def get_rankable_fields(self) -> List[str]:
        """Get all rankable field names."""
        return list(self._rankable_fields)
    
    def get_all_fields(self) -> Dict[str, FieldSchema]:
        """Get all field schemas."""
        return dict(self._fields)
    
    def is_searchable(self, field_name: str) -> bool:
        """Check if a field is searchable."""
        return field_name in self._searchable_fields
    
    def validate_constraints(self, constraints: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate constraints against schema.
        
        Returns:
            Dict mapping invalid field names to error messages
        """
        errors = {}
        for field_name, value in constraints.items():
            schema = self.get_field(field_name)
            if not schema:
                errors[field_name] = f"Unknown field: {field_name}"
            elif not schema.validate_value(value):
                errors[field_name] = f"Invalid value for {field_name}: {value}"
        return errors


# Global singleton instance
_schema_instance: Optional[PropertySchema] = None


def get_property_schema(dataset_path: Optional[Path] = None) -> PropertySchema:
    """Get or create the global property schema instance."""
    global _schema_instance
    
    if _schema_instance is None:
        if dataset_path is None:
            # Default dataset path
            dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset.csv"
        _schema_instance = PropertySchema(dataset_path)
    
    return _schema_instance


def reload_schema(dataset_path: Optional[Path] = None) -> PropertySchema:
    """Force reload of the property schema."""
    global _schema_instance
    _schema_instance = None
    return get_property_schema(dataset_path)
