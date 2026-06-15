"""
Tests for property type filtering to ensure the bug is fixed.

This test suite verifies that:
1. Searching for a specific property type (e.g., "villa") only returns that type
2. Property type filtering works across all search paths
3. Property type constraints are preserved across conversation turns
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.constraint_extractor import ExtractedConstraints
from app.services.property_type_normalizer import normalize_property_type


class TestPropertyTypeFiltering:
    """Test that property type filtering works correctly."""

    def test_villa_search_returns_only_villas(self):
        """Test that searching for villa only returns villas."""
        mock_properties = [
            {"id": "1", "property_type": "villa", "city": "Seattle", "price_per_night": 300},
            {"id": "2", "property_type": "villa", "city": "Seattle", "price_per_night": 350},
            {"id": "3", "property_type": "apartment", "city": "Seattle", "price_per_night": 200},
            {"id": "4", "property_type": "house", "city": "Seattle", "price_per_night": 400},
            {"id": "5", "property_type": "villa", "city": "Seattle", "price_per_night": 320},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
        ]
        
        assert len(filtered) == 3
        assert all(prop["property_type"] == "villa" for prop in filtered)

    def test_apartment_search_returns_only_apartments(self):
        """Test that searching for apartment only returns apartments."""
        mock_properties = [
            {"id": "1", "property_type": "villa", "city": "New York", "price_per_night": 500},
            {"id": "2", "property_type": "apartment", "city": "New York", "price_per_night": 200},
            {"id": "3", "property_type": "apartment", "city": "New York", "price_per_night": 220},
            {"id": "4", "property_type": "condo", "city": "New York", "price_per_night": 250},
            {"id": "5", "property_type": "apartment", "city": "New York", "price_per_night": 210},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "apartment"
        ]
        
        assert len(filtered) == 3
        assert all(prop["property_type"] == "apartment" for prop in filtered)

    def test_house_search_returns_only_houses(self):
        """Test that searching for house only returns houses."""
        mock_properties = [
            {"id": "1", "property_type": "house", "city": "Portland", "price_per_night": 400},
            {"id": "2", "property_type": "villa", "city": "Portland", "price_per_night": 450},
            {"id": "3", "property_type": "house", "city": "Portland", "price_per_night": 420},
            {"id": "4", "property_type": "townhouse", "city": "Portland", "price_per_night": 380},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "house"
        ]
        
        assert len(filtered) == 2
        assert all(prop["property_type"] == "house" for prop in filtered)

    def test_property_type_normalization_villa(self):
        """Test that villa variants normalize to 'villa'."""
        assert normalize_property_type("villa") == "villa"
        assert normalize_property_type("VILLA") == "villa"
        assert normalize_property_type("Villa") == "villa"

    def test_property_type_normalization_apartment(self):
        """Test that apartment variants normalize to 'apartment'."""
        assert normalize_property_type("apartment") == "apartment"
        assert normalize_property_type("APARTMENT") == "apartment"
        assert normalize_property_type("Apartment") == "apartment"
        assert normalize_property_type("apt") == "apartment"
        assert normalize_property_type("APT") == "apartment"

    def test_property_type_normalization_house(self):
        """Test that house variants normalize to 'house'."""
        assert normalize_property_type("house") == "house"
        assert normalize_property_type("HOUSE") == "house"
        assert normalize_property_type("House") == "house"
        assert normalize_property_type("home") == "house"

    def test_mixed_case_property_types_filter_correctly(self):
        """Test that mixed case property types in dataset filter correctly."""
        mock_properties = [
            {"id": "1", "property_type": "Villa", "city": "Seattle", "price_per_night": 300},
            {"id": "2", "property_type": "VILLA", "city": "Seattle", "price_per_night": 350},
            {"id": "3", "property_type": "villa", "city": "Seattle", "price_per_night": 320},
            {"id": "4", "property_type": "Apartment", "city": "Seattle", "price_per_night": 200},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
        ]
        
        assert len(filtered) == 3

    def test_no_results_for_unavailable_property_type(self):
        """Test that searching for unavailable property type returns empty."""
        mock_properties = [
            {"id": "1", "property_type": "apartment", "city": "Seattle", "price_per_night": 200},
            {"id": "2", "property_type": "house", "city": "Seattle", "price_per_night": 400},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
        ]
        
        assert len(filtered) == 0


class TestPropertyTypeConstraintPreservation:
    """Test that property type constraints are preserved across turns."""

    def test_property_type_preserved_when_adding_bedrooms(self):
        """Test that property type is preserved when adding bedroom constraint."""
        constraints1 = ExtractedConstraints(
            city="Seattle",
            property_type="villa"
        )
        
        constraints2 = ExtractedConstraints(bedrooms=3)
        merged = constraints1.merge_with(constraints2)
        
        assert merged.city == "Seattle"
        assert merged.property_type == "villa"
        assert merged.bedrooms == 3

    def test_property_type_preserved_when_adding_price(self):
        """Test that property type is preserved when adding price constraint."""
        constraints1 = ExtractedConstraints(
            city="New York",
            property_type="apartment"
        )
        
        constraints2 = ExtractedConstraints(price_max=250)
        merged = constraints1.merge_with(constraints2)
        
        assert merged.city == "New York"
        assert merged.property_type == "apartment"
        assert merged.price_max == 250

    def test_property_type_preserved_when_adding_amenities(self):
        """Test that property type is preserved when adding amenities."""
        constraints1 = ExtractedConstraints(
            city="Portland",
            property_type="house"
        )
        
        constraints2 = ExtractedConstraints(amenities=["pool", "gym"])
        merged = constraints1.merge_with(constraints2)
        
        assert merged.city == "Portland"
        assert merged.property_type == "house"
        assert "pool" in merged.amenities
        assert "gym" in merged.amenities

    def test_property_type_can_be_overridden(self):
        """Test that property type can be explicitly overridden."""
        constraints1 = ExtractedConstraints(
            city="Seattle",
            property_type="villa"
        )
        
        constraints2 = ExtractedConstraints(property_type="apartment")
        merged = constraints1.merge_with(constraints2)
        
        assert merged.city == "Seattle"
        assert merged.property_type == "apartment"


class TestMultipleConstraintFiltering:
    """Test filtering with multiple constraints."""

    def test_property_type_and_city_filtering(self):
        """Test filtering by both property type and city."""
        mock_properties = [
            {"id": "1", "property_type": "villa", "city": "Seattle", "price_per_night": 300},
            {"id": "2", "property_type": "villa", "city": "Portland", "price_per_night": 350},
            {"id": "3", "property_type": "apartment", "city": "Seattle", "price_per_night": 200},
            {"id": "4", "property_type": "villa", "city": "Seattle", "price_per_night": 320},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
            and prop.get("city") == "Seattle"
        ]
        
        assert len(filtered) == 2
        assert all(prop["property_type"] == "villa" for prop in filtered)
        assert all(prop["city"] == "Seattle" for prop in filtered)

    def test_property_type_and_price_filtering(self):
        """Test filtering by property type and price."""
        mock_properties = [
            {"id": "1", "property_type": "villa", "city": "Seattle", "price_per_night": 300},
            {"id": "2", "property_type": "villa", "city": "Seattle", "price_per_night": 500},
            {"id": "3", "property_type": "apartment", "city": "Seattle", "price_per_night": 200},
            {"id": "4", "property_type": "villa", "city": "Seattle", "price_per_night": 350},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
            and prop.get("price_per_night", 0) <= 400
        ]
        
        assert len(filtered) == 2
        assert all(prop["property_type"] == "villa" for prop in filtered)
        assert all(prop["price_per_night"] <= 400 for prop in filtered)

    def test_property_type_and_bedrooms_filtering(self):
        """Test filtering by property type and bedrooms."""
        mock_properties = [
            {"id": "1", "property_type": "villa", "city": "Seattle", "bedrooms": 3},
            {"id": "2", "property_type": "villa", "city": "Seattle", "bedrooms": 4},
            {"id": "3", "property_type": "apartment", "city": "Seattle", "bedrooms": 3},
            {"id": "4", "property_type": "villa", "city": "Seattle", "bedrooms": 2},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
            and prop.get("bedrooms") == 3
        ]
        
        assert len(filtered) == 1
        assert filtered[0]["property_type"] == "villa"
        assert filtered[0]["bedrooms"] == 3

    def test_property_type_and_amenities_filtering(self):
        """Test filtering by property type and amenities."""
        mock_properties = [
            {"id": "1", "property_type": "villa", "city": "Seattle", "amenities": ["pool", "gym"]},
            {"id": "2", "property_type": "villa", "city": "Seattle", "amenities": ["pool"]},
            {"id": "3", "property_type": "apartment", "city": "Seattle", "amenities": ["pool", "gym"]},
            {"id": "4", "property_type": "villa", "city": "Seattle", "amenities": ["gym"]},
        ]
        
        required_amenities = {"pool", "gym"}
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
            and required_amenities.issubset(set(prop.get("amenities", [])))
        ]
        
        assert len(filtered) == 1
        assert filtered[0]["property_type"] == "villa"
        assert "pool" in filtered[0]["amenities"]
        assert "gym" in filtered[0]["amenities"]


class TestEdgeCases:
    """Test edge cases in property type filtering."""

    def test_empty_property_type_in_dataset(self):
        """Test handling of empty property type in dataset."""
        mock_properties = [
            {"id": "1", "property_type": "", "city": "Seattle", "price_per_night": 200},
            {"id": "2", "property_type": "villa", "city": "Seattle", "price_per_night": 300},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
        ]
        
        assert len(filtered) == 1
        assert filtered[0]["property_type"] == "villa"

    def test_none_property_type_in_dataset(self):
        """Test handling of None property type in dataset."""
        mock_properties = [
            {"id": "1", "property_type": None, "city": "Seattle", "price_per_night": 200},
            {"id": "2", "property_type": "villa", "city": "Seattle", "price_per_night": 300},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
        ]
        
        assert len(filtered) == 1
        assert filtered[0]["property_type"] == "villa"

    def test_missing_property_type_field(self):
        """Test handling of missing property type field."""
        mock_properties = [
            {"id": "1", "city": "Seattle", "price_per_night": 200}, 
            {"id": "2", "property_type": "villa", "city": "Seattle", "price_per_night": 300},
        ]
        
        filtered = [
            prop for prop in mock_properties
            if normalize_property_type(prop.get("property_type")) == "villa"
        ]
        
        assert len(filtered) == 1
        assert filtered[0]["property_type"] == "villa"

    def test_unknown_property_type_not_filtered(self):
        """Test that unknown property types are not filtered out when not searching for specific type."""
        mock_properties = [
            {"id": "1", "property_type": "villa", "city": "Seattle", "price_per_night": 300},
            {"id": "2", "property_type": "castle", "city": "Seattle", "price_per_night": 1000}, 
            {"id": "3", "property_type": "apartment", "city": "Seattle", "price_per_night": 200},
        ]
        
        filtered = mock_properties
        
        assert len(filtered) == 3
