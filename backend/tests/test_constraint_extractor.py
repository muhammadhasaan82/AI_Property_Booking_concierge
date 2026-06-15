"""
Unit tests for the constraint extractor module.

Tests schema-driven constraint extraction from user messages and
constraint merging across conversation turns.
"""
import pytest
from app.services.constraint_extractor import (
    ExtractedConstraints,
    extract_constraints_from_message,
)


def merge_constraints(previous: ExtractedConstraints, current: ExtractedConstraints) -> ExtractedConstraints:
    """Helper function to merge constraints using the merge_with method."""
    return previous.merge_with(current)


class TestExtractedConstraints:
    """Test the ExtractedConstraints dataclass."""

    def test_extracted_constraints_initialization(self):
        """Test that ExtractedConstraints initializes with default values."""
        constraints = ExtractedConstraints()
        assert constraints.city is None
        assert constraints.property_type is None
        assert constraints.bedrooms is None
        assert constraints.bathrooms is None
        assert constraints.price_max is None
        assert constraints.price_min is None
        assert constraints.guests is None
        assert constraints.amenities == []
        assert constraints.check_in is None
        assert constraints.check_out is None
        assert constraints.rating_min is None

    def test_extracted_constraints_with_values(self):
        """Test that ExtractedConstraints can be initialized with values."""
        constraints = ExtractedConstraints(
            city="Seattle",
            property_type="villa",
            bedrooms=3,
            bathrooms=2,
            price_max=300,
            guests=4,
            amenities=["pool", "gym"],
        )
        assert constraints.city == "Seattle"
        assert constraints.property_type == "villa"
        assert constraints.bedrooms == 3
        assert constraints.bathrooms == 2
        assert constraints.price_max == 300
        assert constraints.guests == 4
        assert constraints.amenities == ["pool", "gym"]

    def test_to_dict_excludes_none_values(self):
        """Test that to_dict excludes None values."""
        constraints = ExtractedConstraints(
            city="Seattle",
            property_type="villa",
            bedrooms=3,
        )
        result = constraints.to_dict()
        assert "city" in result
        assert "property_type" in result
        assert "bedrooms" in result
        assert "bathrooms" not in result
        assert "price_max" not in result

    def test_is_empty_with_no_constraints(self):
        """Test is_empty returns True when no constraints are set."""
        constraints = ExtractedConstraints()
        assert constraints.is_empty() is True

    def test_is_empty_with_constraints(self):
        """Test is_empty returns False when constraints are set."""
        constraints = ExtractedConstraints(city="Seattle")
        assert constraints.is_empty() is False


class TestExtractConstraintsFromMessage:
    """Test constraint extraction from user messages."""

    def test_extract_city_only(self):
        """Test extracting city from message."""
        constraints = extract_constraints_from_message("I'm looking for a place in Seattle")
        assert constraints.city == "Seattle"

    def test_extract_property_type_villa(self):
        """Test extracting villa property type."""
        constraints = extract_constraints_from_message("I want a villa")
        assert constraints.property_type == "villa"

    def test_extract_property_type_apartment(self):
        """Test extracting apartment property type."""
        constraints = extract_constraints_from_message("Looking for an apartment")
        assert constraints.property_type == "apartment"

    def test_extract_property_type_house(self):
        """Test extracting house property type."""
        constraints = extract_constraints_from_message("Need a house")
        assert constraints.property_type == "house"

    def test_extract_city_and_property_type(self):
        """Test extracting both city and property type."""
        constraints = extract_constraints_from_message("villa in Seattle")
        assert constraints.city == "Seattle"
        assert constraints.property_type == "villa"

    def test_extract_bedrooms(self):
        """Test extracting bedroom count."""
        constraints = extract_constraints_from_message("I need 3 bedrooms")
        assert constraints.bedrooms == 3

    def test_extract_bathrooms(self):
        """Test extracting bathroom count."""
        constraints = extract_constraints_from_message("with 2 bathrooms")
        assert constraints.bathrooms == 2

    def test_extract_price_max(self):
        """Test extracting maximum price."""
        constraints = extract_constraints_from_message("under $300 per night")
        assert constraints.price_max == 300

    def test_extract_guests(self):
        """Test extracting guest count."""
        constraints = extract_constraints_from_message("for 4 guests")
        assert constraints.guests == 4

    def test_extract_amenities_pool(self):
        """Test extracting pool amenity."""
        constraints = extract_constraints_from_message("with a pool")
        assert "pool" in constraints.amenities

    def test_extract_amenities_wifi(self):
        """Test extracting wifi amenity."""
        constraints = extract_constraints_from_message("need wifi")
        assert "wifi" in constraints.amenities

    def test_extract_multiple_constraints(self):
        """Test extracting multiple constraints from one message."""
        constraints = extract_constraints_from_message(
            "I need a 3 bedroom villa in Seattle with pool and gym, under $500 per night for 4 guests"
        )
        assert constraints.city == "Seattle"
        assert constraints.property_type == "villa"
        assert constraints.bedrooms == 3
        assert constraints.price_max == 500
        assert constraints.guests == 4
        assert "pool" in constraints.amenities
        assert "gym" in constraints.amenities

    def test_extract_property_type_case_insensitive(self):
        """Test that property type extraction is case insensitive."""
        constraints1 = extract_constraints_from_message("I want a VILLA")
        constraints2 = extract_constraints_from_message("I want a Villa")
        constraints3 = extract_constraints_from_message("I want a villa")
        assert constraints1.property_type == "villa"
        assert constraints2.property_type == "villa"
        assert constraints3.property_type == "villa"


class TestMergeConstraints:
    """Test constraint merging across conversation turns."""

    def test_merge_empty_with_constraints(self):
        """Test merging empty constraints with new constraints."""
        previous = ExtractedConstraints()
        current = ExtractedConstraints(city="Seattle", property_type="villa")
        merged = merge_constraints(previous, current)
        assert merged.city == "Seattle"
        assert merged.property_type == "villa"

    def test_merge_preserves_previous_constraints(self):
        """Test that merging preserves previous constraints."""
        previous = ExtractedConstraints(city="Seattle", property_type="villa")
        current = ExtractedConstraints(bedrooms=3)
        merged = merge_constraints(previous, current)
        assert merged.city == "Seattle"
        assert merged.property_type == "villa"
        assert merged.bedrooms == 3

    def test_merge_overrides_with_new_values(self):
        """Test that new constraints override previous ones."""
        previous = ExtractedConstraints(city="Seattle", property_type="villa", bedrooms=2)
        current = ExtractedConstraints(bedrooms=3)
        merged = merge_constraints(previous, current)
        assert merged.city == "Seattle"
        assert merged.property_type == "villa"
        assert merged.bedrooms == 3         

    def test_merge_amenities_accumulates(self):
        """Test that amenities accumulate across turns."""
        previous = ExtractedConstraints(amenities=["pool"])
        current = ExtractedConstraints(amenities=["gym", "wifi"])
        merged = merge_constraints(previous, current)
        assert "pool" in merged.amenities
        assert "gym" in merged.amenities
        assert "wifi" in merged.amenities

    def test_merge_amenities_no_duplicates(self):
        """Test that duplicate amenities are not added."""
        previous = ExtractedConstraints(amenities=["pool", "gym"])
        current = ExtractedConstraints(amenities=["pool", "wifi"])
        merged = merge_constraints(previous, current)
        assert merged.amenities.count("pool") == 1
        assert "gym" in merged.amenities
        assert "wifi" in merged.amenities

    def test_merge_price_override(self):
        """Test that price constraints can be overridden."""
        previous = ExtractedConstraints(price_max=500)
        current = ExtractedConstraints(price_max=300)
        merged = merge_constraints(previous, current)
        assert merged.price_max == 300

    def test_merge_city_override(self):
        """Test that city can be overridden."""
        previous = ExtractedConstraints(city="Seattle")
        current = ExtractedConstraints(city="Portland")
        merged = merge_constraints(previous, current)
        assert merged.city == "Portland"

    def test_merge_property_type_override(self):
        """Test that property type can be overridden."""
        previous = ExtractedConstraints(property_type="villa")
        current = ExtractedConstraints(property_type="apartment")
        merged = merge_constraints(previous, current)
        assert merged.property_type == "apartment"

    def test_merge_none_with_none(self):
        """Test merging two empty constraint sets."""
        previous = ExtractedConstraints()
        current = ExtractedConstraints()
        merged = merge_constraints(previous, current)
        assert merged.is_empty()


class TestConversationTurnScenarios:
    """Test realistic conversation scenarios."""

    def test_initial_search_then_refinement(self):
        """Test initial search followed by refinement."""
        constraints1 = extract_constraints_from_message("villa in Seattle")
        
        constraints2 = extract_constraints_from_message("with 3 bedrooms")
        merged = merge_constraints(constraints1, constraints2)
        
        assert merged.city == "Seattle"
        assert merged.property_type == "villa"
        assert merged.bedrooms == 3

    def test_search_then_price_constraint(self):
        """Test search followed by price constraint."""
        constraints1 = extract_constraints_from_message("apartment in New York")
        
        constraints2 = extract_constraints_from_message("under $200 per night")
        merged = merge_constraints(constraints1, constraints2)
        
        assert merged.city == "New York"
        assert merged.property_type == "apartment"
        assert merged.price_max == 200

    def test_search_then_amenities(self):
        """Test search followed by amenity constraints."""
        constraints1 = extract_constraints_from_message("house in Portland")
        
        constraints2 = extract_constraints_from_message("with pool and gym")
        merged = merge_constraints(constraints1, constraints2)
        
        assert merged.city == "Portland"
        assert merged.property_type == "house"
        assert "pool" in merged.amenities
        assert "gym" in merged.amenities

    def test_change_property_type_mid_conversation(self):
        """Test changing property type mid-conversation."""
        constraints1 = extract_constraints_from_message("villa in Seattle")
        
        constraints2 = extract_constraints_from_message("actually, I want an apartment")
        merged = merge_constraints(constraints1, constraints2)
        
        assert merged.city == "Seattle"
        assert merged.property_type == "apartment"

    def test_change_city_mid_conversation(self):
        """Test changing city mid-conversation."""
        constraints1 = extract_constraints_from_message("villa in Seattle")
        
        constraints2 = extract_constraints_from_message("actually, look in Portland")
        merged = merge_constraints(constraints1, constraints2)
        
        assert merged.city == "Portland"
        assert merged.property_type == "villa"

    def test_multiple_refinements(self):
        """Test multiple constraint refinements."""
        constraints1 = extract_constraints_from_message("villa in Seattle")
        
        constraints2 = extract_constraints_from_message("with 3 bedrooms")
        merged2 = merge_constraints(constraints1, constraints2)
        
        constraints3 = extract_constraints_from_message("and 2 bathrooms")
        merged3 = merge_constraints(merged2, constraints3)
        
        constraints4 = extract_constraints_from_message("under $400 per night")
        merged4 = merge_constraints(merged3, constraints4)
        
        constraints5 = extract_constraints_from_message("with pool")
        merged5 = merge_constraints(merged4, constraints5)
        
        assert merged5.city == "Seattle"
        assert merged5.property_type == "villa"
        assert merged5.bedrooms == 3
        assert merged5.bathrooms == 2
        assert merged5.price_max == 400
        assert "pool" in merged5.amenities
