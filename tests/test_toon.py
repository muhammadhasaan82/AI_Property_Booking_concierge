# tests/test_toon.py
"""
Unit tests for the TOON (Token-Optimized Object Notation) module.
Tests encode/decode round-trips and edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.toon import toon_encode, toon_decode


class TestToonEncode:
    def test_simple_object(self):
        obj = {"name": "John", "age": 30, "active": True}
        toon = toon_encode(obj)
        assert "name: John" in toon
        assert "age: 30" in toon
        assert "active: true" in toon

    def test_nested_object(self):
        obj = {"user": {"name": "Jane", "city": "NYC"}}
        toon = toon_encode(obj)
        assert "user:" in toon
        assert "name: Jane" in toon
        assert "city: NYC" in toon

    def test_array(self):
        obj = {"items": [1, 2, 3]}
        toon = toon_encode(obj)
        assert "items: []" in toon
        assert "- 1" in toon

    def test_null_bool(self):
        obj = {"a": None, "b": True, "c": False}
        toon = toon_encode(obj)
        assert "a: null" in toon
        assert "b: true" in toon
        assert "c: false" in toon

    def test_empty_string_quoted(self):
        obj = {"empty": ""}
        toon = toon_encode(obj)
        assert '""' in toon


class TestToonDecode:
    def test_simple_object(self):
        toon = "name: John\nage: 30\nactive: true"
        obj = toon_decode(toon)
        assert obj["name"] == "John"
        assert obj["age"] == 30
        assert obj["active"] is True

    def test_null(self):
        obj = toon_decode("val: null")
        assert obj["val"] is None

    def test_empty_input(self):
        assert toon_decode("") == {}
        assert toon_decode("   ") == {}


class TestToonEdgeCases:
    def test_string_with_colon(self):
        """Strings containing colons must round-trip correctly."""
        obj = {"time": "12:30:00", "url": "https://example.com"}
        toon = toon_encode(obj)
        decoded = toon_decode(toon)
        assert decoded["time"] == "12:30:00"
        assert decoded["url"] == "https://example.com"

    def test_string_with_newline(self):
        """Strings containing newlines must round-trip correctly."""
        obj = {"message": "line1\nline2\nline3"}
        toon = toon_encode(obj)
        # Newlines should be escaped in TOON format
        assert "\\n" in toon
        decoded = toon_decode(toon)
        assert decoded["message"] == "line1\nline2\nline3"

    def test_numeric_string_preserved(self):
        """A string that looks like a number should stay as a string."""
        obj = {"phone": "12345"}
        toon = toon_encode(obj)
        decoded = toon_decode(toon)
        assert decoded["phone"] == "12345"
        assert isinstance(decoded["phone"], str)

    def test_boolean_string_preserved(self):
        """A string that looks like a boolean should stay as a string."""
        obj = {"answer": "true", "flag": True}
        toon = toon_encode(obj)
        decoded = toon_decode(toon)
        assert decoded["answer"] == "true"  # string
        assert decoded["flag"] is True  # actual boolean

    def test_empty_object_array(self):
        obj = {"empty_obj": {}, "empty_arr": []}
        toon = toon_encode(obj)
        decoded = toon_decode(toon)
        assert decoded["empty_obj"] == {}
        assert decoded["empty_arr"] == []

    def test_key_with_colon(self):
        """Keys containing colons should round-trip correctly."""
        obj = {"time:stamp": "2024-01-01"}
        toon = toon_encode(obj)
        assert "time\\:stamp" in toon
        decoded = toon_decode(toon)
        assert decoded["time:stamp"] == "2024-01-01"


class TestToonRoundTrip:
    def test_complex_roundtrip(self):
        """Full complex object should survive encode→decode unchanged."""
        obj = {
            "ok": True,
            "count": 42,
            "ratio": 3.14,
            "empty": None,
            "message": "Hello world",
            "tags": ["a", "b", "c"],
            "nested": {
                "name": "test",
                "value": 100,
            },
        }
        toon = toon_encode(obj)
        decoded = toon_decode(toon)
        assert decoded["ok"] is True
        assert decoded["count"] == 42
        assert decoded["empty"] is None
        assert decoded["message"] == "Hello world"
        assert decoded["tags"] == ["a", "b", "c"]
        assert decoded["nested"]["name"] == "test"
        assert decoded["nested"]["value"] == 100

    def test_booking_payload_roundtrip(self):
        """Realistic booking payload should round-trip."""
        obj = {
            "data": {
                "location": "Miami",
                "budget": 200,
                "properties": [
                    {"id": "p1", "city": "Miami", "price_per_night": 150, "title": "Beach House"},
                    {"id": "p2", "city": "NYC", "price_per_night": 300, "title": "Studio"},
                ],
            },
        }
        toon = toon_encode(obj)
        decoded = toon_decode(toon)
        assert decoded["data"]["location"] == "Miami"
        assert decoded["data"]["budget"] == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
