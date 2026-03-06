# tests/test_nlp_engine.py
"""
Unit tests for the NLP engine — verifies that dynamic NLP functions
produce the same outputs as the old hardcoded functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services import nlp_engine


# ────────────── Affirmation Classification ──────────────

class TestClassifyAffirmation:
    @pytest.mark.parametrize("text", ["yes", "yeah", "yep", "yup", "sure", "ok", "okay"])
    def test_yes_inputs(self, text):
        assert nlp_engine.classify_affirmation(text) == "yes"

    @pytest.mark.parametrize("text", ["no", "nope", "nah", "cancel", "stop"])
    def test_no_inputs(self, text):
        assert nlp_engine.classify_affirmation(text) == "no"

    def test_neutral(self):
        assert nlp_engine.classify_affirmation("apartment in New York") == "neutral"

    def test_empty(self):
        assert nlp_engine.classify_affirmation("") == "neutral"

    def test_none_like(self):
        assert nlp_engine.classify_affirmation("   ") == "neutral"


# ────────────── Greeting Detection ──────────────

class TestIsGreeting:
    @pytest.mark.parametrize("text", ["hi", "hello", "hey", "good morning", "hiya"])
    def test_greetings(self, text):
        assert nlp_engine.is_greeting(text) is True

    @pytest.mark.parametrize("text", [
        "apartment", "find me a villa", "what is the refund policy",
        "I need 2 bedrooms under $200",
    ])
    def test_non_greetings(self, text):
        assert nlp_engine.is_greeting(text) is False


# ────────────── Acknowledgment ──────────────

class TestIsAcknowledgment:
    @pytest.mark.parametrize("text", ["ok", "okay", "got it", "sounds good", "thanks"])
    def test_ack(self, text):
        assert nlp_engine.is_acknowledgment(text) is True


# ────────────── Status Query ──────────────

class TestIsStatusQuery:
    def test_uuid(self):
        assert nlp_engine.is_status_query(
            "my booking is 12345678-1234-1234-1234-123456789abc"
        ) is True

    def test_status_keywords(self):
        assert nlp_engine.is_status_query("check my booking status") is True

    def test_not_status(self):
        assert nlp_engine.is_status_query("find me an apartment in NYC") is False


# ────────────── Property Search ──────────────

class TestIsPropertySearch:
    @pytest.mark.parametrize("text", [
        "I want a villa in Miami",
        "show me apartments under $200",
        "find me a 2 bedroom condo",
        "looking for a place to stay in NYC",
    ])
    def test_search_intents(self, text):
        assert nlp_engine.is_property_search(text) is True

    def test_status_not_search(self):
        assert nlp_engine.is_property_search("check my booking status") is False


# ────────────── Cardinal Extraction ──────────────

class TestExtractCardinal:
    @pytest.mark.parametrize("text,expected", [
        ("1", 1), ("3", 3),
        ("1st", 1), ("2nd", 2), ("3rd", 3),
        ("first", 1), ("second", 2), ("third", 3),
        ("option 2", 2), ("pick 5", 5),
        ("choose one", 1), ("take three", 3),
    ])
    def test_cardinals(self, text, expected):
        assert nlp_engine.extract_cardinal(text) == expected

    def test_structural_digit_fast_path(self):
        assert nlp_engine.extract_cardinal("9") == 9
        assert nlp_engine.has_cardinal_extraction("9") is True

    def test_structural_digit_fast_path_without_spacy(self, monkeypatch):
        monkeypatch.setattr(nlp_engine, "_get_spacy", lambda: None)
        assert nlp_engine.extract_cardinal("9") == 9
        assert nlp_engine.extract_cardinal("option 9") == 9

    def test_none(self):
        assert nlp_engine.extract_cardinal("hello world") is None

    @pytest.mark.parametrize("text", [
        "no not this one",
        "this one",
        "that one",
    ])
    def test_no_false_selection_for_referential_one(self, text):
        assert nlp_engine.extract_cardinal(text) is None


# ────────────── Name Extraction ──────────────

class TestExtractPersonName:
    def test_my_name_is(self):
        result = nlp_engine.extract_person_name("my name is John Doe")
        assert result is not None
        assert "John" in result

    def test_search_guard(self):
        """Search queries should NOT be parsed as names."""
        assert nlp_engine.extract_person_name("find a villa in Miami") is None


# ────────────── Date Extraction ──────────────

class TestExtractDates:
    def test_iso_date(self):
        dates = nlp_engine.extract_dates("check-in 2027-06-01 checkout 2027-06-05")
        assert "2027-06-01" in dates
        assert "2027-06-05" in dates

    def test_no_dates(self):
        assert nlp_engine.extract_dates("hello world") == []


# ────────────── Phone Extraction ──────────────

class TestExtractPhone:
    def test_phone(self):
        result = nlp_engine.extract_phone("+15551234567")
        assert result == "+15551234567"

    def test_no_phone(self):
        assert nlp_engine.extract_phone("hello world") is None


# ────────────── Email Extraction ──────────────

class TestExtractEmail:
    def test_email(self):
        assert nlp_engine.extract_email("my email is john@example.com") == "john@example.com"

    def test_no_email(self):
        assert nlp_engine.extract_email("hello world") is None


# ────────────── FAQ Intent Detection ──────────────

class TestDetectFaqIntent:
    @pytest.mark.parametrize("text", [
        "What is the refund policy?",
        "Can I cancel my booking?",
        "What are the cancellation terms?",
    ])
    def test_faq_intents(self, text):
        assert nlp_engine.detect_faq_intent(text) is True

    def test_not_faq(self):
        assert nlp_engine.detect_faq_intent("find me an apartment") is False


# ────────────── Triage Intent Parity ──────────────

class TestTriageIntentParity:
    """Ensure triage_intent produces the same outputs as before."""

    @pytest.mark.parametrize("text,expected", [
        ("hi", "greeting"),
        ("hello", "greeting"),
        ("bye", "end"),
        ("goodbye", "end"),
        ("find apartment in NYC", "property_search"),
        ("show me villas under 200", "property_search"),
        ("I want to talk to a human", "handoff"),
    ])
    def test_known_intents(self, text, expected):
        from services.agents import triage_intent
        result = triage_intent(text)
        assert result == expected, f"triage_intent('{text}') = '{result}', expected '{expected}'"


# ────────────── Modification Detection ──────────────

class TestModification:
    def test_wants_modification(self):
        assert nlp_engine.wants_modification("I want to modify my dates") is True
        assert nlp_engine.wants_modification("change my check-in") is True

    def test_not_modification(self):
        assert nlp_engine.wants_modification("find a villa") is False


# ────────────── Field Detection ──────────────

class TestDetectRequestedFields:
    def test_dates(self):
        fields = nlp_engine.detect_requested_fields("change my check-in date")
        assert "check_in" in fields

    def test_property(self):
        fields = nlp_engine.detect_requested_fields("show me different property options")
        assert "property" in fields


class TestSoftIntentSignals:
    def test_wants_modification(self):
        assert nlp_engine.wants_modification("please modify my booking details") is True

    def test_wants_property_search_request(self):
        assert nlp_engine.wants_property_search_request("show me other properties") is True

    def test_is_receipt_request(self):
        assert nlp_engine.is_receipt_request("what is the final cost?") is True

    def test_is_resume_request(self):
        assert nlp_engine.is_resume_request("continue booking please") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
