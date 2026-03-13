import pytest
from app.services.graph import node_triage, ChatState
from app.services.state_keys import SK

def test_triage_routing_greeting():
    # Greetings should always be routed to greeting
    state: ChatState = {"user_text": "hello", "filters": {}}
    res = node_triage(state)
    assert res["intent"] == "greeting"

def test_triage_routing_booking_context_with_fields():
    # If in booking context and providing names, etc., it should be confirmation
    state: ChatState = {
        "user_text": "John Doe",
        "filters": {SK.recent_property_id: "123", SK.awaiting_field: "name"}
    }
    res = node_triage(state)
    assert res["intent"] == "confirmation"

def test_triage_faq_return():
    # After an FAQ is answered, we should resume the previous intent (usually confirmation/property_search)
    state: ChatState = {
        "user_text": "ok sounds good",
        "filters": {SK.faq_answered: True, SK.faq_resume_intent: "confirmation"}
    }
    res = node_triage(state)
    assert res["intent"] == "confirmation"
    assert not res["filters"].get(SK.faq_answered)

def test_triage_status_extraction():
    # Should extract a booking ID automatically if present
    state: ChatState = {
        "user_text": "Where is my booking 9811fbbb-99cf-4180-a8d8-be8ef274737d?",
        "filters": {}
    }
    res = node_triage(state)
    assert res["intent"] == "status_update"
    assert res.get("status_args", {}).get("booking_id") == "9811fbbb-99cf-4180-a8d8-be8ef274737d"

