import services.agents as agents


def test_triage_keeps_greeting_priority(monkeypatch):
    monkeypatch.setattr(agents, "_llm_route_intent", lambda text, filters=None: "property_search")
    assert agents.triage_intent("hi", {}) == "greeting"


def test_triage_uses_llm_router_when_available(monkeypatch):
    monkeypatch.setattr(agents, "_llm_route_intent", lambda text, filters=None: "faq")
    assert agents.triage_intent("what are your wifi rules?", {}) == "faq"


def test_triage_falls_back_to_confirmation_on_slots(monkeypatch):
    monkeypatch.setattr(agents, "_llm_route_intent", lambda text, filters=None: None)
    assert agents.triage_intent("my email is user@example.com", {}) == "confirmation"
