import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import services.agents as agents


def test_numeric_selection_routes_confirmation_only_with_active_results(monkeypatch):
    monkeypatch.setattr(agents, "_llm_route_intent", lambda text, filters=None: None)
    assert agents.triage_intent("9", {"last_results": [{"id": "p1"}]}) == "confirmation"


def test_standalone_number_without_results_falls_back_to_search(monkeypatch):
    monkeypatch.setattr(agents, "_llm_route_intent", lambda text, filters=None: None)
    assert agents.triage_intent("9", {}) == "property_search"


def test_active_results_policy_beats_bad_llm_selection_route(monkeypatch):
    monkeypatch.setattr(agents, "_llm_route_intent", lambda text, filters=None: "property_search")
    assert agents.triage_intent("9", {"last_results": [{"id": "p1"}]}) == "confirmation"
