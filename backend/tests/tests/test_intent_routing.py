import json
from types import SimpleNamespace

import app.services.agents as agents
from app.services.state_keys import SK


def test_triage_keeps_greeting_priority(monkeypatch):
    monkeypatch.setattr(agents, "_llm_route_intent", lambda text, filters=None: "property_search")
    assert agents.triage_intent("hi", {}) == "greeting"


def test_triage_uses_llm_router_when_available(monkeypatch):
    monkeypatch.setattr(agents, "_llm_route_intent", lambda text, filters=None: "faq")
    assert agents.triage_intent("what are your wifi rules?", {}) == "faq"


def test_triage_falls_back_to_confirmation_on_slots(monkeypatch):
    monkeypatch.setattr(agents, "_llm_route_intent", lambda text, filters=None: None)
    assert agents.triage_intent("my email is user@example.com", {}) == "confirmation"


def test_llm_router_injects_dynamic_state_into_system_prompt(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "intent": "confirmation",
                                    "confidence": 0.91,
                                    "brief_reason": "state override",
                                }
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(agents, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(agents, "LLM_STRUCTURED", True)
    monkeypatch.setattr(agents, "SOFT_INTENT_ROUTER", True)
    monkeypatch.setattr(agents, "get_vocabulary", lambda: SimpleNamespace(seed_property_types=["treehouse", "riad"]))
    monkeypatch.setattr(agents.httpx, "Client", FakeClient)

    result = agents._llm_route_intent(
        "9",
        {
            "last_results": [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}],
            SK.awaiting_field: "email",
        },
    )

    assert result == "confirmation"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    payload = captured["payload"]
    messages = payload["messages"]
    assert messages[1]["content"] == "9"

    system_prompt = messages[0]["content"]
    assert "[ACTIVE STATE]" in system_prompt
    assert "- has_last_results: True" in system_prompt
    assert "- last_results_count: 3" in system_prompt
    assert "- awaiting_field: email" in system_prompt
    assert "You just showed the user a numbered list of 3 properties" in system_prompt
    assert "Treat their input as a 'confirmation' of this data." in system_prompt
    assert "Valid property types in our database: treehouse, riad." in system_prompt

    prompt_lower = system_prompt.lower()
    assert "hello i need a loft in nyc" not in prompt_lower
    assert "nyc" not in prompt_lower
    assert "new york" not in prompt_lower
    assert '"context"' not in messages[1]["content"]

