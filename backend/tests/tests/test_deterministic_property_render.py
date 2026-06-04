"""
Deterministic property render tests for adk_runner.

Verifies that when router_output contains status=properties_found
with a real properties array, the reply:
  - Uses ONLY actual titles from the payload (no hallucinated names)
  - Shows the correct count in the header
  - Shows the correct city and property_type
  - Does not contain any hallucinated marker names
  - Contains all real titles from the payload
"""
from __future__ import annotations

import pytest
from app.services.adk_runner import _render_property_results_from_router_output

_HALLUCINATED_NAMES = {"the grand", "city lights", "sky high", "urban oasis",
                       "luxury loft", "cozy studio", "manhattan suite"}


def _make_router_output(
    city: str = "New York",
    prop_type: str = "apartment",
    titles: list[str] | None = None,
    budget: int | None = None,
    pagination: dict | None = None,
) -> dict:
    if titles is None:
        titles = ["Brooklyn Heights Apt", "Midtown Walk-Up", "East Village Flat"]
    props = [
        {
            "number": i + 1,
            "id": f"p_{i}",
            "title": t,
            "city": city,
            "property_type": "Apartment",
            "price_per_night": 120 + i * 10,
            "bedrooms": 2,
            "bathrooms": 1,
            "rating": 4.5,
        }
        for i, t in enumerate(titles)
    ]
    return {
        "status": "properties_found",
        "total_found": len(props),
        "shown_count": len(props),
        "properties": props,
        "query_context": {"city": city, "property_type": prop_type, "budget": budget},
        "pagination": pagination or {"total_pages": 1, "has_next": False, "has_prev": False},
        "summary_mode": False,
    }



def test_header_says_i_found_n_apartments_in_new_york():
    ro = _make_router_output(titles=["A", "B", "C"])
    result = _render_property_results_from_router_output(ro)
    assert "I found 3 apartments in New York" in result


def test_all_real_titles_present():
    titles = ["Brooklyn Heights Apt", "Midtown Walk-Up", "East Village Flat"]
    ro = _make_router_output(titles=titles)
    result = _render_property_results_from_router_output(ro)
    for t in titles:
        assert t in result, f"Expected real title '{t}' in output"


def test_no_hallucinated_names():
    titles = ["Brooklyn Heights Apt", "Midtown Walk-Up", "East Village Flat"]
    ro = _make_router_output(titles=titles)
    result = _render_property_results_from_router_output(ro)
    result_lower = result.lower()
    for fake in _HALLUCINATED_NAMES:
        assert fake not in result_lower, (
            f"Hallucinated name '{fake}' found in deterministic render output"
        )


def test_city_and_type_in_header():
    ro = _make_router_output(city="Dubai", prop_type="villa",
                             titles=["Palm Villa One", "Palm Villa Two"])
    result = _render_property_results_from_router_output(ro)
    assert "villa" in result.lower()
    assert "Dubai" in result


def test_budget_in_header_when_provided():
    ro = _make_router_output(budget=200, titles=["Cheap Apt"])
    result = _render_property_results_from_router_output(ro)
    assert "under $200" in result


def test_no_budget_in_header_when_not_provided():
    ro = _make_router_output(budget=None, titles=["Apt One"])
    result = _render_property_results_from_router_output(ro)
    assert "under $" not in result


def test_option_numbers_present():
    titles = ["A", "B", "C"]
    ro = _make_router_output(titles=titles)
    result = _render_property_results_from_router_output(ro)
    assert "1." in result
    assert "2." in result
    assert "3." in result


def test_price_from_payload():
    ro = _make_router_output(titles=["My Apt"])
    ro["properties"][0]["price_per_night"] = 149
    result = _render_property_results_from_router_output(ro)
    assert "$149/night" in result


def test_rating_from_payload():
    ro = _make_router_output(titles=["Rated Apt"])
    ro["properties"][0]["rating"] = 4.7
    result = _render_property_results_from_router_output(ro)
    assert "4.7" in result



def test_empty_properties_returns_empty_string():
    ro = _make_router_output(titles=[])
    ro["properties"] = []
    result = _render_property_results_from_router_output(ro)
    assert result == ""


def test_missing_properties_key_returns_empty_string():
    result = _render_property_results_from_router_output({"status": "properties_found"})
    assert result == ""



def test_footer_has_next_shows_next_hint():
    ro = _make_router_output(
        titles=["A", "B", "C", "D", "E"],
        pagination={"total_pages": 3, "has_next": True, "has_prev": False,
                    "current_page": 1, "page_start": 1, "page_end": 5},
    )
    ro["total_found"] = 15
    result = _render_property_results_from_router_output(ro)
    assert "next" in result.lower()


def test_footer_no_pagination_shows_option_hint():
    ro = _make_router_output(titles=["X"])
    result = _render_property_results_from_router_output(ro)
    assert "option" in result.lower()


def test_paginated_header_shows_range():
    ro = _make_router_output(
        titles=["A", "B", "C", "D", "E"],
        pagination={"total_pages": 3, "has_next": True, "has_prev": False,
                    "current_page": 1, "page_start": 1, "page_end": 5},
    )
    ro["total_found"] = 15
    result = _render_property_results_from_router_output(ro)
    assert "page 1 of 3" in result


@pytest.mark.asyncio
async def test_run_adk_turn_suppresses_voice_when_properties_found(monkeypatch):
    """When the triage_router tool returns properties_found, the concierge_voice
    LLM must NOT contribute any chunks to the final reply."""
    from app.services import adk_runner

    real_titles = ["Brooklyn Heights Apt", "Midtown Walk-Up", "East Village Flat"]
    fake_router_output = _make_router_output(titles=real_titles)

    voice_was_called = []

    async def fake_run_async(*args, **kwargs):
        from unittest.mock import MagicMock

        tr_event = MagicMock()
        tr_event.author = "triage_router"
        tr_event.content = MagicMock()
        tr_event.content.parts = [
            MagicMock(
                function_response=MagicMock(response=fake_router_output),
                function_call=None,
                text=None,
            )
        ]
        tr_event.is_final_response.return_value = False
        yield tr_event

        voice_event = MagicMock()
        voice_event.author = "concierge_voice"
        voice_event.content = MagicMock()
        voice_event.content.parts = [MagicMock(text="The Grand is a lovely option!", function_call=None, function_response=None)]
        voice_event.is_final_response.return_value = True
        voice_was_called.append("hallucinated chunk")
        yield voice_event

    fake_runner = type("FakeRunner", (), {"run_async": staticmethod(fake_run_async)})()

    class FakeSessionService:
        async def get_session(self, **_kwargs):
            return None

    monkeypatch.setattr(adk_runner, "_get_runner", lambda: fake_runner)
    monkeypatch.setattr(adk_runner, "_get_session_service", lambda: FakeSessionService())
    monkeypatch.setattr(adk_runner, "sanitize_input", lambda m: (m, True))
    monkeypatch.setattr(adk_runner, "sanitize_output", lambda m: m)
    
    async def fake_route_pre_adk(**_kw):
        return None

    async def fake_build_invocation_state_delta(**_kw):
        return {"user_cognitive_context": "", "soft_state": {}}

    async def fake_get_session_snapshot(_sid):
        return {"state": {}, "history": [], "meta": {}}

    monkeypatch.setattr(adk_runner, "route_pre_adk", fake_route_pre_adk)
    monkeypatch.setattr(adk_runner, "_build_invocation_state_delta", fake_build_invocation_state_delta)
    monkeypatch.setattr(adk_runner, "get_session_snapshot", fake_get_session_snapshot)

    chunks = []
    async for chunk in adk_runner.run_adk_turn("u1", "s1", "apartments in New York"):
        chunks.append(chunk)

    full_reply = "".join(chunks)
    full_reply_lower = full_reply.lower()

    for t in real_titles:
        assert t in full_reply, f"Real title '{t}' missing from reply"

    for fake in _HALLUCINATED_NAMES:
        assert fake not in full_reply_lower, (
            f"Hallucinated name '{fake}' leaked into reply via concierge_voice"
        )

    assert "3" in full_reply
    assert "apartment" in full_reply_lower
