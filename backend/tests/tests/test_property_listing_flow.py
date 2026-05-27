import pytest
# ────────────────────────────────────────────────────────────────
# Additional tests for strict property type filtering
# ────────────────────────────────────────────────────────────────

from app.services.property_type_normalizer import normalize_property_type
class _Ctx:
    def __init__(self):
        self.state = {}
# ---------- Normalizer unit tests ----------
def test_normalizer_plural_to_canonical():
    assert normalize_property_type("apartments") == "apartment"
    assert normalize_property_type("flats")      == "apartment"
    assert normalize_property_type("flat")       == "apartment"
    assert normalize_property_type("houses")     == "house"
    assert normalize_property_type("villas")     == "villa"
    assert normalize_property_type("duplexes")   == "duplex"
    assert normalize_property_type("townhomes")  == "townhouse"
    assert normalize_property_type("condos")     == "condo"

def test_normalizer_case_insensitive():
    assert normalize_property_type("APARTMENT") == "apartment"
    assert normalize_property_type("Villa")     == "villa"
    assert normalize_property_type("LOFT")      == "loft"

def test_normalizer_none_and_empty():
    assert normalize_property_type(None) is None
    assert normalize_property_type("")   is None
    assert normalize_property_type("  ") is None

def test_normalizer_already_canonical():
    assert normalize_property_type("apartment") == "apartment"
    assert normalize_property_type("house")     == "house"
    assert normalize_property_type("duplex")    == "duplex"

def test_normalizer_unknown_returns_passthrough():
    # With policy=pass_through, unknown type returns the raw key (lowercased)
    result = normalize_property_type("warehouse")
    assert result == "warehouse"  # pass_through returns raw

# ---------- Search integration tests ----------
@pytest.mark.asyncio
async def test_apartments_only_returned_no_mixed():
    ctx = _Ctx()
    fake = [
        {"id":"a1","city":"New York","price_per_night":100.0,"property_type":"Apartment","bedrooms":1,"bathrooms":1,"rating":4.5,"amenities":["wifi"],"title":"Apt 1","description":""},
        {"id":"d1","city":"New York","price_per_night":110.0,"property_type":"Duplex","bedrooms":2,"bathrooms":1,"rating":4.0,"amenities":[],"title":"Dup 1","description":""},
        {"id":"h1","city":"New York","price_per_night":120.0,"property_type":"House","bedrooms":3,"bathrooms":2,"rating":3.8,"amenities":[],"title":"House 1","description":""},
        {"id":"l1","city":"New York","price_per_night":130.0,"property_type":"Loft","bedrooms":1,"bathrooms":1,"rating":4.2,"amenities":[],"title":"Loft 1","description":""},
        {"id":"a2","city":"New York","price_per_night":95.0,"property_type":"Apartment","bedrooms":2,"bathrooms":1,"rating":4.7,"amenities":["pool"],"title":"Apt 2","description":""},
    ]
    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties",
               return_value={"fallback": True}):
        result = await search_properties(city="New York", property_type="apartments", tool_context=ctx)

    assert result["status"] == "properties_found", f"Unexpected status: {result['status']}"
    returned_types = {p["property_type"].lower() for p in result["properties"]}
    assert returned_types == {"apartment"}, \
        f"Expected only apartment, got: {returned_types}"
    assert result["total_found"] == 2, f"Expected 2 apartments, got: {result['total_found']}"

@pytest.mark.asyncio
async def test_villa_only_returned():
    ctx = _Ctx()
    fake = [
        {"id":"v1","city":"Dubai","price_per_night":200.0,"property_type":"Villa","bedrooms":3,"bathrooms":2,"rating":4.9,"amenities":["pool"],"title":"Villa 1","description":""},
        {"id":"a1","city":"Dubai","price_per_night":150.0,"property_type":"Apartment","bedrooms":2,"bathrooms":1,"rating":4.5,"amenities":[],"title":"Apt 1","description":""},
        {"id":"v2","city":"Dubai","price_per_night":250.0,"property_type":"villa","bedrooms":4,"bathrooms":3,"rating":4.8,"amenities":["pool","gym"],"title":"Villa 2","description":""},
    ]
    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties",
               return_value={"fallback": True}):
        result = await search_properties(city="Dubai", property_type="villa", tool_context=ctx)

    returned_types = {p["property_type"].lower() for p in result["properties"]}
    assert returned_types == {"villa"}, f"Got: {returned_types}"
    assert result["total_found"] == 2

@pytest.mark.asyncio
async def test_duplex_only_returned():
    ctx = _Ctx()
    fake = [
        {"id":"dup1","city":"New York","price_per_night":120.0,"property_type":"Duplex","bedrooms":2,"bathrooms":1,"rating":4.0,"amenities":[],"title":"Dup 1","description":""},
        {"id":"apt1","city":"New York","price_per_night":100.0,"property_type":"Apartment","bedrooms":1,"bathrooms":1,"rating":4.5,"amenities":[],"title":"Apt 1","description":""},
    ]
    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties",
               return_value={"fallback": True}):
        result = await search_properties(city="New York", property_type="duplex", tool_context=ctx)

    returned_types = {p["property_type"].lower() for p in result["properties"]}
    assert returned_types == {"duplex"}
    assert result["total_found"] == 1

@pytest.mark.asyncio
async def test_no_property_type_returns_all_types():
    ctx = _Ctx()
    fake = [
        {"id":"a1","city":"Karachi","price_per_night":80.0,"property_type":"Apartment","bedrooms":1,"bathrooms":1,"rating":4.0,"amenities":[],"title":"A","description":""},
        {"id":"h1","city":"Karachi","price_per_night":90.0,"property_type":"House","bedrooms":2,"bathrooms":1,"rating":4.2,"amenities":[],"title":"H","description":""},
        {"id":"v1","city":"Karachi","price_per_night":150.0,"property_type":"Villa","bedrooms":3,"bathrooms":2,"rating":4.8,"amenities":[],"title":"V","description":""},
    ]
    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties",
               return_value={"fallback": True}):
        result = await search_properties(city="Karachi", property_type=None, tool_context=ctx)

    returned_types = {p["property_type"].lower() for p in result["properties"]}
    assert len(returned_types) == 3, "All types must be returned when no filter"

@pytest.mark.asyncio
async def test_flat_alias_resolves_to_apartment():
    ctx = _Ctx()
    fake = [
        {"id":"a1","city":"London","price_per_night":120.0,"property_type":"Apartment","bedrooms":2,"bathrooms":1,"rating":4.5,"amenities":[],"title":"Flat 1","description":""},
        {"id":"h1","city":"London","price_per_night":200.0,"property_type":"House","bedrooms":3,"bathrooms":2,"rating":4.0,"amenities":[],"title":"House 1","description":""},
    ]
    with patch("app.components.search._DATASET", fake), \
         patch("app.agents.tools.rust_client.search_properties",
               return_value={"fallback": True}):
        # User says "flats" → LLM might pass "flat" or "flats"
        result = await search_properties(city="London", property_type="flat", tool_context=ctx)

    returned_types = {p["property_type"].lower() for p in result["properties"]}
    assert returned_types == {"apartment"}, f"'flat' must resolve to apartment, got: {returned_types}"