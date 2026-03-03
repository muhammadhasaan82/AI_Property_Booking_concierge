# tests/test_rust_integration.py
# Integration tests for the Python ↔ Rust gateway round-trip.
# Requires the Rust gateway to be running on localhost:3001.

import asyncio
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx


RUST_URL = os.getenv("RUST_GATEWAY_URL", "http://localhost:3001")


async def test_health():
    """Rust gateway should respond to /health."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{RUST_URL}/health")
        assert r.status_code == 200, f"Health check failed: {r.status_code}"
        data = r.json()
        assert data["ok"] is True
        assert data["service"] == "rust_gateway"
    print("✅ Health check passed")


async def test_list_tools():
    """Should list all registered tools."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{RUST_URL}/tools")
        assert r.status_code == 200
        data = r.json()
        tools = data["tools"]
        assert "property_search" in tools
        assert "booking_validator" in tools
        assert "pricing" in tools
        assert "sentiment_analysis" in tools
        assert "fraud_check" in tools
    print(f"✅ Tools listed: {tools}")


async def test_execute_search():
    """Schema-agnostic gateway should infer search intent."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(f"{RUST_URL}/execute", json={
            "data": {
                "location": "Miami",
                "budget": 200,
                "properties": [
                    {"id": "p1", "city": "Miami", "price_per_night": 150, "beds": 2, "title": "Beach House"},
                    {"id": "p2", "city": "NYC", "price_per_night": 300, "beds": 1, "title": "Studio"},
                ]
            }
        })
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "search"
        assert data["tool_used"] == "property_search"
        result = data["result"]
        assert result["count"] == 1
        assert result["results"][0]["id"] == "p1"
    print("✅ Search execution passed")


async def test_execute_booking_validation():
    """Gateway should infer booking intent and validate."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(f"{RUST_URL}/execute", json={
            "data": {
                "property_id": "p1",
                "check_in": "2027-06-01",
                "check_out": "2027-06-05",
                "guests": 2,
                "email": "test@example.com"
            }
        })
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "booking"
        assert data["tool_used"] == "booking_validator"
        result = data["result"]
        assert result["valid"] is True
        assert result["nights"] == 4
    print("✅ Booking validation passed")


async def test_execute_pricing():
    """Direct pricing tool endpoint."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(f"{RUST_URL}/tools/pricing", json={
            "price_per_night": 100.0,
            "nights": 3,
            "guests": 2,
            "tax_rate": 0.10
        })
        assert r.status_code == 200
        data = r.json()
        assert data["subtotal"] == 300.0
        assert data["tax"] == 30.0
        assert data["total"] == 330.0
    print("✅ Pricing tool passed")


async def test_execute_sentiment():
    """Sentiment analysis tool endpoint."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(f"{RUST_URL}/tools/sentiment", json={
            "text": "This hotel is amazing and the staff is wonderful"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["label"] == "positive"
        assert data["compound"] > 0
    print("✅ Sentiment tool passed")


async def test_execute_fraud():
    """Fraud check tool endpoint."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(f"{RUST_URL}/tools/fraud", json={
            "check_fraud": True,
            "email": "john@gmail.com",
            "phone": "+15551234567",
            "amount": 500.0,
            "guests": 2
        })
        assert r.status_code == 200
        data = r.json()
        assert data["risk_level"] == "low"
    print("✅ Fraud check passed")


async def test_safety_insufficient_booking_data():
    """Gateway should refuse booking without required fields."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(f"{RUST_URL}/execute", json={
            "data": {
                "property_id": "p1"
                # Missing check_in and check_out
            }
        })
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is False
        assert "insufficient_data" in str(data.get("error", ""))
    print("✅ Safety validation passed")


async def test_cache():
    """Same request should return cached result."""
    payload = {
        "data": {
            "location": "TestCache",
            "budget": 100,
            "properties": [
                {"id": "c1", "city": "TestCache", "price_per_night": 50, "title": "Cache Test"}
            ]
        }
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        r1 = await client.post(f"{RUST_URL}/execute", json=payload)
        assert r1.status_code == 200

        r2 = await client.post(f"{RUST_URL}/execute", json=payload)
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2.get("cached") is True
    print("✅ Caching passed")


async def test_python_rust_client():
    """Test the Python rust_client.py wrapper."""
    from services import rust_client

    # Search
    result = await rust_client.search_properties(
        location="Miami",
        budget=200,
        properties=[
            {"id": "p1", "city": "Miami", "price_per_night": 150, "title": "Beach"},
        ]
    )
    if result.get("fallback"):
        print("⚠️  Rust gateway not available, skipping client test")
        return
    assert result.get("ok") is True
    print("✅ Python rust_client.py integration passed")


async def main():
    print("=" * 60)
    print("RUST GATEWAY INTEGRATION TESTS")
    print("=" * 60)
    print(f"Gateway URL: {RUST_URL}\n")

    # Check if gateway is running
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(f"{RUST_URL}/health")
    except Exception:
        print("❌ Rust gateway is not running. Start it with:")
        print("   cd rust_gateway && cargo run")
        return

    tests = [
        test_health,
        test_list_tools,
        test_execute_search,
        test_execute_booking_validation,
        test_execute_pricing,
        test_execute_sentiment,
        test_execute_fraud,
        test_safety_insufficient_booking_data,
        test_cache,
        test_python_rust_client,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
