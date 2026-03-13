import asyncio
import json
import os

import httpx


BASE = os.getenv("API_BASE", "http://127.0.0.1:8000/api/v1")


async def main() -> None:
    # Force mock mode for booking by clearing Supabase envs for this process
    os.environ.pop("SUPABASE_URL", None)
    os.environ.pop("SUPABASE_KEY", None)
    os.environ.pop("SUPABASE_ANON_KEY", None)
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Health
        r = await client.get(f"{BASE}/health")
        print("HEALTH:", r.status_code, r.json())

        # Property search
        r = await client.post(
            f"{BASE}/properties/search",
            json={"query_text": "apartment", "budget": 200},
        )
        data = r.json()
        print("SEARCH:", r.status_code, {"count": len(data.get("results", []))})

        # Create booking
        r = await client.post(
            f"{BASE}/booking/create",
            json={
                "user_id": "u1",
                "property_id": "p1",
                "check_in": "2025-10-10",
                "check_out": "2025-10-12",
                "guests": 2,
            },
        )
        booking = r.json()
        print("BOOKING_CREATE:", r.status_code, booking)

        booking_id = booking.get("booking_id")
        if booking_id:
            r = await client.get(f"{BASE}/booking/status/{booking_id}")
            print("BOOKING_STATUS:", r.status_code, r.json())
        else:
            print("BOOKING_STATUS: skipped (no booking_id)")

        # Chat message
        r = await client.post(
            f"{BASE}/chat/message",
            json={"message": "Find a 2 bed in NYC under $200"},
        )
        print("CHAT:", r.status_code)


if __name__ == "__main__":
    asyncio.run(main())


