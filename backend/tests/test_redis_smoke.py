from __future__ import annotations

import asyncio
import os
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.security import anomaly
from app.services import redis_store


class FakeRedis:
    def __init__(self):
        self.values: Dict[str, str] = {}
        self.lists: Dict[str, List[str]] = {}
        self.expirations: Dict[str, int] = {}

    async def ping(self):
        return True

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex
        return True

    async def delete(self, key: str):
        removed = 0
        if key in self.values:
            self.values.pop(key, None)
            removed += 1
        if key in self.lists:
            self.lists.pop(key, None)
            removed += 1
        self.expirations.pop(key, None)
        return removed

    async def rpush(self, key: str, value: str):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    async def lrange(self, key: str, start: int, end: int):
        items = list(self.lists.get(key, []))
        if end == -1:
            return items[start:]
        return items[start : end + 1]

    async def expire(self, key: str, ttl: int):
        self.expirations[key] = ttl
        return True


def _reset_test_state() -> None:
    redis_store._LOCAL_FALLBACK.clear()
    redis_store._REDIS_CLIENT = None
    anomaly._session_tool_history.clear()


def test_session_snapshot_roundtrip_with_fake_redis():
    fake_redis = FakeRedis()

    async def fake_client():
        return fake_redis

    _reset_test_state()
    original_client_loader = redis_store._get_redis_client
    redis_store._get_redis_client = fake_client

    try:
        async def scenario():
            await redis_store.save_session_snapshot(
                session_id="smoke-session",
                history=[{"author": "user", "content": {"parts": [{"text": "hello"}]}}],
                state={"final_reply": "hi there"},
                metadata={"user_id": "smoke-user", "app_name": "ai_concierge"},
            )
            snapshot = await redis_store.get_session_snapshot("smoke-session")
            assert snapshot["history"][0]["author"] == "user"
            assert snapshot["state"]["final_reply"] == "hi there"
            assert snapshot["meta"]["user_id"] == "smoke-user"
            assert fake_redis.expirations["adk:session:smoke-session"] == redis_store.REDIS_SESSION_TTL_SECONDS

        asyncio.run(scenario())
    finally:
        redis_store._get_redis_client = original_client_loader
        _reset_test_state()


def test_local_fallback_and_anomaly_smoke():
    async def no_redis_client():
        return None

    _reset_test_state()
    original_client_loader = redis_store._get_redis_client
    original_anomaly_loader = anomaly.get_redis_client
    redis_store._get_redis_client = no_redis_client
    anomaly.get_redis_client = no_redis_client

    try:
        async def scenario():
            await redis_store.save_session_snapshot(
                session_id="fallback-session",
                history=[{"author": "user", "content": {"parts": [{"text": "ping"}]}}],
                state={"router_output": "pong"},
                metadata={"user_id": "fallback-user"},
            )
            snapshot = await redis_store.get_session_snapshot("fallback-session")
            assert snapshot["state"]["router_output"] == "pong"
            assert snapshot["meta"]["user_id"] == "fallback-user"

            await anomaly.clear_session("fallback-anomaly")
            assert await anomaly.check_tool_loop("fallback-anomaly", "search_properties", {"city": "Lahore"}) is False
            await anomaly.record_tool_call("fallback-anomaly", "search_properties", {"city": "Lahore"})
            await anomaly.record_tool_call("fallback-anomaly", "search_properties", {"city": "Lahore"})
            await anomaly.record_tool_call("fallback-anomaly", "search_properties", {"city": "Lahore"})
            await anomaly.record_tool_call("fallback-anomaly", "search_properties", {"city": "Lahore"})
            await anomaly.record_tool_call("fallback-anomaly", "search_properties", {"city": "Lahore"})
            stats = await anomaly.get_session_stats("fallback-anomaly")
            assert stats["total_tool_calls"] == 5
            assert stats["tool_counts"]["search_properties"] == 5
            assert await anomaly.check_tool_loop("fallback-anomaly", "search_properties", {"city": "Lahore"}) is True

        asyncio.run(scenario())
    finally:
        redis_store._get_redis_client = original_client_loader
        anomaly.get_redis_client = original_anomaly_loader
        _reset_test_state()


def test_anomaly_redis_integration():
    """Test anomaly detection with FakeRedis to verify cross-worker compatibility."""
    fake_redis = FakeRedis()

    async def fake_client():
        return fake_redis

    _reset_test_state()
    original_client_loader = redis_store._get_redis_client
    original_anomaly_loader = anomaly.get_redis_client
    redis_store._get_redis_client = fake_client
    anomaly.get_redis_client = fake_client

    try:
        async def scenario():
            session_id = "redis-anomaly-test"
            await anomaly.clear_session(session_id)

            # Record tool calls - should persist to Redis list
            await anomaly.record_tool_call(session_id, "search_properties", {"city": "Dubai"})
            await anomaly.record_tool_call(session_id, "check_faq", {"query": "cancellation"})
            await anomaly.record_tool_call(session_id, "search_properties", {"city": "Dubai"})

            # Verify Redis list contains entries
            key = f"adk:anomaly:{session_id}"
            assert key in fake_redis.lists, "Anomaly history should be stored in Redis"
            assert len(fake_redis.lists[key]) == 3, "Should have 3 tool call entries"

            # Check stats reflect Redis data
            stats = await anomaly.get_session_stats(session_id)
            assert stats["total_tool_calls"] == 3
            assert stats["tool_counts"]["search_properties"] == 2
            assert stats["tool_counts"]["check_faq"] == 1

            # Not yet at threshold (need 5 identical calls within time window)
            assert await anomaly.check_tool_loop(session_id, "search_properties", {"city": "Dubai"}) is False

            # Add more identical calls to hit threshold (5)
            await anomaly.record_tool_call(session_id, "search_properties", {"city": "Dubai"})
            await anomaly.record_tool_call(session_id, "search_properties", {"city": "Dubai"})
            await anomaly.record_tool_call(session_id, "search_properties", {"city": "Dubai"})
            assert await anomaly.check_tool_loop(session_id, "search_properties", {"city": "Dubai"}) is True

            # Different params should not trigger anomaly
            assert await anomaly.check_tool_loop(session_id, "search_properties", {"city": "London"}) is False

            # Clear and verify
            await anomaly.clear_session(session_id)
            assert key not in fake_redis.lists or len(fake_redis.lists.get(key, [])) == 0

        asyncio.run(scenario())
    finally:
        redis_store._get_redis_client = original_client_loader
        anomaly.get_redis_client = original_anomaly_loader
        _reset_test_state()


def test_session_history_helpers():
    """Test get_session_history and save_session_history convenience functions."""
    fake_redis = FakeRedis()

    async def fake_client():
        return fake_redis

    _reset_test_state()
    original_client_loader = redis_store._get_redis_client
    redis_store._get_redis_client = fake_client

    try:
        async def scenario():
            session_id = "history-helper-test"

            # Save history directly
            history = [
                {"author": "user", "content": {"parts": [{"text": "Book a hotel"}]}},
                {"author": "model", "content": {"parts": [{"text": "Sure, which city?"}]}},
            ]
            await redis_store.save_session_history(session_id, history)

            # Retrieve and verify
            retrieved = await redis_store.get_session_history(session_id)
            assert len(retrieved) == 2
            assert retrieved[0]["author"] == "user"
            assert retrieved[1]["author"] == "model"

            # State should be preserved when updating history
            await redis_store.save_session_state(session_id, {"booking_step": "city_selection"})
            state = await redis_store.get_session_state(session_id)
            assert state["booking_step"] == "city_selection"

            # Clear and verify default snapshot
            await redis_store.clear_session_snapshot(session_id)
            snapshot = await redis_store.get_session_snapshot(session_id)
            assert snapshot["history"] == []
            assert snapshot["state"] == {}

        asyncio.run(scenario())
    finally:
        redis_store._get_redis_client = original_client_loader
        _reset_test_state()


def test_graceful_redis_failure_midstream():
    """Test that a Redis failure mid-operation falls back gracefully to local storage."""
    _reset_test_state()

    async def scenario():
        session_id = "fallback-midstream-test"

        # First: save with working Redis (FakeRedis)
        fake_redis = FakeRedis()

        async def working_client():
            return fake_redis

        redis_store._get_redis_client = working_client
        await redis_store.save_session_snapshot(
            session_id=session_id,
            history=[{"author": "user", "content": "test"}],
            state={"step": 1},
        )

        # Verify data is in FakeRedis
        assert "adk:session:" + session_id in fake_redis.values

        # Now simulate Redis going down - return None (triggers local fallback)
        async def broken_client():
            return None

        redis_store._get_redis_client = broken_client

        # Should fall back to local storage (returns default snapshot since local is empty)
        snapshot = await redis_store.get_session_snapshot(session_id)
        assert snapshot["session_id"] == session_id
        # Local fallback returns default empty snapshot since we only saved to FakeRedis
        assert snapshot["history"] == []

        # Save to local fallback
        await redis_store.save_session_snapshot(
            session_id=session_id,
            history=[{"author": "user", "content": "fallback data"}],
            state={"step": 2},
        )

        # Verify local fallback works
        snapshot2 = await redis_store.get_session_snapshot(session_id)
        assert snapshot2["history"][0]["content"] == "fallback data"
        assert snapshot2["state"]["step"] == 2

    asyncio.run(scenario())
    _reset_test_state()


if __name__ == "__main__":
    print("Running test_session_snapshot_roundtrip_with_fake_redis...")
    test_session_snapshot_roundtrip_with_fake_redis()
    print("✓ PASSED")

    print("Running test_local_fallback_and_anomaly_smoke...")
    test_local_fallback_and_anomaly_smoke()
    print("✓ PASSED")

    print("Running test_anomaly_redis_integration...")
    test_anomaly_redis_integration()
    print("✓ PASSED")

    print("Running test_session_history_helpers...")
    test_session_history_helpers()
    print("✓ PASSED")

    print("Running test_graceful_redis_failure_midstream...")
    test_graceful_redis_failure_midstream()
    print("✓ PASSED")

    print("\n All Redis smoke tests passed!")
