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
            stats = await anomaly.get_session_stats("fallback-anomaly")
            assert stats["total_tool_calls"] == 3
            assert stats["tool_counts"]["search_properties"] == 3
            assert await anomaly.check_tool_loop("fallback-anomaly", "search_properties", {"city": "Lahore"}) is True

        asyncio.run(scenario())
    finally:
        redis_store._get_redis_client = original_client_loader
        anomaly.get_redis_client = original_anomaly_loader
        _reset_test_state()


if __name__ == "__main__":
    test_session_snapshot_roundtrip_with_fake_redis()
    test_local_fallback_and_anomaly_smoke()
