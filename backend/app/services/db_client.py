from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    _DB_DRIVER_AVAILABLE = True
except Exception: 
    psycopg = None  
    dict_row = None 
    AsyncConnectionPool = Any  
    _DB_DRIVER_AVAILABLE = False

logger = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_POOL_CONNINFO: Optional[str] = None
_POOL_LOCK = asyncio.Lock()

_DEFAULT_TIMEOUT_SECONDS = float(os.getenv("DB_TIMEOUT_SECONDS", "3"))
_MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "1"))
_RETRY_BACKOFF_SECONDS = float(os.getenv("DB_RETRY_BACKOFF_SECONDS", "0.25"))
_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
_POOL_OPEN_TIMEOUT_SECONDS = float(os.getenv("DB_POOL_OPEN_TIMEOUT_SECONDS", "3"))


def build_conninfo(conninfo: Optional[str] = None) -> str:
    if conninfo:
        return conninfo

    db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL") or os.getenv("POSTGRES_URL")
    if db_url and db_url != "your_supabase_connection_string_here":
        return db_url

    host = os.getenv("SUPABASE_DB_HOST", "127.0.0.1")
    port = os.getenv("SUPABASE_DB_PORT", "54322")
    dbname = os.getenv("SUPABASE_DB_NAME", "postgres")
    user = os.getenv("SUPABASE_DB_USER", "postgres")
    password = os.getenv("SUPABASE_DB_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


async def _close_existing_pool() -> None:
    global _POOL, _POOL_CONNINFO
    if _POOL is not None:
        await _POOL.close()
    _POOL = None
    _POOL_CONNINFO = None


async def get_pool(conninfo: Optional[str] = None) -> AsyncConnectionPool:
    global _POOL, _POOL_CONNINFO
    if not _DB_DRIVER_AVAILABLE:
        raise RuntimeError("Database driver unavailable. Install psycopg[binary] and psycopg-pool.")
    target_conninfo = build_conninfo(conninfo)

    if _POOL is not None and _POOL_CONNINFO == target_conninfo:
        return _POOL

    async with _POOL_LOCK:
        if _POOL is not None and _POOL_CONNINFO == target_conninfo:
            return _POOL

        if _POOL is not None and _POOL_CONNINFO != target_conninfo:
            await _close_existing_pool()

        _POOL = AsyncConnectionPool(
            conninfo=target_conninfo,
            min_size=_POOL_MIN_SIZE,
            max_size=_POOL_MAX_SIZE,
            timeout=_POOL_OPEN_TIMEOUT_SECONDS,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        await _POOL.open(wait=True)
        _POOL_CONNINFO = target_conninfo
        return _POOL


async def close_pool() -> None:
    async with _POOL_LOCK:
        await _close_existing_pool()


async def _run_with_retry(op):
    last_error: Optional[Exception] = None
    retryable_errors = (asyncio.TimeoutError, OSError)
    if psycopg is not None:
        retryable_errors = retryable_errors + (psycopg.OperationalError, psycopg.InterfaceError)
    for attempt in range(_MAX_RETRIES):
        try:
            return await op()
        except retryable_errors as exc: 
            last_error = exc
            if attempt >= _MAX_RETRIES - 1:
                break
            delay = _RETRY_BACKOFF_SECONDS * (2 ** attempt)
            logger.warning("DB operation failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, exc)
            await asyncio.sleep(delay)
    if last_error:
        raise last_error
    raise RuntimeError("DB operation failed without a captured exception")


async def execute(
    query: str,
    params: Optional[Sequence[Any] | Dict[str, Any]] = None,
    *,
    timeout_seconds: Optional[float] = None,
    conninfo: Optional[str] = None,
) -> int:
    timeout = timeout_seconds or _DEFAULT_TIMEOUT_SECONDS

    async def _op() -> int:
        pool = await get_pool(conninfo)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await asyncio.wait_for(cur.execute(query, params), timeout=timeout)
                rowcount = cur.rowcount or 0
            await conn.commit()
        return int(rowcount)

    return await _run_with_retry(_op)


async def fetch_one(
    query: str,
    params: Optional[Sequence[Any] | Dict[str, Any]] = None,
    *,
    timeout_seconds: Optional[float] = None,
    conninfo: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    timeout = timeout_seconds or _DEFAULT_TIMEOUT_SECONDS

    async def _op() -> Optional[Dict[str, Any]]:
        pool = await get_pool(conninfo)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await asyncio.wait_for(cur.execute(query, params), timeout=timeout)
                row = await asyncio.wait_for(cur.fetchone(), timeout=timeout)
                return dict(row) if row else None

    return await _run_with_retry(_op)


async def fetch_all(
    query: str,
    params: Optional[Sequence[Any] | Dict[str, Any]] = None,
    *,
    timeout_seconds: Optional[float] = None,
    conninfo: Optional[str] = None,
) -> List[Dict[str, Any]]:
    timeout = timeout_seconds or _DEFAULT_TIMEOUT_SECONDS

    async def _op() -> List[Dict[str, Any]]:
        pool = await get_pool(conninfo)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await asyncio.wait_for(cur.execute(query, params), timeout=timeout)
                rows = await asyncio.wait_for(cur.fetchall(), timeout=timeout)
                return [dict(row) for row in rows]

    return await _run_with_retry(_op)


@asynccontextmanager
async def transaction(conninfo: Optional[str] = None) -> AsyncIterator[Any]:
    pool = await get_pool(conninfo)
    async with pool.connection() as conn:
        async with conn.transaction():
            yield conn
