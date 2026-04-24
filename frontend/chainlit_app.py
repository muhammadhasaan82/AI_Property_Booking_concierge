import asyncio
import concurrent.futures
import multiprocessing
from pathlib import Path
import os
import sys
from uuid import uuid4

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
_frontend_root = Path(__file__).resolve().parents[1] / "frontend"
os.environ.setdefault("CHAINLIT_APP_ROOT", str(_frontend_root))

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv
from sqlalchemy.engine import make_url
from sqlalchemy import text

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)
load_dotenv(dotenv_path=Path(__file__).resolve().parent[1]/"backend"/".env")
os.environ["SUPABASE_DB_URL"] = os.getenv("SUPABASE_DB_URL", "")
os.environ["SUPABASE_DB_USER"] = os.getenv("SUPABASE_DB_USER", "")
os.environ["SUPABASE_DB_PASSWORD"] = os.getenv("SUPABASE_DB_PASSWORD", "")


from app.services.adk_runner import run_adk_turn

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username == "login_username" and password == "login_password":
        return cl.User(identifier=username, metadata={"role": "admin"})
    return None
LOCAL_HISTORY_DB = Path(__file__).resolve().parents[1] / "local_chat_history.db"
LOCAL_HISTORY_CONNINFO = f"sqlite+aiosqlite:///{LOCAL_HISTORY_DB.as_posix()}"

POSTGRES_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        "id" UUID PRIMARY KEY,
        "identifier" TEXT UNIQUE NOT NULL,
        "metadata" JSONB NOT NULL,
        "createdAt" TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS threads (
        "id" UUID PRIMARY KEY,
        "createdAt" TEXT,
        "name" TEXT,
        "userId" UUID,
        "userIdentifier" TEXT,
        "tags" TEXT[],
        "metadata" JSONB,
        FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS steps (
        "id" UUID PRIMARY KEY,
        "name" TEXT NOT NULL,
        "type" TEXT NOT NULL,
        "threadId" UUID NOT NULL,
        "parentId" UUID,
        "disableFeedback" BOOLEAN NOT NULL DEFAULT FALSE,
        "streaming" BOOLEAN NOT NULL DEFAULT FALSE,
        "waitForAnswer" BOOLEAN,
        "isError" BOOLEAN NOT NULL DEFAULT FALSE,
        "metadata" JSONB,
        "tags" TEXT[],
        "input" TEXT,
        "output" TEXT,
        "createdAt" TEXT,
        "start" TEXT,
        "end" TEXT,
        "generation" JSONB,
        "showInput" TEXT,
        "language" TEXT,
        "indent" INT,
        FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elements (
        "id" UUID PRIMARY KEY,
        "threadId" UUID,
        "type" TEXT,
        "chainlitKey" TEXT,
        "url" TEXT,
        "objectKey" TEXT,
        "name" TEXT NOT NULL,
        "display" TEXT,
        "size" TEXT,
        "language" TEXT,
        "page" INT,
        "autoPlay" BOOLEAN,
        "playerConfig" JSONB,
        "forId" UUID,
        "mime" TEXT,
        "props" JSONB,  -- Added for Chainlit v1.1.0+ compatibility
        FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedbacks (
        "id" UUID PRIMARY KEY,
        "forId" UUID NOT NULL,
        "value" INT NOT NULL,
        "comment" TEXT
    )
    """,
]

SQLITE_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        "id" TEXT PRIMARY KEY,
        "identifier" TEXT UNIQUE NOT NULL,
        "metadata" TEXT NOT NULL,
        "createdAt" TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS threads (
        "id" TEXT PRIMARY KEY,
        "createdAt" TEXT,
        "name" TEXT,
        "userId" TEXT,
        "userIdentifier" TEXT,
        "tags" TEXT,
        "metadata" TEXT,
        FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS steps (
        "id" TEXT PRIMARY KEY,
        "name" TEXT NOT NULL,
        "type" TEXT NOT NULL,
        "threadId" TEXT NOT NULL,
        "parentId" TEXT,
        "disableFeedback" INTEGER NOT NULL DEFAULT 0,
        "streaming" INTEGER NOT NULL DEFAULT 0,
        "waitForAnswer" INTEGER,
        "isError" INTEGER NOT NULL DEFAULT 0,
        "metadata" TEXT,
        "tags" TEXT,
        "input" TEXT,
        "output" TEXT,
        "createdAt" TEXT,
        "start" TEXT,
        "end" TEXT,
        "generation" TEXT,
        "showInput" TEXT,
        "language" TEXT,
        "indent" INTEGER,
        FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elements (
        "id" TEXT PRIMARY KEY,
        "threadId" TEXT,
        "type" TEXT,
        "chainlitKey" TEXT,
        "url" TEXT,
        "objectKey" TEXT,
        "name" TEXT NOT NULL,
        "display" TEXT,
        "size" TEXT,
        "language" TEXT,
        "page" INTEGER,
        "autoPlay" INTEGER,
        "playerConfig" TEXT,
        "forId" TEXT,
        "mime" TEXT,
        "props" TEXT,  -- Added for Chainlit v1.1.0+ compatibility
        FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedbacks (
        "id" TEXT PRIMARY KEY,
        "forId" TEXT NOT NULL,
        "value" INTEGER NOT NULL,
        "comment" TEXT
    )
    """,
]

WELCOME_MESSAGE = "Welcome to AI Booking! How can I help you find a stay today?"


def _normalize_conninfo(conninfo: str) -> str:
    def _with_pgbouncer_safe_psycopg_options(url_text: str) -> str:
        try:
            url = make_url(url_text)
            query = dict(url.query)
            query.setdefault("prepare_threshold", "None")
            return str(url.set(query=query))
        except Exception:
            separator = "&" if "?" in url_text else "?"
            return f"{url_text}{separator}prepare_threshold=None"

    if conninfo.startswith("postgres://"):
        conninfo = "postgresql+psycopg://" + conninfo[len("postgres://") :]
        return _with_pgbouncer_safe_psycopg_options(conninfo)
    if conninfo.startswith("postgresql+asyncpg://"):
        conninfo = "postgresql+psycopg://" + conninfo[len("postgresql+asyncpg://") :]
        return _with_pgbouncer_safe_psycopg_options(conninfo)
    if conninfo.startswith("postgresql://"):
        conninfo = "postgresql+psycopg://" + conninfo[len("postgresql://") :]
        return _with_pgbouncer_safe_psycopg_options(conninfo)
    if conninfo.startswith("postgresql+psycopg://"):
        return _with_pgbouncer_safe_psycopg_options(conninfo)
    if conninfo.startswith("sqlite:///"):
        return "sqlite+aiosqlite:///" + conninfo[len("sqlite:///") :]
    return conninfo


def _resolve_history_conninfo() -> str:
    for env_name in ("DATABASE_URL", "POSTGRES_URL", "SUPABASE_URL", "SUPABASE_DB_URL"):
        raw_value = (os.getenv(env_name) or "").strip()
        if raw_value.startswith(
            (
                "postgres://",
                "postgresql://",
                "postgresql+asyncpg://",
                "postgresql+psycopg://",
                "sqlite:///",
                "sqlite+aiosqlite:///",
            )
        ):
            return _normalize_conninfo(raw_value)
    return LOCAL_HISTORY_CONNINFO


def _schema_statements_for(conninfo: str):
    if conninfo.startswith("sqlite"):
        return SQLITE_SCHEMA_STATEMENTS
    return POSTGRES_SCHEMA_STATEMENTS


@cl.data_layer
def get_data_layer():
    conninfo = _resolve_history_conninfo()
    return SQLAlchemyDataLayer(conninfo=conninfo)

def _get_data_layer():
    try:
        return cl_data.get_data_layer()
    except Exception:
        return None


async def _rename_thread(message: cl.Message, question: str) -> None:
    if not question:
        return

    short_q = question[:15].rstrip()
    if len(question) > 15:
        short_q += "..."

    try:
        data_layer = _get_data_layer()
        thread_id = getattr(message, "thread_id", None)
        if data_layer and thread_id:
            await data_layer.update_thread(thread_id=thread_id, name=f"Booking: {short_q}")
    except Exception:
        pass



@cl.on_chat_resume
async def on_chat_resume(thread):
    if isinstance(thread, dict):
        past_thread_id = thread.get("id")
        if past_thread_id:
            cl.user_session.set("past_thread_id", past_thread_id)
            cl.user_session.set("session_id", past_thread_id)

_pool_initialized = False
@cl.on_chat_start
async def on_chat_start():
    global _pool_initialized
    if not _pool_initialized:
        workers = multiprocessing.cpu_count() *5
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        try:
            asyncio.get_running_loop().set_default_executor(pool)
            print(f"[*] Chainlit Multithread Pool Initialized with {workers} workers")
        except Exception as e:
            print(f"[*] Could not set thread pool: {e}")
        _pool_initialized = True 
    cl.user_session.set("app_name", "AI Booking")
    cl.user_session.set("app_description", "AI Property Booking Concierge")

    data_layer = cl_data.get_data_layer()
    if data_layer and hasattr(data_layer, "engine"):
        try:

            async with data_layer.engine.begin() as conn:
                await conn.run_sync(cl.data.sql_alchemy.Base.metadata.create_all)
            print("Schema automatically created by Chainlit.")
        except Exception as e:
            print(f"Schema Error: {e}")

    await cl.Message(content=WELCOME_MESSAGE).send()


def _resolve_session_id(message: cl.Message) -> str:
    thread_id = getattr(message, "thread_id", None)
    if thread_id:
        cl.user_session.set("session_id", thread_id)
        return thread_id

    stored = (
        cl.user_session.get("session_id")
        or cl.user_session.get("past_thread_id")
        or cl.user_session.get("id")
    )
    if stored:
        return stored

    generated = str(uuid4())
    cl.user_session.set("session_id", generated)
    return generated


@cl.on_message
async def on_message(message: cl.Message):
    await _rename_thread(message, (message.content or "").strip())

    user_obj = cl.user_session.get("user")
    user_id = getattr(user_obj, "identifier", "anonymous") if user_obj else "anonymous"
    session_id = _resolve_session_id(message)
    msg = cl.Message(content="")
    await msg.send()

    async for chunk in run_adk_turn(
        user_id=user_id,
        session_id=session_id,
        message=message.content,
    ):
        await msg.stream_token(chunk)


    await msg.update()
