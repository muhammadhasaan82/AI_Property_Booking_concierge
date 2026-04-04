import asyncio
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure Chainlit loads config and public assets from frontend/
_frontend_root = Path(__file__).resolve().parents[1] / "frontend"
os.environ.setdefault("CHAINLIT_APP_ROOT", str(_frontend_root))

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv
from sqlalchemy import text

# Fix sys.path to allow importing backend/app directly as app.*
_backend_root = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(_backend_root))

from app.services.adk_runner import run_adk_turn

# ---------------------------------------------------------------------------
# Authentication - Password Login
# ---------------------------------------------------------------------------
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username == "admin" and password == "123":
        return cl.User(identifier=username, metadata={"role": "admin"})
    return None

load_dotenv()

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
    if conninfo.startswith("postgres://"):
        return "postgresql+psycopg://" + conninfo[len("postgres://") :]
    if conninfo.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg://" + conninfo[len("postgresql+asyncpg://") :]
    if conninfo.startswith("postgresql://"):
        return "postgresql+psycopg://" + conninfo[len("postgresql://") :]
    if conninfo.startswith("sqlite:///"):
        return "sqlite+aiosqlite:///" + conninfo[len("sqlite:///") :]
    return conninfo


def _resolve_history_conninfo() -> str:
    # Add "SUPABASE_DB_URL" to this list
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
    # Chainlit's official layer automatically creates the schema tables for you!
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


@cl.on_chat_start
async def on_chat_start():
    # --- FORCE UI CONFIGURATION ---
    # This overrides the config.toml file
    cl.user_session.set("app_name", "AI Booking")
    cl.user_session.set("app_description", "AI Property Booking Concierge")

    # Optional: If you want to force the theme via Python
    # cl.user_session.set("theme", "light")
    # ------------------------------

    # --- Safely ensure tables exist ---
    data_layer = cl_data.get_data_layer()
    if data_layer and hasattr(data_layer, "engine"):
        conninfo = _resolve_history_conninfo()
        try:
            async with data_layer.engine.begin() as connection:
                if conninfo.startswith("sqlite"):
                    await connection.execute(text("PRAGMA foreign_keys = ON"))
                for statement in _schema_statements_for(conninfo):
                    await connection.execute(text(statement))
        except Exception as e:
            print(f"Schema Error: {e}")
    # ---------------------------------------

    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message):
    await _rename_thread(message, (message.content or "").strip())

    # ── V2 ADK Pipeline (streaming) — sole path ─────────────
    user_obj = cl.user_session.get("user")
    user_id = getattr(user_obj, "identifier", "anonymous") if user_obj else "anonymous"
    session_id = cl.user_session.get("id", "default_session")

    # Phase 1 — Show a "thinking" step immediately so the UI is never blank.
    # The step closes as soon as the first chunk arrives, then the message streams
    # in clean isolation below it. This gives instant feedback without the step
    # overlapping the streaming reply.
    adk_gen = run_adk_turn(
        user_id=user_id,
        session_id=session_id,
        message=message.content,
    )

    # Prime the generator to get the first token before opening the message
    first_chunk: str = ""
    async with cl.Step(name="Consulting tools…", type="run") as thinking_step:
        thinking_step.input = message.content
        try:
            first_chunk = await adk_gen.__anext__()
        except StopAsyncIteration:
            first_chunk = ""
        thinking_step.output = "Routing complete."
    # Step is now closed — UI shows a clean ✓ tick next to "Consulting tools…"

    # Phase 2 — Stream the reply character-by-character below the closed step.
    msg = cl.Message(content="")
    await msg.send()

    if first_chunk:
        await msg.stream_token(first_chunk)

    async for chunk in adk_gen:
        await msg.stream_token(chunk)

    # Finalise (flushes any buffered content and marks the message complete)
    await msg.update()
