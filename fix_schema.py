import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# The verified working connection string
DB_URL = "postgresql+psycopg://supabase_admin:iNzl5DdQK3F9AOsf@172.21.0.4:5432/postgres"

# The missing columns (defaultOpen, autoCollapse) added for Chainlit compatibility
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (id UUID PRIMARY KEY, identifier TEXT UNIQUE NOT NULL, metadata JSONB NOT NULL, "createdAt" TEXT);""",
    """CREATE TABLE IF NOT EXISTS threads (id UUID PRIMARY KEY, "createdAt" TEXT, name TEXT, "userId" UUID, "userIdentifier" TEXT, tags TEXT[], metadata JSONB, FOREIGN KEY ("userId") REFERENCES users(id) ON DELETE CASCADE);""",
    """CREATE TABLE IF NOT EXISTS steps (id UUID PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, "threadId" UUID NOT NULL, "parentId" UUID, "disableFeedback" BOOLEAN NOT NULL DEFAULT FALSE, streaming BOOLEAN NOT NULL DEFAULT FALSE, "waitForAnswer" BOOLEAN, "isError" BOOLEAN NOT NULL DEFAULT FALSE, metadata JSONB, tags TEXT[], input TEXT, output TEXT, "createdAt" TEXT, start TEXT, "end" TEXT, generation JSONB, "showInput" TEXT, language TEXT, indent INT, "defaultOpen" BOOLEAN DEFAULT FALSE, "autoCollapse" BOOLEAN DEFAULT FALSE, FOREIGN KEY ("threadId") REFERENCES threads(id) ON DELETE CASCADE);""",
    """CREATE TABLE IF NOT EXISTS elements (id UUID PRIMARY KEY, "threadId" UUID, type TEXT, "chainlitKey" TEXT, url TEXT, "objectKey" TEXT, name TEXT NOT NULL, display TEXT, size TEXT, language TEXT, page INT, "autoPlay" BOOLEAN, "playerConfig" JSONB, "forId" UUID, mime TEXT, props JSONB, FOREIGN KEY ("threadId") REFERENCES threads(id) ON DELETE CASCADE);""",
    """CREATE TABLE IF NOT EXISTS feedbacks (id UUID PRIMARY KEY, "forId" UUID NOT NULL, value INT NOT NULL, comment TEXT);"""
]

async def fix():
    engine = create_async_engine(DB_URL)
    async with engine.begin() as conn:
        for stmt in SCHEMA:
            await conn.execute(text(stmt))
    print("Tables created successfully with all required columns.")

if __name__ == "__main__":
    asyncio.run(fix())
