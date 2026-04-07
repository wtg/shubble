"""LangChain PostgreSQL memory for Bubble.

Each Bubble run is stored as a HumanMessage (full context sent to Gemini) paired
with an AIMessage (the JSON response, prefixed with a UTC timestamp).  On the next
run, the last BUBBLE_MEMORY_EXCHANGES AI responses are injected into the prompt so
Bubble can see its recent decision history.
"""
import logging
import uuid
from datetime import datetime, timezone

import psycopg
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory

logger = logging.getLogger(__name__)

# Deterministic UUID derived from "bubble" — stable across restarts, required by langchain-postgres
_SESSION_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "bubble"))
_TABLE_NAME = "bubble_memory"


def _psycopg_url(database_url: str) -> str:
    """Strip SQLAlchemy driver prefix so psycopg3 can use the URL."""
    return (
        database_url
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql+psycopg://", "postgresql://")
        .replace("postgresql+psycopg2://", "postgresql://")
    )


async def ensure_table(database_url: str) -> None:
    """Create the memory table if it doesn't already exist."""
    conn_string = _psycopg_url(database_url)
    async with await psycopg.AsyncConnection.connect(conn_string) as conn:
        await PostgresChatMessageHistory.acreate_tables(conn, _TABLE_NAME)
    logger.debug("Memory table '%s' ready", _TABLE_NAME)


async def load_exchanges(database_url: str, n: int) -> list[tuple[str, str]]:
    """Return the last *n* (user_content, ai_response) pairs, oldest first.

    Only AI responses are used when building the prompt; user_content is
    stored for transparency/debugging.
    """
    conn_string = _psycopg_url(database_url)
    async with await psycopg.AsyncConnection.connect(conn_string) as conn:
        history = PostgresChatMessageHistory(
            _TABLE_NAME,
            _SESSION_ID,
            async_connection=conn,
        )
        messages: list[BaseMessage] = await history.aget_messages()

    pairs: list[tuple[str, str]] = []
    it = iter(messages)
    for human in it:
        ai = next(it, None)
        if isinstance(human, HumanMessage) and isinstance(ai, AIMessage):
            pairs.append((human.content, ai.content))

    return pairs[-n:]


async def save_exchange(database_url: str, user_content: str, ai_response: str) -> None:
    """Persist one run's input and output to the memory table."""
    conn_string = _psycopg_url(database_url)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    async with await psycopg.AsyncConnection.connect(conn_string) as conn:
        history = PostgresChatMessageHistory(
            _TABLE_NAME,
            _SESSION_ID,
            async_connection=conn,
        )
        await history.aadd_messages([
            HumanMessage(content=user_content),
            AIMessage(content=f"[{timestamp}] {ai_response}"),
        ])
    logger.debug("Saved exchange to Bubble memory")
