"""Where each conversation's Submilli session is kept between turns."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, runtime_checkable


class SubmilliSession(TypedDict):
    """The session a conversation runs its programs in."""

    id: str
    server: str
    blueprint: str
    variables: dict[str, str]
    #: A fingerprint of the application tools the session was opened with.
    tools: NotRequired[str | None]


@runtime_checkable
class SessionStore(Protocol):
    """Keeps the Submilli session of each conversation, by conversation id.

    Implement it over the application's own storage to keep sessions across
    restarts and processes. `InMemorySessionStore` and `SQLiteSessionStore`
    are provided.
    """

    async def get(self, conversation_id: str) -> SubmilliSession | None:
        """The conversation's session, or `None`."""
        ...

    async def set(self, conversation_id: str, session: SubmilliSession | None) -> None:
        """Record the conversation's session; `None` forgets it."""
        ...


class InMemorySessionStore:
    """Sessions in this process's memory. They are forgotten when it exits."""

    def __init__(self) -> None:
        self._sessions: dict[str, SubmilliSession] = {}

    async def get(self, conversation_id: str) -> SubmilliSession | None:
        """The conversation's session, or `None`."""
        return self._sessions.get(conversation_id)

    async def set(self, conversation_id: str, session: SubmilliSession | None) -> None:
        """Record the conversation's session; `None` forgets it."""
        if session is None:
            self._sessions.pop(conversation_id, None)
        else:
            self._sessions[conversation_id] = session


class SQLiteSessionStore:
    """Sessions in a SQLite database, in their own table.

    It can share a file with the Agents SDK's `SQLiteSession`, so that a
    conversation's history and its Submilli session are kept together.

    Args:
        db_path: The database file. `":memory:"` keeps it in memory.
        table: The table's name.
    """

    def __init__(self, db_path: str | Path = ":memory:", *, table: str = "submilli_sessions") -> None:
        if not table.isidentifier():
            raise ValueError(f"table must be an identifier; got {table!r}")
        self.db_path: str | Path = db_path
        self.table: str = table
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        with self._lock, self._connection:
            self._connection.execute(
                f"CREATE TABLE IF NOT EXISTS {table} (conversation_id TEXT PRIMARY KEY, session TEXT NOT NULL)"
            )

    async def get(self, conversation_id: str) -> SubmilliSession | None:
        """The conversation's session, or `None`."""
        return await asyncio.to_thread(self._get, conversation_id)

    async def set(self, conversation_id: str, session: SubmilliSession | None) -> None:
        """Record the conversation's session; `None` forgets it."""
        await asyncio.to_thread(self._set, conversation_id, session)

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()

    def _get(self, conversation_id: str) -> SubmilliSession | None:
        with self._lock:
            row = self._connection.execute(
                f"SELECT session FROM {self.table} WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
        if row is None:
            return None
        session: SubmilliSession = json.loads(row[0])
        return session

    def _set(self, conversation_id: str, session: SubmilliSession | None) -> None:
        with self._lock, self._connection:
            if session is None:
                self._connection.execute(f"DELETE FROM {self.table} WHERE conversation_id = ?", (conversation_id,))
            else:
                self._connection.execute(
                    f"INSERT OR REPLACE INTO {self.table} (conversation_id, session) VALUES (?, ?)",
                    (conversation_id, json.dumps(session)),
                )
