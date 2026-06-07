"""Pluggable conversation-history backends.

The :class:`Session` protocol describes a simple CRUD surface for
``LLMMessage`` lists, keyed by a session id. Three built-in backends
are provided:

* :class:`InMemorySession` — ephemeral, useful for tests and notebooks.
* :class:`JsonlFileSession` — single-file append-only JSONL store;
  human-readable and dependency-free.
* :class:`SQLiteSession` — local on-disk persistence keyed by
  ``session_id``, uses a single SQLite file.

The legacy :class:`~clawagents.session.persistence.SessionWriter` /
:class:`~clawagents.session.persistence.SessionReader` remain the
JSONL-based event log used by the loop for trajectory-style replay.
``Session`` is aimed at the *messages*-level abstraction callers want
when they simply say "give me a chat-style memory": append messages on
each turn, read them back on the next.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from clawagents.providers.llm import LLMMessage


@runtime_checkable
class Session(Protocol):
    """Protocol for message-level conversation memory.

    Implementations must be safe for single-process concurrent use; they
    are *not* required to be safe across processes.
    """
    session_id: str

    async def get_items(self, limit: int | None = None) -> list[LLMMessage]: ...
    async def add_items(self, items: Iterable[LLMMessage]) -> None: ...
    async def pop_item(self) -> LLMMessage | None: ...
    async def clear_session(self) -> None: ...


def _message_to_dict(m: LLMMessage) -> dict[str, Any]:
    data: dict[str, Any] = {"role": m.role, "content": m.content}
    if getattr(m, "tool_call_id", None):
        data["tool_call_id"] = m.tool_call_id
    if getattr(m, "tool_calls_meta", None):
        data["tool_calls_meta"] = m.tool_calls_meta
    if getattr(m, "thinking", None):
        data["thinking"] = m.thinking
    return data


def _dict_to_message(d: dict[str, Any]) -> LLMMessage:
    return LLMMessage(
        role=d["role"],
        content=d.get("content", ""),
        tool_call_id=d.get("tool_call_id"),
        tool_calls_meta=d.get("tool_calls_meta"),
        thinking=d.get("thinking"),
    )


class InMemorySession:
    """Ephemeral in-process session; everything lives in a list."""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._items: list[LLMMessage] = []
        self._lock = threading.Lock()

    async def get_items(self, limit: int | None = None) -> list[LLMMessage]:
        with self._lock:
            items = list(self._items)
        if limit is not None and limit >= 0:
            return items[-limit:]
        return items

    async def add_items(self, items: Iterable[LLMMessage]) -> None:
        with self._lock:
            for it in items:
                self._items.append(it)

    async def pop_item(self) -> LLMMessage | None:
        with self._lock:
            if not self._items:
                return None
            return self._items.pop()

    async def clear_session(self) -> None:
        with self._lock:
            self._items.clear()


class JsonlFileSession:
    """JSONL-file-backed session — one line per message, appended on write.

    Good when ``sqlite3`` isn't desirable (read-only filesystems, debugging
    handoffs via ``tail -f``), or when callers want easy human-readable
    logs that can be diffed / version-controlled.

    Access within a process is serialised by an internal
    :class:`threading.Lock`. Cross-process concurrency is *not* guaranteed
    — use :class:`SQLiteSession` if you need that.
    """

    def __init__(
        self,
        session_id: str,
        *,
        file_path: str | Path | None = None,
        dir_path: str | Path | None = None,
    ) -> None:
        self.session_id = session_id
        if file_path is not None:
            self.file_path = Path(file_path).resolve()
        else:
            base = Path(dir_path).resolve() if dir_path is not None else (
                Path.cwd() / ".clawagents" / "sessions-memory"
            )
            self.file_path = (base / f"{session_id}.jsonl").resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _read_all(self) -> list[LLMMessage]:
        if not self.file_path.exists():
            return []
        items: list[LLMMessage] = []
        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    items.append(_dict_to_message(json.loads(stripped)))
                except (json.JSONDecodeError, KeyError, TypeError):
                    # skip malformed lines rather than bail the whole session
                    continue
        return items

    async def get_items(self, limit: int | None = None) -> list[LLMMessage]:
        with self._lock:
            items = self._read_all()
        if limit is not None and limit >= 0:
            return items[-limit:]
        return items

    async def add_items(self, items: Iterable[LLMMessage]) -> None:
        items_list = list(items)
        if not items_list:
            return
        payload = "\n".join(
            json.dumps(_message_to_dict(it), ensure_ascii=False)
            for it in items_list
        ) + "\n"
        with self._lock:
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(payload)

    async def pop_item(self) -> LLMMessage | None:
        with self._lock:
            items = self._read_all()
            if not items:
                return None
            popped = items.pop()
            if items:
                payload = "\n".join(
                    json.dumps(_message_to_dict(it), ensure_ascii=False)
                    for it in items
                ) + "\n"
            else:
                payload = ""
            with self.file_path.open("w", encoding="utf-8") as f:
                f.write(payload)
            return popped

    async def clear_session(self) -> None:
        with self._lock:
            if self.file_path.exists():
                with self.file_path.open("w", encoding="utf-8") as f:
                    f.write("")


class SQLiteSession:
    """SQLite-backed session; messages persist across process restarts.

    The schema is two tables: ``sessions`` (session metadata) and
    ``messages`` (ordered message list, one row per message). Access is
    serialised by an internal :class:`threading.Lock` so that multiple
    coroutines on the same event loop don't race.
    """

    _SCHEMA = [
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at REAL DEFAULT (strftime('%s', 'now'))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            ord INTEGER NOT NULL,
            payload TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, ord)",
    ]

    def __init__(self, session_id: str, db_path: str | Path = ".clawagents/sessions.db"):
        self.session_id = session_id
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._conn() as conn:
            for stmt in self._SCHEMA:
                conn.execute(stmt)
            conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id) VALUES (?)",
                (self.session_id,),
            )
            conn.commit()

    async def get_items(self, limit: int | None = None) -> list[LLMMessage]:
        with self._lock, self._conn() as conn:
            if limit is None or limit < 0:
                cur = conn.execute(
                    "SELECT payload FROM messages WHERE session_id = ? ORDER BY ord ASC",
                    (self.session_id,),
                )
                rows = cur.fetchall()
            else:
                cur = conn.execute(
                    "SELECT payload FROM messages WHERE session_id = ? ORDER BY ord DESC LIMIT ?",
                    (self.session_id, limit),
                )
                rows = list(reversed(cur.fetchall()))
        return [_dict_to_message(json.loads(r[0])) for r in rows]

    async def add_items(self, items: Iterable[LLMMessage]) -> None:
        items_list = list(items)
        if not items_list:
            return
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(ord), -1) FROM messages WHERE session_id = ?",
                (self.session_id,),
            )
            next_ord = (cur.fetchone()[0] or -1) + 1
            conn.executemany(
                "INSERT INTO messages(session_id, ord, payload) VALUES (?, ?, ?)",
                [
                    (self.session_id, next_ord + i, json.dumps(_message_to_dict(it)))
                    for i, it in enumerate(items_list)
                ],
            )
            conn.commit()

    async def pop_item(self) -> LLMMessage | None:
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "SELECT id, payload FROM messages WHERE session_id = ? "
                "ORDER BY ord DESC LIMIT 1",
                (self.session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM messages WHERE id = ?", (row[0],))
            conn.commit()
            return _dict_to_message(json.loads(row[1]))

    async def clear_session(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (self.session_id,),
            )
            conn.commit()

    async def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """FTS5 full-text search over session messages."""
        from clawagents.session.search import search_session_messages

        with self._lock, self._conn() as conn:
            hits = search_session_messages(conn, self.session_id, query, limit=limit)
        return [
            {
                "message_id": h.message_id,
                "ord": h.ord,
                "role": h.role,
                "snippet": h.snippet,
                "rank": h.rank,
            }
            for h in hits
        ]

    async def undo_last(self, count: int = 1) -> list[LLMMessage]:
        """Pop the last *count* messages (Hermes /undo)."""
        removed: list[LLMMessage] = []
        for _ in range(max(1, count)):
            item = await self.pop_item()
            if item is None:
                break
            removed.append(item)
        removed.reverse()
        return removed
