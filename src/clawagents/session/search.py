"""In-session and cross-session SQLite message search (LIKE + optional FTS5)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from clawagents.session.snippet import snippet_from_content


@dataclass
class SessionSearchHit:
    message_id: int
    ord: int
    role: str
    snippet: str
    rank: float


@dataclass
class SqliteSearchRow:
    session_id: str
    message_id: int
    ord: int
    role: str
    content: str


def search_sqlite_messages(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
    session_id: str | None = None,
    order_desc: bool = True,
) -> list[SqliteSearchRow]:
    """Search message payloads in SQLite; optional session filter."""
    token = query.strip()
    if not token:
        return []
    order = "id DESC" if order_desc else "ord ASC"
    pattern = f"%{token.lower()}%"
    if session_id:
        cur = conn.execute(
            f"""
            SELECT session_id, id, ord,
                   coalesce(json_extract(payload, '$.role'), '') AS role,
                   coalesce(json_extract(payload, '$.content'), '') AS content
            FROM messages
            WHERE session_id = ? AND lower(payload) LIKE ?
            ORDER BY {order}
            LIMIT ?
            """,
            (session_id, pattern, limit),
        )
    else:
        cur = conn.execute(
            f"""
            SELECT session_id, id, ord,
                   coalesce(json_extract(payload, '$.role'), '') AS role,
                   coalesce(json_extract(payload, '$.content'), '') AS content
            FROM messages
            WHERE lower(payload) LIKE ?
            ORDER BY {order}
            LIMIT ?
            """,
            (pattern, limit),
        )
    rows: list[SqliteSearchRow] = []
    for row in cur.fetchall():
        rows.append(
            SqliteSearchRow(
                session_id=str(row[0]),
                message_id=int(row[1]),
                ord=int(row[2]),
                role=str(row[3] or ""),
                content=str(row[4] or ""),
            )
        )
    return rows


def search_session_messages(
    conn: sqlite3.Connection,
    session_id: str,
    query: str,
    *,
    limit: int = 20,
) -> list[SessionSearchHit]:
    """Search messages within one session (conversation order)."""
    token = query.strip()
    if not token:
        return []
    rows = search_sqlite_messages(
        conn, query, limit=limit, session_id=session_id, order_desc=False,
    )
    return [
        SessionSearchHit(
            message_id=row.message_id,
            ord=row.ord,
            role=row.role,
            snippet=snippet_from_content(row.content, token),
            rank=1.0,
        )
        for row in rows
    ]


def format_search_hits(hits: list[SessionSearchHit]) -> str:
    if not hits:
        return "No matches."
    lines = []
    for h in hits:
        lines.append(f"- #{h.ord} ({h.role}) {h.snippet.strip()}")
    return "\n".join(lines)


def ensure_fts5(conn: sqlite3.Connection) -> None:
    """Optional FTS5 index for read-heavy search workloads (insert-only)."""
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            session_id UNINDEXED,
            message_id UNINDEXED,
            tokenize='porter'
        )
        """
    )
