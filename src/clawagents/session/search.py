"""In-session and cross-session SQLite message search (LIKE + optional FTS5)."""

from __future__ import annotations

import re
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
    """Search message payloads in SQLite; prefer FTS5 BM25 when indexed."""
    token = query.strip()
    if not token:
        return []
    # Prefer FTS5 when the virtual table exists
    try:
        ensure_fts5(conn)
        fts_q = " OR ".join(re.findall(r"\w+", token)) or token
        if session_id:
            cur = conn.execute(
                """
                SELECT m.session_id, m.id, m.ord,
                       coalesce(json_extract(m.payload, '$.role'), '') AS role,
                       coalesce(json_extract(m.payload, '$.content'), '') AS content,
                       bm25(messages_fts) AS rank
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.message_id
                WHERE messages_fts MATCH ? AND m.session_id = ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_q, session_id, limit),
            )
        else:
            cur = conn.execute(
                """
                SELECT m.session_id, m.id, m.ord,
                       coalesce(json_extract(m.payload, '$.role'), '') AS role,
                       coalesce(json_extract(m.payload, '$.content'), '') AS content,
                       bm25(messages_fts) AS rank
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.message_id
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_q, limit),
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
        if rows:
            return rows
    except Exception:
        pass
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
    rows = []
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
    """Create FTS5 index + triggers so inserts/deletes stay in sync."""
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
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(content, session_id, message_id)
            VALUES (
                coalesce(json_extract(new.payload, '$.content'), new.payload),
                new.session_id,
                new.id
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE message_id = old.id;
        END
        """
    )
    # Backfill once if the FTS table is empty but messages exist.
    try:
        n_fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        n_msg = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        if n_msg and not n_fts:
            conn.execute(
                """
                INSERT INTO messages_fts(content, session_id, message_id)
                SELECT coalesce(json_extract(payload, '$.content'), payload),
                       session_id, id
                FROM messages
                """
            )
    except Exception:
        pass
