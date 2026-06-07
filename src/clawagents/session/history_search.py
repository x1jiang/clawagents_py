"""Cross-session message archive search (SQLite messages + JSONL event logs).

Use :func:`search_session_messages` / ``SQLiteSession.search()`` for the
**current** session. This module searches the **archive** across sessions.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clawagents.paths import get_clawagents_workspace_dir, get_sessions_dir
from clawagents.session.search import search_sqlite_messages
from clawagents.session.snippet import snippet_from_content


@dataclass
class HistorySearchHit:
    session_id: str
    source: str
    role: str
    content: str
    snippet: str
    message_id: int | None = None
    ord: int | None = None
    ts: float | None = None


def _sqlite_rows_to_hits(rows: list, query: str) -> list[HistorySearchHit]:
    token = query.strip()
    hits: list[HistorySearchHit] = []
    for row in rows:
        content = row.content[:4000]
        hits.append(
            HistorySearchHit(
                session_id=row.session_id,
                source="sqlite",
                message_id=row.message_id,
                ord=row.ord,
                role=row.role,
                content=content,
                snippet=snippet_from_content(content, token),
            )
        )
    return hits


def search_sqlite_history(
    db_path: Path,
    query: str,
    *,
    limit: int = 20,
    session_id: str | None = None,
) -> list[HistorySearchHit]:
    token = query.strip()
    if not token or not db_path.is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = search_sqlite_messages(
            conn, query, limit=limit, session_id=session_id, order_desc=True,
        )
        return _sqlite_rows_to_hits(rows, token)
    finally:
        conn.close()


def search_jsonl_history(
    sessions_dir: Path,
    query: str,
    *,
    limit: int = 20,
) -> list[HistorySearchHit]:
    token = query.strip().lower()
    if not token or not sessions_dir.is_dir():
        return []

    hits: list[HistorySearchHit] = []
    for path in sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(hits) >= limit:
            break
        session_id = path.stem
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ev = json.loads(line)
                    ev_type = ev.get("type", "")
                    role = ""
                    content = ""
                    if ev_type == "assistant_message":
                        role = "assistant"
                        content = str(ev.get("content", "") or "")
                    elif ev_type == "tool_result":
                        role = "tool"
                        content = str(ev.get("output", "") or "")
                    else:
                        continue
                    if token not in content.lower():
                        continue
                    hits.append(
                        HistorySearchHit(
                            session_id=session_id,
                            source="jsonl",
                            role=role,
                            content=content[:4000],
                            snippet=snippet_from_content(content, query.strip()),
                            ts=float(ev.get("ts") or 0) or None,
                        )
                    )
                    if len(hits) >= limit:
                        break
        except (OSError, json.JSONDecodeError):
            continue
    return hits


def _resolve_archive_paths(workspace: str | Path | None) -> tuple[Path, Path]:
    """Return (sessions.db path, JSONL sessions dir) for a workspace root."""
    if workspace:
        root = Path(workspace)
        return root / ".clawagents" / "sessions.db", root / ".clawagents" / "sessions"
    ws = get_clawagents_workspace_dir(create=False)
    return ws / "sessions.db", get_sessions_dir(scope="workspace", create=False)


def search_history(
    query: str,
    *,
    limit: int = 20,
    session_id: str | None = None,
    workspace: str | Path | None = None,
    include_jsonl: bool = True,
) -> list[HistorySearchHit]:
    """Search past messages across sessions (SQLite archive + optional JSONL)."""
    db_path, jsonl_dir = _resolve_archive_paths(workspace)
    remaining = max(1, limit)
    hits: list[HistorySearchHit] = []

    sqlite_hits = search_sqlite_history(db_path, query, limit=remaining, session_id=session_id)
    hits.extend(sqlite_hits)
    remaining = limit - len(hits)

    if include_jsonl and remaining > 0 and session_id is None:
        hits.extend(search_jsonl_history(jsonl_dir, query, limit=remaining))

    return hits[:limit]


def serialize_history_hits(hits: list[HistorySearchHit]) -> list[dict[str, Any]]:
    return [asdict(h) for h in hits]


def format_history_hits(hits: list[HistorySearchHit]) -> str:
    if not hits:
        return "No matching messages in past sessions."
    lines: list[str] = []
    for h in hits:
        loc = f"{h.session_id}"
        if h.ord is not None:
            loc += f"#{h.ord}"
        lines.append(f"- [{h.source}] {loc} ({h.role})\n  {h.snippet.strip()}")
    return "\n".join(lines)


def format_search_history_response(
    query: str,
    hits: list[HistorySearchHit],
    *,
    as_json: bool = False,
) -> str:
    if as_json:
        return json.dumps({"query": query, "hits": serialize_history_hits(hits)}, indent=2)
    header = f"Found {len(hits)} match(es) for {query!r}:\n"
    return header + format_history_hits(hits)
