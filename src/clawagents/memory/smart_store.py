"""Smart memory store — access boost, temporal decay, blake2 dedup, hybrid search.

Grok Build parity (xai-grok-memory): dump-all lessons/facts get ranking that
promotes frequently retrieved items and decays stale session memories.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


def _blake2_hex(content: str | bytes) -> str:
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.blake2b(data, digest_size=16).hexdigest()


@dataclass
class MemoryChunk:
    chunk_id: str
    path: str
    content: str
    source: str = "session"  # session | curated | global | workspace
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    content_hash: str = ""
    start_line: int = 1
    end_line: int = 1

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = _blake2_hex(self.content)


@dataclass
class SearchResult:
    chunk_id: str
    path: str
    score: float
    snippet: str
    source: str
    created_at: float
    start_line: int = 1
    end_line: int = 1


@dataclass
class MemorySearchConfig:
    max_results: int = 6
    min_score: float = 0.15
    text_weight: float = 0.7
    vector_weight: float = 0.0  # reserved; hybrid_search is FTS+MMR (Jaccard), not vectors
    temporal_decay: bool = True
    half_life_days: float = 7.0
    access_boost_k: float = 0.05
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7
    evergreen_sources: frozenset[str] = frozenset({"global", "workspace", "curated"})


class SmartMemoryStore:
    """SQLite-backed chunk store with FTS5 when available."""

    def __init__(self, workspace: str | Path | None = None):
        ws = Path(workspace or os.getcwd()).resolve()
        self.workspace = ws
        self.db_path = ws / ".clawagents" / "smart_memory.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'session',
                created_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT NOT NULL,
                start_line INTEGER NOT NULL DEFAULT 1,
                end_line INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
            """
        )
        # Best-effort FTS5
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(chunk_id UNINDEXED, content, path UNINDEXED, "
                "tokenize='porter unicode61')"
            )
            self._fts = True
        except sqlite3.OperationalError:
            self._fts = False
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def is_duplicate_exact(self, content: str) -> bool:
        h = _blake2_hex(content)
        row = self._conn.execute(
            "SELECT 1 FROM chunks WHERE content_hash = ? LIMIT 1", (h,)
        ).fetchone()
        return row is not None

    def upsert(self, chunk: MemoryChunk) -> bool:
        """Insert chunk if not exact-hash duplicate. Returns True if stored."""
        if self.is_duplicate_exact(chunk.content):
            return False
        self._conn.execute(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, path, content, source, created_at, access_count,
             content_hash, start_line, end_line)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.path,
                chunk.content,
                chunk.source,
                chunk.created_at,
                chunk.access_count,
                chunk.content_hash or _blake2_hex(chunk.content),
                chunk.start_line,
                chunk.end_line,
            ),
        )
        if self._fts:
            # REPLACE on chunks leaves orphan FTS rows; delete then insert.
            self._conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                (chunk.chunk_id,),
            )
            self._conn.execute(
                "INSERT INTO chunks_fts(chunk_id, content, path) VALUES (?, ?, ?)",
                (chunk.chunk_id, chunk.content, chunk.path),
            )
        self._conn.commit()
        return True

    def record_access(self, chunk_id: str) -> None:
        self._conn.execute(
            "UPDATE chunks SET access_count = access_count + 1 WHERE chunk_id = ?",
            (chunk_id,),
        )
        self._conn.commit()

    def _temporal_decay(
        self, source: str, created_at: float, cfg: MemorySearchConfig, now: float
    ) -> float:
        if not cfg.temporal_decay or source in cfg.evergreen_sources:
            return 1.0
        if cfg.half_life_days <= 0:
            return 1.0
        age_days = max(0.0, (now - created_at) / 86400.0)
        lam = math.log(2) / cfg.half_life_days
        return math.exp(-lam * age_days)

    def _access_boost(self, access_count: int, cfg: MemorySearchConfig) -> float:
        return 1.0 + math.log1p(max(0, access_count)) * cfg.access_boost_k

    def _fts_candidates(self, query: str, limit: int) -> list[tuple[str, float]]:
        if not self._fts or not query.strip():
            return []
        # Escape FTS5 special chars lightly
        q = re.sub(r'[^\w\s]', " ", query).strip()
        if not q:
            return []
        terms = " OR ".join(t for t in q.split() if t)
        if not terms:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) AS rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (terms, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25: more negative = better → normalize later
        return [(str(r["chunk_id"]), float(r["rank"])) for r in rows]

    def _like_candidates(self, query: str, limit: int) -> list[tuple[str, float]]:
        token = query.strip().lower()
        if not token:
            return []
        rows = self._conn.execute(
            """
            SELECT chunk_id FROM chunks
            WHERE lower(content) LIKE ?
            ORDER BY access_count DESC, created_at DESC
            LIMIT ?
            """,
            (f"%{token}%", limit),
        ).fetchall()
        # Fake ranks: 0, -1, -2 … so first is best
        return [(str(r["chunk_id"]), -float(i)) for i, r in enumerate(rows)]

    def _load_chunk(self, chunk_id: str) -> MemoryChunk | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        return MemoryChunk(
            chunk_id=str(row["chunk_id"]),
            path=str(row["path"]),
            content=str(row["content"]),
            source=str(row["source"]),
            created_at=float(row["created_at"]),
            access_count=int(row["access_count"]),
            content_hash=str(row["content_hash"]),
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
        )

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def hybrid_search(
        self, query: str, config: MemorySearchConfig | None = None
    ) -> list[SearchResult]:
        cfg = config or MemorySearchConfig()
        cand_limit = max(cfg.max_results * 3, 12)
        fts = self._fts_candidates(query, cand_limit)
        if not fts:
            fts = self._like_candidates(query, cand_limit)
        if not fts:
            return []

        ranks = [r for _, r in fts]
        min_r, max_r = min(ranks), max(ranks)
        span = max(max_r - min_r, 1e-9)

        now = time.time()
        scored: list[tuple[float, MemoryChunk, float]] = []
        for cid, rank in fts:
            chunk = self._load_chunk(cid)
            if not chunk or not chunk.content.strip():
                continue
            # FTS5 bm25: more negative better → invert normalize
            fts_norm = 1.0 - (rank - min_r) / span
            decay = self._temporal_decay(chunk.source, chunk.created_at, cfg, now)
            boost = self._access_boost(chunk.access_count, cfg)
            raw = fts_norm * decay * boost
            scored.append((raw, chunk, fts_norm))

        scored.sort(key=lambda x: x[0], reverse=True)

        # MMR diversity
        selected: list[tuple[float, MemoryChunk]] = []
        if cfg.mmr_enabled and scored:
            pool = list(scored)
            while pool and len(selected) < cfg.max_results:
                best_i = 0
                best_mmr = -1e9
                for i, (raw, chunk, _) in enumerate(pool):
                    max_sim = 0.0
                    for _, sel in selected:
                        max_sim = max(max_sim, self._jaccard(chunk.content, sel.content))
                    mmr = cfg.mmr_lambda * raw - (1.0 - cfg.mmr_lambda) * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_i = i
                raw, chunk, _ = pool.pop(best_i)
                if raw >= cfg.min_score or not selected:
                    selected.append((raw, chunk))
        else:
            selected = [(r, c) for r, c, _ in scored[: cfg.max_results] if r >= cfg.min_score]

        results: list[SearchResult] = []
        for raw, chunk in selected:
            self.record_access(chunk.chunk_id)
            snippet = chunk.content[:400].strip()
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    path=chunk.path,
                    score=min(1.0, max(0.0, raw)),
                    snippet=snippet,
                    source=chunk.source,
                    created_at=chunk.created_at,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
            )
        return results


def ingest_text(
    content: str,
    *,
    path: str,
    source: str = "session",
    workspace: str | Path | None = None,
    chunk_id: str | None = None,
) -> bool:
    """Store content if not exact-duplicate. Returns True when written."""
    from clawagents.config.features import is_enabled

    if not is_enabled("smart_memory"):
        return False
    text = (content or "").strip()
    if not text:
        return False
    store = SmartMemoryStore(workspace)
    try:
        import uuid

        cid = chunk_id or f"mem_{uuid.uuid4().hex[:12]}"
        return store.upsert(
            MemoryChunk(chunk_id=cid, path=path, content=text, source=source)
        )
    finally:
        store.close()


def search_memories(
    query: str,
    *,
    workspace: str | Path | None = None,
    config: MemorySearchConfig | None = None,
) -> list[SearchResult]:
    from clawagents.config.features import is_enabled

    if not is_enabled("hybrid_memory_search") and not is_enabled("smart_memory"):
        return []
    store = SmartMemoryStore(workspace)
    try:
        return store.hybrid_search(query, config)
    finally:
        store.close()


__all__ = [
    "MemoryChunk",
    "SearchResult",
    "MemorySearchConfig",
    "SmartMemoryStore",
    "ingest_text",
    "search_memories",
    "_blake2_hex",
]
