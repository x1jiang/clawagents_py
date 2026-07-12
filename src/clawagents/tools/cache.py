"""Tool Result Cache — LRU in-memory cache with per-tool TTLs.

Inspired by ToolUniverse's two-tier caching: avoids redundant API calls,
file reads, and web fetches when the agent re-invokes the same tool with
identical arguments within the TTL window.

Tools opt in via ``cacheable = True`` on the Tool protocol.
"""

import hashlib
import json
import os
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from clawagents.tools.registry import ToolResult

_DEFAULT_PERSIST_DENYLIST = {"read_file", "grep", "web_fetch", "web_search", "explorer_read_source"}


class ResultCacheManager:
    def __init__(self, max_size: int = 256, default_ttl_s: float = 60.0):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl_s = default_ttl_s
        self._tool_ttls: Dict[str, float] = {}

    def set_tool_ttl(self, tool_name: str, ttl_s: float) -> None:
        self._tool_ttls[tool_name] = ttl_s

    @staticmethod
    def _build_key(tool_name: str, args: Dict[str, Any]) -> str:
        payload = json.dumps({"t": tool_name, "a": args}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[ToolResult]:
        key = self._build_key(tool_name, args)
        entry = self._cache.get(key)
        if entry is None:
            return None

        ttl = self._tool_ttls.get(tool_name, self._default_ttl_s)
        if time.monotonic() - entry["created_at"] > ttl:
            del self._cache[key]
            return None

        # LRU promotion
        self._cache.move_to_end(key)
        return entry["result"]

    def set(self, tool_name: str, args: Dict[str, Any], result: ToolResult) -> None:
        key = self._build_key(tool_name, args)

        if len(self._cache) >= self._max_size and key not in self._cache:
            self._cache.popitem(last=False)

        self._cache[key] = {
            "tool_name": tool_name,
            "result": result,
            "created_at": time.monotonic(),
        }
        self._cache.move_to_end(key)

    def invalidate_tool(self, tool_name: str) -> None:
        keys_to_delete = [k for k, v in self._cache.items() if v["tool_name"] == tool_name]
        for k in keys_to_delete:
            del self._cache[k]

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class SqliteResultCacheManager(ResultCacheManager):
    """SQLite-backed result cache with the same surface as ``ResultCacheManager``."""

    def __init__(
        self,
        db_path: str | Path,
        max_size: int = 2048,
        default_ttl_s: float = 60.0,
        persist_denylist: Iterable[str] | bool | None = None,
    ):
        super().__init__(max_size=0, default_ttl_s=default_ttl_s)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.touch(mode=0o600, exist_ok=True)
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass
        self._max_rows = max_size
        self._default_ttl_s = default_ttl_s
        self._tool_ttls: Dict[str, float] = {}
        if persist_denylist is False:
            self._persist_denylist: set[str] = set()
        else:
            names = persist_denylist if persist_denylist is not None else _DEFAULT_PERSIST_DENYLIST
            self._persist_denylist = {str(name).lower() for name in names}
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_cache (
                cache_key TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                args_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_cache_tool ON tool_cache(tool_name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_cache_lru ON tool_cache(last_accessed)")
        self._conn.commit()

    def set_tool_ttl(self, tool_name: str, ttl_s: float) -> None:
        self._tool_ttls[tool_name] = ttl_s

    def _can_persist(self, tool_name: str) -> bool:
        return tool_name.lower() not in self._persist_denylist

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[ToolResult]:
        if not self._can_persist(tool_name):
            return None
        key = self._build_key(tool_name, args)
        row = self._conn.execute(
            "SELECT result_json, created_at FROM tool_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        result_json, created_at = row
        ttl = self._tool_ttls.get(tool_name, self._default_ttl_s)
        now = time.time()
        if now - float(created_at) > ttl:
            self._conn.execute("DELETE FROM tool_cache WHERE cache_key = ?", (key,))
            self._conn.commit()
            return None
        self._conn.execute(
            "UPDATE tool_cache SET last_accessed = ? WHERE cache_key = ?",
            (now, key),
        )
        self._conn.commit()
        data = json.loads(result_json)
        return ToolResult(
            success=bool(data.get("success")),
            output=data.get("output", ""),
            error=data.get("error"),
        )

    def set(self, tool_name: str, args: Dict[str, Any], result: ToolResult) -> None:
        if not self._can_persist(tool_name):
            return
        now = time.time()
        key = self._build_key(tool_name, args)
        payload = {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tool_cache
                (cache_key, tool_name, args_json, result_json, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                tool_name,
                json.dumps(args, sort_keys=True, ensure_ascii=False),
                json.dumps(payload, ensure_ascii=False),
                now,
                now,
            ),
        )
        count = self.size
        if count > self._max_rows:
            self._conn.execute(
                """
                DELETE FROM tool_cache
                WHERE cache_key IN (
                    SELECT cache_key FROM tool_cache
                    ORDER BY last_accessed ASC
                    LIMIT ?
                )
                """,
                (count - self._max_rows,),
            )
        self._conn.commit()

    def invalidate_tool(self, tool_name: str) -> None:
        self._conn.execute("DELETE FROM tool_cache WHERE tool_name = ?", (tool_name,))
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM tool_cache")
        self._conn.commit()

    @property
    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM tool_cache").fetchone()
        return int(row[0] if row else 0)
