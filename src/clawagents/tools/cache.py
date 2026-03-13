"""Tool Result Cache — LRU in-memory cache with per-tool TTLs.

Inspired by ToolUniverse's two-tier caching: avoids redundant API calls,
file reads, and web fetches when the agent re-invokes the same tool with
identical arguments within the TTL window.

Tools opt in via ``cacheable = True`` on the Tool protocol.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from clawagents.tools.registry import ToolResult


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
