"""Append-only event store with recording, replay, and export capabilities.

Stores all context management events during an agent session. Supports:
- Live appending during agent runs
- JSON/CSV export for offline analysis
- Loading from JSON for replay
- Querying by event kind, turn range, etc.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from clawagents.context_observatory.events import (
    BudgetSnapshot,
    CompactionEvent,
    ContextEvent,
    CrushEvent,
    LLMCallEvent,
    MessageSnapshot,
    ToolCallSnapshot,
    TrimEvent,
)


class EventStore:
    """Thread-safe, append-only event store for context observatory."""

    def __init__(self) -> None:
        self._events: list[ContextEvent] = []
        self._session_meta: dict[str, Any] = {}

    @property
    def events(self) -> list[ContextEvent]:
        return list(self._events)

    def set_session_meta(self, **kwargs: Any) -> None:
        """Attach session-level metadata (model, context_window, strategy, etc.)."""
        self._session_meta.update(kwargs)

    @property
    def session_meta(self) -> dict[str, Any]:
        return dict(self._session_meta)

    def append(self, event: ContextEvent) -> None:
        self._events.append(event)

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)

    # ── Queries ──────────────────────────────────────────────────────────

    def get_by_kind(self, kind: str) -> list[ContextEvent]:
        return [e for e in self._events if e.kind == kind]

    def get_by_turn(self, turn: int) -> list[ContextEvent]:
        return [e for e in self._events if e.turn == turn]

    def get_turn_range(self, start: int, end: int) -> list[ContextEvent]:
        return [e for e in self._events if start <= e.turn <= end]

    @property
    def max_turn(self) -> int:
        if not self._events:
            return 0
        return max((e.turn for e in self._events if e.turn is not None), default=0)

    def get_llm_calls(self) -> list[LLMCallEvent]:
        return [e for e in self._events if isinstance(e, LLMCallEvent)]

    def get_compaction_events(self) -> list[CompactionEvent]:
        return [e for e in self._events if isinstance(e, CompactionEvent)]

    def get_crush_events(self) -> list[CrushEvent]:
        return [e for e in self._events if isinstance(e, CrushEvent)]

    def get_trim_events(self) -> list[TrimEvent]:
        return [e for e in self._events if isinstance(e, TrimEvent)]

    def get_budget_snapshots(self) -> list[BudgetSnapshot]:
        return [e for e in self._events if isinstance(e, BudgetSnapshot)]

    # ── Derived metrics ──────────────────────────────────────────────────

    def get_token_curve(self) -> list[dict[str, Any]]:
        """Return per-turn token usage for charting.

        Each entry: {turn, input_tokens, output_tokens, total_tokens,
                     cached_tokens, context_window, utilization_pct}
        """
        curve: list[dict[str, Any]] = []
        for e in self.get_llm_calls():
            curve.append({
                "turn": e.turn,
                "input_tokens": e.total_input_tokens,
                "output_tokens": e.total_output_tokens,
                "total_tokens": e.total_input_tokens + e.total_output_tokens,
                "cached_tokens": e.cached_input_tokens,
                "context_window": e.context_window,
                "utilization_pct": e.utilization_pct,
            })
        return curve

    def get_cumulative_tokens(self) -> dict[str, int]:
        """Return cumulative token totals across all LLM calls."""
        total_in = 0
        total_out = 0
        total_cached = 0
        for e in self.get_llm_calls():
            total_in += e.total_input_tokens
            total_out += e.total_output_tokens
            total_cached += e.cached_input_tokens
        return {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cached_tokens": total_cached,
            "total_tokens": total_in + total_out,
            "calls": len(self.get_llm_calls()),
        }

    def get_cumulative_token_curve(self) -> list[dict[str, Any]]:
        """Return per-turn cumulative token totals for charting.

        Each entry: {turn, cumulative_input, cumulative_output,
                     cumulative_total, input_tokens, output_tokens}
        """
        curve: list[dict[str, Any]] = []
        cum_in = 0
        cum_out = 0
        for e in self.get_llm_calls():
            cum_in += e.total_input_tokens
            cum_out += e.total_output_tokens
            curve.append({
                "turn": e.turn,
                "input_tokens": e.total_input_tokens,
                "output_tokens": e.total_output_tokens,
                "cumulative_input": cum_in,
                "cumulative_output": cum_out,
                "cumulative_total": cum_in + cum_out,
                "context_window": e.context_window,
                "utilization_pct": e.utilization_pct,
            })
        return curve

    def get_compaction_summary(self) -> dict[str, Any]:
        """Return aggregate compaction statistics."""
        events = self.get_compaction_events()
        end_events = [e for e in events if e.phase == "end"]
        total_saved = sum(e.tokens_before - e.tokens_after for e in end_events)
        total_dropped = sum(e.messages_dropped for e in end_events)
        return {
            "compaction_count": len(end_events),
            "total_tokens_saved": total_saved,
            "total_messages_dropped": total_dropped,
            "avg_savings_pct": (
                sum(e.savings_pct for e in end_events) / len(end_events)
                if end_events
                else 0.0
            ),
        }

    def get_crush_summary(self) -> dict[str, Any]:
        """Return aggregate crush statistics."""
        events = self.get_crush_events()
        total_saved = sum(e.saved_chars for e in events)
        by_kind: dict[str, int] = {}
        for e in events:
            by_kind[e.content_kind] = by_kind.get(e.content_kind, 0) + 1
        return {
            "crush_count": len(events),
            "total_chars_saved": total_saved,
            "by_content_kind": by_kind,
        }

    # ── Export ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_meta": self._session_meta,
            "events": [e.to_dict() for e in self._events],
        }

    def export_json(self, path: str | Path) -> None:
        """Export all events to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, default=str, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def export_csv(self, path: str | Path) -> None:
        """Export token curve data to CSV (one row per LLM call turn)."""
        curve = self.get_token_curve()
        if not curve:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
        p.write_text(output.getvalue(), encoding="utf-8")

    # ── Import (replay) & Package Management ──────────────────────────────

    @classmethod
    def load_from_json(cls, path: str | Path) -> "EventStore":
        """Load a previously exported session for replay from json file, directory, or zip."""
        import zipfile

        p = Path(path)

        # Handle .zip file or package
        if p.is_file() and p.suffix.lower() == ".zip":
            history_dir = get_history_dir()
            history_dir.mkdir(parents=True, exist_ok=True)
            extract_dir = history_dir / p.stem
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(extract_dir)
            session_file = extract_dir / "session.json"
            if not session_file.exists():
                # Look for any json file in extracted directory
                jsons = list(extract_dir.glob("*.json"))
                if jsons:
                    session_file = jsons[0]
            if session_file.exists():
                return cls.load_from_json(session_file)

        # Handle session directory
        if p.is_dir():
            session_file = p / "session.json"
            if not session_file.exists():
                jsons = list(p.glob("*.json"))
                if jsons:
                    session_file = jsons[0]
                else:
                    raise ValueError(f"No session.json found in directory {p}")
            p = session_file

        raw = json.loads(p.read_text(encoding="utf-8"))
        store = cls()
        store._session_meta = raw.get("session_meta", {})
        for entry in raw.get("events", []):
            event = _deserialize_event(entry)
            if event is not None:
                store._events.append(event)
        return store

    # ── Auto-save (history) ──────────────────────────────────────────────

    def auto_save(self, chat_id: str | None = None) -> Path | None:
        """Persist this session to the .clawagents/context-observatory/<session_id>/ directory.

        Returns the path written, or None if there was nothing to save.
        Structure:
          .clawagents/context-observatory/<session_id>/
            ├── session.json
            ├── events.jsonl
            └── contexts/ (if large files present)
        """
        if not self._events:
            return None

        import datetime
        import logging

        history_dir = get_history_dir()
        history_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = (chat_id or "session").replace("/", "_")[:40]
        session_id = f"{slug}" if chat_id else f"{ts}_{slug}"

        session_dir = history_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        session_json_path = session_dir / "session.json"
        events_jsonl_path = session_dir / "events.jsonl"
        contexts_dir = session_dir / "contexts"

        try:
            # 1. Save main session.json
            payload = self.to_dict()

            # Separate large message contents if needed (>50KB)
            for event_dict in payload.get("events", []):
                if event_dict.get("kind") == "llm_call":
                    for idx, msg in enumerate(event_dict.get("messages", [])):
                        full_content = msg.get("full_content") or ""
                        if len(full_content) > 50_000:
                            contexts_dir.mkdir(parents=True, exist_ok=True)
                            c_file = contexts_dir / f"turn_{event_dict.get('turn', 0)}_msg_{idx}_{msg.get('role', 'msg')}.txt"
                            c_file.write_text(full_content, encoding="utf-8")
                            msg["external_file"] = str(c_file.relative_to(session_dir))

            session_json_path.write_text(
                json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            # 2. Save raw events.jsonl
            with open(events_jsonl_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps({
                        "__meta__": True,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "session_id": session_id,
                        "session_meta": self._session_meta,
                        "event_count": len(self._events),
                    }) + "\n"
                )
                for event in self._events:
                    f.write(json.dumps(event.to_dict(), default=str) + "\n")

            logging.getLogger(__name__).info("Auto-saved observatory session to %s", session_dir)
            return session_json_path
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to auto-save session", exc_info=True
            )
            return None

    @staticmethod
    def list_history() -> list[dict[str, Any]]:
        """List all saved sessions in the context-observatory directory.

        Returns a list of dicts with session metadata, sizes, and file paths.
        """
        history_dir = get_history_dir()
        if not history_dir.is_dir():
            return []

        entries: list[dict[str, Any]] = []

        # Check subdirectories first
        for item in sorted(history_dir.iterdir(), reverse=True):
            if item.is_dir():
                session_json = item / "session.json"
                if session_json.exists():
                    total_bytes = sum(f.stat().st_size for f in item.glob("**/*") if f.is_file())
                    entry: dict[str, Any] = {
                        "path": str(session_json),
                        "dir_path": str(item),
                        "filename": item.name,
                        "size_bytes": total_bytes,
                        "is_directory": True,
                    }
                    try:
                        raw = json.loads(session_json.read_text(encoding="utf-8"))
                        meta = raw.get("session_meta", {})
                        entry["model"] = meta.get("model", "")
                        entry["context_window"] = meta.get("context_window", 0)
                        entry["started_at"] = meta.get("started_at")
                        entry["completed_at"] = meta.get("completed_at")
                        entry["status"] = meta.get("status", "")
                        entry["session_cost_usd"] = meta.get("session_cost_usd")
                        entry["event_count"] = len(raw.get("events", []))
                        entry["llm_calls"] = sum(
                            1 for e in raw.get("events", []) if e.get("kind") == "llm_call"
                        )
                    except Exception:
                        pass
                    entries.append(entry)

        # Check single .json files (legacy fallback)
        for p in sorted(history_dir.glob("*.json"), reverse=True):
            if p.is_file():
                entry = {
                    "path": str(p),
                    "dir_path": str(p.parent),
                    "filename": p.name,
                    "size_bytes": p.stat().st_size,
                    "is_directory": False,
                }
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    meta = raw.get("session_meta", {})
                    entry["model"] = meta.get("model", "")
                    entry["context_window"] = meta.get("context_window", 0)
                    entry["started_at"] = meta.get("started_at")
                    entry["completed_at"] = meta.get("completed_at")
                    entry["status"] = meta.get("status", "")
                    entry["session_cost_usd"] = meta.get("session_cost_usd")
                    entry["event_count"] = len(raw.get("events", []))
                    entry["llm_calls"] = sum(
                        1 for e in raw.get("events", []) if e.get("kind") == "llm_call"
                    )
                except Exception:
                    pass
                entries.append(entry)

        # Also check legacy observatory_history dir if exists
        legacy_dir = Path.cwd() / ".clawagents" / "observatory_history"
        if legacy_dir.is_dir() and legacy_dir != history_dir:
            for p in sorted(legacy_dir.glob("*.json"), reverse=True):
                if p.is_file():
                    entry = {
                        "path": str(p),
                        "dir_path": str(p.parent),
                        "filename": f"[legacy] {p.name}",
                        "size_bytes": p.stat().st_size,
                        "is_directory": False,
                    }
                    try:
                        raw = json.loads(p.read_text(encoding="utf-8"))
                        meta = raw.get("session_meta", {})
                        entry["model"] = meta.get("model", "")
                        entry["context_window"] = meta.get("context_window", 0)
                        entry["started_at"] = meta.get("started_at")
                        entry["completed_at"] = meta.get("completed_at")
                        entry["status"] = meta.get("status", "")
                        entry["session_cost_usd"] = meta.get("session_cost_usd")
                        entry["event_count"] = len(raw.get("events", []))
                        entry["llm_calls"] = sum(
                            1 for e in raw.get("events", []) if e.get("kind") == "llm_call"
                        )
                    except Exception:
                        pass
                    entries.append(entry)

        return entries

    def export_package_zip(self, session_id: str | None = None) -> bytes:
        """Export session as a downloadable ZIP package containing session files."""
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            session_meta = self.to_dict()
            zf.writestr("session.json", json.dumps(session_meta, indent=2, default=str, ensure_ascii=False))

            # Also include events.jsonl
            events_lines = []
            for e in self._events:
                events_lines.append(json.dumps(e.to_dict(), default=str))
            zf.writestr("events.jsonl", "\n".join(events_lines) + "\n")

        return buffer.getvalue()


def get_history_dir() -> Path:
    """Return the context observatory directory (.clawagents/context-observatory/)."""
    try:
        from clawagents.paths import get_context_observatory_dir
        return get_context_observatory_dir(create=False)
    except Exception:
        return Path.cwd() / ".clawagents" / "context-observatory"



def _deserialize_event(data: dict[str, Any]) -> ContextEvent | None:
    """Reconstruct a typed ContextEvent from its dict representation."""
    kind = data.get("kind", "")
    turn = data.get("turn", 0)
    ts = data.get("timestamp", 0.0)

    if kind == "llm_call":
        messages = [
            MessageSnapshot(**m) for m in data.get("messages", [])
        ]
        tool_calls = [
            ToolCallSnapshot(**tc) for tc in data.get("tool_calls_made", [])
        ]
        return LLMCallEvent(
            turn=turn,
            timestamp=ts,
            model=data.get("model", ""),
            messages=messages,
            system_prompt_breakdown=data.get("system_prompt_breakdown", {}),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            cached_input_tokens=data.get("cached_input_tokens", 0),
            context_window=data.get("context_window", 0),
            utilization_pct=data.get("utilization_pct", 0.0),
            tokens_by_role=data.get("tokens_by_role", {}),
            cache_creation_tokens=data.get("cache_creation_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            tool_calls_made=tool_calls,
            response_text_preview=data.get("response_text_preview", ""),
            response_text_length=data.get("response_text_length", 0),
            cumulative_input_tokens=data.get("cumulative_input_tokens", 0),
            cumulative_output_tokens=data.get("cumulative_output_tokens", 0),
            cumulative_cost_usd=data.get("cumulative_cost_usd", 0.0),
        )
    elif kind == "compaction":
        return CompactionEvent(
            turn=turn,
            timestamp=ts,
            phase=data.get("phase", ""),
            tokens_before=data.get("tokens_before", 0),
            tokens_after=data.get("tokens_after", 0),
            messages_before=data.get("messages_before", 0),
            messages_after=data.get("messages_after", 0),
            messages_dropped=data.get("messages_dropped", 0),
            savings_pct=data.get("savings_pct", 0.0),
            budget=data.get("budget", 0),
            summary_preview=data.get("summary_preview", ""),
        )
    elif kind == "crush":
        return CrushEvent(
            turn=turn,
            timestamp=ts,
            tool_name=data.get("tool_name", ""),
            content_kind=data.get("content_kind", ""),
            original_chars=data.get("original_chars", 0),
            crushed_chars=data.get("crushed_chars", 0),
            saved_chars=data.get("saved_chars", 0),
            original_tokens=data.get("original_tokens", 0),
            crushed_tokens=data.get("crushed_tokens", 0),
        )
    elif kind == "trim":
        return TrimEvent(
            turn=turn,
            timestamp=ts,
            role=data.get("role", ""),
            original_chars=data.get("original_chars", 0),
            trimmed_chars=data.get("trimmed_chars", 0),
            saved_chars=data.get("saved_chars", 0),
        )
    elif kind == "budget":
        return BudgetSnapshot(
            turn=turn,
            timestamp=ts,
            system_tokens=data.get("system_tokens", 0),
            tool_tokens=data.get("tool_tokens", 0),
            user_assistant_tokens=data.get("user_assistant_tokens", 0),
            image_tokens=data.get("image_tokens", 0),
            budget_limits=data.get("budget_limits", {}),
            actual_usage=data.get("actual_usage", {}),
        )
    # Unknown kind — return generic
    return ContextEvent(turn=turn, kind=kind, timestamp=ts)
