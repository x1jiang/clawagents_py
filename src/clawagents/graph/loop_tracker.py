"""Tool loop detection and consecutive failure tracking.

Two independent trackers used by the main agent loop to detect and break
out of unproductive tool-call cycles.

Extracted from ``agent_loop.py`` for modularity.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

from clawagents.tools.registry import ParsedToolCall

if TYPE_CHECKING:
    from clawagents.loop_detection import LoopDetectionConfig


# ─── Tool Loop Detection ──────────────────────────────────────────────────


# Tools whose answer depends on wall-clock time, not just their arguments.
# Serving these from the duplicate-suppression cache would be wrong rather
# than merely wasteful: "wait for this job again" is the *intended* way to keep
# waiting past one wait budget, and replaying the earlier "still running"
# answer would make the job look permanently stuck.
TIME_DEPENDENT_TOOLS: frozenset[str] = frozenset(
    {"task_wait", "task_status", "task_output", "task_list", "finish_coordination"}
)

# ─── Repeated-failure escalation ─────────────────────────────────────────
# Identical-argument loops are caught by ``is_soft_looping``; a weaker model
# that keeps retrying the *same failing approach with different arguments*
# (six ``unsandboxed=true`` retries with six commands, five ``cat`` variants of
# a path the sandbox denies) is not. Key failures by (tool, normalised error
# line) and escalate the tool result itself, where the model actually reads.
_FAILURE_PATH_RE = re.compile(r"(?<![\w-])/[^\s'\"`|;)]+")
_FAILURE_NUM_RE = re.compile(r"\d+")
_FAILURE_HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b")
_FAILURE_WS_RE = re.compile(r"\s+")
_FAILURE_SIG_MAX = 160
_FAILURE_DIRECTIVE_AT = 2


def failure_signature(tool_name: str, output: str) -> str:
    """Stable key for "the same error again": tool + first error-ish line with
    paths / numbers / hashes normalised away."""
    text = output if isinstance(output, str) else str(output or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chosen = ""
    for ln in lines:
        low = ln.lower()
        if "error" in low or "denied" in low or "not permitted" in low or "not authorized" in low:
            chosen = ln
            break
    if not chosen and lines:
        chosen = lines[0]
    chosen = chosen.lower()
    chosen = _FAILURE_PATH_RE.sub("<path>", chosen)
    chosen = _FAILURE_HEX_RE.sub("<hex>", chosen)
    chosen = _FAILURE_NUM_RE.sub("#", chosen)
    chosen = _FAILURE_WS_RE.sub(" ", chosen).strip()
    return f"{tool_name}|{chosen[:_FAILURE_SIG_MAX]}"


SHELL_TOOLS: frozenset[str] = frozenset({"execute", "exec", "bash"})
_PROBE_STREAK_AT = 8
_PROBE_STREAK_EVERY = 4


def probe_streak_directive(streak: int) -> str | None:
    """Nudge after many consecutive shell commands with no edit in between."""
    if streak < _PROBE_STREAK_AT or (streak - _PROBE_STREAK_AT) % _PROBE_STREAK_EVERY:
        return None
    return (
        f"[System] {streak} shell commands have run since the last file edit. "
        "If the checks you needed have passed, stop probing and give the final "
        "answer now; if something is still wrong, edit the code instead of running "
        "more commands."
    )


def repeated_failure_directive(tool_name: str, count: int) -> str | None:
    """Escalating instruction appended to a tool result on the Nth identical failure."""
    if count < _FAILURE_DIRECTIVE_AT:
        return None
    if count == _FAILURE_DIRECTIVE_AT:
        return (
            f"[System] {tool_name} has now failed twice with the same error. "
            "Do not retry the same approach with cosmetic changes. Change strategy: "
            "use a different tool (for example read_file/write_file/edit_file instead "
            "of shell cat/echo, or an in-workspace path instead of an absolute one), "
            "or report the blocker to the user."
        )
    return (
        f"[System] {tool_name} failed {count} times with the same error. STOP retrying "
        "it — the environment will not change. Use a different tool or approach now, "
        "or finish with a clear report of what is blocked and why."
    )


# Tools after which "the same call again" is verification, not a loop, and
# after which a cached read is stale. ``execute`` is deliberately absent: a
# test command re-run three times with no edit in between IS the loop the
# detector exists for; the same command after each edit is the normal
# edit-test cycle and used to trip the hard stop (critical_threshold=3 on the
# Glimmer/Luna profiles) and end the run mid-task.
MUTATING_TOOLS: frozenset[str] = frozenset(
    {
        "write_file", "edit_file", "apply_patch", "hashline_edit", "create_file",
        "replace_in_file", "insert_in_file", "insert_lines", "patch_file",
        "delete_file", "git_commit", "git_undo_ai", "checkpoint_restore",
        "task", "subagent", "compose",
    }
)


# Read-only tools: a repeat is cheap (cached) and, for a model that lost the
# content to crushing/compaction, necessary. Killing the run on the third
# identical read ("Tool loop detected (hashline_read). Stopping.") cost
# Glimmer 2 of 12 benchmark trials; the no-progress circuit breaker remains
# the backstop. Policy: 1st runs, 2nd gets the cached stub, 3rd+ re-executes.
READ_TOOLS: frozenset[str] = frozenset(
    {
        "read_file", "hashline_read", "hashline_grep", "read_and_grep", "grep",
        "glob", "ls", "tree", "search_history", "retrieve_tool_result",
    }
)

# Planning text is model-authored, not fresh evidence about the workspace.
_PLANNING_TOOLS = frozenset({"think", "write_todos", "update_todo", "enter_plan_mode", "exit_plan_mode"})
_OBSERVATION_TOOLS = READ_TOOLS | SHELL_TOOLS | frozenset(
    {"web_search", "web_fetch", "memory_search"}
)
# Delegation and committing can succeed without changing any source files.
_FILE_MUTATING_TOOLS = MUTATING_TOOLS - {"task", "subagent", "compose", "git_commit"}
_MAX_PROGRESS_NUDGES = 2


class _ToolCallTracker:
    def __init__(
        self,
        window_size: int = 30,
        soft_limit: int = 3,
        hard_limit: int = 6,
        circuit_breaker_limit: int = 30,
        loop_config: "LoopDetectionConfig | None" = None,
        progress_nudge_after: int = 0,
    ):
        from clawagents.loop_detection import resolve_loop_detection_config

        self._history: list[str] = []
        # Parallel to _history: the mutation epoch each call was made in.
        self._history_epochs: list[int] = []
        self._mutation_epoch = 0
        self._poll_history: list[tuple[str, str, str | None]] = []
        self._window_size = window_size
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._circuit_breaker_limit = circuit_breaker_limit
        self._loop_config = resolve_loop_detection_config(loop_config)
        self._result_hashes: dict[str, str] = {}
        self._result_outputs: dict[str, str] = {}
        self._read_history: list[tuple[str, dict, str]] = []
        self._no_progress_count = 0
        self._soft_warnings = 0
        self._poll_warnings: set[str] = set()
        self._failure_counts: dict[str, int] = {}
        # Consecutive shell commands since the last file edit ("verification
        # churn": dozens of tiny python -c probes after the code is already
        # correct, until the round cap).
        self._probe_streak = 0
        # Opt-in model policy. Output novelty is only a weak progress signal:
        # it postpones the checkpoint, but cannot postpone it indefinitely.
        self._progress_nudge_after = max(0, progress_nudge_after)
        self._progress_calls = 0
        self._stale_observations = 0
        self._progress_outputs: dict[str, None] = {}
        self._progress_nudges = 0

    def _key(self, tool_name: str, args: dict) -> str:
        try:
            return f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        except (TypeError, ValueError):
            return f"{tool_name}:{args}"

    @staticmethod
    def _hash_result(output: str) -> str:
        sample = output[:500]
        h = 0
        for ch in sample:
            h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
        return str(h)

    def record(self, tool_name: str, args: dict) -> None:
        if tool_name in MUTATING_TOOLS:
            self.note_mutation(confirmed=False)
        self._history.append(self._key(tool_name, args))
        self._history_epochs.append(self._mutation_epoch)
        if len(self._history) > self._window_size:
            self._history.pop(0)
            self._history_epochs.pop(0)

    def note_mutation(self, *, confirmed: bool = True) -> None:
        """Workspace state changed: cached reads are stale and identical calls
        made before this point no longer count toward a loop."""
        self._mutation_epoch += 1
        self._probe_streak = 0
        self._result_outputs.clear()
        self._read_history.clear()
        if confirmed:
            self._reset_progress_window()

    def _reset_progress_window(self) -> None:
        self._progress_calls = 0
        self._stale_observations = 0
        self._progress_outputs.clear()

    def _progress_directive(
        self, tool_name: str, output: str, *, success: bool, can_emit: bool
    ) -> str | None:
        """Advisory only: never suppress a read or assume every task needs edits."""
        threshold = self._progress_nudge_after
        if not threshold or not self._loop_config.enabled:
            return None
        if success and tool_name in _FILE_MUTATING_TOOLS:
            self._reset_progress_window()
            return None
        if tool_name not in _OBSERVATION_TOOLS | _PLANNING_TOOLS | _FILE_MUTATING_TOOLS:
            return None
        self._progress_calls += 1
        if success and tool_name in _OBSERVATION_TOOLS and output.strip():
            # Hash the complete result: new evidence may occur after the first
            # 500 characters used by the legacy duplicate detector. Do not key
            # by arguments; cosmetic path/query changes are not new evidence.
            digest = hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()
            if digest not in self._progress_outputs:
                self._stale_observations = 0
            else:
                self._stale_observations += 1
            self._progress_outputs[digest] = None
            if len(self._progress_outputs) > self._window_size:
                self._progress_outputs.pop(next(iter(self._progress_outputs)))
        else:
            self._stale_observations += 1
        if (
            not can_emit
            or self._progress_nudges >= _MAX_PROGRESS_NUDGES
            or self._progress_calls < threshold
            or (
                self._stale_observations < max(1, threshold // 2)
                and self._progress_calls < 2 * threshold
            )
        ):
            return None
        calls = self._progress_calls
        self._progress_nudges += 1
        self._reset_progress_window()
        return (
            f"[System] Progress checkpoint after {calls} inspection/planning calls "
            "without a confirmed file change. Use the evidence collected to take "
            "the next task-appropriate action. If changes are requested and the "
            "cause is clear, make the smallest justified edit and verify it. For "
            "read-only tasks, synthesize findings when sufficient. Otherwise "
            "identify the specific missing evidence and retrieve it; avoid "
            "repeating checks whose outcome is already known."
        )

    def note_context_cleared(self) -> None:
        """Compaction / micro-compact removed earlier tool output from the
        transcript. The model can no longer see those results, so a repeat of
        the same read is recovery: serve it fresh and do not count it."""
        self.note_mutation()

    def cache_result_output(self, tool_name: str, args: dict, output: str) -> None:
        """Store truncated output for identical/overlapping reuse stubs."""
        if tool_name in TIME_DEPENDENT_TOOLS:
            return
        key = self._key(tool_name, args)
        text = output if isinstance(output, str) else str(output or "")
        self._result_outputs[key] = text[:2_000]
        if tool_name in {"read_file", "hashline_read"}:
            self._read_history.append((tool_name, dict(args or {}), text[:2_000]))
            if len(self._read_history) > self._window_size:
                self._read_history.pop(0)

    def reuse_tool_output(self, tool_name: str, args: dict) -> str | None:
        """Return a short stub if this call (or an overlapping read) already ran."""
        from clawagents.loop_detection import detect_overlapping_read

        if tool_name in TIME_DEPENDENT_TOOLS:
            return None
        key = self._key(tool_name, args)
        if tool_name in READ_TOOLS and self._count_occurrences(tool_name, args) > 2:
            # Third identical read in this epoch: the stub was not enough,
            # return the real content again instead of arguing.
            return None
        prior = self._result_outputs.get(key)
        if prior is not None:
            return (
                f"[Reused identical {tool_name} result] Same arguments already ran "
                f"this turn — do not re-call. Prior excerpt "
                f"({min(500, len(prior))} chars):\n{prior[:500]}"
            )
        return detect_overlapping_read(
            tool_name=tool_name,
            params=args or {},
            prior_reads=self._read_history,
        )

    def record_result(
        self, tool_name: str, args: dict, output: str, *, success: bool = True
    ) -> str | None:
        """Record the result of a tool call for no-progress detection.

        Returns an escalation directive when this failure repeats an earlier
        one (same tool, same normalised error) so the caller can append it to
        the tool result the model reads; ``None`` otherwise.
        """
        from clawagents.loop_detection import hash_tool_call

        key = self._key(tool_name, args)
        result_hash = self._hash_result(output)
        prev_hash = self._result_hashes.get(key)
        if prev_hash == result_hash:
            self._no_progress_count += 1
        else:
            self._no_progress_count = max(0, self._no_progress_count - 1)
        self._result_hashes[key] = result_hash
        directive: str | None = None
        if success:
            self.cache_result_output(tool_name, args, output)
        else:
            # Failure is progress evidence, never a reusable successful result.
            self._result_outputs.pop(key, None)
            self._read_history = [
                row for row in self._read_history
                if self._key(row[0], row[1]) != key
            ]
            if tool_name not in TIME_DEPENDENT_TOOLS:
                directive = self.note_failure(tool_name, output)
        if tool_name in SHELL_TOOLS:
            self._probe_streak += 1
            if directive is None and not self._progress_nudge_after:
                directive = probe_streak_directive(self._probe_streak)
        progress_directive = self._progress_directive(
            tool_name, output, success=success, can_emit=directive is None
        )
        if directive is None:
            directive = progress_directive
        call_hash = hash_tool_call(tool_name, args)
        self._poll_history.append((tool_name, call_hash, result_hash))
        if len(self._poll_history) > self._window_size:
            self._poll_history.pop(0)
        return directive

    def note_failure(self, tool_name: str, output: str) -> str | None:
        """Count a failure by error signature; return the escalation directive if any."""
        sig = failure_signature(tool_name, output)
        count = self._failure_counts.get(sig, 0) + 1
        self._failure_counts[sig] = count
        return repeated_failure_directive(tool_name, count)

    def failure_count(self, tool_name: str, output: str) -> int:
        return self._failure_counts.get(failure_signature(tool_name, output), 0)

    def is_ping_ponging(self) -> bool:
        """Detect A->B->A->B ping-pong oscillation (last 6 entries)."""
        if len(self._history) < 4:
            return False
        recent = self._history[-6:]
        if len(recent) < 4:
            return False
        unique = set(recent)
        if len(unique) != 2:
            return False
        for i in range(len(recent) - 1):
            if recent[i] == recent[i + 1]:
                return False
        return True

    def is_circuit_broken(self) -> bool:
        """Global circuit breaker: too many no-progress calls."""
        return self._no_progress_count >= self._circuit_breaker_limit

    def _count_occurrences(self, tool_name: str, args: dict) -> int:
        key = self._key(tool_name, args)
        epoch = self._mutation_epoch
        return sum(
            1
            for entry, entry_epoch in zip(self._history, self._history_epochs)
            if entry == key and entry_epoch == epoch
        )

    def is_soft_looping(self, tool_name: str, args: dict) -> bool:
        if tool_name in TIME_DEPENDENT_TOOLS or tool_name in READ_TOOLS:
            return False
        return self._count_occurrences(tool_name, args) >= self._soft_limit

    def is_hard_looping(self, tool_name: str, args: dict) -> bool:
        # Repeat count is the wrong signal for a wait: a job that outlives the
        # per-call wait budget legitimately needs more calls than the hard
        # limit, and stopping the batch there would abandon it. Stalls are
        # still caught by the no-progress circuit breaker, which keys off
        # changing results rather than call count. Reads are served, not
        # stopped (see READ_TOOLS).
        if tool_name in TIME_DEPENDENT_TOOLS or tool_name in READ_TOOLS:
            return False
        return self._count_occurrences(tool_name, args) >= self._hard_limit

    def is_soft_looping_batch(self, calls: list[ParsedToolCall]) -> bool:
        return any(self.is_soft_looping(c.tool_name, c.args) for c in calls)

    def is_hard_looping_batch(self, calls: list[ParsedToolCall]) -> bool:
        return any(self.is_hard_looping(c.tool_name, c.args) for c in calls)

    def record_batch(self, calls: list[ParsedToolCall]) -> None:
        for c in calls:
            self.record(c.tool_name, c.args)

    def bump_soft_warning(self) -> int:
        self._soft_warnings += 1
        return self._soft_warnings

    def check_known_poll_no_progress(self, tool_name: str, args: dict):
        from clawagents.loop_detection import detect_known_poll_no_progress

        result = detect_known_poll_no_progress(
            tool_name=tool_name,
            params=args,
            history=self._poll_history,
            config=self._loop_config,
        )
        if result and result.stuck and result.warning_key in self._poll_warnings:
            if result.level == "warning":
                return None
        if result and result.stuck and result.warning_key:
            self._poll_warnings.add(result.warning_key)
        return result


# ─── Consecutive Failure Detection ────────────────────────────────────────
# Tracks tool-call success/failure to detect persistent failure streaks.
# When N consecutive tool calls fail, injects a "step back and rethink"
# message — lightweight online adaptation inspired by OpenClaw-RL's
# next-state reward signal.

_RETHINK_THRESHOLD = 3
_MAX_RETHINKS = 3

_RETHINK_MESSAGE = (
    "[System] Your last {n} tool calls all failed. "
    "Stop before trying another workaround. Classify the failure as code/format, "
    "local environment, dependency availability, permission, or external service. "
    "Retry only after relevant state changed. If the evidence says authentication "
    "was rejected, a package is unavailable, or user-owned configuration is needed, "
    "report the exact error and request that action instead of changing runtimes or tools."
)


_SCORELESS_TOOLS: frozenset[str] = frozenset({
    "think", "todolist", "todo_write", "todo_read", "use_skill", "ask_user",
})


class _FailureTracker:
    """Track consecutive tool failures to trigger rethink injection.

    Scoreless tools (think, todolist, etc.) are excluded — their results
    are not meaningful signals for failure detection.
    """

    def __init__(self, threshold: int = _RETHINK_THRESHOLD, max_rethinks: int = _MAX_RETHINKS):
        self._results: list[bool] = []  # True = success, False = failure
        self._threshold = threshold
        self._max_rethinks = max_rethinks
        self._rethink_count = 0

    def record(self, success: bool, tool_name: str = "") -> None:
        if tool_name in _SCORELESS_TOOLS:
            return
        self._results.append(success)

    def record_batch(self, results: list[tuple[bool, str]]) -> None:
        for success, name in results:
            self.record(success, name)

    def should_rethink(self) -> bool:
        if self._rethink_count >= self._max_rethinks:
            return False
        if len(self._results) < self._threshold:
            return False
        return all(not s for s in self._results[-self._threshold:])

    def bump_rethink(self) -> int:
        self._rethink_count += 1
        self._results.clear()
        return self._rethink_count

    @property
    def consecutive_failures(self) -> int:
        count = 0
        for s in reversed(self._results):
            if not s:
                count += 1
            else:
                break
        return count
