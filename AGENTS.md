# AGENTS.md — clawagents (Python)

Operational guide for AI/automation agents working *on* the `clawagents` Python
codebase. This file is intentionally short. It captures the conventions that
are easy to violate accidentally and hard to fix retroactively.

## 1. Prompt-cache policy

Every long-running agent session relies on the LLM provider's **prompt
prefix cache**. A handful of slash-commands and tools mutate state that lives
in the system prompt (skills, permission mode, persona, model routing, …).
If those mutations apply *immediately*, the cached prefix is invalidated and
the next turn pays the full re-prefill cost. For multi-turn tool runs this
typically dominates latency and cost.

The policy mirrors the one we adopted from Hermes:

- **`cache_impact = "none"`** — read-only commands. Always safe mid-run.
  Examples: `/help`, `/status`, `/version`, `/tools`, `/models`, `/trace`,
  `/profile`, `/redact`, `/history`.
- **`cache_impact = "deferred"`** — *default* for any command that mutates
  system-prompt state. The change is staged for the **next session** so the
  current prefix cache survives. Pass `--now` to opt into immediate
  invalidation. Examples: `/plan`, `/accept-edits`, `/default`, `/bypass`,
  and any future `/skills install`, `/model`, `/personality` commands.
- **`cache_impact = "immediate"`** — commands that *cannot* be deferred
  because they rewrite history or start a fresh context. Examples: `/new`,
  `/clear`, `/compress`, `/undo`.

Implementation lives in `src/clawagents/commands.py`:

- `CommandDef.cache_impact: CacheImpact` (declared per command).
- `ResolvedCommand.apply_now: bool` (computed by `resolve_command()`,
  combining `cache_impact` with the `--now` flag).
- The `--now` token is recognised in any position in the argument list and
  is stripped from `ResolvedCommand.args` so consumers never see it.
- The bare word `now` is **not** a flag, so steer/queue prompts like
  `/q now do the next thing` round-trip cleanly.

When you add a new slash-command or skill that touches system-prompt state,
default to `"deferred"`. Use `"immediate"` only when correctness requires
a cache flush (history rewrites, persona swaps that must take effect on the
very next turn).

## 2. User-facing paths

Tool descriptions and user-visible messages must render configuration paths
through `display_clawagents_home()` / `display_clawagents_workspace_dir()`
instead of hardcoding `~/.clawagents/...`. This keeps the message correct
when the user is on a non-default profile (`CLAWAGENTS_PROFILE`) or has
relocated home (`CLAWAGENTS_HOME`).

```python
from clawagents.paths import display_clawagents_home

description = f"Read a memory file from {display_clawagents_home()}/memories/"
```

## 3. Subagent boundaries (do not violate)

- Subagent depth is capped at **2**. The `task` tool refuses to spawn a
  subagent from inside a subagent. This prevents recursive blowup and
  protects the iteration / token budget.
- Subagents run with `skip_memory=True`. They do **not** load the parent's
  memory directory, lessons, or skill state. Pass anything they need
  explicitly via the prompt or a tool argument.
- Agent loops respect a per-agent `IterationBudget` (default 50). The
  delegating agent must reserve at least one iteration for itself.

## 4. Parallel tool execution

The agent loop will run independent read-only tools in parallel when their
**path scopes do not overlap**. Two `read_file` calls on different paths
go in parallel; a `read_file` and `write_file` on overlapping paths do not.
If you add a new tool, declare its path scope with the standard helper so
the scheduler can reason about it; otherwise it will be treated as
"unknown scope" and run serially.

## 5. Tests

Always run the hermetic test entry point before shipping:

```bash
scripts/run_tests.sh
```

This pins xdist worker count, scrubs `CLAWAGENTS_*` env vars, and runs both
the unit suite and the typed parity checks against `clawagents/` (TS).
Don't run `pytest` directly when validating a release — the worker count
and env scrubbing matter.

## 6. Plugin hooks

Plugins can implement:

- `pre_tool_call(name, args)` — return `{"veto": True, "reason": "..."}` to
  block a call. Plain raises are also caught and surfaced as veto reasons.
- `post_tool_call(name, args, result)` — observation only.
- `transform_tool_result(name, args, result)` — *return* a replacement
  result (or `None` to leave it unchanged). Use this for redaction,
  summarisation, or schema migration.

`pre_tool_call` runs first, then the tool itself, then `transform_tool_result`,
then `post_tool_call`. Hooks must be deterministic-ish; if a hook is slow,
the entire agent loop is slow.
