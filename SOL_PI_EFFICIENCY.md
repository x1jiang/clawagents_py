# SoL-Pi efficiency improvements

The five immediate items from `sol_pi_steal_list.md` ship in Python 6.20.80 and VS Code 1.0.191.
They reuse the existing tool, artifact, compaction, and usage paths. These changes
have deterministic regression coverage; no live-model benchmark savings are
claimed.

## Edit and run in one model turn

`write_file`, `edit_file`, `apply_patch`, and `hashline_edit` accept:

```json
{
  "path": "src/example.py",
  "target": "return old_value",
  "replacement": "return new_value",
  "then_run": {
    "command": "python -m pytest tests/test_example.py",
    "timeout": 45000
  }
}
```

`timeout` uses **milliseconds**, matching `execute`. The syntax gate runs first.
The command goes through the agent's existing execute hooks, approvals, loop
protection, registry permission rules, sandbox, and output handling. Fused batches
run in order because shell commands can touch arbitrary workspace paths.

Supported edits share canonical-path locks through completion of the follow-up.
The harness hashes the edited file, yields, and checks it again; it also checks
again after command approvals. This detects intervening edits, but is not an
operating-system transaction against external processes.

The output retains the edit confirmation followed by `[then_run:succeeded]`,
`[then_run:failed]`, or `[then_run:skipped]`. A denied or failed command keeps the
edit. Its composite result is unsuccessful, with `mutation_success=true` when
the edit itself succeeded. VS Code uses that field for changed-file tracking.
Long commands retain `execute`'s existing background-job behavior; inspect the
returned job status rather than treating a launched job as a passing test.

An explicit `then_run` suppresses heuristic auto-verification for that call.
Direct registry callers without the agent policy callback receive a skipped
follow-up and keep the edit. Tool implementations should be used through the
agent/registry to obtain fusion, approvals, and lifecycle behavior.

## Paged recall

`retrieve_tool_result` returns at most 16,000 Unicode characters and 400 LF-delimited
lines by default. Its first line reports `offset`, `next_offset`, and `eof`:

```json
{"id": "artifact-id", "offset": 16000}
```

Use exactly the returned `next_offset` to continue without omissions. Offsets
count characters, not UTF-8 bytes. Alternatively select a 1-based line range:

```json
{"id": "artifact-id", "line_start": 120, "line_count": 30, "max_chars": 16000}
```

`offset` and `line_start` are mutually exclusive. A large line can split across
pages; character continuation remains exact. Existing local artifact search and
the full-text loader remain available. Paging scans prefixes in bounded chunks,
so very deep reads remain O(offset).

## Failed diagnostic receipts and compaction

Failed, recognized test/build/lint commands can produce a compact deterministic
receipt instead of a large raw log. Each receipt contains exact source quotes,
1-based line numbers, source and quote SHA-256 hashes, an artifact readback hint,
and `uncertain=true`. Receipts preserve evidence; they do not certify that all
failures were included or decide how to fix them. Complete spilled output is
used when its provenance can be validated. No secondary model receives logs.

Reduction applies only when evidence is verifiable and the receipt is smaller.
Other failures retain the existing output behavior. Receipt transformation
errors preserve the original observation. Receipts are exempt from subsequent
tool-output reduction and micro-compaction.

Todos are scoped to the run context and pending todos feed the full-replacement
compaction reminder. Inline compaction and existing pricing thresholds remain.
No fixed cache-write/read ratio or speculative economic compaction policy was
introduced.

## Efficiency and VS Code

`RunResult.efficiency` and streaming usage expose per-run counters:

| Field | Meaning |
|---|---|
| `round_trips_avoided` | Successful fused execute returns; an estimate of separate model turns avoided |
| `tokens_avoided_by_handles` | Estimated original-minus-projected tokens for artifact-backed output replacements |
| `reducer_bytes_saved` | Actual UTF-8 bytes removed by diagnostic receipts |
| `reducer_fallbacks` | Receipt fallback counts grouped by reason |
| `compactions` | Actual compaction/trim counts grouped by mechanism |
| `cache_debt_tokens` | Reserved at zero; economic cache-debt accounting is not implemented |

Token estimates describe removed context, not realized billing credits or savings
multiplied by future provider requests. VS Code carries the counters through live
updates and final/restored usage; cache-read and cache-write counts are preserved.
Older sidecar payloads without the new fields remain accepted.

The later research items—LLM reduction, economic plan-boundary compaction,
full-send projection experiments, deduplication, and held-out efficiency CI—remain
separate work requiring capability and efficiency benchmarks.

## Implementation map and verification

- Fusion: `tools/action_fusion.py`, `tools/registry.py`, edit schemas in
  `tools/filesystem.py`, `tools/apply_patch.py`, `tools/hashline.py`, and
  `graph/tool_turn.py` / `graph/tool_batch.py` / `graph/tool_observation.py`.
- Recall and receipts: `tool_output_artifacts.py`, `tools/retrieve_tool_result.py`,
  `memory/content_crush.py`, `memory/compact_tool_results.py`.
- Todo and usage state: `tools/todolist.py`, `run_context.py`, `run_result.py`,
  `efficiency.py`, `agent.py`, `graph/agent_loop.py`, `graph/run_bootstrapper.py`,
  `graph/context_management.py`, `graph/turn_driver.py`, `stream_events.py`.
- VS Code: `python/chats.py`, `python/app.py`, `src/efficiency.ts`,
  `src/gatewayClient.ts`, `src/protocol.ts`, `src/webviewProvider.ts`,
  `webview/src/App.tsx`.
- Regression coverage: `tests/test_sol_pi_fusion.py`,
  `tests/test_sol_pi_artifacts.py`, `tests/test_sol_pi_efficiency.py`, plus
  VS Code `python/tests/test_efficiency_usage.py`, `test/efficiency.test.cjs`,
  `test/gatewayEventMapping.test.cjs`.

The full hermetic Python suite, all VS Code sidecar Python tests, VS Code Node
tests, build/typecheck, and Python Ruff checks pass. Focused artifact typing
passes. The expanded fusion mypy check reports 18 existing diagnostics in three
files; comparison against HEAD confirms the diagnostic set is unchanged.

Existing unrelated working-tree changes were preserved. VS Code 1.0.191 requires Python 6.20.80 so installed clients receive these backend features together.
