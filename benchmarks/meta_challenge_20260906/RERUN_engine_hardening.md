# Re-run after the harness hardening (2026-09-06, same day)

Same frozen suite, runner (`scripts/benchmark_meta_challenge.py`), tasks, budgets
(196,608 context, 6,144 max output tokens, 32 rounds, 240 s) and sandbox as
[REPORT.md](REPORT.md). Only the engine changed (uncommitted working tree at
pyproject 6.20.76; see `META_GLIMMER.md` "Reasoning channel and output budget").
No task-specific tuning was done after inspecting scored results; every change
is a generic loop/transport behaviour that also applies to Luna.

| Arm | Run | Clean passes | Correct artifacts | Timeouts | Total seconds | Prompt tokens | Output tokens | Reasoning tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| glimmer | baseline | 1/12 | 5/12 | 9 | 2,739 | 2,133,160 | 229,307 | not metered |
| glimmer | after | **4/12** | 5/12 | 7 | 2,345 | 1,570,858 | 204,391 | 189,598 |
| luna | baseline | 12/12 | 12/12 | 0 | 605 | 1,142,516 | 47,819 | — |
| luna | after | 12/12 | 12/12 | 0 | **463** | **806,442** | 39,842 | — |

Raw rows: [rerun2_glimmer_luna_engine_final.json](rerun2_glimmer_luna_engine_final.json).
An intermediate Glimmer-only pass with a partially updated engine is kept as
[rerun1_glimmer_engine_wip.json](rerun1_glimmer_engine_wip.json) (2/12 clean; two
runs were killed by the identical-read hard stop that the final engine removes).

## What changed the outcome

- **Output-limit recovery.** A turn that hit `max_tokens` mid-reasoning used to
  arrive as empty content with no finish reason and be routed into the "write
  the answer in plain language" nudge; it is now recognised, the model is asked
  to continue briefly, and the output budget grows ×1.5. Fired in most Glimmer
  trials; no more wasted 70 s turns followed by a premature final answer.
- **Reads are served, not stopped.** Two run-2 trials ended with `Tool loop
  detected (hashline_read)` / `(read_file)` on the third identical read. Read
  tools no longer hard-stop or withhold; the third repeat re-executes.
- **Edit-test cycles are not loops.** The same test command after each edit no
  longer counts toward the hard stop, and a file write invalidates cached read
  stubs (a `read_file` after `edit_file` of the same path was served stale
  pre-edit content).
- **Repeated identical failures escalate in the tool result** instead of
  letting the model retry `unsandboxed=true` or an absolute-path `cat` six times.
- **Luna got cheaper for free** (−29% prompt tokens, −23% wall time) from the
  same loop changes; 12/12 unchanged.

## What did not change

Seven Glimmer trials still time out. Each of them produced 16–21K output
tokens, of which ~93% is chain-of-thought at roughly 85 tokens/s — that is the
240 s budget on its own. The deployment ignores every reasoning-control field
(`enable_thinking`, `reasoning_effort`, `separate_reasoning`), so the harness
cannot cap it server-side. On this suite and budget the remaining gap to Luna
is model throughput, not harness overhead. A fair next comparison would raise
the wall budget (e.g. 480 s) or the output cap (the `meta` profile now defaults
to 16,384 outside this frozen runner) and re-run both arms.

One non-timeout failure remains: `dependency_planner` r1 produced a correct
artifact, then issued ~20 tiny `python -c` probe commands until the 32-round
cap. A probe-streak nudge (after 8 shell commands with no edit) was added after
this run started and is not reflected in these numbers.
