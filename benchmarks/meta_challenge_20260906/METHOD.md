# Harder Glimmer / Luna comparison

Frozen synthetic engineering tasks: decimal financial-event processing with deduplication and a CLI; deterministic dependency waves with target closure and full-graph validation; TTL/LRU cache interactions using injected clocks; idempotent SQLite migration with forced-failure rollback.

Protocol: 4 tasks × 3 independent repeats × 2 models. Order is randomized with seed 917. Both use the release's Python harness, the same core tools, a 196,608 context cap, 6,144 maximum output tokens, 32 tool rounds, and a 240-second wall timeout. Luna uses medium reasoning; Glimmer uses server-default reasoning. No prompt cache flush is performed.

Each task begins in a fresh workspace containing its specification, broken starter, and a minimal public unittest. The agent process and shell use this workspace as cwd. A fail-closed macOS sandbox-exec boundary denies child network, confines writes, and denies reads of the release source tree and grader files. The actual graders execute afterward in an independent process; their code is never written into the agent workspace. A direct negative-control test confirms that fixture access succeeds while grader access fails. A transcript integrity check also looks for leaked grader markers.

Graders use deterministic edge cases and seeded randomized cases. Every grader was tested against a broken starter and a working reference before the scored run. No model was used as a judge. A clean pass requires a valid artifact, normal completion within budget, and an intact isolation check. Artifact correctness and clean completion are recorded separately.

An initial attempt was invalidated because the shell cwd differed from the task workspace and a trial read benchmark sources. That entire attempt is excluded. The restarted scored run uses the corrected isolation boundary. No model-specific tuning was performed after inspecting scored challenge results.

These are newly authored synthetic tasks, not SWE-bench or another established suite. Three repeats do not support significance or broad quality claims. Different servers, tokenizers, reasoning modes, network conditions and cache state limit latency comparisons. Client wall time excludes the independent grader. All raw scored rows and patches are provided with the final report.

Reporting correction: the harness can return status `done` with result `[cancelled]` after timeout cancellation. The final report treats this as incomplete. The live run used the runner hash recorded in raw results; the released runner additionally rejects this cancellation marker. This changes completion classification only, never prompts, tools, outputs, graders, or budgets.

Some scored tool calls report filesystem permission errors. Direct checks through the release execute tool confirmed fixture reads and public unittest execution succeed across fresh workspaces; both arms use the same OS boundary. Raw diagnostic tails are retained. Results characterize this sandboxed harness configuration, including its tool failures, rather than an unrestricted coding environment.
