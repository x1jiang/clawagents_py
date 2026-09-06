# Meta Glimmer integration and benchmark

Date: 2026-09-06. Final results only; exploratory pilot/diagnostic runs are excluded.

## Final matched benchmark

| ClawAgents Python arm | Clean passes | Median seconds/task | Total seconds | Prompt tokens | Output tokens | Tool calls |
|---|---:|---:|---:|---:|---:|---:|
| glimmer-baseline | 5/9 | 13.917 | 125.198 | 850,916 | 9,729 | 70 |
| glimmer-tuned | 9/9 | 6.223 | 72.892 | 338,072 | 7,183 | 45 |
| luna | 9/9 | 6.273 | 57.911 | 148,240 | 2,156 | 28 |

Tuning reduced Glimmer prompt tokens by 60.3% and total task time by 41.8% relative to the recreated baseline. Clean passes changed from 5/9 to 9/9.

Compared with Luna, tuned Glimmer took 1.26× total wall time and consumed 2.28× prompt tokens. These short tasks do not establish general model superiority.

## Per-task results

| Task | Arm | Clean passes | Correct artifacts | Median seconds |
|---|---|---:|---:|---:|
| read | glimmer-baseline | 3/3 | 3/3 | 4.104 |
| read | glimmer-tuned | 3/3 | 3/3 | 4.328 |
| read | luna | 3/3 | 3/3 | 5.564 |
| aggregate | glimmer-baseline | 0/3 | 2/3 | 19.232 |
| aggregate | glimmer-tuned | 3/3 | 3/3 | 6.223 |
| aggregate | luna | 3/3 | 3/3 | 5.428 |
| repair | glimmer-baseline | 2/3 | 3/3 | 13.917 |
| repair | glimmer-tuned | 3/3 | 3/3 | 11.271 |
| repair | luna | 3/3 | 3/3 | 7.623 |

## Method and interpretation

- 3 tasks × 3 repeats × 3 arms, serial execution with randomized arm order per task/repeat (seed 72).
- Tasks: read a configuration value; aggregate paid orders into JSON; repair an inclusive-range function. Grading checks actual files, JSON parsing, and 289 hidden range pairs, without an LLM judge.
- Same registered tools, fresh temporary workspace/session, streaming, 196,608-token cap, 4,096 max output tokens, 12 tool rounds. Personal memory/learning and advisor models are disabled. Baseline starts with 67 active tools; tuned Glimmer and Luna start with 25.
- Baseline recreates the pre-integration Glimmer profile: no model-specific efficiency profile and legacy structured-value-to-Python-repr coercion. Tuned Glimmer uses the final core tool profile and JSON coercion fix. Luna uses its existing GPT-5.6 efficiency profile and receives the same shared coercion fix.
- Glimmer: Muse-Glimmer-30B at http://129.106.31.72:7790/v1, SGLang Chat Completions. The live models endpoint advertised max_model_len=196608. Luna: gpt-5.6-luna at the official OpenAI Responses endpoint, medium reasoning. Glimmer uses server-default reasoning.
- A correct artifact does not count as a clean pass if the loop exhausts its tool-round budget. Earlier pilot summaries counted artifact success; they are superseded by this stricter final report.
- Prompt caches were not flushed, and server load, hardware, tokenizers, and reasoning policies differ. Latency includes client/harness/tool overhead. Timings are a small smoke benchmark, not throughput or statistical significance estimates. No GPU cost or model quality beyond these tasks was measured.
- Raw final rows and usage: [final-comparison.json](final-comparison.json).

## Integration

- Python supports create_claw_agent(profile="meta"), the bare Muse-Glimmer-30B name, and PROVIDER=meta. glimmer_30B_backend/model (or uppercase aliases) override defaults; explicit arguments win. META_API_KEY is isolated from OpenAI credentials.
- Uses the existing OpenAI-compatible streaming/native-tool client. The Glimmer profile sets context budgeting, earlier old-tool clearing, loop warnings, concise tool instructions, and core-tool activation. No dependencies were added.
- Fixed structured values passed to string tool parameters: JSON serialization preserves objects, arrays, booleans, and null rather than silently writing Python repr. Already-string values remain byte-for-byte unchanged; non-JSON values yield validation errors.
- VS Code Settings and per-chat model pickers expose Meta (Glimmer). Environment forwarding, context meter, explicit endpoint trust, and cross-provider key/endpoint isolation are covered. Extension build is complete; this work was not published to a marketplace or installed into a running editor.
- Setup: [META_GLIMMER.md](../../META_GLIMMER.md). Reproducer: [benchmark_meta_glimmer.py](../../scripts/benchmark_meta_glimmer.py).

## Verification

- Python canonical hermetic suite: 2,031 passed, 9 skipped, 5 warnings. Focused integration/validator tests: 15 passed.
- JSON coercion regression reproduced 2 failures before the fix; all 5 coercion tests pass after it.
- Python changed-code Ruff: passed. Full mypy: 184 existing errors in 56 files; same error signatures as a shadow-file baseline excluding this integration. No new type errors.
- VS Code build (including both TypeScript type checks): passed. Node tests: 179 passed. Sidecar tests: 156 passed, 1 warning. Sidecar-to-core Meta construction also verified.
- Sidecar Ruff finds one existing unused import (read_project_instructions in chats.py); it predates this change. Git diff whitespace checks pass in both repositories.
- Existing unrelated computer-use edits were preserved. No release version, dependency, commit, or marketplace publication was changed.

## Reproduce

```bash
# From clawagents_py, configure LUNA_API_KEY locally (or OPENAI_API_KEY).
.venv/bin/python scripts/benchmark_meta_glimmer.py --repeats 3 --output results/meta-glimmer.json
```
