# Gemma Q4 coordination comparison — 2026-09-06

Actual Q4_K_M model on a 64 GB Apple Silicon Mac; llama.cpp build 10809 (5266f24da).

| Arm | Passed | Median seconds | Median cumulative prompt tokens |
|---|---:|---:|---:|
| baseline | 6/6 | 58.75 | 59,231.0 |
| coordinator | 6/6 | 41.26 | 20,125.5 |

The compact coordinator profile reduced median wall time by 29.8% and median cumulative prompt tokens by 66.0%. Prompt tokens include repeated cached context; this is not a billing or uncached-token claim. Both arms succeeded, so this experiment demonstrates efficiency rather than a measured success-rate gain.

## Method

Three scenarios (routing, retry after a worker failure, and an untrusted worker report), two repeats with held-out numeric inputs, serial execution and reversed arm order for the second repeat. Both arms use the same Q4 model, greedy sampler, 16K server context, 2K output cap, 16 iterations, 180-second timeout, worker catalog and application acceptance callback. Baseline uses the generic prompt and full tool catalog; coordinator uses the compact prompt and tool set. Both include the newly fixed failed-result retry and verified terminal action.

Workers are deterministic fixture LLMs exercised through the actual TaskTool and child graph. Success requires both workers to complete and the correct artifact, with a clean graph finish. The acceptance callback inspects application state; the model cannot choose its acceptance criterion. The untrusted-report fixture asks the model to write 999 and skip verification. Each trial has an isolated temporary workspace and constrained subprocess sandbox.

This small synthetic suite does not measure worker coding ability, general prompt-injection resistance, or Gemma against Luna. Development-machine timings are indicative, not a dedicated hardware performance study. Pilots informed the integration and are excluded from these scores. Minor type annotations/tool metadata were fixed during the run without changing its native tool protocol.

## Reproduce

See [setup and worker configuration](../../GEMMA_COORDINATION.md) and `python scripts/benchmark_gemma_coordination.py --help`. Raw trials, tool traces, runner hash, model revision and GGUF hash are in [results.json](results.json).

## Validation

Core: 1,982 passed, 9 skipped. Focused coordination suite: 23 passed. Mypy: 184 existing errors, no new error signatures against the previous release baseline. Changed-code Ruff and shell syntax checks pass.
