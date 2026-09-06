# Harder coding benchmark: Glimmer vs Luna

24 trials: four synthetic engineering tasks, three repeats per model, same clawagents 6.20.74 harness. Each trial has a four-minute limit. See [methodology](METHOD.md) for isolation, grading, and limitations.

| Model | Clean passes | Correct artifacts | Median seconds | Total seconds | Prompt tokens | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| glimmer | 1/12 | 5/12 | 240.02 | 2738.84 | 2,133,160 | 229,307 |
| luna | 12/12 | 12/12 | 47.12 | 604.64 | 1,142,516 | 47,819 |

Median latency includes failed and timed-out runs; it is not a speed comparison restricted to correct solutions. Token accounting reflects server-reported usage and may exclude an interrupted request.

| Task | Glimmer clean passes | Luna clean passes |
|---|---:|---:|
| ledger | 0/3 | 3/3 |
| dependency_planner | 0/3 | 3/3 |
| ttl_cache | 1/3 | 3/3 |
| sqlite_migration | 0/3 | 3/3 |

Luna completed all 12 trials correctly within budget. Glimmer completed one correctly, produced five correct artifacts overall, and hit the timeout in nine trials. On these tasks and settings, completion reliability is the main gap. The median includes failures and timeouts; it should not be interpreted as equal-quality throughput.

Some model tool calls reported filesystem permission errors, despite direct execute-tool fixture and public-test probes passing. These diagnostics remain in the raw data; the comparison measures this sandboxed harness configuration.

## Trial outcomes

| Model | Task | Repeat | Clean pass | Artifact correct | Seconds |
|---|---|---:|---|---|---:|
| glimmer | dependency_planner | 1 | False | False | 240.09 |
| luna | dependency_planner | 1 | True | True | 80.44 |
| glimmer | ttl_cache | 1 | False | True | 240.02 |
| luna | ttl_cache | 1 | True | True | 24.42 |
| glimmer | sqlite_migration | 1 | False | False | 240.02 |
| luna | sqlite_migration | 1 | True | True | 62.17 |
| glimmer | ledger | 1 | False | False | 240.02 |
| luna | ledger | 1 | True | True | 78.12 |
| glimmer | sqlite_migration | 2 | False | False | 240.01 |
| luna | sqlite_migration | 2 | True | True | 40.84 |
| luna | ttl_cache | 2 | True | True | 33.81 |
| glimmer | ttl_cache | 2 | True | True | 214.30 |
| glimmer | dependency_planner | 2 | False | False | 240.02 |
| luna | dependency_planner | 2 | True | True | 35.24 |
| glimmer | ledger | 2 | False | False | 240.02 |
| luna | ledger | 2 | True | True | 44.26 |
| luna | dependency_planner | 3 | True | True | 49.98 |
| glimmer | dependency_planner | 3 | False | True | 208.33 |
| luna | sqlite_migration | 3 | True | True | 66.49 |
| glimmer | sqlite_migration | 3 | False | True | 240.02 |
| luna | ledger | 3 | True | True | 58.00 |
| glimmer | ledger | 3 | False | False | 240.02 |
| luna | ttl_cache | 3 | True | True | 30.87 |
| glimmer | ttl_cache | 3 | False | True | 155.99 |

The earlier easy smoke suite scored tuned Glimmer 9/9 and Luna 9/9. These harder results apply to this frozen suite and budget; they do not establish general coding rankings. No model-specific tuning followed inspection of these scored results.

The initial attempt was excluded after detecting access to benchmark sources. All 24 reported trials use the corrected sandbox boundary and pass the transcript integrity check. Timeout cancellation classification was corrected for reporting; raw results and the exact executed runner are retained.

Artifacts: [normalized results and patches](results.json), [original results](raw_results.json), [executed runner](runner_used.py.txt), and [SHA-256 manifest](SHA256SUMS).
