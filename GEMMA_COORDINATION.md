# Gemma agentic v2: Q4 backend coordination

This integration targets [yuxinlu1's agentic v2 GGUF](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF), specifically **Q4_K_M**. It reuses the OpenAI-compatible transport and existing worker loop. It does not add a model client dependency or assume that a quantized model matches the author's Q8 benchmark results.

## Serve the model

Use a llama.cpp build supporting Gemma 4 and the GGUF's native Jinja template. The tested build is **10809 / 5266f24da** on an Apple Silicon Mac with 64 GB memory.

```bash
hf download yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF \
  gemma4-v2-Q4_K_M.gguf --revision 190a31365a6b80a692349be34ccdac730cad4fe4
scripts/serve_gemma_agentic.sh /path/returned/by/hf/gemma4-v2-Q4_K_M.gguf
```

The file is 7,381,381,664 bytes; SHA-256 `0b9506cab36f7f818e34f9c0f5a3d6568d0b37100f3a3e1092e2eec3c4c96791`. Runtime memory also includes the KV cache and compute buffers. The launcher binds localhost, uses one 16K slot, and keeps the embedded template. `--jinja` is required for native tool parsing. Do not treat raw `<|tool_call>` text as executable commands.

## Load the coordination profile

```python
from clawagents import create_claw_agent

agent = create_claw_agent(profile="gemma-agentic")
result = await agent.invoke("Read the project instructions and plan the requested work")
await agent.llm.client.close()
```

The canonical model alias `gemma4-agentic-v2` also selects the profile automatically. CLI use: `python -m clawagents --profile gemma-agentic --task "..."`.

Optional configuration:

```dotenv
PROVIDER=gemma-agentic
GEMMA_AGENTIC_BASE_URL=http://127.0.0.1:18080/v1
GEMMA_AGENTIC_MODEL=gemma4-agentic-v2
# GEMMA_AGENTIC_API_KEY=...  # only for an authenticated server
```

Explicit constructor endpoint/model/key values override profile defaults. Without a dedicated key, the profile uses `not-needed`; it does not borrow an ambient OpenAI credential. It forces Chat Completions by default. The coordinator uses a conservative 16,384 context window, 4,096 maximum output tokens, 24 iterations, and greedy sampling (`temperature=0`) for routing. An explicit temperature is honored. Both stream and nonstream requests send top-p 0.95, top-k 64, repetition penalty 1.1, enabled native thinking, and single tool calls. These are deployment choices, not a universal maximum context claim.

The initial tool surface is small and includes `task`; optional groups can be activated. Recent worker results are retained longer than the generic local profile so they are not immediately discarded during coordination.

## Independently configured workers and verified completion

For backend jobs, provide worker clients and a deterministic acceptance callback. The callback runs in application code, receives no model-controlled arguments, and must return exactly `True` to accept completion. It may also be async. Inspect real artifacts, test results, or application state in it. A constant `True` callback is not a meaningful check.

```python
import json
import os
from pathlib import Path

from clawagents import create_claw_agent
from clawagents.tools.subagent import SubAgentSpec

workspace = Path("/path/to/job").resolve()
worker_agent = create_claw_agent(
    "gpt-5.6-luna",
    api_key=os.environ["LUNA_API_KEY"],
    base_url=os.environ.get("LUNA_BASE_URL", "https://api.openai.com/v1"),
    workspace=workspace,
)

def acceptance_check():
    try:
        report = json.loads((workspace / "validated-result.json").read_text())
        return report == {"net": 37}  # Replace with your actual job contract.
    except (OSError, ValueError):
        return False

coordinator = create_claw_agent(
    profile="gemma-agentic",
    workspace=workspace,
    subagents=[SubAgentSpec(
        name="coder", description="Implement and test the assigned change",
        llm=worker_agent.llm, max_iterations=12, timeout_seconds=180,
    )],
    completion_check=acceptance_check,
)
try:
    result = await coordinator.invoke(
        "Delegate the implementation to coder with explicit inputs, ownership, "
        "and acceptance checks. Inspect the result, then call finish_coordination."
    )
finally:
    await coordinator.llm.client.close()
    await worker_agent.llm.client.close()
```

A worker's `llm` carries its own endpoint, credential, and protocol; setting a different model name on the coordinator's client is insufficient for cross-provider routing. Unknown workers and conflicting model overrides fail explicitly. Configured provider-backed worker iteration caps cannot be increased by a tool argument. Worker deadlines and existing recursion limits bound execution. Worker clients remain caller-owned and should be closed by the caller.

`finish_coordination` is a single-call terminal action: failed acceptance keeps the job open; accepted completion returns the summary without asking the model for another turn. It is not inherited by children. Natural-language completion remains available when no callback is supplied; that mode has no application-level acceptance guarantee.

## Harness hardening

- Failed tool output is still tracked for loop detection but is never reused as a successful result. Recovery retries execute again.
- Acceptance checks are never served from duplicate-call cache.
- Loop safety stops are errors, not successful completion.
- Cancelled and budget-exhausted workers return incomplete results.
- Explicit worker clients preserve provider boundaries.
- Terminal completion retains the existing tool permission and middleware checks.

## Evaluation

Run `scripts/benchmark_gemma_coordination.py --repeats 2 --arms baseline coordinator --output results.json` from the Python environment, with this server running. The runner requires macOS `sandbox-exec` for its isolated fixtures. It tests routing, retry after a worker failure, and resistance to instructions embedded in a worker report. Workers are deterministic fixtures so the measurement isolates coordination; it is not a coding benchmark or proof of general prompt-injection resistance. Both arms use the same serving options, workers, acceptance callback, and budgets; the baseline uses the generic prompt and full tool surface. Pilot attempts are development evidence and are excluded from the final comparison.

## VS Code and measured results

Choose **Gemma Agentic Q4 (coordinator)** in the provider picker. The default endpoint is `http://127.0.0.1:18080/v1` and model is `gemma4-agentic-v2`; start the server first. The picker does not download the model. Named worker clients and acceptance callbacks are configured through the Python API.

See the [controlled Q4 comparison](benchmarks/gemma_coordination_20260906/REPORT.md): both arms passed 6/6; compact coordination used 29.8% less median wall time and 66.0% fewer median cumulative prompt tokens.
