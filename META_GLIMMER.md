# Meta / Muse-Glimmer-30B

The `meta` entry connects to an OpenAI-compatible SGLang deployment. It is a
named provider profile, so it reuses the existing streaming and native-tool
transport rather than adding another client or dependency.

```python
from clawagents import create_claw_agent

agent = create_claw_agent(profile="meta", base_url="https://your-meta-server/v1")
result = await agent.invoke("Read pyproject.toml and report the version")
```

`create_claw_agent("Muse-Glimmer-30B")` selects the same profile. Explicit
`model`, `base_url`, and `api_key` arguments override profile defaults.

For CLI or environment-based use:

```dotenv
PROVIDER=meta
glimmer_30B_backend=https://your-meta-server/v1
glimmer_30B_model=Muse-Glimmer-30B
```

```bash
python -m clawagents --profile meta --task "Read pyproject.toml and report the version"
```

Meta has no built-in server endpoint. Supply a Base URL explicitly, through
`glimmer_30B_backend`, or in your named profile. Missing or blank endpoints
raise a configuration error before creating a client. The Python core
also accepts `GLIMMER_30B_BACKEND` and `GLIMMER_30B_MODEL`. Optional `META_API_KEY`
authenticates a protected deployment. Otherwise the SDK receives `not-needed`;
it does not forward an ambient OpenAI key to Meta. Existing `profiles.json`
entries override built-ins/environment defaults, and explicit constructor
arguments win over those entries. An explicitly empty key also uses the inert
placeholder.

The default transport is Chat Completions. The deployment's `/v1/models`
reported a 196,608-token context on 2026-09-06; the model profile reserves 20%
headroom. This is a deployment-specific limit, not a claim about every Glimmer
server. Tune the profile if your server's context limit differs.

The `meta-glimmer` harness keeps optional tools registered but starts with the
same smaller core tool surface as Luna. `activate_tool_group` reveals optional
tools. Concise tool instructions, repeated-call detection, and earlier clearing
of old tool output reduce avoidable context growth. Structured values supplied
for string tool parameters are serialized as JSON instead of Python repr,
avoiding invalid file contents and repair loops. Existing text is unchanged.
Reasoning remains at the
server default: an `enable_thinking=false` probe still produced reasoning on
this deployment, so we do not advertise a working no-thinking mode.

In the companion VS Code extension, select **Meta (Glimmer)** in Settings, then
**Muse-Glimmer-30B**. Enter your server endpoint in the required **Base URL**
field (or configure it in the environment); save and approve the
custom URL through the extension's existing endpoint prompt. Per-chat model
selection also lists Meta. Build/use the updated core together with the updated
extension. Server credentials can be provided through the sidecar environment
as `META_API_KEY`.

## Reproduce the benchmark

```bash
# Configure OPENAI_API_KEY or LUNA_API_KEY locally; never commit it.
# Optional LUNA_BASE_URL selects an OpenAI-compatible gateway.
.venv/bin/python scripts/benchmark_meta_glimmer.py \
  --repeats 3 --output results/meta-glimmer.json
```

The runner compares untuned Glimmer, tuned Glimmer, and GPT-5.6-Luna in the
Python harness. The baseline recreates the earlier full tool surface and
Python-repr coercion. It randomizes arm order within each task/repeat, uses fresh
workspaces, records token counts/latency/tool calls, and grades artifacts with
deterministic checks rather than a judge model. A correct artifact with an
exhausted tool-round budget is reported separately and does not count as a
clean pass. Three short tasks are a smoke
benchmark, not evidence of general coding superiority. Prompt caches are not
flushed; tokenizers, serving hardware, and reasoning modes differ.

## Harder coding suite

See [challenge results](benchmarks/meta_challenge_20260906/REPORT.md) and [methodology](benchmarks/meta_challenge_20260906/METHOD.md). On macOS, run:

```bash
.venv/bin/python scripts/benchmark_meta_challenge.py --output results/challenges.json
```

The harness requires `sandbox-exec` and fails closed if it is unavailable. Hidden
graders stay outside the agent workspace, with OS-level source-read denial.
