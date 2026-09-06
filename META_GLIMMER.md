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

## Reasoning channel and output budget

Glimmer always reasons before acting. The deployment returns chain-of-thought
on a separate `reasoning_content` field (SGLang reasoning parser) and counts it
in `usage.completion_tokens` and `usage.reasoning_tokens`. Probed on
2026-09-06: `chat_template_kwargs={"enable_thinking": false}`,
`reasoning_effort`, and `separate_reasoning` are accepted and ignored, so there
is no server-side way to cap it. The harness compensates from the outside:

- The OpenAI-compatible transport captures `reasoning_content` (streaming and
  non-streaming) into `LLMResponse.thinking`, so the doom-loop detector's
  thinking channel and the `assistant_message` event see it, and
  `usage.reasoning_tokens` is metered. Reasoning never reaches the visible
  stream and is never re-sent to the server. Inline `<think>` blocks (Qwen,
  DeepSeek, llama.cpp with thinking on) get the same treatment, including a
  block cut before `</think>`.
- `profile="meta"` defaults `max_tokens` to 16,384 (explicit values win).
  Reasoning turns of 2K–6K tokens were common on real coding tasks; a 6,144
  cap truncated four of twelve benchmark trials mid-thought.
- When a turn still stops with `finish_reason="length"` and no tool call, the
  loop no longer treats the empty content as the answer or routes it into the
  "write the answer now" nudge. It records the turn as `[output truncated at
  the token limit]`, asks the model to continue with brief reasoning, grows
  `max_tokens` by 1.5× (capped), and does this at most twice per run. A tool
  call whose JSON arguments were cut by the limit is dropped rather than run
  on repaired-but-wrong arguments.
- A tool that fails twice with the same normalised error (paths, numbers and
  hashes stripped) gets an escalation appended to its result; the third time
  it says to stop retrying and change approach. This targets the observed
  pattern of retrying `unsandboxed=true` or an absolute-path `cat` with cosmetic
  changes, which identical-argument loop detection cannot see.

A custom served name (`glimmer_30B_model=Custom-Glimmer`) keeps all of the
above: `create_claw_agent(profile="meta")` registers the name as an alias for
the `meta-glimmer` harness and the `muse-glimmer-30b` context profile, and pins
the context window to the deployment's 196,608 tokens.

`.clawagents/profiles.json` and `.clawagents/harness-profiles.json` are read
from `~/.clawagents/` always, and from the workspace only with
`CLAW_FEATURE_WORKSPACE_PROFILES=1`. A cloned repository could otherwise point
the builtin `openai` profile at its own host (the env API key follows) or
replace the system prompt for every model.

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
