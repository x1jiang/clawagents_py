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

The `meta-glimmer` harness keeps optional tools registered but starts ordinary
coding with ten tools: file reads/edits/writes, shell execution, listing,
glob/grep, tool-group activation, result retrieval, and user questions.
`activate_tool_group` with the `coding_full` group restores the full coding surface, including
alternative editors and planning tools. Explore, plan, and goal modes retain
their mode-specific controls; connected context-protection tools stay visible.
Luna's initial tools are unchanged. Concise instructions and earlier clearing
of old tool output reduce avoidable context growth. Structured values supplied
for string tool parameters are serialized as JSON instead of Python repr,
avoiding invalid file contents and repair loops. Existing text is unchanged.

## Reasoning channel and output budget

Glimmer's reasoning strength is controlled by a system-prompt line:
`Reasoning strength: low`, `medium`, `high`, or `xhigh`. The harness maps
`reasoning_effort` to this documented control, defaulting to `medium`.
Use explicit `low` for the measured latency-oriented configuration; the default
remains `medium` because the small benchmark does not establish general coding
quality at lower effort. An explicit system-prompt directive takes precedence. `none` / `minimal` map
to `low` (not reasoning disabled); `max` maps to `xhigh`.

```python
agent = create_claw_agent(
    profile="meta", base_url="https://your-meta-server/v1",
    reasoning_effort="low",
)
```

This corrects the earlier conclusion that reasoning was uncontrollable.
The tested server ignores generic `enable_thinking`, top-level
`reasoning_effort`, `thinking_budget`, and `separate_reasoning` switches;
that does not mean it ignores the model's documented system directive.
Reasoning strength is a soft control rather than an exact token cap.
See [Meta's best practices](https://huggingface.co/meta-models/Muse-Glimmer-30B#best-practices)
for the supported system-prompt syntax.

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
  `max_tokens` by 1.5× (capped), and does this at most twice per run.
  With a run timeout, recovery avoids increasing the cap in the final reserved
  interval and reports exhausted recovery as incomplete. Temporary increases
  are restored when the run ends, including cancellation. A tool
  call whose JSON arguments were cut by the limit is dropped rather than run
  on repaired-but-wrong arguments.
- Explicit smaller context windows now constrain both the initial request and
  later turns. Input budgeting reserves the live output cap, native tool schemas,
  and template headroom; internal summarization uses the same reserve. If a
  protected user request cannot fit, the run stops with an explicit incomplete
  message instead of clipping that request. Estimates remain approximate across
  server tokenizers, so context-overflow recovery remains available.
- A Glimmer progress checkpoint watches mixed reads, searches, planning and
  shell commands. Novel evidence postpones the checkpoint; successful edits
  reset it. At most two advisory notices are appended, and reads remain
  available. Read-only tasks are never required to edit files.
- OpenAI-compatible SDK retries are disabled; the harness owns the retry
  budget, avoiding multiplied requests after transient failures.
- A tool that fails twice with the same normalised error (paths, numbers and
  hashes stripped) gets an escalation appended to its result; the third time
  it says to stop retrying and change approach. This targets the observed
  pattern of retrying `unsandboxed=true` or an absolute-path `cat` with cosmetic
  changes, which identical-argument loop detection cannot see.

A custom served name (`glimmer_30B_model=Custom-Glimmer`) keeps all of the
above: `create_claw_agent(profile="meta")` registers the name as an alias for
the `meta-glimmer` harness and the `muse-glimmer-30b` context profile. The default
context window is 196,608 tokens; pass `context_window` explicitly for a smaller
deployment.

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

For a server configured with a 32,768-token context, set the client limit
explicitly and reserve space for the requested output:

```python
agent = create_claw_agent(
    profile="meta",
    base_url="https://your-meta-server/v1",
    context_window=32768,
    max_tokens=16384,
    reasoning_effort="low",
)
result = await agent.invoke("Your coding task", timeout_s=240)
```

Use the server's actual context capacity. The reasoning setting is a soft
control; lower effort is a latency tradeoff, not a guarantee of equal accuracy.

See [challenge results](benchmarks/meta_challenge_20260906/REPORT.md) and [methodology](benchmarks/meta_challenge_20260906/METHOD.md). On macOS, run:

```bash
.venv/bin/python scripts/benchmark_meta_challenge.py --output results/challenges.json
```

The harness requires `sandbox-exec` and fails closed if it is unavailable. Hidden
graders stay outside the agent workspace, with OS-level source-read denial.
