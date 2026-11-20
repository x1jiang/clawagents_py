# Configuration & Provider Guides

Back to [Documentation Index](README.md) · [README](../README.md)

### Where does `.env` go?

ClawAgents loads `.env` from **the directory you run the command from** (your current working directory). Different projects can have different configurations.

```
~/my-project/
├── .env              ← ClawAgents reads this when you run from ~/my-project/
├── run_agent.py
├── AGENTS.md
└── src/
```

**Three ways to configure** (in priority order, highest → lowest):

1. **`create_claw_agent()` parameters** — explicit values passed to the factory always win.
2. **`.env` file values** — by default loaded with `override=True`, so they take precedence over any pre-existing shell env vars. This is intentional for CLI use: it prevents a stale `OPENAI_API_KEY` from `~/.zshrc` silently shadowing the fresh key in `.env`. Hosts that inject keys (e.g. the VS Code extension SecretStorage) set `CLAWAGENTS_DOTENV_OVERRIDE=0` so the injected key wins; keys marked in `CLAW_KEY_SOURCES` as SecretStorage are also protected even when override is on.
3. **Shell environment variables** — used as a fallback when no `.env` is found, or for keys the `.env` doesn't define.

**Where ClawAgents looks for `.env`** (first match wins):

1. **`$CLAWAGENTS_ENV_FILE`** — explicit absolute path (handy for CI / Docker / multi-project setups).
2. **`./.env`** — the directory you ran the command from.
3. **`../.env`** — parent directory (monorepo-friendly).

A ready-to-use template is included in the repo:

```bash
cp .env.example .env   # then fill in your API key
```

Or run `clawagents --init` to generate one interactively.

## Configuration

### 1. Configure your environment

Create a `.env` file (or run `clawagents --init` to generate one):

```env
PROVIDER=gemini                    # or "openai"
GEMINI_API_KEY=AIza...             # Your Gemini API key
GEMINI_MODEL=gemini-3-flash-preview
STREAMING=1
CONTEXT_WINDOW=1000000
MAX_TOKENS=8192
TEMPERATURE=0                      # Model-specific overrides apply (see below)

# Optional: RL-inspired agent improvements
CLAW_TRAJECTORY=1                  # Enable trajectory logging + scoring
CLAW_RETHINK=1                     # Enable consecutive-failure detection
CLAW_LEARN=1                       # Enable PTRL (lessons from past runs)
```

<details>
<summary><strong>OpenAI configuration</strong></summary>

```env
PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.6-terra
STREAMING=1
CONTEXT_WINDOW=1000000
MAX_TOKENS=8192
TEMPERATURE=0                      # 0 for deterministic output
CLAW_TRAJECTORY=1
CLAW_RETHINK=1
CLAW_LEARN=1
```
</details>

### 2. One-line agent

```python
from clawagents import create_claw_agent

agent = create_claw_agent("gemini-3-flash")
result = await agent.invoke("List all Python files in src/")
print(result.result)
```

### 3. With custom instructions

```python
agent = create_claw_agent(
    "gpt-5",
    instruction="You are a senior code reviewer. Be thorough and concise."
)
result = await agent.invoke("Review this codebase and suggest improvements")
```

`instruction=` **replaces** the built-in base prompt. To change the built-in
text itself (used whenever no `instruction` is given) see the next section.

### 3b. Customizing the built-in base prompt

The default base prompt (identity, core behavior, efficiency rules) is
`clawagents.prompts.base.DEFAULT_BASE_SYSTEM_PROMPT`. Override it without
forking, highest precedence first:

| Source | Example |
|---|---|
| `base_prompt=` parameter (inline text or file path) | `create_claw_agent("gpt-5", base_prompt="prompts/base.md")` |
| `--base-prompt` CLI flag | `clawagents --task "..." --base-prompt prompts/base.md` |
| `CLAW_BASE_PROMPT_FILE` env var | `CLAW_BASE_PROMPT_FILE=prompts/base.md` |
| `CLAW_BASE_PROMPT` env var (inline text) | `CLAW_BASE_PROMPT="You are ..."` |
| `.clawagents/base-prompt.md` in the workspace | drop a file, no code change |
| `~/.clawagents/base-prompt.md` | user-wide default |
| built-in text | — |

Pass `base_prompt=""` to send no base prompt at all.

**Appending instead of replacing.** To keep the base prompt (or your
`instruction=`) and add rules after it, use any of these, highest precedence
first:

| Source | Example |
|---|---|
| `base_prompt_append=` parameter (inline text or file path) | `create_claw_agent("gpt-5", base_prompt_append="Always answer in French.")` |
| `--base-prompt-append` CLI flag | `clawagents --task "..." --base-prompt-append prompts/extra.md` |
| `CLAW_BASE_PROMPT_APPEND_FILE` env var | `CLAW_BASE_PROMPT_APPEND_FILE=prompts/extra.md` |
| `CLAW_BASE_PROMPT_APPEND` env var (inline text) | `CLAW_BASE_PROMPT_APPEND="Always answer in French."` |
| `.clawagents/base-prompt-append.md` in the workspace | drop a file, no code change |
| `~/.clawagents/base-prompt-append.md` | user-wide default |

The append block is added after the base prompt, or after `instruction=`
when one is given (an instruction replaces the base, not the append).

```python
agent = create_claw_agent(
    "gpt-5",
    instruction="You are a senior code reviewer.",
    base_prompt_append="Always answer in French.",
)
# system prompt == "You are a senior code reviewer.\n\nAlways answer in French."
```

Mode instructions (`mode=`) and model harness suffixes (e.g. GPT-5.6
efficiency rules) are layered on top of whichever base prompt wins.

### 4. With trajectory logging & rethink

```python
agent = create_claw_agent(
    "gpt-5-mini",
    trajectory=True,   # logs every turn + scores the run
    rethink=True,       # auto-injects "rethink" after 3 consecutive failures
)
result = await agent.invoke("Refactor the auth module and add tests")
# Run summary written to .clawagents/trajectories/runs.jsonl
```

### 5. With PTRL (Prompt-Time Reinforcement Learning)

```python
agent = create_claw_agent(
    "gpt-5-mini",
    learn=True,    # enables all 3 PTRL layers (implies trajectory=True)
    rethink=True,  # enhanced rethink uses past lessons
)
result = await agent.invoke("Build the data pipeline")
# After the run: lessons extracted and saved to .clawagents/lessons.md
# Next run: lessons injected into system prompt automatically
```

### 6. With Advisor Model (smart model guides cheap model)

```python
# GPT-5.4-nano executes, GPT-5.4 advises 2-3 times per task
agent = create_claw_agent(
    "gpt-5.4-nano",
    advisor_model="gpt-5.4",
)

# Cross-provider: Haiku executes, GPT-5.4 advises
agent = create_claw_agent(
    "claude-haiku-4-5",
    advisor_model="gpt-5.4",
    advisor_api_key="sk-...",
)
```

The advisor is consulted at three points: (1) after initial orientation, before committing to an approach, (2) when stuck (consecutive failures trigger rethink), and (3) before declaring the task complete. Set `ADVISOR_MODEL` in `.env` or pass `advisor_model` in code.

### 7. Multi-Sample Comparison (GRPO-inspired) 

```python
agent = create_claw_agent("gpt-5-mini", rethink=True)
# Run the task 3 times, pick the best based on objective scoring
result = await agent.compare("Fix the bug in app.py", n_samples=3)
print(result["best_result"])   # best answer
print(result["best_score"])    # objective score
print(result["all_scores"])    # all samples with scores
```

### 8. Azure OpenAI

```python
agent = create_claw_agent(
    "gpt-4o",                    # your Azure deployment name
    api_key="your-azure-key",
    base_url="https://myresource.openai.azure.com/",
    api_version="2024-12-01-preview",
    learn=True,
)
result = await agent.invoke("Analyze the codebase")
```

Or via `.env`:

```env
PROVIDER=openai
OPENAI_API_KEY=your-azure-key
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://myresource.openai.azure.com/
OPENAI_API_VERSION=2024-12-01-preview
```

### 9. Amazon Bedrock (native IAM — Claude / HIPAA)

Install the optional extra, then use a Bedrock model ID. Auth uses the standard AWS credential chain (env keys, shared credentials, instance/task role) — no Anthropic API key:

```bash
pip install 'clawagents[bedrock]'
```

```python
# Claude on Bedrock (AsyncAnthropicBedrock)
agent = create_claw_agent("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
# or: create_claw_agent(profile="bedrock")

# Amazon Nova / other non-Claude models (Converse API)
agent = create_claw_agent("amazon.nova-pro-v1:0")
```

```env
PROVIDER=bedrock
AWS_REGION=us-east-1
# Optional overrides:
# AWS_PROFILE=...
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# BEDROCK_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

If you prefer an OpenAI-compatible proxy ([Bedrock Access Gateway](https://github.com/aws-samples/bedrock-access-gateway) / [LiteLLM](https://docs.litellm.ai/docs/proxy/quick_start)), set `base_url` — routing stays on the OpenAI client:

```python
agent = create_claw_agent(
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    base_url="http://localhost:8080/v1",
    api_key="bedrock",
)
# or: create_claw_agent(profile="bedrock-gateway", base_url="http://localhost:8080/v1")
```

### 10. Local Models (Ollama / vLLM / LM Studio)

Any OpenAI-compatible local server works out of the box:

```python
# Ollama (default port 11434)
agent = create_claw_agent("llama3.1", base_url="http://localhost:11434/v1")

# vLLM
agent = create_claw_agent("Qwen/Qwen3-8B", base_url="http://localhost:8000/v1")

# LM Studio
agent = create_claw_agent("local-model", base_url="http://localhost:1234/v1")
```

Or via `.env`:

```env
# No API key needed for local models — just omit OPENAI_API_KEY
OPENAI_MODEL=llama3.1
OPENAI_BASE_URL=http://localhost:11434/v1
```

> **Tip:** For local models that emit `<think>...</think>` tokens (Qwen3, DeepSeek), thinking content is automatically detected, stripped from output, and preserved in trajectory records (Feature H).

### 11. MCP Servers (Model Context Protocol)

Wire any external **MCP server** into the agent and its tools become first-class
clawagents tools — no boilerplate. Three transports are supported (stdio, HTTP+SSE,
Streamable HTTP):

```python
from clawagents import create_claw_agent, MCPServerStdio

agent = create_claw_agent(
    "gpt-5-mini",
    mcp_servers=[
        MCPServerStdio(
            params={"command": "python", "args": ["-m", "my_mcp_server"]},
            name="my-mcp",
            cache_tools_list=True,
        ),
    ],
)
result = await agent.invoke("Use the my-mcp tools to do X")
```

Install the optional dependency once: `pip install 'clawagents[mcp]'`.
If `mcp_servers=` is non-empty without the SDK installed, the factory raises
a clear `ImportError`. The manager connects each server, lists its tools,
bridges them into the existing `ToolRegistry`, and registers a shutdown
finalizer. Every lifecycle phase (`Idle → Connecting → Initializing →
DiscoveringTools → Ready → Invoking → Errored / Shutdown`) emits a tracing
span, so MCP activity is visible in the standard tracing exporters.

For HTTP-based servers:

```python
from clawagents import MCPServerSse, MCPServerStreamableHttp

mcp_servers = [
    MCPServerSse(params={"url": "https://example.com/mcp/sse"}),
    MCPServerStreamableHttp(params={"url": "https://example.com/mcp"}),
]
```

### 12. Browser tools

Give the agent a Playwright-backed browser. Install once: `pip install 'clawagents[browser]' && playwright install chromium`.

```python
from clawagents import create_claw_agent
from clawagents.browser import create_browser_tools

agent = create_claw_agent(
    "gpt-5-mini",
    tools=create_browser_tools(),  # navigate / snapshot / click / type / screenshot / ...
)
result = await agent.invoke("Open https://example.com and summarise the page")
```

`create_browser_tools()` lazily instantiates a sandboxed `BrowserSession` on first use, applies SSRF + scheme checks before every navigation, and registers a shutdown hook so the headless Chromium is torn down when the agent exits. Cloud providers (Browserbase, browser-use) plug in via `BrowserConfig(provider="browserbase")` — see `clawagents.browser.providers.get_provider()`.

### 13. Scheduled jobs / cron

Run agent prompts on a schedule. Interval (`every 5m`) and one-shot (`@once`) schedules work out of the box; cron expressions (`0 9 * * *`) require `pip install 'clawagents[cron]'`.

```python
from clawagents import create_claw_agent, Scheduler, create_job

# Persisted to ~/.clawagents/<profile>/cron/jobs.json
create_job("Summarise overnight logs", "0 9 * * *", name="daily-summary")
create_job("Heartbeat ping", "every 5m")

async def run_prompt(job: dict) -> str:
    agent = create_claw_agent("gpt-5-nano")
    return (await agent.invoke(job["prompt"])).result

scheduler = Scheduler(runner=run_prompt)
await scheduler.start()        # poll every 30s, dispatch due jobs
# ... later ...
await scheduler.stop()
```

`list_jobs()`, `pause_job()`, `trigger_job()`, and `remove_job()` round out the management API. Each successful run records its output under `~/.clawagents/<profile>/cron/runs/<job_id>/<timestamp>.json` so you can audit history.

### 14. ACP adapter

Serve a ClawAgents agent over Zed's [Agent Client Protocol](https://github.com/zed-industries/agent-client-protocol) (JSON-RPC over stdio) so any ACP-compatible client (Zed, Cursor with ACP plugin, custom UIs) can drive the agent. Install: `pip install 'clawagents[acp]'`.

```python
from clawagents import create_claw_agent, AcpServer

agent = create_claw_agent("gpt-5-mini")
AcpServer(agent=agent).serve()  # blocks on stdin/stdout until EOF
```

Streaming chunks (`AgentMessageChunk`, `AgentThoughtChunk`), tool-call updates, and permission prompts are all bridged to ACP `SessionUpdate` events. Pass `permission_requester=` to wire HITL approval into the host UI.

### 15. RL fine-tuning hooks

Capture agent runs as training-ready trajectories and export them to TRL / SLIME / Atropos / generic JSONL. The recorder works without any RL framework installed; `trl` and `atropos` are only needed when you actually drive a trainer.

```python
from clawagents import create_claw_agent, RLRecorder
from clawagents.rl import export_jsonl

recorder = RLRecorder(task="Fix the bug in app.py", model="gpt-5-mini")
agent = create_claw_agent("gpt-5-mini", on_event=recorder.observe)
result = await agent.invoke("Fix the bug in app.py")
recorder.finalise(final=result.result, reward=1.0 if result.status == "done" else 0.0)

export_jsonl([recorder.trajectory], "runs.jsonl")
```

For online rollouts, swap `export_jsonl` for the `AtroposAdapter` HTTP submitter, or hand the trajectory to `to_trl_sft()` / `to_trl_dpo()` for offline SFT / DPO fine-tuning.

### Explicit Python Selection for Local Commands

Set `CLAWAGENTS_PYTHON` to the full interpreter path before starting ClawAgents, for example `/path/to/venv/bin/python`. Local shell execution prepends that interpreter's directory to PATH without resolving venv symlinks, sets VIRTUAL_ENV when applicable, and reports missing interpreters instead of falling back. With no selection, the inherited PATH remains unchanged. In VS Code, set `clawagents.pythonRuntime` to `custom` and `clawagents.pythonPath` in User/Remote settings; the extension supplies this environment automatically. Use the selected interpreter with `-m pip` for package operations. Explicit commands and container/remote backends are not rewritten.

## Environment Variables

All environment variables are **optional**. They serve as defaults when the corresponding `create_claw_agent()` parameter is not provided. Explicit parameters always take priority.

**General**

| Variable | Default | Required? | Description |
|:---|:---|:---:|:---|
| `CLAWAGENTS_ENV_FILE` | *(unset)* | No | Explicit path to a `.env` file. Overrides default `cwd/.env` discovery. Useful for CI, Docker, or multi-project setups |
| `CLAWAGENTS_DOTENV_OVERRIDE` | `1` | No | When `0`/`false`, workspace `.env` does not overwrite pre-set provider secrets (used by the VS Code sidecar so SecretStorage wins) |
| `CLAWAGENTS_SKIP_DOTENV` | *(unset)* | No | When `1`/`true`, skip discovering/loading workspace `.env` entirely (long-lived hosts that already injected secrets) |

**Provider & Model** — set at least one API key (or `OPENAI_BASE_URL` for local models)

| Variable | Default | Required? | Description |
|:---|:---|:---:|:---|
| `PROVIDER` | auto-detect | No | Hint: `"openai"`, `"gemini"`, `"anthropic"`, or `"bedrock"` / `"aws"`. Auto-detected from keys / AWS region |
| `OPENAI_API_KEY` | — | **Yes** *(for OpenAI/Azure)* | OpenAI or Azure API key. **Not needed for local models** — when `OPENAI_BASE_URL` is set, a placeholder is used automatically |
| `OPENAI_MODEL` | `gpt-5.6-terra` | No | Model name, Azure deployment name, or local model ID (e.g. `llama3.1`) |
| `OPENAI_BASE_URL` | *(unset)* | No | Custom endpoint for OpenAI-compatible APIs: Azure, Bedrock gateway, Ollama, vLLM, LM Studio. Omit to use `api.openai.com` |
| `OPENAI_API_VERSION` | *(unset)* | No | **Azure only.** API version string (e.g. `2024-12-01-preview`). Ignored by all other providers |
| `GEMINI_API_KEY` | — | **Yes** *(for Gemini)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | No | Gemini model name |
| `ANTHROPIC_API_KEY` | — | **Yes** *(for Anthropic)* | Anthropic API key (not used for native Bedrock) |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | No | Anthropic model name (e.g. `claude-sonnet-4-5`, `claude-opus-4`) |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | `us-east-1` | **Yes** *(for native Bedrock)* | Region for Bedrock Runtime / AsyncAnthropicBedrock |
| `AWS_PROFILE` | — | No | Shared-credentials profile for native Bedrock |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` | — | No | Explicit keys (prefer IAM role / profile in production) |
| `BEDROCK_MODEL` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | No | Default Bedrock model when `PROVIDER=bedrock` |
| `TAVILY_API_KEY` | — | **Yes** *(for `web_search`)* | Tavily API key for the built-in `web_search` tool |

**LLM Tuning**

| Variable | Default | Required? | Description |
|:---|:---|:---:|:---|
| `STREAMING` | `1` | No | `1` = streaming enabled, `0` = disabled |
| `CONTEXT_WINDOW` | `1000000` | No | Token budget. Older messages are compacted when exceeded |
| `MAX_TOKENS` | `8192` | No | Max output tokens per response (`max_completion_tokens` for OpenAI, `max_output_tokens` for Gemini) |
| `TEMPERATURE` | `0.0` | No | Sampling temperature. Auto-forced to `1.0` for reasoning models (o-series + bare `gpt-5` + `gpt-5-nano` / `gpt-5-mini` / `gpt-5-turbo`). Non-reasoning models (`gpt-5-micro`, `gpt-4o`, `gpt-4o-mini`) use the configured value |
| `MAX_ITERATIONS` | `200` | No | Max tool rounds before the agent stops. Override per-run: `agent.invoke(task, max_iterations=N)` |

**PTRL & Trajectory Flags** — all off by default, opt-in with `1`/`true`/`yes`

| Variable | Default | Required? | Description |
|:---|:---|:---:|:---|
| `CLAW_TRAJECTORY` | `0` | No | Enable trajectory logging. Records every turn + scores each run to `.clawagents/trajectories/` |
| `CLAW_RETHINK` | `0` | No | Enable consecutive-failure detection + adaptive rethink injection |
| `CLAW_LEARN` | `0` | No | Enable full PTRL: lesson extraction, injection, LLM-as-Judge, and thinking token preservation. Implies `CLAW_TRAJECTORY=1` |
| `CLAW_PREVIEW_CHARS` | `120` | No | Max chars for tool-output previews in trajectory logs |
| `CLAW_RESPONSE_CHARS` | `500` | No | Max chars for LLM response text in trajectory records |

**Claude Code Features** — mostly off by default, opt-in with `1`/`true`/`yes`

| Variable | Default | Required? | Description |
|:---|:---|:---:|:---|
| `CLAW_FEATURE_MICRO_COMPACT` | `1` | No | Aggressively clear old tool result contents to save context |
| `CLAW_FEATURE_FILE_SNAPSHOTS` | `1` | No | Safely copy files to `.clawagents/snapshots/` before writing |
| `CLAW_FEATURE_CACHE_TRACKING` | `0` | No | Extract and log detailed Anthropic/OpenAI prompt cache stats |
| `CLAW_FEATURE_TYPED_MEMORY` | `0` | No | Parse YAML frontmatter in `AGENTS.md` to classify memory types |
| `CLAW_FEATURE_WAL` | `0` | No | Persistent Write-Ahead Logging to `.clawagents/wal.jsonl` (crash recovery) |
| `CLAW_FEATURE_PERMISSION_RULES` | `0` | No | Enforce declarative glob-based `Allow`/`Deny` execution bounds |
| `CLAW_FEATURE_BACKGROUND_MEMORY` | `0` | No | Background thread extracting agent state/metadata implicitly |
| `CLAW_FEATURE_FORKED_AGENTS` | `0` | No | Enable the `run_forked_agent` sandboxed sub-agent API |
| `CLAW_FEATURE_COORDINATOR` | `0` | No | Enable the `run_coordinator` swarm routing orchestration mode |
| `CLAW_FEATURE_TRANSCRIPT_ARCHIVAL` | `0` | No | Archive full pre-compaction messages to `.clawagents/transcripts/pre_compact_*.md` (audit trail) |
| `CLAW_FEATURE_CREDENTIAL_PROXY` | `0` | No | Route subagent credentials through a least-privilege proxy instead of inheriting parent env |

**v5.28.0 Features** — inspired by [claw-code-main](https://github.com/anthropics/claw-code) (Rust reference)

| Variable | Default | Required? | Description |
|:---|:---|:---:|:---|
| `CLAW_FEATURE_CACHE_BOUNDARY` | `1` | No | Split system prompt at `__CACHE_BOUNDARY__` for Anthropic prompt caching. Static prefix cached, dynamic suffix fresh each turn. |
| `CLAW_FEATURE_SESSION_PERSISTENCE` | `0` | No | Save sessions as append-only JSONL to `.clawagents/sessions/`. Enables `--sessions` and `--resume`. |
| `CLAW_FEATURE_ERROR_TAXONOMY` | `1` | No | Classify LLM/tool errors into 7 discrete classes (`context_window`, `provider_auth`, `provider_rate_limit`, etc.) with recovery hints. |
| `CLAW_FEATURE_EXTERNAL_HOOKS` | `0` | No | Run shell hooks before/after tool calls and LLM calls. Config via `.clawagents/hooks.json` or `CLAW_HOOK_*` env vars. |

**External Hook Env Vars** (requires `CLAW_FEATURE_EXTERNAL_HOOKS=1`)

| Variable | Description |
|:---|:---|
| `CLAW_HOOK_PRE_TOOL_USE` | Shell command run before each tool. Receives JSON on stdin, can block or modify args. |
| `CLAW_HOOK_POST_TOOL_USE` | Shell command run after each tool. Can modify results. |
| `CLAW_HOOK_PRE_LLM` | Shell command run before each LLM call. Can inject extra messages. |
| `CLAW_HOOK_POST_LLM` | Shell command run after each LLM response. Fire-and-forget logging. |

---
