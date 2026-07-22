<p align="center">
  <h1 align="center">🦞 ClawAgents</h1>
  <p align="center"><strong>A lean, full-stack agentic AI framework — ~2,500 LOC</strong></p>
  <p align="center">
    <img src="https://img.shields.io/badge/version-6.20.47-blue" alt="Version">
    <img src="https://img.shields.io/badge/python-≥3.10-green" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
    <img src="https://img.shields.io/badge/LOC-~2500-purple" alt="LOC">
  </p>
</p>

---

ClawAgents is a **production-ready agentic framework** that gives LLMs the ability to read, write, and execute code — with built-in planning, memory, sandboxing, and a gateway server. It supports **OpenAI**, **Google Gemini**, **Anthropic Claude**, and **Amazon Bedrock** (native IAM / Converse) out of the box, with a pluggable provider architecture for any LLM.

Built by extracting and unifying the best architectural patterns from [OpenClaw](https://github.com/anthropics/openclaw) (~5,800 files) and [DeepAgents](https://github.com/langchain-ai/deepagents) (~1,400 LOC core), ClawAgents delivers **the same power at a fraction of the complexity**.

## Apps & extensions

This repo is the **Python framework** (`pip install clawagents`). Ready-made clients (all ship with Bedrock + the current agent stack):

| Product | Latest | What it is | Link |
|---------|--------|------------|------|
| **ClawAgents Desktop** | **v0.4.26** | Native macOS app — project chats, file editor, SSH remotes, Settings (incl. AWS Bedrock), Developer ID signed + notarized | [Repo](https://github.com/x1jiang/clawagents-desktop) · [Download DMG](https://github.com/x1jiang/clawagents-desktop/releases/tag/v0.4.26) |
| **ClawAgents for VS Code / Cursor** | **v1.0.138** | Editor extension — fork hardening, Mantle Kimi ids, companion lockstep | [Repo](https://github.com/x1jiang/clawagents-vscode) · [Releases](https://github.com/x1jiang/clawagents-vscode/releases) |
| **Python package** | **v6.20.47** | This library — Context Observatory, modular agent loop · `pip install -U 'clawagents[bedrock]'` | [PyPI](https://pypi.org/project/clawagents/) · [Release](https://github.com/x1jiang/clawagents_py/releases) |
| **TypeScript package** | **v6.12.13** | Node/TS sibling — `npm install git+https://github.com/x1jiang/clawagents.git` | [Repo](https://github.com/x1jiang/clawagents) |

## Installation

```bash
pip install -U clawagents              # Core (OpenAI only)
pip install -U 'clawagents[gemini]'    # + Google Gemini support
pip install -U 'clawagents[anthropic]' # + Anthropic Claude support
pip install -U 'clawagents[bedrock]'   # + Amazon Bedrock (Claude via IAM + Nova/Converse)
pip install -U 'clawagents[all]'       # All providers + tiktoken
```

> **Version 6.20.47** — Context Observatory, modular RunBootstrapper & ContextLayer pipeline, aiohttp → httpx migration (July 2026).
>
> **Version 6.20.46** — Mantle Grok/Sonnet-5 omit temperature; Kimi `moonshotai.*`; Fable retention hint (July 2026).
>
> **Version 6.20.45** — Mantle Claude via `AsyncAnthropicBedrockMantle` (Bearer) (July 2026).
>
> **Version 6.20.44** — Mantle frontier base is `…/openai/v1` (fixes GPT-5.x 404) (July 2026).
>
> **Version 6.20.43** — Mantle xAI Grok routes via `/openai/v1` (fixes Berm access_denied) (July 2026).
>
> **Version 6.20.42** — Lower-churn patch, sandbox, timeout, and PTY recovery (July 2026).
>
> **Version 6.20.41** — Resilient skill paging and actionable audit findings (July 2026).
>
> **Version 6.20.40** — Fail-closed external-action reconciliation contracts (July 2026).
>
> **Version 6.20.39** — Context Mode binary-input guard and concise MCP failures (July 2026).
>
> **Version 6.20.38** — Stable OpenAI prompt-cache affinity, incremental token ledger, TTFT/RSS telemetry, bounded exec head/tail (July 2026).
>
> **Version 6.20.8** — Artifact path containment + preserve raw tool output for retrieval (July 2026).

> **Version 6.20.0** — Grok harness ports (July 2026).

> **Version 6.19.0** — Companion lockstep (July 2026).

> **Version 6.18.0** — Grok-inspired edit/execute harness (July 2026).

### New In v6.20.47
- **Context Observatory:** real-time LLM context inspector with token analytics, message timeline, budget visualization, session export/import (.zip), and auto-saved history browser
- **RunBootstrapper:** extracted 800+ lines of initialization from `agent_loop.py` into a 7-phase ordered bootstrapper — agent_loop.py reduced from 1382 to 676 lines (−51%)
- **ContextLayer pipeline:** pluggable system-prompt injection with 9 built-in layers (Lessons, Goal, CoreMemory, ContextLedger, MemoryBank, FactStore, Plan, RepoMap, WorkspaceEnv)
- **Gateway Observatory toggle:** `POST /observatory/toggle` endpoint to enable/disable context event streaming at runtime
- **aiohttp → httpx:** eliminated `aiohttp` dependency in SSE client (httpx already present via openai)

### New In v6.20.46
- Omit `temperature` for Mantle Grok and Claude Sonnet 5 / Fable 5 (models reject sampling params)
- Rewrite legacy `moonshot.*` Mantle ids to catalog `moonshotai.*`
- Clearer recovery hint when Claude Fable 5 requires account `provider_data_share`

### New In v6.20.45
- Mantle Claude uses `AsyncAnthropicBedrockMantle` (Bearer token) instead of plain `AsyncAnthropic` (`X-Api-Key`)
- Catalog routing matrix tests for chat / openai/v1 / anthropic messages families

### New In v6.20.44
- Mantle frontier rewrite uses `…/openai/v1` (not bare `…/openai`) so the OpenAI SDK hits `…/openai/v1/responses`
- Clearer HTTP 404 recovery hint (wrong Mantle path or GPT-5.6 Sol region: us-east-1 / us-east-2 only)

### New In v6.20.43
- Route Mantle `xai.grok-*` through `…/openai` + Responses (plain `…/v1` chat returned Berm `access_denied`)
- Keep catalog id `xai.grok-4.3` (do not rewrite to `openai.xai.*`)
- Clearer recovery hint when Berm mis-route is detected

### New In v6.20.42
- Route ambiguous/stale patch failures toward refreshed single-hunk or hashline edits
- Refuse unauthorized `unsandboxed=true` before execution and explain temporary private gcloud config
- Auto-background timed-out local commands through default OS-sandbox profiles
- Retain completed PTY screens and exit diagnostics instead of returning `unknown session_id`

### New In v6.20.41
- Treat repeated same-name `use_skill` calls as continuation pages instead of restarting the load
- Explain when a tool is outside a skill boundary and will remain unavailable after loading
- Classify nonzero `npm audit` reports as security findings without weakening their failed status
- Keep deploy safeguards framework-generic and require reconciliation commands to be read-only

### New In v6.20.40
- Require approved pre-action verification and post-action reconciliation for external publish/deploy actions
- Consume authorization before execution so failures, crashes, and timeouts cannot hide partial remote state
- Block retries, mutations, and final completion until reconciliation succeeds
- Keep external-action safeguards framework-generic while requiring exact verification and read-only reconciliation

### New In v6.20.39
- Reject binary PDF/DOCX/image/ZIP inputs to `ctx_execute_file` before UTF-8 decoding
- Route binary analysis toward `ctx_execute` or dedicated document/PDF tooling
- Keep MCP failure details in output without duplicating them in the error field

### New In v6.20.38
- Stable hashed OpenAI `prompt_cache_key` + session affinity; incremental context token ledger
- TTFT / input-cache / peak RSS usage telemetry; Gemini cache-read accounting
- Bounded streaming exec output (head+tail) with spill-to-artifact adoption
- Context-protection MCP tools remain active under Luna's reduced tool profile

### New In v6.20.3
- Execute harden: background seatbelt wrap + scrubbed env; bwrap missing-`.env` touch; killpg cancel; profile warnings visible; AST guard against unbound local imports

### New In v6.20.2
- **Seatbelt execute:** fix unbound `shlex` crash on macOS sandbox profiles

### New In v6.20.1
- Trailer-only shell markers (mid-output marker poisoning fixed); sticky env caps + denser denylist
- `create_if_missing` refuses any existing path; truthy bool parsing; streaming output retention cap
- `hashline_grep` file/size/binary/pattern caps; `pty_start` validates cwd

### New In v6.20.0
- **edit_file:** external-mod miss hint; optional `create_if_missing` (empty target → new file only); soft read-before-edit description
- **execute:** `block_until_ms` (alias of timeout; `0` → background); streaming `tool_progress` events; sticky shell env overlay (`__CLAW_ENV__`)
- **hashline_grep:** regex search returning hashline anchors; description nudge grep→edit
- **PTY routing:** execute vs pty_* descriptions; `pty_start` inherits shell_session cwd

### New In v6.19.0
- **Companions module:** `clawagents.companions` probes `context-mode` (≥1.0.169) and `rtk` (≥0.43.0)
- **Doctor:** reports companion versions + install/upgrade hints
- **RTK wrap:** one-shot stderr warning when `rtk` is below the floor

### New In v6.18.0
- **Hashline tools:** feature-gated `hashline_read` / `hashline_edit` (chunk_v1 anchors, atomic batches, stale recovery)
- **Execute:** optional `is_background` + `description`; shell-session cwd persistence; auto-background on FG timeout (adopt live process)
- **RTK wrap:** auto-wrap noisy shell cmds (`pytest`, `git status/log/diff`, `ls`, `rg`, …) when `rtk` is installed
- **Aggressive tool crush:** tighter in-loop crush thresholds (artifact + `retrieve_tool_result`)
- **Edit diagnostics:** NFKC / whitespace / nearest-line hints on `edit_file` misses

### New In v6.17.8
- **Hunk/rewind secrets:** agent-write + snapshot + hunk baselines skip `.env`/pem/credentials (also fixed `.env`→`env` `lstrip` bug)
- **Webhook SSRF:** DNS-pinned HTTPS POST (Host/SNI) closes rebind TOCTOU
- **Circuit breaker:** wired into OpenAI Responses + Gemini + Anthropic streams (and non-stream keys)
- **Doom-loop:** force response-channel instruction on resample (no `<think>` retry)

### New In v6.17.7
- **Circuit breaker:** per-endpoint keys; streaming covered; `BreakerOpen` waits without burning retries
- **Structured output:** `FallbackProvider` propagates schema to children
- **Session FTS5:** triggers + backfill (MATCH no longer silently empty)
- **Smart memory / docs:** FTS replace cleans orphans; hybrid = FTS5 + Jaccard MMR (not vectors)
- **Memory flush / dream:** cycle-0 guard; lock always released; no orphan timeout task
- **Doom-loop / HistoryThenSteps / PTY / interject:** response-channel + temp bump; graduated fold; to_thread+reaper; thread-safe export

### New In v6.17.6
- **Hooks:** `hook_taxonomy` default off; taxonomy dispatcher also requires `external_hooks` (no SessionStart RCE from cloned `hooks.json`)
- **Seatbelt:** exec wrap uses `shlex.quote` (no `repr` quote-flip sandbox escape)
- **Secrets:** ProfileBackend denies `.env`/credential path reads in-process; hunk watcher skips secret files; PTY env scrubbed; PTY tools are plan-mode write-class
- **Webhooks:** SSRF fail-closed; no auto-redirect; each hop re-validated
- **Dream:** writes `.clawagents/MEMORY.md` only (never overwrites workspace-root `MEMORY.md`)
- **Rewind:** prompt_index advances across VS Code RunContext resets

### New In v6.17.5
- **Skills:** Claude YAML `allowed-tools:` block lists parse correctly; `Bash`/`Read`/… map to clawagents tools
- **grep:** `path: "*.js"` treated as glob filter (not a missing path)
- **apply_patch:** clear error for unsupported `*** Begin Patch` envelopes

### New In v6.17.4
- **Act ≠ Goal** — Goal tools, standing reminder, and final verifier only run when `goal_mode=True`. Switching to Act/Plan pauses an active disk-backed goal.

### New In v6.17.3

- **Hooks:** StopFailure, PostToolUseFailure, PermissionDenied, Notification, SubagentStart/Stop now fire
- **Hunks:** typed attribution `AgentEdit{n}` / `ExternalEditOnAgentFile` / `External`
- **Rewind:** conversation truncate hints + host chat truncation; Linux bwrap `--ro-bind /dev/null` secret overlays

> Older release notes live in [Changelog](#changelog).

---

## 30-Second Quick Start

The fastest way to get going — scaffolds a `.env`, a `run_agent.py` starter script, and an `AGENTS.md` memory file:

```bash
pip install clawagents
cd ~/my-project         # any project directory
clawagents --init       # creates .env, run_agent.py, AGENTS.md
```

Then edit `.env` with your API key and run:

```bash
python run_agent.py
```

That's it. The generated `run_agent.py` includes commented-out examples for every provider (OpenAI, Gemini, Azure, Ollama, vLLM).

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

### CLI One-Liner

```bash
clawagents --task "List all Python files and summarize the project"
```

### Minimal Python Code

```python
import asyncio
from clawagents import create_claw_agent

async def main():
    agent = create_claw_agent("gpt-5-mini")  # or "gemini-3-flash", "llama3.1", etc.
    result = await agent.invoke("List all Python files in src/")
    print(result.result)

asyncio.run(main())
```

### Examples

See the [`examples/`](examples/) directory for ready-to-run scripts:

| File | Provider |
|:---|:---|
| [`01_openai.py`](examples/01_openai.py) | OpenAI (GPT-5, GPT-4o) |
| [`02_gemini.py`](examples/02_gemini.py) | Google Gemini |
| [`03_azure.py`](examples/03_azure.py) | Azure OpenAI |
| [`04_local_ollama.py`](examples/04_local_ollama.py) | Ollama (local) |
| [`05_local_vllm.py`](examples/05_local_vllm.py) | vLLM (local) |
| [`06_bedrock.py`](examples/06_bedrock.py) | AWS Bedrock native IAM + gateway |
| [`07_with_custom_tools.py`](examples/07_with_custom_tools.py) | Custom tools |
| [`08_compare_samples.py`](examples/08_compare_samples.py) | Multi-sample comparison |

---

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
OPENAI_MODEL=gpt-5-nano
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

### 16. CLI

```bash
# Scaffold a project (generates .env, run_agent.py, AGENTS.md)
clawagents --init

# Check your configuration
clawagents --doctor

# Run a task directly
clawagents --task "Find all TODO comments in the codebase"

# Inspect past run trajectories
clawagents --trajectory        # last run
clawagents --trajectory 5      # last 5 runs

# Start the gateway server
clawagents --serve --port 3000

# Show all options
clawagents --help
```

### Typical First-Time Flow

```bash
pip install clawagents           # 1. Install
clawagents --init                # 2. Scaffold .env, run_agent.py, AGENTS.md
# edit .env with your API key    # 3. Configure
clawagents --doctor              # 4. Verify setup
clawagents --task "hello world"  # 5. Run your first task
python run_agent.py              # 6. Or use the generated script
```

### CLI Reference

| Command | Description |
|:---|:---|
| `clawagents --init` | Scaffold a starter project: `.env` (config template), `run_agent.py` (starter script with 5 provider options), `AGENTS.md` (memory file). Skips existing files. |
| `clawagents --doctor` | Check configuration health: `.env` discovery, API keys, active model, LLM settings, PTRL flags, local endpoint reachability, trajectory history, `AGENTS.md` presence. |
| `clawagents --tools [--json]` | Inspect built-in tool schemas without starting a model client. Useful for release checks and native-tool schema debugging. |
| `clawagents --task "..."` | Run a single task. Prints a startup banner (`provider=X model=Y env=Z ptrl=...`), executes the agent, prints the result to stdout. |
| `clawagents --task "..." --output-format json` | Same as `--task`, but emit a single JSON object (`status`, `result`, `iterations`, …). Use `stream-json` for NDJSON events. |
| `clawagents --trajectory [N]` | Inspect the last N run summaries (default: 1). Shows run ID, model, task, duration, turns, tool calls, score, quality, failure breakdown, verified score, and judge verdict. Requires `CLAW_TRAJECTORY=1`. |
| `clawagents --serve [--port N]` | Start the HTTP gateway server (default port 3000). Endpoints: `POST /chat`, `POST /chat/stream` (SSE), `WS /ws`, `GET /queue`, `GET /health`. |
| `clawagents --sessions` | List saved sessions (requires `CLAW_FEATURE_SESSION_PERSISTENCE=1`). Shows session ID, turn count, status, and task. |
| `clawagents --resume [ID\|latest]` | Resume a saved session. Loads messages from JSONL and continues the conversation. Defaults to `latest`. |
| `clawagents --help` | Show all options with examples. |
| `clawagents --advisor MODEL` | Pair a stronger model for strategic guidance (e.g. `--advisor gpt-5.4`). |

---

## 🏆 Performance: ClawAgents vs Traditional Frameworks

ClawAgents v5.10 outperforms traditional multi-layer agentic frameworks through **architectural simplicity**. Here's how it stacks up against DeepAgents (LangGraph/LangChain-based) in head-to-head benchmarks.

### Benchmark Results (February 2026)

#### TypeScript — 5 tasks × 2 models × 2 frameworks (20/20 ✅)

| Framework | Gemini-2.5-flash | GPT-5-mini |
|-----------|:---:|:---:|
| **ClawAgents v5.5** | **2.3s avg** · 1.4 tools | **13.6s avg** · 1.4 tools |
| DeepAgents | 2.5s avg · 1.8 tools | 15.7s avg · 2.4 tools |

#### Per-Task Breakdown

| Task | ClawAgents (Gemini) | DeepAgents (Gemini) | ClawAgents (GPT-5) | DeepAgents (GPT-5) |
|:---|:---:|:---:|:---:|:---:|
| File Listing | 3.7s, 1 tool | 1.9s, 1 tool | 8.9s, 1 tool | 8.4s, 1 tool |
| Read & Analyze | **1.6s**, 1 tool | 3.6s, 3 tools | **5.4s**, 1 tool | 13.0s, 2 tools |
| Write File | **2.1s**, 2 tools | 2.6s, 2 tools | **5.2s**, 2 tools | 7.5s, 2 tools |
| Multi-Step | **3.4s**, 3 tools | 3.7s, 3 tools | 46.2s, 3 tools | 46.9s, 7 tools |
| Reasoning | **0.7s**, 0 tools | 0.9s, 0 tools | **2.3s**, 0 tools | 2.8s, 0 tools |

#### Python — 18/20 completed (DeepAgents hung on GPT-5 multi_step)

| Task | ClawAgents (Gemini) | DeepAgents (Gemini) | ClawAgents (GPT-5) | DeepAgents (GPT-5) |
|:---|:---:|:---:|:---:|:---:|
| File Listing | **2.8s**, 1 tool | 1.0s, 0 tools\* | **9.9s**, 1 tool | 3.4s, 1 tool |
| Read & Analyze | **2.0s**, 1 tool | 9.8s, 4 tools | **5.5s**, 1 tool | 8.4s, 3 tools |
| Write File | **2.0s**, 2 tools | 1.0s, 0 tools\* | **5.0s**, 2 tools | 9.3s, 3 tools |
| Multi-Step | **4.1s**, 3 tools | 0.9s, 0 tools\* | **16.0s**, 3 tools | ❌ hung >5min |
| Reasoning | **0.7s**, 0 tools | 1.0s, 0 tools | — | — |

> \* *DeepAgents 0-tool results mean the model answered without using filesystem tools — faster but lower-quality (unverified answers). ClawAgents consistently uses tools to verify answers.*

### Why ClawAgents Wins

```
Traditional Stack (DeepAgents):           ClawAgents:
┌─────────────────────────┐               ┌──────────────────┐
│  Your Code              │               │  Your Code       │
├─────────────────────────┤               ├──────────────────┤
│  LangGraph              │               │  ClawAgents      │
├─────────────────────────┤               │  (direct SDK)    │
│  LangChain              │               └────────┬─────────┘
├─────────────────────────┤                        │
│  ChatOpenAI / ChatGemini│                        ▼
├─────────────────────────┤               ┌──────────────────┐
│  Responses API          │               │  Responses API   │
└─────────────────────────┘               └──────────────────┘
        4 layers                                1 layer
```

| Advantage | Impact |
|:---|:---|
| **Direct SDK calls** (1 layer vs 4) | Lower latency, fewer failure points |
| **Working directory awareness** | Tools operate from CWD; DeepAgents has no CWD concept |
| **Soft + hard loop detection** | Catches repetitive tool calls at 3 repeats, hard-stops at 6 |
| **Efficiency rules in system prompt** | ~30% reduction in redundant tool calls |
| **Fewer tool calls overall** | 1.4 avg vs 1.8–2.4 (20–40% more efficient) |
| **No OpenAI lock-in** | Native Gemini + OpenAI support with FallbackProvider chain |

---

## Feature Matrix

> Compares **ClawAgents v6.10.0** against four peer agent frameworks: **Hermes Agent**
> ([metaspartan/hermes-agent](https://github.com/metaspartan/hermes-agent)), **DeepAgents**
> ([langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)), and **OpenClaw**, plus **OpenHarness** ([HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness)).
> The v6.8.1 prompt/packaging polish, v6.8.0 OpenHarness-inspired operational
> surfaces, v6.7.1 compact tool-discovery recovery, v6.7.0 security fixes, and
> v6.5/v6.6 Hermes-parity areas now ship together in the current release —
> every row in the ClawAgents column is ✅. `◐` means partial or comparable
> coverage rather than exact feature parity.

| Feature | ClawAgents v6.10.0 | Hermes Agent | DeepAgents | OpenClaw | OpenHarness |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Core** |  |  |  |  |  |
| ReAct loop | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool loop detection (soft + hard + ping-pong) | ✅ | ✅ | ❌ | ✅ | ❌ |
| Circuit breaker (no-progress / tool failure) | ✅ | ✅ | ❌ | ❌ | ◐ |
| Efficiency rules (system prompt) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Adaptive token estimation (tiktoken) | ✅ | ✅ | ❌ | ❌ | ✅ |
| Model-aware context budgeting | ✅ | ✅ | ❌ | ❌ | ◐ |
| Fraction-based summarization triggers | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Tools** |  |  |  |  |  |
| Pluggable sandbox backend | ✅ | ✅ | ✅ | ✅ | ◐ |
| In-memory VFS (testing) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Cross-provider conformance tests | ✅ | ✅ | ✅ | ❌ | ◐ |
| Lazy tool registry (deferred imports) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Compact tool-universe discovery | ✅ | ❌ | ❌ | ❌ | ◐ |
| Tool lookup over names, descriptions, and keywords | ✅ | ❌ | ❌ | ❌ | ✅ |
| Tool result caching (LRU) | ✅ | ❌ | ❌ | ❌ | ❌ |
| JSON Schema param validation + coercion | ✅ | ✅ | ❌ | ❌ | ✅ |
| ComposeTool (deterministic pipelines) | ✅ | ❌ | ❌ | ❌ | ❌ |
| `think` tool (structured reasoning) | ✅ | ✅ | ❌ | ❌ | ❌ |
| LangChain tool adapter | ✅ | N/A | N/A | ❌ | N/A |
| MCP server integration (stdio / SSE / Streamable HTTP) | ✅ | ✅ | ❌ | ❌ | ✅ |
| Path-scoped parallel tool execution | ✅ | ✅ | ❌ | ❌ | ◐ |
| **Agents & Orchestration** |  |  |  |  |  |
| Sub-agent delegation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Subagent depth limit (≤ 2, no recursion) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Subagent / forked-agent memory isolation | ✅ | ✅ | ✅ | ❌ | ◐ |
| Per-agent IterationBudget | ✅ | ✅ | ❌ | ❌ | ❌ |
| Coordinator / swarm mode | ✅ | ❌ | ❌ | ✅ | ✅ |
| Barrier-based request scheduling | ✅ | ❌ | ❌ | ❌ | ❌ |
| Planning / TodoList | ✅ | ✅ | ✅ | ❌ | ✅ |
| Plugin hook expansion (priority chain) | ✅ | ✅ | ❌ | ❌ | ◐ |
| **Providers & Resilience** |  |  |  |  |  |
| Three-tier provider fallback + quarantine | ✅ | ✅ | ❌ | ❌ | ❌ |
| Native + text tool call repair | ✅ | ✅ | ✅ | ❌ | ❌ |
| Structured nonzero `execute` output | ✅ | ❌ | ❌ | ❌ | ❌ |
| Repeated command-failure recovery hints | ✅ | ❌ | ❌ | ❌ | ❌ |
| Streaming with stall detection | ✅ | ✅ | ❌ | ✅ | ◐ |
| Truncated JSON repair + retry | ✅ | ✅ | ❌ | ❌ | ❌ |
| Model-specific temperature override | ✅ | ✅ | ❌ | ❌ | ❌ |
| Gemini 3 thought_signature support | ✅ | ❌ | ❌ | ❌ | ❌ |
| Thinking token preservation (`<think>`) | ✅ | ✅ | ❌ | ❌ | ◐ |
| Model control token stripping | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Memory & Context** |  |  |  |  |  |
| Persistent memory (AGENTS.md) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Auto-summarization + history offloading | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pre-compact transcript archival | ✅ | ✅ | ❌ | ❌ | ◐ |
| Atomic file writes (crash-safe) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Session persistence + resume | ✅ | ✅ | ❌ | ❌ | ✅ |
| Session heartbeat + auto-cleanup | ✅ | ✅ | ❌ | ❌ | ❌ |
| Background memory extraction | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Security & Hooks** |  |  |  |  |  |
| Rich hook result model (block/redirect/inject) | ✅ | ✅ | ✅ | ✅ | ◐ |
| Credential proxy for sandboxed agents | ✅ | ✅ | ❌ | ✅ | ❌ |
| External shell hooks (pre/post tool + LLM) | ✅ | ✅ | ❌ | ✅ | ✅ |
| Declarative permission rules | ✅ | ✅ | ❌ | ❌ | ✅ |
| Tool access control (block/allow) | ✅ | ✅ | ❌ | ❌ | ✅ |
| Human-in-the-loop | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Skills** |  |  |  |  |  |
| SKILL.md with constraint documents | ✅ | ✅ | ✅ | ✅ | ✅ |
| Skill eligibility gating (OS/bins/env) | ✅ | ✅ | ✅ | ❌ | ❌ |
| Runtime `display_clawagents_home()` (path rendering in tool descriptions) | ✅ | ✅ | ❌ | ❌ | ❌ |
| **RL & Self-Improvement** |  |  |  |  |  |
| Prompt-Time RL (PTRL) — learn from past runs | ✅ | ❌ | ❌ | ❌ | ❌ |
| Trajectory logging + run scoring | ✅ | ✅ | ❌ | ❌ | ❌ |
| Trajectory compression (RLAIF / fine-tuning ready) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Consecutive-failure rethink | ✅ | ❌ | ❌ | ❌ | ❌ |
| Adaptive rethink threshold | ✅ | ❌ | ❌ | ❌ | ❌ |
| Deterministic verification (exit codes, tests) | ✅ | ✅ | ❌ | ❌ | ◐ |
| GRPO-inspired multi-sample comparison | ✅ | ❌ | ❌ | ❌ | ❌ |
| Task-type-aware verification | ✅ | ❌ | ❌ | ❌ | ❌ |
| LLM-as-Judge verification | ✅ | ✅ | ❌ | ❌ | ❌ |
| RL fine-tuning hooks (TRL / SLIME / Atropos) | ✅ | ✅ | ❌ | ❌ | ❌ |
| RFT-ready transition export | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Infrastructure** |  |  |  |  |  |
| Gateway HTTP server + SSE | ✅ | ✅ | ❌ | ✅ | ✅ |
| WebSocket gateway | ✅ | ✅ | ❌ | ✅ | ◐ |
| Activity heartbeats (prevent gateway false-timeouts) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Multi-channel messaging (Telegram, WhatsApp, Signal) | ✅ | ✅ (+ Discord, Slack, Feishu, WeChat, QQ) | ❌ | ✅ | ✅ (+ Feishu, Slack, Discord) |
| Per-session message serialization | ✅ | ✅ | ❌ | ✅ | ◐ |
| Error taxonomy + recovery recipes | ✅ | ✅ | ❌ | ❌ | ❌ |
| Prompt cache boundary (Anthropic) | ✅ | ✅ | ✅ | ❌ | ❌ |
| Prompt-cache-aware `CommandDef` (deferred state mutation) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Lane-based command queue | ✅ | ✅ | ❌ | ✅ | ◐ |
| Hermetic test runner with concurrency pinning | ✅ | ✅ | ❌ | ❌ | ❌ |
| Cron / scheduled jobs | ✅ | ✅ | ❌ | ❌ | ❌ |
| ACP (Agent Communication Protocol) adapter | ✅ | ✅ | ❌ | ❌ | ❌ |
| Browser tools (Playwright / CDP / Camoufox) | ✅ | ✅ | ❌ | ❌ | ◐ |

---

## Architecture

### Core Components

```
clawagents/
├── agent.py              # ClawAgent class, create_claw_agent factory
├── __main__.py            # CLI entrypoint (--init, --doctor, --task, --serve, --trajectory)
├── config/
│   ├── config.py          # EngineConfig, .env discovery, model resolution
│   └── features.py        # 15 feature flags (CLAW_FEATURE_* env vars)
├── providers/
│   ├── llm.py             # LLMProvider ABC + OpenAI/Gemini/Anthropic implementations
│   └── fallback.py        # FallbackProvider — 3-tier failover + quarantine (v6.0)
├── tools/
│   ├── registry.py        # ToolRegistry, LazyTool, parallel execution, LRU cache (v6.0)
│   ├── filesystem.py      # ls, read_file, write_file, edit_file, grep, glob
│   ├── advanced_fs.py     # tree, diff, insert_lines
│   ├── exec.py            # Shell command execution with dangerous command blocking
│   ├── subagent.py        # Sub-agent delegation with state isolation (v6.0)
│   ├── skills.py          # SKILL.md loading with constraint documents (v6.0)
│   ├── think.py           # Structured reasoning (no side effects)
│   ├── web.py             # URL fetching with HTML cleanup
│   ├── todolist.py        # write_todos, update_todo
│   ├── compose.py         # ComposeTool — deterministic multi-tool pipelines
│   ├── interactive.py     # ask_user (stdin-based)
│   ├── cache.py           # ResultCacheManager (SHA-256, TTL-based)
│   ├── validate.py        # JSON Schema param validation + lenient coercion
│   └── permissions.py     # Declarative permission rules (glob-based)
├── graph/
│   ├── agent_loop.py      # Core ReAct loop — pure control flow (676 lines, v6.20.47)
│   ├── run_bootstrapper.py # 7-phase initialization (RunBootstrapper, RunSession, AdvisorController)
│   ├── context_layers.py  # Pluggable system-prompt injection (ContextLayer protocol + 9 impls)
│   ├── run_config.py      # AgentRunConfig — stable parameter object
│   ├── run_runtime.py     # RunEvents, HookDispatcher, SessionMessageJournal
│   ├── run_finalizer.py   # Output guardrails, output type coercion, session flush
│   ├── turn_driver.py     # IncrementalTokenLedger, per-turn LLM orchestration
│   ├── turn_llm.py        # LLM call wrapper with usage tracking + TTFT
│   ├── turn_response.py   # Response parsing (native tool calls, text extraction)
│   ├── round_scheduler.py # Iteration budget, timeout, cancel checks
│   ├── round_dispatcher.py # Per-round dispatch (LLM → tools → handoff decision)
│   ├── tool_batch.py      # ToolBatchSafety, ToolCallRunner, ToolPolicyGate, RethinkController
│   ├── tool_turn.py       # ToolTurnExecutor — serial/parallel tool execution
│   ├── tool_observation.py # Tool result formatting, truncation, side effects
│   ├── completion_handler.py # Final-answer detection + output post-processing
│   ├── handoff_router.py  # Agent handoff dispatch (v6.4)
│   ├── context_management.py # Context window GC — compaction, trimming, WAL, history offload
│   ├── loop_tracker.py    # Tool loop detection (soft + hard + ping-pong)
│   ├── message_repair.py  # Dangling tool-call repair, orphan tool-result dropping
│   ├── model_profiles.py  # Model-aware context budgets + long-context thresholds
│   ├── coordinator.py     # Coordinator/swarm orchestration mode
│   └── forked_agent.py    # Background forked agent pattern
├── context_observatory/   # Real-time LLM context inspector (v6.20.47)
│   ├── app.py             # Streamlit UI — chat panel, context inspector, token analytics
│   ├── events.py          # Typed event dataclasses (LLMCallEvent, CompactionEvent, etc.)
│   ├── store.py           # EventStore — in-memory event log with session export/import
│   ├── hooks.py           # ContextObserverHooks — RunHooks bridge for context events
│   ├── sse_client.py      # SSE client for sidecar gateway (httpx-based)
│   ├── sse_hooks_bridge.py # SSE → EventStore translator
│   └── components/        # Streamlit UI panels (history browser, etc.)
├── sandbox/
│   ├── backend.py         # SandboxBackend protocol (15+ methods)
│   ├── local.py           # LocalBackend (pathlib + asyncio)
│   ├── memory.py          # InMemoryBackend (VFS for testing)
│   └── credential_proxy.py # Credential proxy for sandboxed agents (v6.0)
├── trajectory/            # RL-inspired run analysis
│   ├── recorder.py        # TrajectoryRecorder, scoring, quality grading
│   ├── lessons.py         # PTRL — post-run self-analysis + lesson injection
│   ├── verifier.py        # Deterministic verification, task-type detection
│   ├── compare.py         # GRPO-inspired multi-sample comparison
│   ├── judge.py           # LLM-as-Judge verification
│   └── background_memory.py # Continuous memory extraction
├── session/
│   ├── persistence.py     # Append-only JSONL session events
│   └── heartbeat.py       # Session heartbeat + auto-cleanup (v6.0)
├── memory/                # AGENTS.md discovery + LLM compaction
├── channels/              # Multi-channel messaging (Telegram, WhatsApp, Signal)
├── hooks/                 # External shell hook system
├── errors/                # Error taxonomy + recovery recipes
├── gateway/               # HTTP + WebSocket gateway server
├── process/               # Lane-based command queue with barriers (v6.0)
├── utils/                 # Atomic file writes (v6.0)
└── logging/               # Structured diagnostic logging
```

### Built-in Tools

Every agent includes these — no setup needed:

| Tool | Description |
|:---|:---|
| `ls` | List directory with size + modified time |
| `read_file` | Read file with line numbers + pagination |
| `write_file` | Write/create file (auto-creates directories) |
| `edit_file` | Replace text with pattern matching |
| `grep` | Search — single file or recursive with glob filter |
| `glob` | Find files by pattern (`**/*.py`) |
| `execute` | Shell command execution |
| `tree` | Recursive directory tree with smart ignoring |
| `diff` | Unified diff between two files |
| `insert_lines` | Precise line-level insertion |
| `think` | Structured reasoning without side effects |
| `web_fetch` | URL fetching with HTML stripping (50KB cap) |
| `web_search` | Web search via Tavily (`TAVILY_API_KEY` required) |
| `write_todos` | Plan tasks as a checklist |
| `tool_program` | Bounded read-only multi-tool sequence with `${step.output}` substitutions |
| `update_todo` | Mark plan items complete |
| `task` | Delegate to a sub-agent with isolated context |
| `ask_user` | Interactive stdin-based user input |
| `use_skill` | Load a skill's instructions (when skills exist) |
| `search_history` | Cross-session raw message recall from archived sessions |
| `skill_workshop` | Governed skill proposals (create/update/apply/reject/rollback) |

### Tool Examples

<details>
<summary><strong>📂 Filesystem — ls, read_file, write_file, edit_file</strong></summary>

The agent calls tools by emitting JSON blocks. Here's what happens under the hood when you ask the agent to work with files:

```python
# The agent autonomously emits tool calls like:

# List a directory
{"tool": "ls", "args": {"path": "src/"}}
# → Returns:  drwxr-xr-x  4.0 KB  2026-02-24  components/
#             -rw-r--r--  1.2 KB  2026-02-24  main.py

# Read a file with pagination
{"tool": "read_file", "args": {"path": "src/main.py", "offset": 0, "limit": 50}}
# → Returns:  1 | import asyncio
#             2 | from clawagents import create_claw_agent
#             ...

# Write a new file (parent directories auto-created)
{"tool": "write_file", "args": {"path": "src/utils/helpers.py", "content": "def greet(name):\n    return f'Hello, {name}!'"}}
# → Returns:  ✅ Wrote 45 bytes to src/utils/helpers.py

# Edit an existing file by pattern match
{"tool": "edit_file", "args": {
    "path": "src/main.py",
    "old": "print('hello')",
    "new": "print('Hello, World!')"
}}
# → Returns:  ✅ 1 replacement made in src/main.py
```

</details>

<details>
<summary><strong>🔍 Search — grep, glob</strong></summary>

```python
# Recursive grep across all Python files
{"tool": "grep", "args": {"pattern": "TODO", "path": "src/", "include": "*.py"}}
# → Returns:  src/agent.py:42:  # TODO: add retry logic
#             src/tools/web.py:15:  # TODO: handle redirects

# Single-file search
{"tool": "grep", "args": {"pattern": "class.*Tool", "path": "src/tools/registry.py"}}
# → Returns:  15: class ToolResult:
#             24: class Tool(Protocol):

# Find files by pattern
{"tool": "glob", "args": {"pattern": "**/*.md", "path": "."}}
# → Returns:  ./README.md (15.3 KB)
#             ./docs/ARCHITECTURE.md (4.1 KB)
#             ./AGENTS.md (892 B)
```

</details>

<details>
<summary><strong>⚡ Shell Execution</strong></summary>

```python
# Run any shell command
{"tool": "execute", "args": {"command": "python -m pytest tests/ -v"}}
# → Returns full stdout/stderr with exit code

# With custom timeout (in milliseconds)
{"tool": "execute", "args": {"command": "pip install requests", "timeout": 60000}}

# Dangerous commands are auto-blocked
{"tool": "execute", "args": {"command": "rm -rf /"}}
# → Error: Blocked potentially destructive command
```

</details>

<details>
<summary><strong>🧠 Think — structured reasoning</strong></summary>

```python
# The agent can reason without side effects
{"tool": "think", "args": {
    "thought": "The user wants me to refactor the database layer. Let me plan: 1) Read the current schema, 2) Identify coupled components, 3) Extract a repository pattern, 4) Update tests."
}}
# → [Thought recorded] — no files touched, no commands run
```

This reduces unnecessary tool calls by giving the agent a structured space to plan.
</details>

<details>
<summary><strong>📋 Planning — write_todos, update_todo</strong></summary>

```python
# Create a structured plan
{"tool": "write_todos", "args": {
    "todos": ["Read the existing codebase", "Fix the auth bug", "Add unit tests", "Update docs"]
}}
# → ## Progress: 0/4 complete
#   0. [ ] Read the existing codebase
#   1. [ ] Fix the auth bug
#   2. [ ] Add unit tests
#   3. [ ] Update docs

# Mark steps complete as you go
{"tool": "update_todo", "args": {"index": 0}}
# → ## Progress: 1/4 complete
#   0. [x] Read the existing codebase
#   1. [ ] Fix the auth bug
#   ...
```

</details>

<details>
<summary><strong>🤖 Sub-agent delegation</strong></summary>

```python
# Delegate to a fresh sub-agent with isolated context
{"tool": "task", "args": {
    "description": "Analyze all Python files in src/ and create a summary of the module structure",
    "max_iterations": 10
}}
# → [Sub-agent completed: 6 tool calls, 4 iterations]
#   The src/ directory contains 3 modules: ...

# With named specialized sub-agents (configured at creation)
{"tool": "task", "args": {
    "description": "Review this pull request for security issues",
    "agent": "security-reviewer"
}}
```

**Registering named sub-agents:**
```python
from clawagents import create_claw_agent
from clawagents.tools.subagent import SubAgentSpec

agent = create_claw_agent(
    "gemini-3-flash",
    subagents=[
        SubAgentSpec(
            name="researcher",
            description="Deep research on a topic",
            system_prompt="You are a thorough researcher. Always cite sources.",
            max_iterations=15,
        ),
        SubAgentSpec(
            name="coder",
            description="Write and test code",
            system_prompt="You are a senior engineer. Write clean, tested code.",
            max_iterations=10,
        ),
    ],
)
```

</details>

<details>
<summary><strong>🌐 Web Fetch</strong></summary>

```python
# Fetch and read a web page (HTML stripped automatically)
{"tool": "web_fetch", "args": {"url": "https://docs.python.org/3/library/asyncio.html"}}
# → [200] https://docs.python.org/3/library/asyncio.html
#   asyncio — Asynchronous I/O ...

# Fetch a JSON API
{"tool": "web_fetch", "args": {"url": "https://api.github.com/repos/python/cpython", "timeout": 10}}
# → Returns raw JSON response
```

</details>

<details>
<summary><strong>❓ AskUserQuestion — structured HITL</strong></summary>

#### Structured HITL

`ask_user_question` lets the agent ask 1-3 multiple-choice questions in a single batch — useful for upfront clarification with a small, well-defined option set. Each question carries a short `header` (≤80 chars), the `question` text (≤256 chars), and 2-4 unique `options`. Headers must be unique across the batch; an implicit `Other (please specify)` option is appended automatically so the user can break out of the menu.

The actual rendering and answer collection is delegated to a callback you supply, so the same tool plugs into a CLI prompt, a TUI, a web UI, or a channel adapter (Telegram/Signal/etc.) without code changes:

```python
from clawagents import create_claw_agent, ask_user_question_tool

async def my_ui(questions):
    # Render questions with your UI of choice; return a dict keyed by header.
    return {
        q["header"]: {"question": q["question"], "answer": q["options"][0]}
        for q in questions
    }

agent = create_claw_agent("gpt-5", tools=[ask_user_question_tool(on_ask=my_ui)])
```

If no `on_ask` is supplied the tool fails fast with a clear error rather than hanging on stdin — safe to install in headless gateways.

</details>

<details>
<summary><strong>🖼️ Image Sanitization (Tool Output Hygiene)</strong></summary>

#### Multimodal — Tool Output Hygiene

Anthropic's Messages API rejects images > 5MB and tends to fail on images much larger than ~2000px on a side. When tool results surface large screenshots or attachments, they can silently break the conversation. `clawagents.media.images` clamps base64 image blocks down to safe limits via Pillow:

```python
from clawagents.media.images import sanitize_image_block, sanitize_tool_output

clean_block = sanitize_image_block(block, max_dim=1200, max_bytes=5 * 1024 * 1024)
clean_output = sanitize_tool_output(tool_result_blocks)  # walks a list of content blocks
```

- Base64 sources: decode → resize the longest side down to `max_dim` (aspect-preserving), recompress as JPEG (or PNG when the input is a PNG with alpha) walking through `quality_steps=(90, 75, 60)` until under `max_bytes`. If still too big at the lowest quality, the block is replaced with a `[image too large after sanitization, dropped]` text block.
- URL sources and non-image blocks pass through unchanged.
- Pillow is **optional** (`pip install 'clawagents[media]'`). Without it, the helpers no-op and emit a one-time warning. `is_pillow_available()` reports the runtime state.

</details>

### Custom Tools

Create your own tools by implementing the `Tool` protocol:

```python
from clawagents import create_claw_agent
from clawagents.tools.registry import Tool, ToolResult

class DatabaseQueryTool:
    name = "query_db"
    description = "Run a read-only SQL query against the application database."
    parameters = {
        "sql": {"type": "string", "description": "The SQL SELECT query", "required": True},
        "limit": {"type": "number", "description": "Max rows to return. Default: 100"},
    }

    async def execute(self, args):
        sql = args.get("sql", "")
        limit = int(args.get("limit", 100))
        # ... your database logic here ...
        rows = await run_query(sql, limit=limit)
        return ToolResult(success=True, output=format_table(rows))

# Register custom tools alongside built-ins
agent = create_claw_agent("gpt-5", tools=[DatabaseQueryTool()])
```

You can also wrap **LangChain tools** directly:

```python
from langchain_community.tools import WikipediaQueryRun

agent = create_claw_agent("gpt-5", tools=[WikipediaQueryRun()])
# LangChain tools are automatically adapted via LangChainToolAdapter
```

---

## Skills System

Skills are **reusable instruction sets** that teach the agent domain-specific knowledge without blindly injecting every body. Retrieval is quality-first: a compact name index preserves discoverability, structured aliases/triggers/anti-triggers and morphology-aware ranking retain every plausible intent up to a generous ambiguity cap, and short follow-ups inherit the prior substantive request. `list_skills` uses the same ranked search when the model needs a wider view.

`use_skill` delivers large instructions in contiguous, content-hash-bound pages. No data-plane tool can run until every page is read; skipped, stale, or reordered continuations fail. Multiple completed skills compose, and their declared `allowed-tools` boundaries intersect so loading another skill cannot widen authority. An explicit empty boundary allows no data-plane tools.

### Skill Directory Structure

```
your-project/
├── skills/                  # Auto-discovered (or .skills/, skill/, .skill/, Skills/)
│   ├── code_review/
│   │   └── SKILL.md         # ← Skill defined as a folder + SKILL.md
│   ├── sql_expert.md         # ← Skill defined as a single .md file
│   └── deploy_checklist.md
├── AGENTS.md                 # Project memory (auto-injected)
└── src/
    └── ...
```

### Writing a Skill

Every skill is a Markdown file with optional YAML frontmatter:

**Example 1 — `skills/code_review/SKILL.md`**

```markdown
---
name: code_review
description: "Perform thorough code reviews following team standards"
allowed-tools: read_file grep glob think
---

# Code Review Skill

When reviewing code, follow these steps:

## 1. Structure Check
- Verify the file follows our module pattern (one class per file)
- Check imports are grouped: stdlib → third-party → local
- Ensure `__init__.py` exports are up to date

## 2. Logic Review
- Look for unhandled edge cases (empty inputs, None values)
- Verify error messages are actionable
- Check that async functions are properly awaited

## 3. Security
- No hardcoded secrets or API keys
- SQL queries use parameterized statements
- User input is sanitized before use

## 4. Output Format
Provide your review as:
- ✅ **Approved** — no issues found
- ⚠️ **Changes requested** — list specific issues with file:line references
- 🚫 **Blocked** — critical issues that must be fixed
```

**Example 2 — `skills/sql_expert.md`** (single-file skill)

```markdown
---
name: sql_expert
description: "Write optimized SQL queries for PostgreSQL"
allowed-tools: execute read_file think
---

# SQL Expert

You are a PostgreSQL expert. When writing queries:

## Rules
1. Always use explicit `JOIN` syntax (never implicit joins in WHERE)
2. Use CTEs (`WITH` clauses) for complex multi-step queries
3. Add `EXPLAIN ANALYZE` when the user asks about performance
4. Use parameterized queries — never interpolate user values
5. Default to `LIMIT 100` unless the user specifies otherwise

## Patterns

### Pagination
Use keyset pagination for large tables:
```sql
SELECT * FROM events
WHERE id > :last_seen_id
ORDER BY id
LIMIT 50;
```

### Aggregation
Always include the raw count alongside percentages:
```sql
SELECT
    status,
    COUNT(*) AS n,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM orders
GROUP BY status
ORDER BY n DESC;
```
```

**Example 3 — `skills/deploy_checklist.md`**

```markdown
---
name: deploy_checklist
description: "Step-by-step production deployment checklist"
---

# Deployment Checklist

Before deploying to production, complete every step:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] No lint errors: `ruff check src/`
- [ ] Version bumped in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] Docker image builds: `docker build -t app:latest .`
- [ ] Smoke test on staging environment
- [ ] Database migrations reviewed and tested
- [ ] Rollback plan documented
```

### How Skills Work at Runtime

```python
# Skills are auto-discovered from ./skills/ directory
agent = create_claw_agent("gemini-3-flash")

# Or specify custom skill directories
agent = create_claw_agent("gpt-5", skills=["./my-skills", "./shared-skills"])
```

When skills are available, the agent gets two additional tools:

```python
# 1. List available skills
{"tool": "list_skills", "args": {}}
# → Available skills (3):
#   - **code_review**: Perform thorough code reviews following team standards
#     → Allowed tools: read_file, grep, glob, think
#   - **sql_expert**: Write optimized SQL queries for PostgreSQL
#     → Allowed tools: execute, read_file, think
#   - **deploy_checklist**: Step-by-step production deployment checklist

# 2. Load a specific skill's instructions
{"tool": "use_skill", "args": {"name": "sql_expert"}}
# → Returns the full skill content, injected into the agent's context
```

The agent **decides on its own** when to use a skill. If you ask it to "write a query to find all overdue orders," and a `sql_expert` skill exists, it will load the skill first, then write the query following those rules.

---

## API Reference

### `create_claw_agent(model, instruction, ...)`

All parameters are **optional** — zero-config usage (`create_claw_agent()`) works if you have a `.env` with at least one API key.

**Model & Provider**

| Param | Type | Default | Required? | Description |
|:---|:---|:---|:---:|:---|
| `model` | `str \| LLMProvider \| None` | `None` | No | Model name (e.g. `"gpt-5-mini"`, `"gemini-3-flash"`, `"llama3.1"`), a pre-built `LLMProvider` instance, or `None` to auto-detect from env |
| `api_key` | `str \| None` | `None` | No | API key. Auto-routed to OpenAI or Gemini based on model name. Falls back to `OPENAI_API_KEY` / `GEMINI_API_KEY` env vars. For local models: omit entirely (a placeholder is used automatically) |
| `base_url` | `str \| None` | `None` | No | Custom endpoint URL for OpenAI-compatible APIs. Set this for **Azure OpenAI**, **AWS Bedrock** (via gateway), **Ollama**, **vLLM**, **LM Studio**, or any OpenAI-compatible server. Falls back to `OPENAI_BASE_URL` env var. Omit to use `api.openai.com` |
| `api_version` | `str \| None` | `None` | No | API version string. **Only needed for Azure OpenAI** (e.g. `"2024-12-01-preview"`). Falls back to `OPENAI_API_VERSION` env var. Ignored for all other providers |

**Agent Behavior**

| Param | Type | Default | Required? | Description |
|:---|:---|:---|:---:|:---|
| `name` | `str \| None` | `None` | No | Optional human-readable name for this agent. Used in handoff routing and tracing |
| `instruction` | `str \| None` | `None` | No | System prompt — what the agent should do and how it should behave |
| `tools` | `list \| None` | `None` | No | Additional tools to register. Built-in tools (filesystem, exec, grep, etc.) are always included |
| `skills` | `str \| list \| None` | auto-discover | No | Skill directories to load. Default: checks `./skills`, `./.skills`, `./skill`, `./.skill`, `./Skills`. Bundled OpenViking is included when eligible. |
| `memory` | `str \| list \| None` | auto-discover | No | Memory files to inject into system prompt. Default: checks `./AGENTS.md`, `./CLAWAGENTS.md` |
| `sandbox` | `SandboxBackend` | `LocalBackend()` | No | Pluggable sandbox backend for file/shell operations. Use `InMemoryBackend` for testing |
| `streaming` | `bool` | `True` | No | Enable streaming responses |
| `use_native_tools` | `bool` | `True` | No | Use provider native function calling. Set `False` for text-based JSON tool calls |
| `on_event` | `callable \| None` | `None` | No | Callback for agent events (tool calls, errors, context messages, etc.) |
| `handoffs` | `list[Handoff] \| None` | `None` | No | Sub-agents this agent can delegate to. See the **Handoffs** docs for the routing protocol |
| `mcp_servers` | `list \| None` | `None` | No | MCP servers to expose as tools. See the **MCP Servers** section for configuration |
| `fallback_models` | `list[str] \| None` | env `CLAWAGENTS_FALLBACK_MODELS` / `None` | No | Ordered fallback model names; tried in order if the primary provider fails. Precedence between env and arg is controlled by `CLAWAGENTS_PROVIDER_CONFIG_MODE` (`env_override` \| `default` \| `fallback`) |
| `advisor_model` | `str \| LLMProvider \| None` | env `ADVISOR_MODEL` / `None` | No | A stronger model that advises the primary model 2–3 times per task. See **Configuration § Advisor Model** |
| `advisor_api_key` | `str \| None` | env `ADVISOR_API_KEY` / `None` | No | API key for the advisor model when it lives on a different provider |
| `advisor_max_calls` | `int \| None` | env `ADVISOR_MAX_CALLS` / `3` | No | Maximum advisor consultations per task |

**LLM Tuning**

| Param | Type | Default | Required? | Description |
|:---|:---|:---|:---:|:---|
| `context_window` | `int \| None` | env `CONTEXT_WINDOW` / `1000000` | No | Token budget. When messages exceed this, older turns are compacted |
| `max_tokens` | `int \| None` | env `MAX_TOKENS` / `8192` | No | Max output tokens per LLM response. Sent as `max_completion_tokens` (OpenAI) or `max_output_tokens` (Gemini) |
| `temperature` | `float \| None` | env `TEMPERATURE` / `0.0` | No | LLM sampling temperature. Automatically forced to `1.0` for reasoning models (o1 / o3 / o4-mini, bare `gpt-5`, and `gpt-5-nano` / `gpt-5-mini` / `gpt-5-turbo`). Non-reasoning models (`gpt-5-micro`, `gpt-4o`, `gpt-4o-mini`) respect the configured value |
| `max_iterations` | `int \| None` | env `MAX_ITERATIONS` / `200` | No | Max tool rounds before the agent stops and returns |

**PTRL & Trajectory**

| Param | Type | Default | Required? | Description |
|:---|:---|:---|:---:|:---|
| `trajectory` | `bool \| None` | env `CLAW_TRAJECTORY` / `False` | No | Enable trajectory logging. Records every turn as NDJSON to `.clawagents/trajectories/` and scores each run |
| `rethink` | `bool \| None` | env `CLAW_RETHINK` / `False` | No | Enable consecutive-failure detection. Injects a "rethink" prompt with adaptive threshold after repeated tool failures |
| `learn` | `bool \| None` | env `CLAW_LEARN` / `False` | No | Enable Prompt-Time Reinforcement Learning. Includes: post-run self-analysis, pre-run lesson injection, LLM-as-Judge verification (Feature G), and thinking token preservation (Feature H). Implies `trajectory=True` |
| `preview_chars` | `int \| None` | env `CLAW_PREVIEW_CHARS` / `120` | No | Max chars for tool-output previews in trajectory logs |
| `response_chars` | `int \| None` | env `CLAW_RESPONSE_CHARS` / `500` | No | Max chars for LLM response text in trajectory records |

> **Priority:** Explicit parameter > environment variable > default value. You never need to set both.

### Hooks & Access Control

```python
agent = create_claw_agent("gemini-3-flash", instruction="Code reviewer")

# Block dangerous tools at runtime
agent.block_tools("execute", "write_file")

# Or whitelist only safe tools
agent.allow_only_tools("read_file", "ls", "grep", "glob")

# Inject context into every LLM call
agent.inject_context("Always respond in Spanish")

# Limit tool output size
agent.truncate_output(3000)
```

**Advanced — raw hooks:**

```python
agent.before_llm = lambda messages: messages        # modify messages before LLM
agent.before_tool = lambda name, args: True          # return False to block
agent.after_tool = lambda name, args, result: result # modify tool results
```

### Instance Methods

| Method | Description |
|:---|:---|
| `await agent.invoke(task, max_iterations=None)` | Run the agent on a task. Returns `AgentState` with `.result`, `.status` (`"running" \| "done" \| "error" \| "max_iterations"`), `.iterations`, `.tool_calls` |
| `await agent.compare(task, n_samples=3, max_iterations=None, on_event=None)` | Run the task N times and return the best result based on objective scoring (GRPO-inspired). Returns `{"best_result", "best_score", "best_index", "all_scores", "comparison_method", "n_samples"}` |
| `agent.block_tools(*names)` | Block specific tools at runtime |
| `agent.allow_only_tools(*names)` | Whitelist-only mode — all other tools blocked |
| `agent.inject_context(text)` | Inject extra context into every LLM call |
| `agent.truncate_output(max_chars)` | Limit tool output size |

---

## Auto-Discovery

The agent factory automatically discovers project files:

| What | Default locations checked |
|:---|:---|
| **Memory** | `./AGENTS.md`, `./CLAWAGENTS.md` |
| **Skills** | `./skills`, `./.skills`, `./skill`, `./.skill`, `./Skills`. Bundled skills are auto-included based on eligibility (see below). |

### Bundled Skills

| Skill | Purpose | Prerequisite | Auto-enabled? |
|:---|:---|:---|:---:|
| **[OpenViking](https://github.com/volcengine/OpenViking)** | Tiered context retrieval (L0/L1/L2) over repos and docs | `pip install openviking` + running `openviking-server` | Only when `ov` CLI is on PATH |

**OpenViking** is a structured context database. Use `ov add-resource` to ingest repos or docs, then `ov find` for semantic search. Results use a virtual filesystem (`viking://`) with **L0** (abstract), **L1** (overview), **L2** (full content).

**OpenViking prerequisites:**
1. Install: `pip install openviking --upgrade`
2. Configure: create `~/.openviking/ov.conf` with embedding model and VLM settings (see [OpenViking docs](https://github.com/volcengine/OpenViking))
3. Start server: `openviking-server`
4. Put `ov` on your PATH (often `~/Library/Python/3.11/bin` on macOS)

> **Note:** ByteRover was removed — its default LLM provider is a cloud internet service.

Override with explicit paths:
```python
agent = create_claw_agent(
    "gpt-5",
    memory="./docs/AGENTS.md",
    skills=["./my-skills", "./shared-skills"]
)
```

---

## Memory & Context Management

### Project Memory
Loads `AGENTS.md` (and `CLAWAGENTS.md`) from the working directory and injects their content into every LLM call. Use for project-level context and conventions.

### Auto-Compaction
When the conversation exceeds **75% of `CONTEXT_WINDOW`**:
1. Full history **offloaded** to `.clawagents/history/compacted_<ts>_<N>msgs.json`
2. Older messages **summarized** into a single placeholder message tagged `[System — Compacted History]`
3. Last 20 messages kept intact

This provides **unlimited conversation length** with full audit trail preservation.

---

## Gateway Server

Launch an HTTP server with one line:

```python
from clawagents.gateway import start_gateway

start_gateway(port=3000)            # binds to 127.0.0.1 by default (loopback only)
start_gateway(port=3000, host="0.0.0.0")  # explicit LAN exposure — REQUIRES auth
```

### Bind & auth

The gateway binds to **`127.0.0.1` (loopback)** by default in v6.2+. To expose
it on the LAN, pass `host="0.0.0.0"` or set `GATEWAY_HOST=0.0.0.0` (the env
var wins over the `host=` argument), and *also* set `GATEWAY_API_KEY=<secret>`
to require Bearer auth. Starting on a non-loopback address without an API key
prints a loud warning at startup — anyone on the network can otherwise hit
`/chat`, `/chat/stream`, and `/ws`.

### Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/chat` | POST | Synchronous agent invocation |
| `/chat/stream` | POST | SSE streaming (events: `queued`, `started`, `agent`, `done`, `error`) |
| `/ws` | WS | WebSocket session (bidirectional, same Bearer-auth as `/chat`) |
| `/queue` | GET | Queue status for all lanes |
| `/health` | GET | Health check |

### Lane-Based Concurrency

4 lanes with configurable `max_concurrent` per lane:
- `main` — primary user requests
- `cron` — scheduled tasks
- `subagent` — sub-agent delegation
- `nested` — nested sub-agent calls

---

## Trust Boundaries & Hardening

A few surfaces are deliberately powerful — they exist for trusted operators,
and you should treat them as such when running ClawAgents in environments with
untrusted prompts or LAN exposure:

- **`execute` tool** — runs arbitrary commands inside the configured sandbox.
  Pair with the `LocalBackend(cwd=...)` constraint and ideally a containerized
  runtime; the tool's blocklist is a guardrail, not a security boundary.
- **External hooks** (`CLAW_FEATURE_EXTERNAL_HOOKS=1`, `CLAW_HOOK_*`) execute
  shell commands defined in your env or `.clawagents/hooks.json`. Anyone who
  controls those configs has code execution. Treat hooks as **trusted-only**.
- **`web_fetch` tool** — refuses loopback / RFC1918 / link-local / multicast
  IPs by default to block SSRF. Set `CLAWAGENTS_WEB_ALLOW_PRIVATE=1` only in
  trusted dev environments.
- **`web_search` tool** — calls fixed host `api.tavily.com` with `TAVILY_API_KEY`.
- **Gateway** — defaults to loopback (`127.0.0.1`) bind. Set `GATEWAY_API_KEY`
  if you bind to `0.0.0.0`.

---

## Sandbox Backends

ClawAgents uses a **pluggable sandbox protocol** for all file and shell operations:

```python
from clawagents.sandbox import InMemoryBackend, LocalBackend

# Production: real filesystem
agent = create_claw_agent("gpt-5", sandbox=LocalBackend())

# Testing: pure in-memory VFS
mem = InMemoryBackend()
mem.seed({"src/main.py": "print('hello')", "README.md": "# My Project"})
agent = create_claw_agent("gpt-5", sandbox=mem)
snapshot = mem.snapshot()  # deterministic state capture
```

---

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
| `OPENAI_MODEL` | `gpt-5-nano` | No | Model name, Azure deployment name, or local model ID (e.g. `llama3.1`) |
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

## Testing

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest -q

# Hermetic runner — exactly the environment CI uses (pinned xdist=4,
# TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0, credentials scrubbed)
bash scripts/run_tests.sh

# Run benchmarks (requires API keys)
python -m pytest tests/ -v -m benchmark

# Static type check
python -m mypy
```

The test suite includes regression tests for every Hermes-inspired pattern
landed in the v6.5/v6.6 line — `tests/test_subagent_depth.py`,
`tests/test_compaction_hardened.py`, `tests/test_mcp_env_scrub.py`,
`tests/test_paths.py`, `tests/test_redact.py`, `tests/test_steer.py`,
`tests/test_transport.py`, `tests/test_commands.py`, `tests/test_aux_models.py`,
`tests/test_background.py` — and the four v6.6 feature suites
(`tests/test_browser.py`, `tests/test_cron.py`, `tests/test_acp.py`,
`tests/test_rl.py`). Current v6.8.1 coverage adds `tests/test_prompts.py` for
shared prompt assembly and legacy hook injection, while v6.8.0 added
`tests/test_openharness_inspired_surfaces.py` for dry-run previews, provider
profiles, structured permission decisions, background task tools, plugin
metadata compatibility loading, and MCP auth/reconnect helpers. v6.7.1 added
`tests/test_infra_improvements.py` for compact tool discovery, structured
tool failure observations, recovery hints, and infrastructure behavior,
alongside the v6.3/v6.4 regression sets and the broad `tests/simulated_test.py`
parity sweep.

---

## Changelog

Full history: [docs/CHANGELOG.md](docs/CHANGELOG.md).
