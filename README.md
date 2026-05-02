<p align="center">
  <h1 align="center">🦞 ClawAgents</h1>
  <p align="center"><strong>A lean, full-stack agentic AI framework — ~2,500 LOC</strong></p>
  <p align="center">
    <img src="https://img.shields.io/badge/version-6.8.1-blue" alt="Version">
    <img src="https://img.shields.io/badge/python-≥3.10-green" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
    <img src="https://img.shields.io/badge/LOC-~2500-purple" alt="LOC">
  </p>
</p>

---

ClawAgents is a **production-ready agentic framework** that gives LLMs the ability to read, write, and execute code — with built-in planning, memory, sandboxing, and a gateway server. It supports **OpenAI GPT-5**, **Google Gemini**, and **Anthropic Claude** out of the box, with a pluggable provider architecture for any LLM.

Built by extracting and unifying the best architectural patterns from [OpenClaw](https://github.com/anthropics/openclaw) (~5,800 files) and [DeepAgents](https://github.com/langchain-ai/deepagents) (~1,400 LOC core), ClawAgents delivers **the same power at a fraction of the complexity**.



## Installation

```bash
pip install clawagents              # Core (OpenAI only)
pip install clawagents[gemini]      # + Google Gemini support
pip install clawagents[anthropic]   # + Anthropic Claude support
pip install clawagents[all]         # All providers + tiktoken
```

> **Version 6.8.1** — Prompt architecture and packaged-surface polish (May 2026). Adds shared prompt assembly helpers, preserves cache-boundary behavior across memory/skill injection, refreshes the OpenHarness comparison, and keeps the v6.8.0 operational surfaces plus v6.7.1 compact tool-discovery recovery and v6.7.0 security hardening. See [Changelog](#changelog).

### New In v6.8.1

```bash
clawagents --dry-run --profile ollama --task "inspect this repo"
```

- **Shared prompt assembly** centralizes system prompt construction, lesson preambles, cache-boundary placement, and dynamic memory/skill injection in `clawagents.prompts`.
- **Legacy hook compatibility** keeps dict-shaped `before_llm` messages working while exposing reusable prompt helpers for downstream integrations.
- **OpenHarness comparison** adds [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness) as a peer in the feature matrix with conservative full/partial markers.
- **Dry-run previews** report provider resolution, auth readiness, inspectable tools, likely matching tools, and next actions without calling an LLM or executing tools.
- **Provider profiles** give stable aliases for common backends while still letting explicit `create_claw_agent()` parameters override profile values.
- **Background task tools** expose long-running command management (`task_create`, `task_status`, `task_output`, `task_stop`, `task_list`) through the normal tool registry.
- **Plugin compatibility loading** reads `plugin.json` / `.claude-plugin/plugin.json` metadata, skills, commands, hooks, and MCP server declarations without executing plugin code.
- **MCP auth refresh** lets agents update MCP server auth material and reconnect configured servers deliberately.

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
2. **`.env` file values** — loaded with `override=True`, so they take precedence over any pre-existing shell env vars. This is intentional: it prevents the "stale `OPENAI_API_KEY` exported from `~/.zshrc` silently shadows the fresh key in `.env`" bug class.
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
| [`06_bedrock.py`](examples/06_bedrock.py) | AWS Bedrock (via gateway) |
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

### 9. AWS Bedrock (via OpenAI-compatible gateway)

Use [Bedrock Access Gateway](https://github.com/aws-samples/bedrock-access-gateway) or [LiteLLM proxy](https://docs.litellm.ai/docs/proxy/quick_start) to expose Bedrock models as an OpenAI-compatible API:

```python
agent = create_claw_agent(
    "anthropic.claude-3-sonnet-20240229-v1:0",
    base_url="http://localhost:8080/v1",
    api_key="bedrock",           # gateway handles AWS auth
)
```

Or via `.env`:

```env
OPENAI_API_KEY=bedrock
OPENAI_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
OPENAI_BASE_URL=http://localhost:8080/v1
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

> Compares **ClawAgents v6.8.1** against four peer agent frameworks: **Hermes Agent**
> ([metaspartan/hermes-agent](https://github.com/metaspartan/hermes-agent)), **DeepAgents**
> ([langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)), and **OpenClaw**, plus **OpenHarness** ([HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness)).
> The v6.8.1 prompt/packaging polish, v6.8.0 OpenHarness-inspired operational
> surfaces, v6.7.1 compact tool-discovery recovery, v6.7.0 security fixes, and
> v6.5/v6.6 Hermes-parity areas now ship together in the current release —
> every row in the ClawAgents column is ✅. `◐` means partial or comparable
> coverage rather than exact feature parity.

| Feature | ClawAgents v6.8.1 | Hermes Agent | DeepAgents | OpenClaw | OpenHarness |
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
│   ├── agent_loop.py      # Core ReAct loop, HookResult, context management (v6.0)
│   ├── coordinator.py     # Coordinator/swarm orchestration mode
│   └── forked_agent.py    # Background forked agent pattern
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
| `write_todos` | Plan tasks as a checklist |
| `tool_program` | Bounded read-only multi-tool sequence with `${step.output}` substitutions |
| `update_todo` | Mark plan items complete |
| `task` | Delegate to a sub-agent with isolated context |
| `ask_user` | Interactive stdin-based user input |
| `use_skill` | Load a skill's instructions (when skills exist) |

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

Skills are **reusable instruction sets** that teach the agent domain-specific knowledge — without polluting the system prompt. They use a progressive disclosure pattern: the agent loads skill instructions on demand via the `use_skill` tool.

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
| `skills` | `str \| list \| None` | auto-discover | No | Skill directories to load. Default: checks `./skills`, `./.skills`, `./skill`, `./.skill`, `./Skills`. Bundled skills (ByteRover, OpenViking) are always included when eligible. |
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

ClawAgents ships with two complementary bundled skills that work together:

| Skill | Purpose | Prerequisite | Auto-enabled? |
|:---|:---|:---|:---:|
| **[ByteRover](https://clawhub.ai/byteroverinc/byterover)** | **Write** decisions, patterns, and rules to local Markdown files | Node/npx (`brv` runs via `npx byterover-cli`) | Always |
| **[OpenViking](https://github.com/volcengine/OpenViking)** | **Read** context from repos, docs, and large knowledge bases with tiered L0/L1/L2 loading | `pip install openviking` + running `openviking-server` | Only when `ov` CLI is on PATH |

**How they complement each other:**

- **ByteRover** is a fast, serverless notebook for the agent. Use `brv curate` to persist decisions ("We chose Postgres for ACID compliance") and `brv query` to recall them. No infrastructure needed — context is stored as Markdown in `.brv/context-tree/`.
- **OpenViking** is a structured context database. Use `ov add-resource` to ingest entire repos or doc sites, then `ov find` for semantic search across all indexed content. Results are organized in a virtual filesystem (`viking://`) with three tiers: **L0** (abstract, ~100 tokens), **L1** (overview, ~2k tokens), **L2** (full content) — the agent loads only what it needs, saving tokens.

**Typical workflow:** OpenViking **retrieves** context → agent works on the task → ByteRover **curates** the decisions made.

**OpenViking prerequisites:**
1. Install: `pip install openviking --upgrade`
2. Configure: create `~/.openviking/ov.conf` with embedding model and VLM settings (see [OpenViking docs](https://github.com/volcengine/OpenViking))
3. Start server: `openviking-server`
4. The `ov` CLI must be on your PATH — the skill auto-enables when detected |

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

**Provider & Model** — set at least one API key (or `OPENAI_BASE_URL` for local models)

| Variable | Default | Required? | Description |
|:---|:---|:---:|:---|
| `PROVIDER` | auto-detect | No | Hint: `"openai"`, `"gemini"`, or `"anthropic"`. Auto-detected from which API key is set |
| `OPENAI_API_KEY` | — | **Yes** *(for OpenAI/Azure)* | OpenAI or Azure API key. **Not needed for local models** — when `OPENAI_BASE_URL` is set, a placeholder is used automatically |
| `OPENAI_MODEL` | `gpt-5-nano` | No | Model name, Azure deployment name, or local model ID (e.g. `llama3.1`) |
| `OPENAI_BASE_URL` | *(unset)* | No | Custom endpoint for OpenAI-compatible APIs: Azure, Bedrock gateway, Ollama, vLLM, LM Studio. Omit to use `api.openai.com` |
| `OPENAI_API_VERSION` | *(unset)* | No | **Azure only.** API version string (e.g. `2024-12-01-preview`). Ignored by all other providers |
| `GEMINI_API_KEY` | — | **Yes** *(for Gemini)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | No | Gemini model name |
| `ANTHROPIC_API_KEY` | — | **Yes** *(for Anthropic)* | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | No | Anthropic model name (e.g. `claude-sonnet-4-5`, `claude-opus-4`) |

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

### v6.8.1 — Prompt architecture and release packaging polish (May 2026)

Patch release focused on keeping the Python and TypeScript packages aligned for
installed users after the OpenHarness-inspired operational surface work.

- **Prompt assembly module** — `clawagents.prompts` now owns system prompt
  construction, lesson preambles, `__CACHE_BOUNDARY__` placement, and dynamic
  memory/skill prompt injection.
- **Hook compatibility** — prompt injection remains compatible with the legacy
  dict-shaped messages used by older `before_llm` integrations.
- **OpenHarness comparison** — the feature matrix now includes
  [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness) with conservative
  full/partial markers.

Release verification: **Python 851 passed, 3 skipped** plus bytecode
compilation; TypeScript sibling: **526 passed, 4 skipped**, `tsc --noEmit`, and
build.

### v6.8.0 — OpenHarness-inspired operational surfaces (May 2026)

Minor release focused on making ClawAgents easier to inspect, configure,
recover, and integrate without changing the core agent loop contract.

- **Static readiness previews** — `clawagents --dry-run --profile <name> --task
  "<prompt>"` reports resolved provider settings, auth readiness, inspectable
  tools, likely matching tools, and next actions without calling a model or
  executing tools.
- **Named provider profiles** — built-in `openai`, `gemini`, `anthropic`, and
  `ollama` profiles plus project/user profile files give stable provider
  aliases. Explicit factory parameters still take precedence.
- **Structured permission decisions** — permission evaluation now returns a
  reusable decision object with allow/confirmation/reason fields and feeds the
  registry hard-block path for plan-mode and sensitive-path decisions.
- **Background task tools** — the registry can expose task create/status/output
  /stop/list tools backed by the existing background job manager, so long-running
  work can be tracked instead of blocking an agent turn.
- **Plugin compatibility loader** — metadata-only loading for `plugin.json` and
  `.claude-plugin/plugin.json` reads plugin manifests, markdown skills/commands,
  hooks, and MCP server declarations without executing arbitrary plugin code.
- **MCP auth/reconnect helper** — MCP manager configs can be updated with new
  environment/header auth material and reconnected deliberately.

Release verification: **Python 844 passed, 3 skipped** plus bytecode
compilation and dry-run smoke; TypeScript sibling: **520 passed, 4 skipped**,
`tsc --noEmit`, build, and matching dry-run smoke.

### v6.7.1 — Tool discovery and compact-agent recovery (April 2026)

Patch release focused on generalizable low-latency tool use for compact
models. `tool_discover` is registered by default so agents can inspect the
available tool universe before committing to a call, and lookup now searches
tool names, descriptions, and keyword metadata. That makes discovery robust
when a model remembers the action it needs but not the exact tool name.

Native-tool failures now keep useful output in the observation stream instead
of reducing everything to a generic error. The built-in `execute` tool returns
structured JSON for nonzero exits (`command`, `exit_code`, `stdout`,
`stderr`, `output`, `timed_out`), and repeated identical `execute` failures
include a recovery hint that nudges the agent to inspect the captured output
or change command strategy.

Planning/todo guidance was also tightened so quick read-only or single-step
tasks do not pay unnecessary planning overhead, while multi-step repair tasks
still get explicit progress tracking. Focused release verification covers the
infra-improvement regression tests and bytecode compilation for Python, plus
TypeScript typecheck and matching infra-improvement tests.

### v6.7.0 — Security hardening across validator, web_fetch, redact, sandbox (April 2026)

Minor release. Adversarial probing of the v6.6.4 surfaces uncovered a
cluster of bypasses; this release closes them. Test totals after this
release: **Python 835 passed, 3 skipped**; **TypeScript 511 passed,
4 skipped** plus parity checks; `tsc --noEmit` clean. **49 new
regression tests** ride alongside the fixes (44 Python, 5 TypeScript).

**Bash validator hardening** — `validate_bash` now walks every shell
clause, including the contents of `(...)`, `$(...)`, backticks, and
`bash -c '<cmd>'`/`sh -c '<cmd>'` wrappers; the strictest verdict
across all clauses wins. The previous head-only inspection meant
`ls && rm -rf /var/log`, `(rm -rf /)`, `echo $(rm -rf /)`, and
`bash -c 'rm -rf /'` all silently passed. Additional shapes now
`BLOCK`: `rm -rf "$HOME"` / `rm -rf $HOME/x` and any `rm` of a system
directory (`/etc`, `/var`, `/usr`, `/home`, …); `tee /dev/sda` and
`tee /etc/passwd` / `tee -a /etc/sudoers`; quoted block-device
redirects (`>'/dev/sda'`); FD-prefixed redirects (`1>/dev/sda`);
`find -exec sh -c '…'` and `find -execdir`; `chmod -R 777 /`;
`sed --in-place` (long form, previously unrecognised). Null bytes
and unprintable control characters in any command are also `BLOCK`
(closes the C-string truncation evasion).

**Web fetch SSRF — DNS-rebinding TOCTOU eliminated** — `web_fetch` now
resolves the host once per hop and connects to the validated IP
directly, sending the original hostname via the `Host` header and SNI.
A controlled DNS server can no longer return a public address to the
validator and a private one (loopback, `169.254.169.254` / cloud
metadata) to the actual fetch. Body reads are bounded at 4 MiB and
truncated streamingly so a hostile server can't OOM the agent. Each
redirect hop gets its own timeout. `Location` headers that downgrade
HTTPS → HTTP across a redirect are refused.

**Obfuscation detector — host-suffix bypass closed** — the curl-pipe-
shell installer allowlist used `\b`-anchored regexes, but `.` is a
non-word character so `brew\.sh\b` matched `brew.sh.evil.com`.
Allowlist is now keyed on parsed hostname (with required path prefix
for `raw.githubusercontent.com`), not regex.

**`edit_file` empty-target corruption** — `target=""` plus
`replace_all=true` previously inserted the replacement between every
character of the file, silently corrupting it. Now refused.

**Redaction coverage** — `redact()` now scrubs PEM private-key blocks
(any `-----BEGIN […] PRIVATE KEY-----` / `END` block), `Authorization:
Bearer <token>` / `Authorization: Basic …` headers, AWS *secret* access
keys (the previous regex covered only the access-key ID), URL
basic-auth credentials (`https://user:pass@host`), and shorter
generic-secret values. The Docker sandbox env-name policy now reuses
`is_secret_name()` from `redact.py` plus a small extras regex covering
vendor-prefixed shapes (`GITHUB_PAT`, `STRIPE_SK_LIVE`,
`DATABASE_URL`, `DSN`); the previous end-anchored regex missed
`AWS_SECRET_ACCESS_KEY`, `GITHUB_PAT`, `DATABASE_PASSWORD_PROD`, etc.
and forwarded them into containers via `-e`.

**Subprocess timeouts no longer orphan children** — the local sandbox
now starts each shell in a new session and `SIGKILL`s the whole
process group on timeout, so long-running grandchildren of `sh -c`
don't outlive the parent.

**Concurrency** — `RunContext.iteration_budget` lazy-init is serialised
under an `asyncio.Lock`; sub-agents sharing a context can no longer
clobber each other's budget. Callsite is `await
run_context.ensure_iteration_budget(size)`.

**Other quality fixes** — `RetryPolicy.shouldRetry` now correctly
allows `maxRetries=N` to perform `N` retries (was off-by-one); `jitter`
is clamped to `[0, 1]` to prevent zero-delay retry storms; the MCP
manager tracks connected servers so a partial-failure `start()`
doesn't double-register tools on retry, and shutdown errors are
aggregated into a thrown `Error` instead of a span no caller observes;
`compressMessagesSafe` no longer produces two consecutive same-role
messages when the head is empty (Anthropic rejects that). The
overbroad `"curl http"` / `"wget http"` legacy substring (which also
matched `https://` because `https` starts with `http`) is removed —
the bash validator's NETWORK classification now applies cleanly.

### v6.6.4 — Keyword discovery and infrastructure parity (April 2026)

Patch release for the v6.6 line. Test totals after this release:
**Python 786 passed, 3 skipped**; **TypeScript 509 passed, 4 skipped**
plus **49 parity checks**; `tsc --noEmit` clean.

- **Keyword-backed compact discovery** — tools can now declare explicit
  keyword aliases, `tool_discover` searches names, descriptions, and those
  aliases, and `tool_describe`/registry inspection expose the metadata so
  compact tool universes stay useful even when the model uses a near-synonym.
- **Bounded tool profiles** — catalog helpers can publish smaller tool views
  for focused agents while preserving the full registry for callers that need
  it.
- **Infrastructure parity** — Docker sandbox support, resumable `RunResult`
  metadata, SQLite result caching for safe cacheable tools, explorer helpers,
  gym-style eval aliases, and next-state trajectory export helpers now ship in
  both the Python and TypeScript packages.
- **Cache safety defaults** — read/search-style filesystem outputs remain
  uncached by default to avoid persisting sensitive repository contents, while
  explicitly cacheable pure tools can reuse results across runs.

### v6.6.3 — Efficiency and release hardening (April 2026)

Patch release for the v6.6 line. Test totals after this release:
**Python 778 passed, 3 skipped**; **TypeScript 497 passed, 4 skipped**
plus **49 parity checks**; `tsc --noEmit` clean. Real `.env` smoke tests
passed for Gemini and OpenAI, including read-only `read_file` tool use and
`task` subagent delegation in both ports.

- **Non-blocking local filesystem backend** — async `LocalBackend` file,
  directory, and stat operations now offload synchronous pathlib work with
  `asyncio.to_thread()`, so parallel-safe tool calls can yield the event loop
  instead of serializing on local disk I/O.
- **Append-only run summaries** — trajectory finalization now appends one
  JSONL row to `runs.jsonl` instead of reading and atomically rewriting the
  full historical log for every run.
- **Bounded session preload** — agent session hydration now passes a default
  preload limit of 200 prior messages to session backends, with
  `session_preload_limit=None` available when callers explicitly want the
  full persisted history.
- **Cross-package efficiency parity** — the TypeScript sibling now caps large
  in-process diffs and single-file grep matches, and its session preload uses
  the same bounded default.

### v6.6.1 — Approval, proxy, ACP, and release hardening (April 2026)

Patch/security release for the v6.6 line. Test totals after this release:
**Python 769 passed, 3 skipped**; **TypeScript 489 passed, 4 skipped**;
mypy clean, `tsc --noEmit` clean.

- **Parallel tool approvals** — batched/native tool execution now checks
  `RunContext` approval state before dispatch, so sticky denials and pending
  approvals cannot be bypassed by a multi-tool response.
- **Credential proxy SDK mode** — the sandbox credential proxy now forwards
  provider SDK path requests such as `/v1/models`, restricts upstream origins,
  and refuses redirects that would leak injected credentials across origins or
  protocol downgrades.
- **Lazy tool schema parity** — factory-published schemas now match the
  implementation arguments for `edit_file`, `grep`, and `tree`
  (`target` / `replacement`, `glob_filter`, `max_depth`).
- **ACP default runner parity** — `AcpServer.serve(create_claw_agent(...))`
  now accepts real ClawAgents instances via `invoke()` and normalizes
  `AgentState.result` into protocol messages.
- **Hermetic runner override** — `CLAW_TEST_WORKERS` is preserved before the
  runner scrubs credentials and other `CLAW_*` variables.

### v6.6.0 — Hermes-parity feature release: browser tools, scheduler, ACP, RL hooks (April 2026)

Feature release. Four big Hermes-side capabilities now ship on both
Python and TypeScript ports, each behind an optional dependency so the
core install stays slim. Test totals after this release: **Python 762
passed**, **TypeScript 478 passed**, mypy clean, `tsc --noEmit` clean.

- **🌐 Browser tools** (`clawagents.browser`) — Playwright-driven browser
  control for agents that need to read or interact with the live web.
  `BrowserSession` exposes a stable async API (`navigate`, `snapshot`,
  `click`, `type_text`, `fill_form`, `scroll`, `wait_for_selector`,
  `screenshot`, `close`) over a pluggable provider (`LocalProvider` for
  Playwright; `BrowserbaseProviderStub` / `BrowserUseProviderStub` ready
  to be filled in for cloud back-ends). `create_browser_tools()` adapts
  the session into ClawAgents tools with per-action accessibility-tree
  snapshots so the model sees the page through the same axtree Hermes
  uses. Playwright is an optional peer (`pip install clawagents[browser]`);
  importing the module without it works fine — only `session.start()`
  raises `MissingPlaywrightError`. `MAX_NODES = 800`-cap on snapshots,
  navigation allow-/deny-lists, and a `renderSnapshot()` helper for
  prompt-friendly trees.
- **⏰ Cron / scheduled jobs** (`clawagents.cron`) — minimal but
  production-shaped scheduler for agent-driven cron, one-shots, and
  intervals. `parse_schedule()` handles `every 30s`, `at 2026-04-23T18:00`,
  and 5-field cron expressions; cron support uses the optional
  `croniter` package and degrades cleanly when missing. `Scheduler`
  provides `create_job` / `get_job` / `pause_job` / `resume_job` /
  `trigger_job` / `remove_job` plus a `run_due` driver that emits
  `JobNotifier` events (`job_started`, `job_finished`, `job_failed`,
  `job_skipped`). Job store is plain JSON on disk; runners can be any
  callable, so users can wire it to `agent.invoke(...)` or shell.
  Mirrors Hermes' "agents as a workflow engine" pattern.
- **🔌 ACP adapter** (`clawagents.acp`) — bridges any ClawAgents agent
  to **Zed's Agent Client Protocol** over stdio so editors / IDEs that
  speak ACP can drive a ClawAgents agent the same way they drive
  Claude Code or Codex. `AcpServer.serve()` registers an
  `AgentSessionFactory`, accepts ACP `initialize` / `newSession` /
  `prompt` / `cancel` requests, and translates ClawAgents stream events
  into ACP `session/update` messages (`agent_message_chunk`,
  `agent_thought_chunk`, `tool_call.start` / `.complete`, `permission`).
  Per-session `AgentSession` wraps prompt history, permission
  callbacks, and `StopReason` propagation. The optional
  `agent-client-protocol` package is loaded lazily — importing
  `clawagents.acp` works without it; only `serve()` raises
  `MissingAcpDependencyError`. Round-trip tested against Hermes'
  reference message shape.
- **🎯 RL fine-tuning hooks** (`clawagents.rl`) — capture live agent
  runs as training-ready trajectories and export them to **TRL**,
  **Atropos**, **SLIME**, or generic JSONL. `RLRecorder` plugs into
  `agent.on_event` and assembles a `Trajectory` (system / user /
  assistant + `tool_calls` / tool messages) in correct ChatML order,
  with config knobs for `max_tool_result_chars`, `redact_tool_args`,
  and `capture_system_prompt`. Pluggable `RewardScorer`s (`Contains`,
  `ExactMatch`, `Regex`, `LengthPenalty`, `Composite`) attach a scalar
  reward + per-component breakdown. Export helpers: `export_jsonl`,
  `to_chatml`, `to_trl_sft`, `to_trl_dpo`, `to_atropos_rollout`. Lazy
  `TrlAdapter` and `AtroposAdapter` only import `trl` / `atropos` when
  the user actually drives a trainer or rollout collector — install
  hints surface as `MissingRLDependencyError`.

**Backwards compatibility:** All four features are additive and
opt-in. Importing the new submodules has no side effects; nothing in
the core `create_claw_agent()` / `agent.invoke()` path changed. The
optional peers (`playwright`, `croniter`, `agent-client-protocol`,
`trl`, `atropos`) are only required at the moment you actually
`session.start()` / parse a cron expression / `serve()` over ACP /
build a TRL dataset.

### v6.5.0 — Hermes-inspired hardening: depth, isolation, heartbeats, path-scoped parallelism (April 2026)

Architecture/correctness release. Ten patterns ported from the Hermes agent are
now live on **both** Python and TypeScript ports — every change comes with
regression tests on both. Test totals after this release: **Python 662 passed**,
**TypeScript 370 passed**, mypy clean, `tsc --noEmit` clean.

**Tier 1 — runtime safety & isolation:**

- **🪜 Subagent depth limits** (`graph/coordinator`, `tools/subagent`, `graph/forked_agent`) — `RunContext` now tracks `subagent_depth`. The `task` tool refuses to delegate when the parent is already at `depth >= 2`, returning a structured error instead of silently spawning a third tier. Forks inherit the depth counter; the cap mirrors Hermes' "no recursive delegation" rule and prevents exponential subagent fan-out.
- **🧠 Memory-isolated forks/subagents** (`graph/forked_agent`, `memory/loader`) — both `forked_agent` and the built-in `task` tool now accept `skip_memory=True` (default for forks). When set, memory loaders are bypassed so a sandboxed fork cannot see the parent's `AGENTS.md`/skills/notes — closing a previously-silent context-leak path. Forks also get their own `IterationBudget` so a runaway research fork cannot starve the parent's remaining turns.
- **💓 Activity heartbeats** (`session/heartbeat`, `gateway/server`, `graph/agent_loop`) — long-running tool calls now emit periodic `tool_heartbeat` events (`tool_name`, `call_id`, `elapsed_s`) every ~20s through `run_with_heartbeat`. Gateway clients can use these to keep WebSocket channels alive and surface progress, eliminating false timeouts on slow shell/web/sandbox calls. Best-effort: emitter exceptions are swallowed so they never mask the real result.
- **⏱️ Per-agent IterationBudget** (`iteration_budget`, `graph/agent_loop`, `graph/forked_agent`) — replaces the implicit `max_turns` counter with an explicit `IterationBudget` object that lives on `RunContext`. Subagents and forks each get their own budget sized from `delegation.max_iterations` (default `DEFAULT_DELEGATION_MAX_ITERATIONS`), so one chatty fork can't drain the parent's turn pool. Surfaces the same `consume()`/`refund()`/`exhausted` shape Hermes uses, making it easy to tee budgets across recursive delegation.
- **🌿 Path-scoped parallel tool execution** (`tools/registry`) — `execute_tools_parallel` no longer fans out blindly. Tools are tagged `parallel_safe` (read-only by default for `read_file` / `list_dir` / `glob` / `search_files` / `grep` / `web_fetch`) with optional `path_scoped_arg` ("path", "url", …); the registry partitions calls into ordered batches so reads run concurrently while any writer or path-scope collision serialises behind them. Capped at `MAX_PARALLEL_TOOL_WORKERS = 8` to keep file-handle pressure bounded. Mirrors Hermes' parallel-read / serial-write contract.

**Tier 2 — extensibility & cache-discipline:**

- **🔌 Plugin hook expansion** (`plugins`) — new top-level `Plugin` + `PluginManager` (`from clawagents import Plugin, PluginManager`). Plugins compose three hook families with priority-based ordering: `pre_tool` (first-deny veto / args-rewrite, alias `before_tool`), `transform_tool_result` (sequential post-execution rewrite, alias `after_tool`), and `before_llm` (prompt-massage). Replaces the previous "single hook wins" model with a deterministic chain that's easy to unit-test.
- **📁 `display_clawagents_home()`** (`paths`) — runtime helper that resolves the package install root and rewrites it to a placeholder (`<clawagents-home>`) for tool descriptions, error messages, and traces. Makes prompt cache hits stable across user homes / dev / CI by stripping absolute paths from anything that ends up in the LLM context window.
- **🧊 Prompt-cache-aware `CommandDef`** (`commands`) — slash-command definitions now carry an explicit `cache_impact` (`"none" | "soft_break" | "hard_break"`) and parse a `--now` flag (`/skills install foo --now`) so users can opt into immediate state mutation; default is `cache_impact="none"`, `--now` upgrades to `"hard_break"` and forces a fresh prompt build. Mirrors Hermes' "deferred by default to preserve prompt cache" contract.
- **📜 Prompt-cache policy** (`AGENTS.md`) — new top-level rule documents the cache invariants (stable system prompt prefix, no per-turn timestamps in cached blocks, deferred slash-command state mutations, `display_clawagents_home()` for paths) so contributors keep the cache hit rate above the 80%+ Hermes target.

**Tier 3 — testing infrastructure:**

- **🧪 Hermetic test runner + pinned xdist** (`scripts/run_tests.sh`, `pyproject.toml`) — canonical CI-mirrored runner that pins `pytest-xdist` to 4 workers (override via `CLAW_TEST_WORKERS`), forces `TZ=UTC` / `LANG=C.UTF-8` / `PYTHONHASHSEED=0`, and scrubs credentials plus non-runner `CLAW_*` env vars before pytest sees them. Gives every contributor the exact environment CI runs in, eliminating local-vs-CI flakes. Mirrored by `clawagents/scripts/run_tests.sh` for the TypeScript port (`node:test --test-concurrency=4` via `tsx`).

**Backwards compatibility:** All 10 features are additive. Existing
`create_claw_agent()` / `agent.invoke()` call sites keep working; the new
machinery activates automatically (depth tracking, heartbeats, parallel-safe
tagging) or via opt-in (`Plugin`, `--now`, `skip_memory`, `IterationBudget`).

### v6.4.1 — Public-API export polish (no behavior change)

Patch release. Surfaces `PromptHook` and `PromptHookVerdict` at the top-level
`clawagents` package (Python) and `clawagents` module (TypeScript) so users
can `from clawagents import PromptHook` instead of reaching into
`clawagents.hooks.prompt_hook`. No code-path changes; both ports remain at
516/226 passing.

### v6.4.0 — Tracing, MCP, Handoffs, Plan Mode (April 2026)

Big feature release. Nine new subsystems shipped on **both** Python and TypeScript ports — every change comes with regression tests on both. Test totals: **Python 516 passed**, **TypeScript 226 passed**, mypy clean, `tsc --noEmit` clean.

**Tier 1 — production interop & safety:**

- **🔭 Tracing infrastructure** (`clawagents.tracing`) — hierarchical Span model with 8 kinds (`agent` / `turn` / `generation` / `tool` / `handoff` / `guardrail` / `subagent` / `custom`), pluggable `TracingProcessor` + `TracingExporter` ABCs, batched `BatchTraceProcessor` with background flush, ready-made `JsonlSpanExporter` / `ConsoleSpanExporter` / `NoopSpanExporter`, and `agent_span` / `turn_span` / `generation_span` / `tool_span` / `handoff_span` context managers. Spans propagate via Python `contextvars` (TS: `AsyncLocalStorage`). Replaces flat trajectory JSONL — drop in OTLP/Langfuse/Logfire by writing one exporter.
- **🔌 MCP (Model Context Protocol) integration** (`clawagents.mcp`) — full client supporting **stdio**, **SSE**, and **Streamable-HTTP** transports. `MCPServerStdio` / `MCPServerSse` / `MCPServerStreamableHttp` follow openai-agents-python's shape; `MCPServerManager` lifecycles a list of servers; `MCPBridgedTool` adapts MCP tools into `ToolRegistry` so they coexist with native tools, hooks, and approval flows. SDK is an optional dep (`pip install clawagents[mcp]` / `npm install @modelcontextprotocol/sdk`). 11 lifecycle phases tracked per server with tracing spans.
- **🔁 Handoffs + `Agent.as_tool()`** — fills the previously-stub `on_handoff` lifecycle hook. `Handoff` dataclass + `handoff()` builder lets one agent transfer control to another (with optional `input_filter` for history trimming). `agent.as_tool(tool_name=…, tool_description=…)` is the complementary primitive: expose any agent as a callable tool to a parent agent. Built-in `remove_all_tools` filter strips tool calls/results before handoff. New `HandoffOccurredEvent` typed stream event.
- **🛡️ Exec safety v2** (`clawagents.permissions`, `clawagents.tools.{plan_mode,bash_validator,exec_obfuscation}`) — three security upgrades shipped together: (1) `PermissionMode` enum (`DEFAULT|PLAN|ACCEPT_EDITS|BYPASS`) on `RunContext` plus `enter_plan_mode` / `exit_plan_mode` built-in tools — write-class tools refuse in `PLAN`. (2) Bash semantic validator classifies every command (`READ_ONLY|WRITE|DESTRUCTIVE|NETWORK|PROCESS|PACKAGE|SYSTEM_ADMIN|UNKNOWN`) with a 47-row corpus and decision (`ALLOW|WARN|BLOCK`). (3) Command obfuscation detector catches base64/hex/printf decode-then-exec, `<(curl …)`, `curl … | sh`, `eval` decoders, and 9 other patterns — with an allowlist for known-safe installers (rustup, brew, nvm, …).
- **🪝 Hook event taxonomy expansion + `PromptHook`** — extended `RunHooks` with 8 additive events: `on_pre_compact`, `on_post_compact`, `on_subagent_start`, `on_subagent_end`, `on_user_prompt_submit`, `on_session_start`, `on_session_end`, `on_tool_failure`. New `PromptHook(prompt, model)` evaluates a guardrail using a small/cheap model with strict-JSON `{"ok":bool, "reason":str}` verdict — write a natural-language guardrail in `settings.json` instead of Python code. Fails open on timeout/error so a noisy hook can't deadlock the agent.

**Tier 2 — ergonomics & correctness:**

- **❓ AskUserQuestion structured tool** (`clawagents.tools.ask_user_question`) — structured HITL primitive: 1-3 multi-choice questions per call, 2-4 options each, implicit `"Other (please specify)"` always appended. Renders cleanly to Telegram inline buttons / WhatsApp quick-replies. Delegates rendering via `on_ask` callback.
- **⚙️ Settings hierarchy** (`clawagents.settings`) — `user → project → local → flag → policy` precedence, deep-merged. Policy layer (`/etc/clawagents/policy-settings.json`) ALWAYS wins, so even runtime flags can't override an MDM-style enforced rule. Repo root walks up looking for `.git`/`pyproject.toml`/`package.json`. `get_setting("hooks.before_tool")` for dotted-path access.
- **🖼️ Image sanitization** (`clawagents.media.images`) — clamps tool-result base64 image blocks to ≤1200px / ≤5MB before transcript ingest, walking quality steps `(90, 75, 60)` until under limit. Closes a silent-failure path on Anthropic's 5MB limit. Pillow is **optional** (`pip install clawagents[media]`).

**Tier 3 — testing infrastructure:**

- **🎭 Mock-provider parity harness** (`clawagents.testing.mock_provider`) — deterministic fake LLM service (`MockLLMService`) bound to `127.0.0.1:0`. Real provider clients point at it via `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` env vars. Routes via `X-Parity-Scenario:` header or `PARITY_SCENARIO: <name>` system message. Five built-in scenarios. Pure stdlib, zero new deps.

**v6.5 backlog (deferred):** Anthropic prompt-cache tracking + cache-break detection, auth-profile rotation with cooldowns, multi-provider routing prefix + LiteLLM extension, file checkpoint snapshots, cache-TTL provider eligibility map, `tool_use_behavior` / `StopAtTools`, granular lifecycle payload widening, skills hot-reload watcher, `finalize` cleanup hook, `edit_scope` allowlist in skills, multi-tier numeric verifier reward, replayable per-task archives.

### v6.3.0.post1 — Docs Re-publish (no code changes)

PEP 440 post-release. Identical code to `6.3.0`; re-published so the PyPI page
shows the corrected README (version badge, feature-matrix header, latest-release
callout). `pip install clawagents` resolves to this artifact.

### v6.3.0 — Sandbox & Security Hardening, Strict Type Checking

Security/correctness release. Eleven bugs fixed across both the Python and TypeScript ports, plus a full mypy cleanup. All tests green: **334 passed**, **mypy clean** (0 errors, exit 0).

**Security fixes:**
- **Sandbox escape via symlink (TS)** — `LocalBackend.safePath` was lexical-only (`path.resolve`), so an agent that ran `ln -s /etc evil` could read `/etc/*` through the symlink. Now uses `realpathSync` for both cwd and resolved paths so symlinks are followed before the containment check. Python was already safe via `Path.resolve()`.
- **SSRF gap (TS)** — `web_fetch`'s IPv6 link-local check only matched `fe8X`, missing `fe9X`/`feaX`/`febX`. Now matches the full `fe80::/10` range (`/^fe[89ab]/i`). Python uses `ipaddress.is_link_local`, no change needed.
- **`> /dev/null` blocked legitimate use (both)** — `BLOCKED_PATTERNS` had `"> /dev/null"` (typo for `"> /dev/sd"`), which blocked the common shell idiom `cmd > /dev/null`. Removed.
- **`rm /` regex parity (TS)** — `DANGEROUS_RE` was missing the `*` quantifier on the flag group, so `rm /` (no flags) slipped past while Python's regex blocked it. Aligned.
- **`wget http` / `curl http` parity (TS)** — added to TS `BLOCKED_PATTERNS` to match Python. Agents should use the `web_fetch` tool (with SSRF guards) for HTTP, not raw shell utilities.

**Correctness fixes:**
- **Multimodal system message crashed context shedding (Py)** — `_preflight_context_check` called `.replace()` and string-slicing on system messages without checking if `content` was a `list[dict]` (multimodal). Now guards each tier with `isinstance(content, str)` and emits a `warn` event if the system message is multimodal.
- **Arbitrary role from `pre_llm` hook (Py)** — external hooks could pass any string as `role`, blowing up Pydantic validation in `LLMMessage`. Now coerces unknown roles to `"user"` and emits a `warn`.
- **Parallel native tool-call indexing (Py)** — when `before_tool` rejected a call OR returned `updated_args`, `native_tool_call_objects` was indexed by approved-list index (off-by-one) and the identity check `tc is approved_calls[i]` failed (because `updated_args` constructs a new `ParsedToolCall`). Tool-call IDs sent back to the LLM were wrong, causing native function-calling failures. Now tracks `(orig_idx, call)` pairs through the approval loop.
- **Subagent env-mutation race (Py)** — concurrent subagent runs with `credential_proxy` enabled raced on `os.environ`. The second run captured the first's overrides as its "original" env, then stamped them back into place after the first run had already stopped its proxy. Wrapped the env-mutate / run / env-restore window in an `asyncio.Lock`. No-proxy path is unaffected.
- **`classify_error` rejected `BaseException` (Py)** — `asyncio.CancelledError` and similar inherit from `BaseException`, not `Exception`. Widened `classify_error`, `_extract_status`, and `ErrorDescriptor.original` to accept `BaseException`.
- **Gemini provider `None` parts iteration (Py)** — streaming chunks could surface `None` for `candidate.content.parts` after a `hasattr` check that says only the attribute exists. Switched to `getattr(getattr(_cand, "content", None), "parts", None)` and explicit truthiness check.

**Type checking:**
- Full mypy cleanup: 46 errors → 0. Real bugs fixed (None-iter, `AsyncOpenAI`/`AsyncAzureOpenAI` mismatch, missing telegram updater check, kwargs widening). False positives addressed by renaming reused variables, adding explicit `dict[str, Any]` annotations on union-typed locals, and `parameters: Dict[str, Dict[str, Any]]` annotations on tool implementations to satisfy the `Tool` protocol.
- Added `[tool.mypy]` block to `pyproject.toml` with `warn_unused_ignores = true` and `ignore_missing_imports = true`. Run `python -m mypy` — clean run shows `Success: no issues found in 72 source files`. Mypy now exits non-zero on errors so CI can gate on it.

**Regression coverage added:**
- `tests/test_exec_safety.py` — denylist behavior (legitimate idioms allowed, destructive patterns blocked)
- `tests/test_agent_loop_bugs.py` — multimodal shedding paths + role coercion
- `tests/test_parallel_native_indexing.py` — both rejection-skip and updated-args indexing paths
- `tests/test_subagent_env_race.py` — concurrent credential-proxy runs don't corrupt env

### v6.2.1 — Release Hardening, Redirect-Safe `web_fetch`, and Parity Smokes

Patch release focused on making the v6.2 line safer to install, test, and operate.

- **Redirect-aware SSRF protection** — `web_fetch` disables automatic redirects and manually revalidates every hop before network I/O. Public-to-private redirects to loopback, RFC1918, link-local, reserved, multicast, or cloud metadata IPs are refused by default.
- **Hermetic SSRF regression tests** — added `tests/test_web_fetch_ssrf.py` covering public-to-private redirects, redirect loops, direct private IP refusal, and legitimate public-to-public redirects.
- **Local-source pytest resolution** — `pyproject.toml` now sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so local test runs cannot accidentally import an older installed wheel from `site-packages`.
- **Cross-package parity smoke** — added `scripts/smoke_gemma4.py`, mirroring the TypeScript smoke script and printing provider, base URL, and stored model for Ollama/Gemma4, `gpt-5.4`, `gemini-3.1-pro`, and `claude-opus-4-6`.
- **Release verification** — `python -m pytest` reports **319 passed, 2 skipped**; the SSRF-specific suite reports **5 passed**.

### v6.2.0 — OpenAI-Agents Parity, Ollama/Gemma4 First-Class Routing, 63 Model Profiles

A substantial additive release. Everything is backward compatible — existing `create_claw_agent()` calls, env vars, and tool registrations work unchanged.

**1. Ten OpenAI-Agents-SDK parity surfaces** (all additive, all new modules)

| Surface | Module | What it adds |
|:---|:---|:---|
| **Run Context** | `clawagents.run_context` | `RunContext` carries per-run state, approvals, and arbitrary user data through hooks and tools. |
| **Usage Tracking** | `clawagents.usage` | `Usage` + `RequestUsage` aggregate token/latency stats across turns, providers, and sub-agents. |
| **Lifecycle Hooks** | `clawagents.lifecycle` | `RunHooks` / `AgentHooks` with typed `LLMStart/LLMEnd/ToolStart/ToolEnd/AgentStart/AgentEnd/RunStart/RunEnd/Handoff` payloads. `composite_hooks` chains multiple observers without interference. |
| **Guardrails** | `clawagents.guardrails` | `input_guardrail` / `output_guardrail` decorators, `GuardrailTripwireTriggered`, behavior modes (raise / log / filter). |
| **Stream Events** | `clawagents.stream_events` | First-class `TurnStartedEvent`, `AssistantDeltaEvent`, `ToolCallPlannedEvent`, `ApprovalRequiredEvent`, `UsageEvent`, `GuardrailTrippedEvent`, `FinalOutputEvent`, `ErrorStreamEvent`. Consumable via `on_stream_event` callback. |
| **Retry Policy** | `clawagents.retry` | `RetryPolicy` dataclass + `DEFAULT_RETRY_POLICY`. Exponential backoff with jitter, per-error-class overrides. |
| **Function Tools** | `clawagents.function_tool` | `@function_tool` decorator auto-derives JSON Schema from Python type hints. Zero boilerplate. |
| **Session Backends** | `clawagents.session` | Unified `Session` protocol with `InMemorySession`, `JsonlFileSession`, `SQLiteSession`. Drop-in persistence. |
| **Structured Outputs** | `output_type=` arg on `create_claw_agent` / `agent.invoke` | Return typed objects via Pydantic model, dataclass, `dict`, `list`, or `str`. Coerced after run completes; failures emit a `warn` stream event. |
| **Tool Approval** | `approval_handler=` arg + `ApprovalRequiredEvent` | HITL gate — async callable receives `{tool, args}` and returns `True` / `False` / a redirect dict. Integrates with `ApprovalRequiredEvent` for streaming UIs. |

**2. Ollama & Gemma 4 first-class routing**

`create_provider()` now auto-routes 24 Ollama-family prefixes to `http://localhost:11434/v1` with no config needed. Use either the bare tag (`gemma4:e4b`) or the explicit routing form (`ollama/gemma4:e4b`).

| Family | Examples | Routed to |
|:---|:---|:---|
| **Gemma 4** | `gemma4`, `gemma4:e2b`, `gemma4:e4b`, `gemma4:26b`, `gemma4:31b` | Ollama @ :11434/v1 |
| **Gemma 3 / 3n / 2** | `gemma3`, `gemma3n:e4b`, `gemma2`, `gemma` | Ollama @ :11434/v1 |
| **Llama / Qwen / Mistral / Phi / Deepseek / Codellama** | `llama3`, `qwen2`, `mistral`, `mixtral`, `phi4`, `deepseek-r1`, `codellama`, … | Ollama @ :11434/v1 |
| **Explicit routing** | `ollama/<any-tag>` | Ollama @ :11434/v1 (prefix stripped) |

Override with `OPENAI_BASE_URL` if you run Ollama on a different host/port. API key is auto-set to the placeholder `"ollama"`.

**3. 63 model profiles + model-aware context budget**

The `_MODEL_PROFILES` table now covers frontier (GPT-5.4 → 400K, Gemini 3.1 → 1M, Claude 4.6 Opus), Ollama (Gemma4 e2b/e4b → 128K, 26b/31b → 256K), and a long tail of OSS variants. `_resolve_context_budget()` walks insertion order for deterministic prefix matching (most-specific first).

**4. Cross-package parity** — the TypeScript sibling `clawagents` (see [x1jiang/clawagents](https://github.com/x1jiang/clawagents)) has the identical 24-entry Ollama prefix list, 63-entry model profile table with the same (window, ratio) values, and the same `create_provider` routing logic. Parity can be exercised manually with the matching smoke scripts in each repo (`clawagents_py/scripts/smoke_gemma4.py` and `clawagents/scripts/smoke-gemma4.ts`); both print the same provider, base URL and stored model for `gemma4:*`, `ollama/...`, `gpt-5.4`, `gemini-3.1-pro` and `claude-opus-4-6`. The GitHub Actions workflow added in v6.2.1 runs `pytest`, `python -m build`, and `twine check` on every push.

**5. Quality / debug pass**

- Async agent loop hardening — new turn-started events, tighter cancellation semantics, cleaner state hand-off to sub-agents.
- Added `tests/test_openai_agents_surfaces.py` — full coverage for RunContext, Usage, Hooks, Guardrails, StreamEvents, Retry, FunctionTool, Session backends.
- Test suite: **314 passed, 2 skipped**.

**New public exports** (from `clawagents`):
`RunContext`, `ApprovalRecord`, `Usage`, `RequestUsage`, `RunHooks`, `AgentHooks`, `composite_hooks`, `InputGuardrail`, `OutputGuardrail`, `input_guardrail`, `output_guardrail`, `GuardrailBehavior`, `GuardrailResult`, `GuardrailTripwireTriggered`, `StreamEvent` (+ 10 concrete event types), `stream_event_from_kind`, `RetryPolicy`, `DEFAULT_RETRY_POLICY`, `function_tool`, `InMemorySession`, `JsonlFileSession`, `SQLiteSession`.

### v6.1.1 — Credential Isolation & Lazy Tool Provisioning

| Feature | Description |
|:---|:---|
| **Credential Isolation** | `execute` tool strips sensitive env vars (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) from subprocess environment. Claude-generated code can no longer read API keys via `env` or `os.environ`. |
| **Lazy Tool Provisioning** | Sandbox-backed tools (filesystem, exec, advanced-fs, web) defer module import to first `execute()` call. Schema is available immediately for the LLM. Reduces startup overhead. |

### v6.1.0 — Advisor Model: Smart Model Guides Cheap Model

Pair a stronger "advisor" model with a cheaper "executor" model. The executor runs every turn; the advisor is consulted 2-3 times per task for strategic guidance. Cross-provider supported — any model can advise any other model.

| Feature | Description |
|:---|:---|
| **Advisor Model** | New `advisor_model` config field. Set it and the agent gets smarter. Don't set it, nothing changes. Fully backward compatible. |
| **Three Trigger Points** | (1) After initial orientation, before planning. (2) When stuck (consecutive failures). (3) Before declaring done. |
| **Cross-Provider** | Mix providers freely: `gpt-5.4-nano` executor + `claude-opus-4-6` advisor, or any combination. |
| **CLI Flag** | `--advisor MODEL` flag for one-line usage. |
| **Env Config** | `ADVISOR_MODEL`, `ADVISOR_API_KEY`, `ADVISOR_MAX_CALLS` env vars. |

```python
agent = create_claw_agent(
    "gpt-5.4-nano",
    advisor_model="gpt-5.4",
)
```

### v6.0.0 — Production Hardening: 17 Improvements

**High Priority**

| Feature | Description |
|:---|:---|
| **Native Tool Call Patching (H1)** | `_patch_dangling_tool_calls` now handles native function calling (`tool_calls_meta`), not just text-mode JSON. Injects synthetic cancelled responses for orphaned tool_call IDs. Prevents 400 API errors in HITL scenarios. |
| **Three-Tier Provider Fallback (H2)** | New `FallbackProvider` wraps any LLM with `primary → named fallback → global fallback` chain. Quarantines providers after consecutive failures, periodic health-check restores. Config via `fallback_models` param or `CLAWAGENTS_FALLBACK_MODELS` env var. |
| **Credential Proxy (H3)** | New `CredentialProxy` — local HTTP proxy that injects API keys into outbound requests so sandboxed sub-agents never see raw credentials. Opt-in via `CLAW_FEATURE_CREDENTIAL_PROXY=1`. |
| **Rich Hook Result Model (H4)** | `BeforeToolHook` now accepts `HookResult` return (backward-compatible with bool). Hooks can block with reason, redirect args, inject messages. New `HookResult` dataclass exported from public API. |
| **Fraction-Based Summarization (H5)** | Soft-trim threshold now derives from per-model `budget_ratio` instead of hardcoded 0.60. GPT=0.60, Gemini=0.675, Claude=0.6375. Auto-adapts to any model's context window. |
| **Lazy Static Tool Registry (H7)** | New `LazyTool` class + `ToolRegistry.register_lazy()`. Tools are imported only on first `execute()` call. Fast startup with large tool sets. |

**Medium Priority**

| Feature | Description |
|:---|:---|
| **Subagent State Isolation (M1)** | `EXCLUDED_STATE_KEYS` prevents parent state (messages, todos, trajectory, lessons, session) from leaking into child sub-agents. |
| **SKILL.md Constraint Documents (M4)** | Skills now support `forbidden-actions`, `workspace-layout`, `success-criteria`, `workflow-steps` in YAML frontmatter. Structured constraints for sandboxed code execution. |
| **Pre-Compact Transcript Archival (M5)** | Before context compaction, full transcript is archived to `.clawagents/transcripts/`. Opt-in via `CLAW_FEATURE_TRANSCRIPT_ARCHIVAL=1`. |
| **Atomic File Writes (M7)** | Trajectory recorder and session persistence now use temp-then-rename pattern via `atomic_write_text()`. Prevents corruption on crash. |
| **Barrier-Based Scheduling (M8)** | Command queue now supports barrier entries. Destructive ops wait for active tasks to complete before executing. |
| **Session Heartbeat (M9)** | New `SessionHeartbeat` class auto-releases stale sessions after timeout. Resource management for multi-user deployments. |
| **Cross-Provider Test Suite (M10)** | 14 conformance tests (7 per backend) ensuring `LocalBackend` and `InMemoryBackend` both satisfy the `SandboxBackend` protocol. |

**New files:** `providers/fallback.py`, `sandbox/credential_proxy.py`, `utils/atomic_write.py`, `session/heartbeat.py`, `tests/test_cross_provider.py`

**New feature flags:** `transcript_archival` (off), `credential_proxy` (off)

**New exports:** `HookResult`, `FallbackProvider`, `CredentialProxy`, `SessionHeartbeat`, `LazyTool`, `atomic_write_text`, `atomic_write_bytes`

### v5.28.0 — Error Taxonomy, Prompt Caching, Session Persistence & External Hooks

Four production-grade features ported from the [claw-code-main](https://github.com/anthropics/claw-code) Rust reference implementation:

| Feature | Description |
|:---|:---|
| **Prompt Cache Boundary** | Inserts `__CACHE_BOUNDARY__` marker in system prompt. Anthropic provider splits into static (cached via `cache_control: ephemeral`) + dynamic blocks. Reduces input token costs on multi-turn sessions. ON by default. |
| **Error Taxonomy & Recovery** | Classifies all LLM/tool errors into 7 discrete classes (`context_window`, `provider_auth`, `provider_rate_limit`, `provider_retry_exhausted`, `provider_internal`, `provider_transport`, `runtime_io`). Each class has `retryable`, `recovery_hint`, and optional `failover_model`. Structured error events emitted via `onEvent`. ON by default. |
| **Session Persistence** | Saves agent sessions as append-only JSONL to `.clawagents/sessions/`. Events: `system_prompt`, `turn_started`, `assistant_message`, `tool_result`, `usage`, `turn_completed`. New CLI: `--sessions` (list) and `--resume [ID\|latest]` (continue). Opt-in. |
| **External Hook System** | Shell commands that run before/after tool execution and LLM calls. Config via `.clawagents/hooks.json` or `CLAW_HOOK_*` env vars. Hooks receive JSON on stdin, return JSON on stdout. `pre_tool_use` can block or modify args. 10s timeout, fail-open. Opt-in. |

Also:
- **Anthropic cache token extraction** — `cache_creation_tokens` and `cache_read_tokens` now populated from both streaming and non-streaming Anthropic responses.
- **`AgentState.session_file`** — New field tracks the session JSONL path when persistence is enabled.
- **New public exports** — `ErrorClass`, `ErrorDescriptor`, `classify_error`, `get_recovery_recipe`, `SessionWriter`, `SessionReader`, `list_sessions`, `HooksConfig`, `ExternalHookRunner`, `load_hooks_config`.

### v5.27.3 — Gemini Signature Regression Coverage
- **Gemini signature regression test** — Added targeted tests for `_serialize_gemini_parts` to ensure `thought_signature` is propagated to sibling parallel `function_call` parts.
- **Parallel integration test reliability** — Fixed integration test fixture validation mismatch so large-output parallel execution is validated correctly.

### v5.27.2 — Gemini 3 Thought Signature Fix
- **Gemini 3 Propagation** — Propagated `thought_signature` to all parallel `function_call` parts in the response, preventing `400 INVALID_ARGUMENT` during multi-tool execution.

### v5.27.1 — Timeout Bugfix
- **Fixed NameError** — Added `timeout_s` parameter to `ClawAgent.invoke` to prevent an exception when a global timeout is not provided.

### v5.27.0 — Claude Code Architectural Patterns

Ported 10 production-grade architectural patterns from Anthropic's Claude Code directly into ClawAgents. These features are controllable via environment variables or constructor injection:

| Feature | Description |
|:---|:---|
| **Micro-Compact Memory** | Aggressively clears giant tool results to save context. |
| **File History Snapshots** | Safely backs up files to `.clawagents/snapshots/` before writing. |
| **Prompt Cache Tracking** | Real-time stats on Anthropic/OpenAI prompt cache hits. |
| **Typed Memory Taxonomy** | Auto-parses `project`, `user`, and `feedback` memories via frontmatter. |
| **Write-Ahead Logging (WAL)** | Crash-resilient interaction logging. |
| **Granular Permission Rules** | Define glob-based `Allow`/`Deny` execution policies. |
| **Background Memory Extraction** | Periodically scans conversations and extracts metadata. |
| **Orchestration** | Access to `run_forked_agent` and `run_coordinator` (swarm routing). |

### v5.26.0 — Bundled OpenViking Skill, Updated ByteRover Skill

| Feature | Description |
|:---|:---|
| **OpenViking skill** | Bundled `skills/openviking/SKILL.md` teaches the agent to use the `ov` CLI for tiered context retrieval (L0/L1/L2). Auto-enabled when `ov` is on PATH |
| **ByteRover skill updated** | Refreshed to match `byterover-cli` v1.8.0 — added `--headless`, `--folder`, removed obsolete commands |
| **Generic bundled skill loader** | Skill loader now scans the entire bundled `skills/` directory instead of hardcoding individual skills |

### v5.25.0 — Gemini Streaming Fix

| Feature | Description |
|:---|:---|
| **Fix Gemini SDK warning** | Eliminated "non-text parts in the response" warning by iterating `candidates[].content.parts[]` instead of accessing the `.text` property on streaming chunks containing function calls |
| **Consistent text extraction** | Streaming path now uses the same parts-based extraction as the non-streaming `_request_once`, filtering out thought parts |

### v5.24.0 — Zero-Config Channel Auto-Detection

| Feature | Description |
|:---|:---|
| **Auto-detect channels from env vars** | `clawagents --serve` now reads `TELEGRAM_BOT_TOKEN`, `WHATSAPP_AUTH_DIR`, `SIGNAL_ACCOUNT` from `.env` and auto-starts the ChannelRouter — zero code required |
| **`--doctor` channel status** | `clawagents --doctor` reports which messaging channels are configured |
| **`.env.example` updated** | All channel env vars documented with inline comments |
| **`--init` scaffold** | `clawagents --init` generates `.env` with channel variables pre-commented |

### v5.23.0 — WebSocket Gateway, Multi-Channel Messaging (Telegram, WhatsApp, Signal)

Full multi-platform messaging support inspired by OpenClaw's channel architecture:

| Feature | Description |
|:---|:---|
| **WebSocket gateway** | FastAPI native WebSocket endpoint at `/ws` alongside existing HTTP. Methods: `chat.send` (streaming events), `chat.history`, `chat.inject`, `ping`. Auth via `?token=` query param |
| **Channel adapter interface** | `ChannelAdapter` protocol + `ChannelMessage` dataclass — standard contract for any messaging platform |
| **Telegram adapter** | Uses [python-telegram-bot](https://python-telegram-bot.org/). Config: `{"bot_token": "..."}` |
| **WhatsApp adapter** | Baileys subprocess (Node.js) or WhatsApp Business API. Config: `{"mode": "baileys", "auth_dir": ".whatsapp-auth"}` |
| **Signal adapter** | Uses [signal-cli](https://github.com/AsamK/signal-cli) subprocess with JSON-RPC. Config: `{"account": "+1234567890"}` |
| **Channel router** | `ChannelRouter` dispatches inbound messages to agents, routes replies back. Per-session serialization via `KeyedAsyncQueue`, optional debouncer, hooks |

```python
from clawagents import create_claw_agent, ChannelRouter
from clawagents.channels.telegram import TelegramAdapter
from clawagents.channels.whatsapp import WhatsAppAdapter

router = ChannelRouter(lambda: create_claw_agent("gpt-5-mini"))
router.register(TelegramAdapter())
router.register(WhatsAppAdapter())
await router.start_all({
    "telegram": {"bot_token": "123456:ABC..."},
    "whatsapp": {"mode": "baileys", "auth_dir": ".whatsapp-auth"},
})
```

### v5.22.0 — Tool Result Caching, Parameter Validation & ComposeTool

3 features inspired by ToolUniverse's tool management patterns:

| Feature | Description |
|:---|:---|
| **Tool result caching** | LRU in-memory cache (`ResultCacheManager`) avoids redundant tool calls. Tools opt in with `cacheable = True`. Per-tool TTL overrides via `result_cache.set_tool_ttl()`. Built-in cacheable tools: `read_file`, `grep`, `web_fetch`. Default: 256 entries, 60s TTL |
| **Parameter validation + coercion** | `validate_tool_args()` checks required params and type-matches before execution. Lenient coercion handles common LLM quirks: `"42"` → `42`, `"true"` → `True`, JSON strings → objects/arrays. Enabled by default on `ToolRegistry` |
| **ComposeTool** | `create_compose_tool()` chains multiple tools in a deterministic pipeline without an LLM in the loop. Lighter than sub-agents for predictable workflows. Steps receive previous results and a `call_tool` helper. Failures short-circuit with clear error messages |

### v5.21.0 — Context Engine, Loop Detection & Compaction Overhaul

8 improvements inspired by the latest OpenClaw architecture:

| Feature | Description |
|:---|:---|
| **Chunked compaction with retry** | Compaction now splits old messages into ~30K-token chunks, summarizes each separately with up to 3 retries (exponential backoff), and explicitly preserves file paths, function names, error messages, and commands verbatim |
| **Better loop detection** | Result hashing detects "different args, same result" stalls; ping-pong detection catches A→B→A→B oscillation; global circuit breaker hard-stops at 30 no-progress calls |
| **Context pruning (soft-trim)** | New `_soft_trim_messages` runs at 60% context usage (before the 75% compaction trigger). Trims old tool results >1000 chars, removes duplicates, and stubs stale image data |
| **Skill eligibility gating** | Skills can declare `requires:` in YAML frontmatter (`os`, `bins`, `env`). Ineligible skills are filtered at load time |
| **Skill prompt budget** | Max 20 skills / 4000 chars injected into the system prompt. Full list accessible via `list_skills` |
| **Control token sanitization** | Strips leaked model control tokens (`<\|assistant\|>`, `<\|endoftext\|>`, full-width variants) from final output |
| **Head+tail truncation** | Eviction fallback and content preview now use head+tail (preserving error messages at the end). Also fixes a bug where few-line, huge-character content bypassed preview truncation |
| **Pluggable context engine** | New `ContextEngine` ABC with `after_turn`, `compact`, `bootstrap`, `cleanup` lifecycle hooks. `DefaultContextEngine` is a no-op pass-through. Registry: `register_context_engine()` / `resolve_context_engine()` |

### v5.20.4 — Gemini MALFORMED_FUNCTION_CALL Retry

| Feature | Description |
|:---|:---|
| **Gemini malformed FC retry** | When Gemini returns `finish_reason=MALFORMED_FUNCTION_CALL` with 0 parts (common with complex parallel tool calls), the provider now automatically retries with `tool_config.mode=ANY` instead of stopping the agent |
| **Streaming + non-streaming** | Fix applied to both streaming (`_stream_with_retry`) and non-streaming (`_request_once`) code paths |
| **Recursion guard** | `_malformed_retry` flag prevents infinite retry loops if mode=ANY also fails |

### v5.20.3 — GPT-5 Temperature Corrections

| Feature | Description |
|:---|:---|
| **GPT-5-nano temperature** | Live API tests confirmed `gpt-5-nano` requires `temperature=1` (not 0). Fixed in `_FIXED_TEMPERATURE_MODELS` |

### v5.20.0 — Temperature & Compaction Fixes

| Feature | Description |
|:---|:---|
| **Temperature fix** | GPT-5 models no longer forced to `temperature=1.0`. Only o-series models (o1, o3, o4-mini) retain the fixed override. This restores deterministic behavior when `TEMPERATURE=0` is set |
| **Compaction overhaul** | Context compaction no longer causes the agent to "forget" what it was doing. Five improvements: (1) `RECENT_MESSAGES_TO_KEEP` increased from 6 → 20, (2) tool call/result pairs are never split, (3) summary prompt now includes original task + structured preservation instructions, (4) compacted summary inserted as `role="user"` with `[System — Compacted History]` prefix instead of `role="assistant"`, (5) text log for summarization includes structured `[TOOL CALLS]` and `[TOOL RESULT]` markers |
| **Debug cleanup** | All development instrumentation removed from production code |

### v5.19.0 — Anthropic Provider, Security, Architecture Overhaul

| Feature | Description |
|:---|:---|
| **Anthropic/Claude provider** | First-class support for Claude models via `ANTHROPIC_API_KEY`. Install with `pip install clawagents[anthropic]` |
| **Optional Gemini** | `google-genai` is now an optional dependency. Install with `pip install clawagents[gemini]` or `pip install clawagents[all]` |
| **`py.typed` + `__version__`** | PEP 561 type stub marker and `clawagents.__version__` export for downstream tools |
| **Lazy config loading** | No more module-level side effects — `.env` discovery happens on first `load_config()` call |
| **Lazy `Path.cwd()`** | All module-level `Path.cwd()` calls replaced with lazy functions — safe for import from any directory |
| **Gateway authentication** | `GATEWAY_API_KEY` env var enables Bearer token auth on POST endpoints |
| **CORS support** | Gateway now supports `GATEWAY_CORS_ORIGINS` for cross-origin requests |
| **Improved blocked patterns** | Expanded dangerous command detection with regex matching |
| **API key masking** | `clawagents --doctor` now masks keys (shows `********...last4`) |
| **Azure detection** | New `OPENAI_API_TYPE=azure` env var for explicit Azure OpenAI configuration |
| **Global timeout** | `--timeout N` CLI flag and `CLAW_TIMEOUT` env var for agent run time limits |
| **`--verbose` / `--quiet`** | CLI flags for controlling output verbosity |
| **`--prune-trajectories N`** | Delete trajectory files older than N days |
| **Lesson export/import** | `export_lessons()` / `import_lessons()` for sharing lessons between projects |
| **Trajectory pruning** | `prune_trajectories(max_age_days)` utility function |
| **`pydantic-settings`** | Now properly listed as a dependency (was missing) |
| **pyproject.toml metadata** | Added license, authors, classifiers, URLs, optional dependency groups |
| **New tests** | Tests for `_repair_json`, trajectory recorder, config module |

### v5.18.0 — Doctor, Trajectory Inspector & Config Improvements

| Feature | Description |
|:---|:---|
| **`clawagents --doctor`** | New diagnostic command checks `.env` discovery, API keys, active model, LLM settings, PTRL flags, local endpoint reachability, trajectory history, and `AGENTS.md` presence |
| **`clawagents --trajectory [N]`** | Inspect the last N run summaries: score, quality, failures, judge verdict, duration — human-readable trajectory output |
| **Startup banner** | Every `--task` and `--serve` now prints `provider=X model=Y env=Z ptrl=...` for instant visibility into active config |
| **`CLAWAGENTS_ENV_FILE`** | New env var to explicitly point to a `.env` file path. Priority: `CLAWAGENTS_ENV_FILE` > `cwd/.env` > `cwd/../.env`. Useful for CI, Docker, multi-project |
| **Publish hygiene** | GitHub releases no longer include `.clawagents/`, `.pytest_cache/`, logs, or other runtime artifacts |
| **Config/docs consistency tests** | 6 pytest tests verify every `EngineConfig` field appears in `.env.example` and `README.md` |
| **`--port` in TypeScript** | Gateway server port now configurable via `--port N` in TypeScript CLI |

### v5.17.0 — Quick Start Scaffold & Examples

| Feature | Description |
|:---|:---|
| **`clawagents --init`** | New CLI command scaffolds a starter project in the current directory: generates `.env` (with all providers commented out), `run_agent.py` (ready-to-run starter script with 5 provider options), and `AGENTS.md` (memory template) |
| **`clawagents --help`** | Shows usage with examples, quick start instructions |
| **`clawagents --task`** | Run a single task from the command line |
| **`clawagents --serve`** | Start the HTTP gateway server from CLI |
| **Examples directory** | 8 ready-to-run example scripts: OpenAI, Gemini, Azure, Ollama, vLLM, Bedrock, custom tools, and multi-sample comparison |
| **README overhaul** | New "30-Second Quick Start" section, examples table, clearer onboarding flow |

### v5.16.0 — LLM-as-Judge & Thinking Token Preservation

| Feature | Description |
|:---|:---|
| **G. LLM-as-Judge verification** | After each run (when `learn=True`), a separate, focused LLM call evaluates whether the task was actually accomplished. Returns a 0-3 score with justification — more reliable than heuristic scoring. Results stored as `judge_score` and `judge_justification` on `RunSummary` |
| **H. Thinking token preservation** | Models like Qwen3 and DeepSeek that emit `<think>...</think>` blocks are now fully supported. Thinking content is extracted before tool-call parsing, preserved on messages and trajectory records, and stripped from visible output. Available via `strip_thinking_tokens()` utility |

### v5.15.0 — Deterministic Verification & GRPO-Inspired Comparison

| Feature | Description |
|:---|:---|
| **A. Deterministic rewards** | Tool execution results (exit codes, test pass/fail counts) are now used as objective ground truth for scoring, replacing pure LLM self-assessment. Each turn and run summary includes `deterministic_score` and `verified_score` fields |
| **B. Multi-sample comparison** | New `agent.compare(task, n_samples=3)` method runs the same task N times and picks the best result using objective scoring — inspired by SkyRL's Group Relative Policy Optimization (GRPO) |
| **C. Task-type-aware verification** | Auto-detects task type (coding/file/search/refactor/general) and applies type-specific verifiers. Coding tasks use test results; file tasks check write success; refactoring checks edits + tests |
| **D. Progressive context caching** | System prompt token count is computed once and cached, avoiding redundant re-counting on every turn. Logged at startup for budget visibility |
| **E. RFT-ready transitions** | Each trajectory now exports `{run_id}_rft.json` with (observation, action, reward, done) tuples per step — structured for future Rejection Fine-Tuning pipelines |
| **F. Adaptive rethink threshold** | Rethink trigger threshold now adjusts dynamically: complex tasks (coding/refactor) get more patience (threshold=5), simple tasks (search/file) trigger sooner (threshold=3), and late in runs threshold drops to minimum (2) |

### v5.14.0 — SkyRL-Inspired PTRL Improvements

| Feature | Description |
|:---|:---|
| 🚦 **Quality gate for lesson extraction** | Lessons only extracted from runs with mixed outcomes (both successes and failures). Zero-variance runs (all-success or all-failure with no contrast) are skipped — inspired by SkyRL's GRPO dynamic sampling |
| ⏰ **Lesson staleness decay** | Each lesson block is now timestamped + model-tagged (`@timestamp [model]`). `load_lessons(max_age_s=N)` filters out stale lessons. Prevents prompt pollution from outdated advice |
| 🔤 **Format vs. logic failure classification** | Every failed tool call is classified as `"format"` (bad JSON, wrong params) or `"logic"` (valid call, wrong approach). Rethink messages now include format-specific or strategy-specific guidance |
| 📊 **Per-step reward attribution** | Each `TurnRecord` now includes `observation_context` (what the agent saw before deciding), `productivity_score` (-1.0 to 1.0), and `failure_type` per tool call. `RunSummary` adds `format_failures`, `logic_failures`, `has_mixed_outcomes`, and `finish_reason` |
| 🧠 **Enhanced self-analysis prompt** | Post-run LLM analysis now receives failure type breakdown and productivity scores for targeted lesson extraction |

### v5.13.0 — Prompt-Time Reinforcement Learning (PTRL)

| Feature | Description |
|:---|:---|
| 🧠 **PTRL: Post-run self-analysis** | After each run, the LLM reviews its own trajectory and extracts 2-5 actionable lessons, saved to `.clawagents/lessons.md` |
| 📖 **PTRL: Pre-run lesson injection** | On subsequent runs, stored lessons are injected into the system prompt so the agent avoids past mistakes |
| 🔄 **PTRL: Enhanced mid-run rethink** | When consecutive failures trigger a rethink, relevant past lessons are included in the rethink message |
| 🎛️ **`learn` flag / `CLAW_LEARN` env** | Opt-in via `learn=True` or `CLAW_LEARN=1`. Automatically enables trajectory logging |
| 📐 **Default `context_window` → 1,000,000** | Increased from 128,000 to support modern large-context models |
| 🔧 **macOS sandbox symlink fix** | `LocalBackend` now resolves symlinks at init (fixes `/var` → `/private/var` on macOS) |
| ✅ **All 150 tests passing** | Fixed 48 pre-existing test failures (sandbox path traversal, LLMMessage subscript, mock assertions) |

### v5.12.1 — Streamlit / Jupyter Compatibility

| Feature | Description |
|:---|:---|
| 🔧 **Signal handler fix** | `add_signal_handler` now catches `RuntimeError` in addition to `NotImplementedError`/`OSError`, fixing crashes in Streamlit, Jupyter, and other non-main-thread environments |

### v5.12.0 — Gemini 3 Thought Signature Support

| Feature | Description |
|:---|:---|
| 🧠 **`thought_signature` preservation** | Gemini 3 thinking models (e.g. `gemini-3-flash-preview`) require `thought` and `thought_signature` fields to be echoed back during multi-turn function calling. ClawAgents now captures the full response parts and replays them verbatim, preventing 400 errors. |
| 🔄 **`gemini_parts` field** | New optional field on `LLMMessage` and `LLMResponse` carries raw Gemini response parts through the conversation history. Used automatically — no user action required. |

### v5.11.0 — Configurable Limits

| Feature | Description |
|:---|:---|
| 🔢 **`max_iterations`** | Now settable at construction or via `MAX_ITERATIONS` env (default 200, was hardcoded in caller) |
| 📏 **`preview_chars`** | Tool-output preview length configurable via `CLAW_PREVIEW_CHARS` env (default 120) |
| 📄 **`response_chars`** | Response text length in trajectory records via `CLAW_RESPONSE_CHARS` env (default 500) |
| ⚙️ **Priority** | Explicit param > env var > default for all three |

### v5.10.0 — Discrete Reward Bands & Weighted Scoring

| Feature | Description |
|:---|:---|
| 🎯 **Discrete reward bands** | Run scores mapped to -1 … +3 bands (inspired by CUDA-Agent PPO reward shaping) |
| ⚖️ **Weighted execution scoring** | `execute`, `shell`, `run_code` weighted 2× higher than generic tools |
| 🏷️ **Run quality grading** | Each run classified as `clean`, `noisy`, or `failed` for trajectory filtering |
| 🛡️ **Gameable tool exclusion** | `think`, `todolist`, `use_skill`, etc. excluded from scoring to prevent reward hacking |

### v5.9.0 — Trajectory Logging & Rethink

| Feature | Description |
|:---|:---|
| 📊 **Trajectory logging** | Structured recording of every turn, tool call, and outcome to `runs.jsonl` |
| 🔄 **Consecutive-failure rethink** | After 3 consecutive meaningful failures, injects a system "rethink" prompt |
| 🎛️ **Opt-in flags** | `trajectory=True` / `CLAW_TRAJECTORY=1` and `rethink=True` / `CLAW_RETHINK=1` |

### v5.8.0 — JSON Resilience

| Feature | Description |
|:---|:---|
| 🔧 **JSON repair** | `_repair_json()` utility fixes truncated JSON from hitting `max_completion_tokens` |
| 🔁 **Truncated JSON retry** | Detects incomplete JSON tool calls and prompts the LLM to resend |

### v5.7.0 — Model-Specific Temperature

| Feature | Description |
|:---|:---|
| 🌡️ **Fixed-temperature models** | Reasoning models (o-series, gpt-5, gpt-5-mini, gpt-5-turbo) auto-override to `temperature=1.0`. Non-reasoning models (gpt-5-nano, gpt-5-micro, gpt-4o) respect configured temperature |
| 🌡️ **Configurable temperature** | `TEMPERATURE` env var + `temperature` parameter on `create_claw_agent` |

### v5.6.0 — LLM Parameter Fixes

| Feature | Description |
|:---|:---|
| 🔑 **`max_completion_tokens`** | OpenAI calls now use `max_completion_tokens` (replacing deprecated `max_tokens`) |
| 🔑 **`max_output_tokens`** | Gemini calls now pass `max_output_tokens` correctly |
| ⚙️ **Config priority** | Explicit param > `.env` > default — no more shadowing of env values |

### v5.5.0 — Foundation

| Feature | Description |
|:---|:---|
| 🔌 **Pluggable Sandbox** | `SandboxBackend` protocol with `LocalBackend` + `InMemoryBackend` |
| 🌐 **Gateway Server** | FastAPI server with SSE streaming and 4-lane queue |
| 🗂️ **Advanced FS Tools** | `tree`, `diff`, `insert_lines` |
| 🧠 **Think Tool** | Structured reasoning without side effects |
| 🌍 **Web Fetch** | URL fetching with HTML cleanup |
| 💬 **Ask User** | Interactive stdin-based input |
| 📜 **History Offloading** | Full audit trail preserved after compaction |
| 🔒 **Tool Access Control** | `block_tools()` / `allow_only_tools()` at runtime |
| 💉 **Context Injection** | `inject_context()` hook for every LLM call |
| ✂️ **Output Truncation** | `truncate_output()` to cap tool output size |

---

## Trajectory Logging & RL-Inspired Scoring

ClawAgents includes an optional **trajectory system** inspired by reinforcement learning techniques from [CUDA-Agent](https://github.com/NexaAI/CUDA-Agent) and [OpenClaw-RL](https://github.com/anthropics/openclaw-rl). Enable it with `trajectory=True` or `CLAW_TRAJECTORY=1`.

### What gets logged

Every agent run records:
- **Turn-level data**: tool calls, arguments, success/failure, output previews
- **Weighted turn scores**: execution tools (shell, code runners) weighted 2× higher than generic tools
- **Run summary**: total turns, tool calls, successes/failures, elapsed time

### Discrete reward bands

Each run receives a score from **-1 to +3**:

| Score | Meaning |
|:---:|:---|
| **+3** | All tools succeeded, task completed cleanly |
| **+2** | Minor hiccups but overall success |
| **+1** | Partial success with some failures |
| **0** | Inconclusive — mixed results |
| **-1** | Majority of tool calls failed |

### Quality grading

Runs are classified for downstream filtering:

| Quality | Criteria |
|:---|:---|
| `clean` | Score ≥ 2 and ≤ 2 mid-run failures |
| `noisy` | Score ≥ 0 but too many mid-run failures |
| `failed` | Score < 0 |

### Anti-gaming protections

Tools like `think`, `todolist`, `use_skill`, `list_skills`, and `update_todo` are excluded from scoring — they can't inflate success rates.

### Consecutive-failure rethink

With `rethink=True` or `CLAW_RETHINK=1`, the agent monitors tool outcomes in real-time. After **3 consecutive meaningful failures**, it injects a system message:

> *"You have had 3 consecutive tool failures. Stop and rethink your approach before continuing."*

This simple mechanism prevents the agent from spiraling into repeated failed attempts.

### Output

Run summaries are appended to `.clawagents/trajectories/runs.jsonl`:

```json
{
  "run_id": "a1b2c3d4",
  "model": "gpt-5-mini",
  "total_turns": 8,
  "tool_calls": 12,
  "successes": 10,
  "failures": 2,
  "run_score": 2,
  "quality": "clean",
  "elapsed_ms": 45230,
  "turns": [...]
}
```

---

## Roadmap

- [ ] Docker sandbox backend (protocol ready)
- [ ] Semantic browser automation (accessibility tree)
- [ ] Prompt caching (Anthropic-style)
- [ ] Persistent memory learning from trajectory data (advanced — RFT-style rule extraction)
- [x] Post-run self-analysis + lesson extraction ✅ (v5.13 — PTRL)
- [x] Pre-run lesson injection ✅ (v5.13 — PTRL)
- [x] Enhanced mid-run rethink with past lessons ✅ (v5.13 — PTRL)
- [x] Trajectory logging + discrete reward bands ✅ (v5.9–5.10)
- [x] Consecutive-failure rethink injection ✅ (v5.9)
- [x] Weighted execution scoring + quality grading ✅ (v5.10)
- [x] JSON repair + truncated JSON retry ✅ (v5.8)
- [x] Model-specific temperature override ✅ (v5.7)
- [x] Configurable temperature / max_completion_tokens ✅ (v5.6)
- [x] Pluggable sandbox backend ✅ (v5.5)
- [x] Lane-based queue serialization ✅ (v5.5)
- [x] Skill progressive disclosure ✅ (v5.5)
- [x] Gateway HTTP server ✅ (v5.5)

---

## License

MIT

---

<p align="center">
  <strong>Built with 🦞 by the ClawAgents team</strong>
</p>
