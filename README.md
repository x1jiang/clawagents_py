<p align="center">
  <h1 align="center">🦞 ClawAgents</h1>
  <p align="center"><strong>A lean, full-stack agentic AI framework — ~2,500 LOC</strong></p>
  <p align="center">
    <img src="https://img.shields.io/badge/version-5.24.0-blue" alt="Version">
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

> **Version 5.24.0** — Latest stable release (March 2026)

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

**Four ways to configure** (in priority order):

1. **`create_claw_agent()` parameters** — highest priority, overrides everything
2. **Shell environment variables** — `export OPENAI_API_KEY=sk-...` in `~/.zshrc` (works globally)
3. **`CLAWAGENTS_ENV_FILE`** — set this env var to point to an explicit `.env` file path (useful for CI/Docker/multi-project)
4. **`.env` file** — project-level config, loaded from `cwd/.env` or `cwd/../.env`

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

### 6. Multi-Sample Comparison (GRPO-inspired)

```python
agent = create_claw_agent("gpt-5-mini", rethink=True)
# Run the task 3 times, pick the best based on objective scoring
result = await agent.compare("Fix the bug in app.py", n_samples=3)
print(result["best_result"])   # best answer
print(result["best_score"])    # objective score
print(result["all_scores"])    # all samples with scores
```

### 7. Azure OpenAI

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

### 8. AWS Bedrock (via OpenAI-compatible gateway)

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

### 9. Local Models (Ollama / vLLM / LM Studio)

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

### 10. CLI

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
| `clawagents --task "..."` | Run a single task. Prints a startup banner (`provider=X model=Y env=Z ptrl=...`), executes the agent, prints the result to stdout. |
| `clawagents --trajectory [N]` | Inspect the last N run summaries (default: 1). Shows run ID, model, task, duration, turns, tool calls, score, quality, failure breakdown, verified score, and judge verdict. Requires `CLAW_TRAJECTORY=1`. |
| `clawagents --serve [--port N]` | Start the HTTP gateway server (default port 3000). Endpoints: `POST /chat`, `POST /chat/stream` (SSE), `GET /queue`, `GET /health`. |
| `clawagents --help` | Show all options with examples. |

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

| Feature | ClawAgents v5.23 | DeepAgents | OpenClaw |
|:---|:---:|:---:|:---:|
| ReAct loop | ✅ | ✅ | ✅ |
| Tool loop detection | ✅ **soft + hard** | ❌ | ✅ |
| Efficiency rules (system prompt) | ✅ | ❌ | ❌ |
| Adaptive token estimation | ✅ | ❌ | ❌ |
| Model-aware context budgeting | ✅ | ❌ | ❌ |
| Pluggable sandbox backend | ✅ | ✅ | ✅ |
| In-memory VFS (testing) | ✅ | ❌ | ❌ |
| Sub-agent delegation | ✅ | ✅ | ✅ |
| Planning / TodoList | ✅ | ✅ | ❌ |
| Persistent memory (AGENTS.md) | ✅ | ✅ | ✅ |
| Human-in-the-loop | ✅ | ✅ | ✅ |
| Dangling tool call repair | ✅ | ✅ | ❌ |
| Auto-summarization + offloading | ✅ | ✅ | ✅ |
| Lane-based command queue | ✅ | ❌ | ✅ |
| Gateway HTTP server + SSE | ✅ | ❌ | ✅ |
| Tool access control | ✅ | ❌ | ❌ |
| `think` tool (structured reasoning) | ✅ | ❌ | ❌ |
| LangChain tool adapter | ✅ | N/A | ❌ |
| Streaming with stall detection | ✅ | ❌ | ✅ |
| Trajectory logging + run scoring | ✅ | ❌ | ❌ |
| Consecutive-failure rethink | ✅ | ❌ | ❌ |
| Discrete reward bands (RL-inspired) | ✅ | ❌ | ❌ |
| Weighted execution scoring | ✅ | ❌ | ❌ |
| Truncated JSON repair + retry | ✅ | ❌ | ❌ |
| Model-specific temperature override | ✅ | ❌ | ❌ |
| Gemini 3 thought_signature support | ✅ | ❌ | ❌ |
| Prompt-Time RL (PTRL) — learn from past runs | ✅ | ❌ | ❌ |
| Deterministic verification (exit codes, tests) | ✅ | ❌ | ❌ |
| GRPO-inspired multi-sample comparison | ✅ | ❌ | ❌ |
| Task-type-aware verification | ✅ | ❌ | ❌ |
| RFT-ready transition export | ✅ | ❌ | ❌ |
| Adaptive rethink threshold | ✅ | ❌ | ❌ |
| LLM-as-Judge verification | ✅ | ❌ | ❌ |
| Thinking token preservation (`<think>`) | ✅ | ❌ | ❌ |
| Tool result caching (LRU) | ✅ | ❌ | ❌ |
| JSON Schema param validation + coercion | ✅ | ❌ | ❌ |
| ComposeTool (deterministic pipelines) | ✅ | ❌ | ❌ |
| WebSocket gateway | ✅ | ❌ | ✅ |
| Multi-channel messaging (Telegram, WhatsApp, Signal) | ✅ | ❌ | ✅ |
| Per-session message serialization | ✅ | ❌ | ✅ |

---

## Architecture

### Core Components (~2,500 LOC)

```
clawagents/
├── agent.py            # ClawAgent class — ReAct loop, hooks, compaction
├── __main__.py          # CLI entrypoint
├── config/              # Env-based configuration (incl. TEMPERATURE, CLAW_*)
├── providers/           # LLM backends (OpenAI, Gemini, Fallback)
│   └── llm.py           # max_completion_tokens, temperature override, JSON repair
├── tools/               # 14+ built-in tools
│   ├── filesystem.py    # ls, read_file, write_file, edit_file
│   ├── advanced_fs.py   # tree, diff, insert_lines
│   ├── search.py        # grep, glob
│   ├── execute.py       # Shell command execution
│   ├── planning.py      # write_todos, update_todo
│   ├── delegation.py    # Sub-agent task delegation
│   ├── think.py         # Structured reasoning (no side effects)
│   ├── web.py           # URL fetching with HTML cleanup
│   └── interactive.py   # ask_user (stdin-based)
├── sandbox/             # Pluggable backend protocol
│   ├── protocol.py      # SandboxBackend interface (15+ methods)
│   ├── local.py         # LocalBackend (pathlib + asyncio)
│   └── in_memory.py     # InMemoryBackend (VFS for testing)
├── trajectory/          # RL-inspired run analysis (v5.9+)
│   └── recorder.py      # TrajectoryRecorder, discrete scoring, quality grading
├── gateway/             # Production HTTP server
│   ├── server.py        # FastAPI + SSE streaming
│   └── queue.py         # 4-lane FIFO command queue
├── graph/               # Agent loop orchestration + failure tracking
├── memory/              # AGENTS.md discovery + compaction
├── process/             # Process management
└── logging/             # Structured logging
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
| `instruction` | `str \| None` | `None` | No | System prompt — what the agent should do and how it should behave |
| `tools` | `list \| None` | `None` | No | Additional tools to register. Built-in tools (filesystem, exec, grep, etc.) are always included |
| `skills` | `str \| list \| None` | auto-discover | No | Skill directories to load. Default: checks `./skills`, `./.skills`. Built-in ByteRover skill is always included. |
| `memory` | `str \| list \| None` | auto-discover | No | Memory files to inject into system prompt. Default: checks `./AGENTS.md`, `./CLAWAGENTS.md` |
| `sandbox` | `SandboxBackend` | `LocalBackend()` | No | Pluggable sandbox backend for file/shell operations. Use `InMemoryBackend` for testing |
| `streaming` | `bool` | `True` | No | Enable streaming responses |
| `use_native_tools` | `bool` | `True` | No | Use provider native function calling. Set `False` for text-based JSON tool calls |
| `on_event` | `callable \| None` | `None` | No | Callback for agent events (tool calls, errors, context messages, etc.) |

**LLM Tuning**

| Param | Type | Default | Required? | Description |
|:---|:---|:---|:---:|:---|
| `context_window` | `int \| None` | env `CONTEXT_WINDOW` / `1000000` | No | Token budget. When messages exceed this, older turns are compacted |
| `max_tokens` | `int \| None` | env `MAX_TOKENS` / `8192` | No | Max output tokens per LLM response. Sent as `max_completion_tokens` (OpenAI) or `max_output_tokens` (Gemini) |
| `temperature` | `float \| None` | env `TEMPERATURE` / `0.0` | No | LLM sampling temperature. Automatically overridden for reasoning models (o1/o3/o4-mini, gpt-5/gpt-5-mini/gpt-5-turbo → 1.0). Non-reasoning models (gpt-5-nano, gpt-5-micro, gpt-4o) respect the configured value |
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
| `await agent.invoke(task, max_iterations=None)` | Run the agent on a task. Returns `AgentState` with `.result`, `.status`, `.iterations`, `.tool_calls` |
| `await agent.compare(task, n_samples=3, max_iterations=None)` | Run the task N times and return the best result based on objective scoring (GRPO-inspired). Returns `{"best_result", "best_score", "best_index", "all_scores"}` |
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
| **Skills** | `./skills`, `./.skills`, `./skill`, `./.skill`, `./Skills`. Built-in [ByteRover](https://clawhub.ai/byteroverinc/byterover) skill is always included. **CLI:** when the agent runs `brv`, it is executed via `npx byterover-cli` so Node/npx is sufficient (no global install required). |

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
Loads `AGENTS.md` files and injects content into every LLM call. Use for project-level context and conventions.

### Auto-Compaction
When the conversation exceeds **75% of `CONTEXT_WINDOW`**:
1. Full history **offloaded** to `.clawagents/history/compacted_*.json`
2. Older messages **summarized** into `[Compacted History]`
3. Last 20 messages kept intact

This provides **unlimited conversation length** with full audit trail preservation.

---

## Gateway Server

Launch a production-ready HTTP server with one line:

```python
from clawagents.gateway import start_gateway

start_gateway(port=3000)
```

### Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/chat` | POST | Synchronous agent invocation |
| `/chat/stream` | POST | SSE streaming (events: `queued`, `started`, `agent`, `done`, `error`) |
| `/queue` | GET | Queue status for all lanes |
| `/health` | GET | Health check |

### Lane-Based Concurrency

4 lanes with configurable `max_concurrent` per lane:
- `main` — primary user requests
- `cron` — scheduled tasks
- `subagent` — sub-agent delegation
- `nested` — nested sub-agent calls

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
| `PROVIDER` | auto-detect | No | Hint: `"openai"` or `"gemini"`. Auto-detected from which API key is set |
| `OPENAI_API_KEY` | — | **Yes** *(for OpenAI/Azure)* | OpenAI or Azure API key. **Not needed for local models** — when `OPENAI_BASE_URL` is set, a placeholder is used automatically |
| `OPENAI_MODEL` | `gpt-5-nano` | No | Model name, Azure deployment name, or local model ID (e.g. `llama3.1`) |
| `OPENAI_BASE_URL` | *(unset)* | No | Custom endpoint for OpenAI-compatible APIs: Azure, Bedrock gateway, Ollama, vLLM, LM Studio. Omit to use `api.openai.com` |
| `OPENAI_API_VERSION` | *(unset)* | No | **Azure only.** API version string (e.g. `2024-12-01-preview`). Ignored by all other providers |
| `GEMINI_API_KEY` | — | **Yes** *(for Gemini)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | No | Gemini model name |

**LLM Tuning**

| Variable | Default | Required? | Description |
|:---|:---|:---:|:---|
| `STREAMING` | `1` | No | `1` = streaming enabled, `0` = disabled |
| `CONTEXT_WINDOW` | `1000000` | No | Token budget. Older messages are compacted when exceeded |
| `MAX_TOKENS` | `8192` | No | Max output tokens per response (`max_completion_tokens` for OpenAI, `max_output_tokens` for Gemini) |
| `TEMPERATURE` | `0.0` | No | Sampling temperature. Auto-overridden for reasoning models (o-series + gpt-5/gpt-5-mini/gpt-5-turbo → 1.0). Non-reasoning models (gpt-5-nano, gpt-5-micro, gpt-4o) use the configured value |
| `MAX_ITERATIONS` | `200` | No | Max tool rounds before the agent stops. Override per-run: `agent.invoke(task, max_iterations=N)` |

**PTRL & Trajectory Flags** — all off by default, opt-in with `1`/`true`/`yes`

| Variable | Default | Required? | Description |
|:---|:---|:---:|:---|
| `CLAW_TRAJECTORY` | `0` | No | Enable trajectory logging. Records every turn + scores each run to `.clawagents/trajectories/` |
| `CLAW_RETHINK` | `0` | No | Enable consecutive-failure detection + adaptive rethink injection |
| `CLAW_LEARN` | `0` | No | Enable full PTRL: lesson extraction, injection, LLM-as-Judge, and thinking token preservation. Implies `CLAW_TRAJECTORY=1` |
| `CLAW_PREVIEW_CHARS` | `120` | No | Max chars for tool-output previews in trajectory logs |
| `CLAW_RESPONSE_CHARS` | `500` | No | Max chars for LLM response text in trajectory records |

---

## Testing

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run benchmarks (requires API keys)
python -m pytest tests/ -v -m benchmark
```

---

## Changelog

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
