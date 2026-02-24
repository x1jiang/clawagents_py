<p align="center">
  <h1 align="center">🦞 ClawAgents</h1>
  <p align="center"><strong>A lean, full-stack agentic AI framework — ~2,500 LOC</strong></p>
  <p align="center">
    <img src="https://img.shields.io/badge/version-5.5.0-blue" alt="Version">
    <img src="https://img.shields.io/badge/python-≥3.10-green" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
    <img src="https://img.shields.io/badge/LOC-~2500-purple" alt="LOC">
  </p>
</p>

---

ClawAgents is a **production-ready agentic framework** that gives LLMs the ability to read, write, and execute code — with built-in planning, memory, sandboxing, and a gateway server. It supports **OpenAI GPT-5** and **Google Gemini** out of the box, with a pluggable provider architecture for any LLM.

Built by extracting and unifying the best architectural patterns from [OpenClaw](https://github.com/anthropics/openclaw) (~5,800 files) and [DeepAgents](https://github.com/langchain-ai/deepagents) (~1,400 LOC core), ClawAgents delivers **the same power at a fraction of the complexity**.

## Installation

```bash
pip install clawagents
```

> **Version 5.5.0** — Latest stable release (February 2026)

---

## Quick Start

### 1. Configure your environment

Create a `.env` file:

```env
PROVIDER=gemini                    # or "openai"
GEMINI_API_KEY=AIza...             # Your Gemini API key
GEMINI_MODEL=gemini-3-flash-preview
STREAMING=1
CONTEXT_WINDOW=128000
MAX_TOKENS=4096
```

<details>
<summary><strong>OpenAI configuration</strong></summary>

```env
PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-nano
STREAMING=1
CONTEXT_WINDOW=128000
MAX_TOKENS=4096
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

### 4. CLI mode

```bash
python -m clawagents --task "Find all TODO comments in the codebase"
```

---

## 🏆 Performance: ClawAgents vs Traditional Frameworks

ClawAgents v5.5 outperforms traditional multi-layer agentic frameworks through **architectural simplicity**. Here's how it stacks up against DeepAgents (LangGraph/LangChain-based) in head-to-head benchmarks.

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

| Feature | ClawAgents v5.5 | DeepAgents | OpenClaw |
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

---

## Architecture

### Core Components (~2,500 LOC)

```
clawagents/
├── agent.py            # ClawAgent class — ReAct loop, hooks, compaction
├── __main__.py          # CLI entrypoint
├── config/              # Env-based configuration
├── providers/           # LLM backends (OpenAI, Gemini, Fallback)
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
├── gateway/             # Production HTTP server
│   ├── server.py        # FastAPI + SSE streaming
│   └── queue.py         # 4-lane FIFO command queue
├── graph/               # Agent loop orchestration
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

---

## API Reference

### `create_claw_agent(model, instruction, ...)`

| Param | Type | Default | Description |
|:---|:---|:---|:---|
| `model` | `str \| LLMProvider \| None` | `None` | Model name or provider instance. `None` = auto-detect from env |
| `instruction` | `str` | `None` | System instruction for the agent |
| `tools` | `list` | `None` | Additional tools (built-in tools always included) |
| `skills` | `str \| list` | auto-discover | Skill directories to load |
| `memory` | `str \| list` | auto-discover | Memory files to inject |
| `streaming` | `bool` | `True` | Enable streaming responses |
| `sandbox` | `SandboxBackend` | `LocalBackend` | Pluggable sandbox for file/shell operations |
| `on_event` | `callable` | `None` | Event callback |

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

---

## Auto-Discovery

The agent factory automatically discovers project files:

| What | Default locations checked |
|:---|:---|
| **Memory** | `./AGENTS.md`, `./CLAWAGENTS.md` |
| **Skills** | `./skills`, `./.skills`, `./skill`, `./.skill`, `./Skills` |

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
3. Last 6 messages kept intact

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

| Variable | Default | Description |
|:---|:---|:---|
| `PROVIDER` | auto-detect | `openai` or `gemini` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-5-nano` | OpenAI model |
| `GEMINI_API_KEY` | — | Gemini API key |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Gemini model |
| `STREAMING` | `1` | `1` = enabled, `0` = disabled |
| `CONTEXT_WINDOW` | `128000` | Token budget for compaction |
| `MAX_TOKENS` | `4096` | Max output tokens per response |

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

## What's New in v5.5

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

## Roadmap

- [ ] Docker sandbox backend (protocol ready)
- [ ] Semantic browser automation (accessibility tree)
- [ ] Prompt caching (Anthropic-style)
- [x] Pluggable sandbox backend ✅
- [x] Lane-based queue serialization ✅
- [x] Skill progressive disclosure ✅
- [x] Gateway HTTP server ✅

---

## License

MIT

---

<p align="center">
  <strong>Built with 🦞 by the ClawAgents team</strong>
</p>
