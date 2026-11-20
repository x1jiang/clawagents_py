# Architecture, Tools & Benchmarks

Back to [Documentation Index](README.md) · [README](../README.md)

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
│   ├── llm.py                # Compatibility facade re-exporting all provider symbols
│   ├── base.py               # LLMProvider ABC, LLMMessage, LLMResponse, Usage
│   ├── stream_retry.py       # Mid-stream retry + assistant_reset divergence handling
│   ├── tool_convert.py       # Provider tool-schema and message conversion helpers
│   ├── openai_provider.py    # OpenAI Chat Completions, Responses, and compatible APIs
│   ├── gemini_provider.py    # Google Gemini (google-genai) implementation
│   ├── anthropic_provider.py # Direct Anthropic Messages API implementation
│   ├── bedrock_provider.py   # AWS Bedrock native IAM + Bedrock Mantle implementations
│   ├── factory.py            # create_provider routing across profiles and model families
│   └── fallback.py           # FallbackProvider — 3-tier failover + quarantine (v6.0)
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
