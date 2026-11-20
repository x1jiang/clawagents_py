# Python API Reference

Back to [Documentation Index](README.md) · [README](../README.md)

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
