<p align="center">
  <h1 align="center">🦞 ClawAgents</h1>
  <p align="center"><strong>Lean, full-stack Python agentic AI framework with PTRL &amp; Harness²</strong></p>
  <p align="center">
    <img src="https://img.shields.io/badge/version-6.20.102-blue" alt="Version">
    <img src="https://img.shields.io/badge/python-≥3.10-green" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
  </p>
</p>

---

ClawAgents is a **production-ready agentic framework** for Python that gives LLMs the tools to read, edit, and execute code with built-in planning, memory (`AGENTS.md`), skills (`SKILL.md`), sandboxing, Prompt-Time Reinforcement Learning (**PTRL**), recursive harness optimization (**Harness²**), and an optional HTTP/WebSocket gateway.

It supports **OpenAI**, **Anthropic Claude**, **Google Gemini**, **Amazon Bedrock & Mantle**, **xAI Grok**, **Meta / Muse**, and **local OpenAI-compatible servers** (Ollama, vLLM, LM Studio, Gemma Q4) through a unified provider architecture.

## Ecosystem

| Product | Latest | Description | Links |
| --- | --- | --- | --- |
| **Python SDK (`clawagents`)** | **v6.20.102** | Core agent runtime, CLI, PTRL / Harness², MCP, Gateway, Context Observatory | [PyPI](https://pypi.org/project/clawagents/) · [Releases](https://github.com/x1jiang/clawagents_py/releases) |
| **VS Code / Cursor Extension** | **v1.0.217** | Sidebar chat, concurrent tabs, checkpoints, Harness² optimizer, telemetry | [Repo](https://github.com/x1jiang/clawagents-vscode) · [Releases](https://github.com/x1jiang/clawagents-vscode/releases) |
| **ClawAgents Desktop** | **macOS** | Native macOS app for project chats, SSH remotes, and local gateway | [Repo](https://github.com/x1jiang/clawagents-desktop) · [Releases](https://github.com/x1jiang/clawagents-desktop/releases) |
| **TypeScript SDK** | **Node / TS** | Sibling TypeScript package (`npm install git+https://github.com/x1jiang/clawagents.git`) | [Repo](https://github.com/x1jiang/clawagents) |

## Installation

```bash
pip install -U clawagents                       # Core (OpenAI + OpenAI-compatible)
pip install -U 'clawagents[gemini]'             # + Google Gemini
pip install -U 'clawagents[anthropic]'          # + Anthropic Claude
pip install -U 'clawagents[bedrock]'            # + AWS Bedrock (IAM / Converse) & Mantle
pip install -U 'clawagents[mcp,media,pty]'      # + MCP client, media resizing, PTY tools
pip install -U 'clawagents[gateway]'            # + FastAPI / Uvicorn HTTP+WS gateway
pip install -U 'clawagents[browser]'            # + Playwright browser tools (then: playwright install chromium)
pip install -U 'clawagents[all]'                # All providers, gateway, observatory, MCP, browser, PTY
```

Other optional extras: `accurate-tokens` (`tiktoken`), `observatory` (Streamlit Context Observatory), `cron`, `watch`, `rl`, `acp`, and `dev`.

## 30-Second Quick Start

### CLI

```bash
clawagents --init                 # Scaffolds .env, run_agent.py, and AGENTS.md
# Add an API key to .env
clawagents --doctor               # Verify config, keys, and model setup
clawagents --task "Summarize this repository and run the test suite"
```

### Minimal Python

```python
import asyncio
from clawagents import create_claw_agent

async def main():
    # Reads .env (or CLAWAGENTS_ENV_FILE) and auto-discovers AGENTS.md + ./skills/
    agent = await create_claw_agent()
    result = await agent.invoke("List the files in this project and summarize the architecture")
    print(result.result)

asyncio.run(main())
```

## Supported Providers & Models

| Provider | Extra / Setup | Highlights |
| --- | --- | --- |
| **OpenAI** | core (`OPENAI_API_KEY`) | GPT-6 / GPT-6.1 Terra, Sol, Luna, Astra (`Responses` API, reasoning effort, Fast mode), Azure OpenAI |
| **Anthropic** | `clawagents[anthropic]` | Claude Fable 5.2, Claude 5.5 (Opus / Sonnet / Haiku), adaptive & signed thinking, direct Fast mode |
| **Google Gemini** | `clawagents[gemini]` | Gemini 4 Hydrogen (`xhigh` thinking), Carbon, Argon, Gemini 3.8 Flash |
| **AWS Bedrock & Mantle** | `clawagents[bedrock]` | Native IAM / Converse profiles (`us.`, `global.`, etc.) and Mantle bearer endpoints |
| **xAI Grok** | core (`XAI_API_KEY`) | Grok 4.6 (`500K` context, reasoning effort) |
| **Meta / Muse** | core (`MODEL_API_KEY` / `profile="meta"`) | Hosted Meta Model API (`muse-spark-*`, `muse-wonder-*`, `muse-ember-*`) & self-hosted [Muse-Glimmer-30B](docs/meta-glimmer.md) |
| **Local Models** | core (`OPENAI_BASE_URL`) | Ollama, vLLM, LM Studio, and local [Gemma Q4 coordinator](docs/gemma-coordination.md) |

See [docs/model-catalog-audit.md](docs/model-catalog-audit.md) for verified model IDs, context limits, request rules, and pricing sources.

## Key Capabilities

- **Built-in coding tools:** Filesystem (`ls`, `read_file`, `write_file`, `edit_file`, `grep`, `glob`), sandboxed shell `execute`, `think`, structured `todolist`, isolated `subagent`, `web_fetch` / `web_search`, and optional Playwright browser tools.
- **PTRL & Harness²:** Trajectory logging, rethink-on-failure, cross-run lesson extraction (`CLAW_LEARN`), and Propose–Probe–Compose workspace harness optimization with rollback (`clawagents --adapt` / `--rollback-harness`).
- **Skills & Memory:** Progressive-disclosure `SKILL.md` skills with tool allowlists, `AGENTS.md` project memory, pinned context, and automatic context compaction.
- **MCP & Gateway:** Connect external MCP servers (`stdio`, SSE, Streamable HTTP) and expose agents via a loopback-authenticated FastAPI/WebSocket gateway (`clawagents --serve`).
- **Security by default:** Command allowlists/blocklists, macOS Seatbelt / workspace sandboxing, SSRF-checked `web_fetch`, and explicit workspace profile trust gates. See [SECURITY.md](SECURITY.md).

## Configuration & Core Environment Variables

ClawAgents looks for `.env` in `./.env`, parent directories, and `~/.clawagents/.env`, or at an explicit path via `CLAWAGENTS_ENV_FILE`.

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI API key and model name |
| `OPENAI_BASE_URL` / `OPENAI_API_VERSION` | Custom OpenAI-compatible endpoint (Ollama/vLLM/Azure) and Azure API version |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Google Gemini API key and model name |
| `MAX_TOKENS` / `TEMPERATURE` | Output token cap and sampling temperature (omitted automatically for reasoning models) |
| `CONTEXT_WINDOW` / `STREAMING` | Override context budget (tokens) and enable/disable response streaming (`1` / `0`) |
| `CLAW_TRAJECTORY` / `CLAW_RETHINK` / `CLAW_LEARN` | Enable trajectory logging, failure rethink, and PTRL cross-run lesson extraction (`1` / `0`) |
| `CLAW_PREVIEW_CHARS` / `CLAW_RESPONSE_CHARS` | Truncation thresholds for tool previews and stored trajectory responses |
| `CLAWAGENTS_ENV_FILE` | Explicit path to a `.env` file (overrides automatic `.env` discovery) |

See [docs/configuration.md](docs/configuration.md) for provider recipes (Bedrock, Azure, Local, MCP, Browser, Cron, Advisor, GRPO) and the full environment variable reference.

## CLI Quick Reference

```bash
clawagents --init                    # Scaffold starter .env, run_agent.py, AGENTS.md
clawagents --doctor                  # Check environment, credentials, and endpoints
clawagents --task "..."              # Run a one-shot task
clawagents --trajectories            # Inspect recorded run summaries
clawagents --adapt                   # Run Harness² workspace optimization
clawagents --rollback-harness        # Restore workspace from Harness² backup
clawagents --serve --port 8080       # Start the HTTP/WebSocket gateway
```

See [docs/cli-and-gateway.md](docs/cli-and-gateway.md) for full CLI flags, gateway authentication, and channel allowlists.

## Documentation

| Guide | Topic |
| --- | --- |
| [docs/README.md](docs/README.md) | Full documentation index |
| [docs/configuration.md](docs/configuration.md) | Provider recipes (Bedrock, Azure, Local, MCP, Browser, PTRL, Advisor, GRPO) & all env vars |
| [docs/api-reference.md](docs/api-reference.md) | `create_claw_agent(...)` parameters, hooks, instance methods, and auto-discovery |
| [docs/cli-and-gateway.md](docs/cli-and-gateway.md) | CLI reference, gateway server, auth, channel allowlists, and concurrency lanes |
| [docs/architecture.md](docs/architecture.md) | Architecture, built-in & custom tools, sandbox backends, feature matrix, and benchmarks |
| [docs/skills.md](docs/skills.md) | Writing & loading `SKILL.md` skills, bundled skills, and `AGENTS.md` memory |
| [docs/model-catalog-audit.md](docs/model-catalog-audit.md) | Verified model catalogs, context windows, thinking contracts, and pricing sources |
| [docs/meta-glimmer.md](docs/meta-glimmer.md) | Self-hosted Meta / Muse-Glimmer-30B setup and benchmarks |
| [docs/gemma-coordination.md](docs/gemma-coordination.md) | Local Gemma Q4 coordinator setup and verification |
| [docs/efficiency.md](docs/efficiency.md) | Agent efficiency mechanics (fused follow-ups, paged recall, diagnostic reduction) |
| [SECURITY.md](SECURITY.md) | Security architecture, trust boundaries, and vulnerability reporting |
| [CHANGELOG.md](CHANGELOG.md) | Release notes (`v6.20.80`–present; older history in [docs/changelog-archive.md](docs/changelog-archive.md)) |

## Testing & Development

```bash
pip install -e ".[dev]"
pytest tests/ -q                 # Full unit test suite
./scripts/run_tests.sh           # Hermetic CI-parity runner
ruff check .                     # Linter
mypy                             # Static type check
```

## License

MIT — see [LICENSE](LICENSE).
