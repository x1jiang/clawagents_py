# CLI & Gateway Server

Back to [Documentation Index](README.md) · [README](../README.md)

## CLI

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

## Gateway Server

The gateway is an optional extra: `pip install 'clawagents[gateway]'` (adds
FastAPI and uvicorn). Launch an HTTP server with one line:

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
`/chat`, `/chat/stream`, and `/ws`. `clawagents serve` follows the same rule
whether or not messaging channels are configured.

Requests from a browser are origin-checked. A `POST` or WebSocket connection
that carries an `Origin` header is refused (HTTP 403 / WS close 4003) unless
the origin is listed in `GATEWAY_CORS_ORIGINS` (default
`http://localhost:3000,http://127.0.0.1:3000`) or is the gateway's own
loopback address. Clients that send no `Origin` — curl, SDKs, server-side
code — are unaffected. This stops a web page you happen to visit from
starting agent runs on an unauthenticated local gateway.

### Messaging channels: sender allowlist

`clawagents serve` auto-starts Telegram, WhatsApp and Signal adapters when
their env vars are set (`TELEGRAM_BOT_TOKEN`, `WHATSAPP_AUTH_DIR` /
`WHATSAPP_API_URL`, `SIGNAL_ACCOUNT`). A channel message runs an agent with
shell and file tools on the host, so each channel **refuses every sender
until its allowlist is set**:

| Channel | Env var |
|---------|---------|
| Telegram | `TELEGRAM_ALLOWED_SENDERS` |
| WhatsApp | `WHATSAPP_ALLOWED_SENDERS` |
| Signal | `SIGNAL_ALLOWED_SENDERS` |

Each is a comma-separated list of sender ids or conversation/group ids. A
refused sender is logged once to stderr with its id, so you can message the
bot, copy the id from the log, and add it. Set a list to `*` only for a bot
you intend to be public.

```bash
TELEGRAM_BOT_TOKEN=123456:ABC... TELEGRAM_ALLOWED_SENDERS=123456789 clawagents serve
```

`ChannelRouter` used directly is unchanged: pass your own `on_inbound`
callback (or `clawagents.channels.auto.make_sender_gate`) to gate senders.

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
