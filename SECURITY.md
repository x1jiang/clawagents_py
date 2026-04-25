# Clawagents Security Policy

This document outlines the security protocols, trust model, and deployment
hardening guidelines for **clawagents** — both the Python package
([`clawagents` on PyPI](https://pypi.org/project/clawagents/),
[x1jiang/clawagents_py](https://github.com/x1jiang/clawagents_py)) and the
TypeScript sibling ([x1jiang/clawagents](https://github.com/x1jiang/clawagents)).
The two packages share a single trust model and the same defence-in-depth
layers, so this policy applies to both unless explicitly noted.

## 1. Vulnerability Reporting

Clawagents does **not** operate a bug bounty program. Security issues should be
reported via [GitHub Security Advisories](https://github.com/x1jiang/clawagents_py/security/advisories/new)
on the relevant repository. Do not open public issues for security
vulnerabilities.

### Required Submission Details

- **Title & Severity:** Concise description and CVSS score/rating.
- **Affected Component:** Exact file path and line range
  (e.g., `src/clawagents/tools/web.py:148-180`).
- **Affected Package(s):** `clawagents` (Python) and/or `clawagents` (npm), with
  package version (`pip show clawagents` / `npm ls clawagents`) and commit SHA.
- **Environment:** OS, Python version (or Node version), and any non-default
  configuration that the PoC depends on.
- **Reproduction:** Step-by-step Proof-of-Concept (PoC) against `main` or the
  latest release, including the exact command and inputs.
- **Impact:** Explanation of what trust boundary was crossed.

---

## 2. Trust Model

The core assumption is that clawagents is a **personal agent** with one trusted
operator. The library is designed to be embedded in operator-controlled
processes and exposed to the operator (and only the operator) via terminal,
gateway, or library calls.

### Operator & Session Trust

- **Single tenant.** The system protects the operator from LLM-issued actions,
  not from malicious co-tenants. Multi-user isolation must happen at the
  OS / host / network level.
- **Gateway security.** The gateway WS server (`clawagents.gateway.server`,
  `src/gateway/`) is fail-closed by default: when bound to a non-loopback
  interface without `GATEWAY_API_KEY`, the server refuses to start and logs a
  loud warning. Loopback (`127.0.0.1` / `::1`) without an API key is permitted
  for local demos but logs an explicit "anonymous-localhost" warning.
- **Execution.** Defaults to local subprocess execution. Container isolation
  (Docker / sandbox) is the operator's responsibility; the library does not
  ship a built-in sandbox backend.

### Permission Mode

`PermissionMode` (`clawagents.permissions.mode` /
`clawagents/permissions/mode.ts`) gates write-class tools at the registry
level:

- `default` — normal behaviour.
- `plan` — read-only exploration. Write-class tools (filesystem writers,
  `execute`/`exec`/`bash`, subagent dispatch) refuse before executing.
- `acceptEdits` — auto-approve write-class edits.
- `bypassPermissions` — disable the gate entirely (break-glass; see § 3).

The write-class registry is defined once in
`clawagents/permissions/mode.py` (`WRITE_CLASS_TOOLS`) and kept in sync with
the TypeScript side.

### Dangerous-Command Denylist

`clawagents.tools.exec` runs a structural denylist before invoking a shell
command (see `_is_dangerous_command` in `tools/exec.py` and
`tools/exec.ts`). This is **not** a security boundary on its own — it is a
last-resort guard against obvious foot-guns (`rm -rf /`, fork-bombs, etc.).
Real isolation must come from the host (containers, unprivileged user, etc.).

### Output Redaction

`clawagents.redact` (`src/clawagents/redact.py` and `src/redact.ts`) strips
secret-like patterns (OpenAI / Anthropic / Google / GitHub / AWS keys, JWTs,
generic high-entropy strings, secret-named env vars) from all display output
*before* it reaches the terminal, gateway platform, log file, or trajectory
recording. This prevents accidental credential leakage in chat logs, tool
previews, response text, and persisted run history.

- The `DiagnosticLogger` in both packages routes every emitted message through
  `redact()`.
- The trajectory recorder (`clawagents.trajectory.recorder`) redacts tool-call
  arguments, output previews, response text, observation context, thinking
  blocks, metadata, and the run-summary task.
- Operators can set `CLAW_REDACT=off` to disable redaction (e.g., for
  debugging) or `CLAW_REDACT=warn` to log a warning each time a redaction
  fires. The default is `on`.
- Redaction operates on the **display layer only**. Underlying values remain
  intact so the agent can still call APIs that need the real key.

### MCP Server Trust

External Model Context Protocol (MCP) servers are treated as **lower trust**
than the host process:

- **stdio MCP servers** (`MCPServerStdio`) receive a **scrubbed environment**
  via `scrub_env_for_stdio` (Python) / `scrubEnvForStdio` (TypeScript). Only a
  small allowlist of safe variables (`PATH`, `HOME`, `USER`, `SHELL`, `TERM`,
  `LANG`, `TZ`, `TMPDIR`, `LC_*`) is inherited from the parent by default.
  Host secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AWS_*`, etc.) are
  dropped unless the operator opts in via `env_allowlist` /
  `envAllowlist`. A debug log lists the secret-named variables that were
  dropped on each spawn.
- **SSE / streamable-HTTP MCP servers** speak over user-supplied URLs; the
  library does not scrub auth headers but does honour the same SSRF rules
  documented in § "Network egress" below when reachable from the agent's
  `web_fetch`.
- The legacy "inherit everything" behaviour is recoverable via
  `CLAW_MCP_INHERIT_ALL_ENV=1` (escape hatch, **not** recommended in
  production).

Operators are expected to vet MCP server packages before configuring them; the
library does not automatically scan `npx`/`uvx` packages for malware.

### Network Egress (web_fetch)

`clawagents.tools.web` (`web.py` / `web.ts`) defends against SSRF when the
agent calls `web_fetch`:

- The hostname is resolved and rejected if it points at loopback, link-local,
  RFC1918 private space, unspecified, multicast, or known cloud-metadata IPs
  (`169.254.169.254`, `fd00:ec2::254`).
- Redirects are followed manually, **not** via the HTTP client's automatic
  redirect handling. Every hop is re-validated, with a hop limit. This blocks
  the classic "public URL → 302 → `http://169.254.169.254/...`" SSRF chain.
- Operators who genuinely need to hit private endpoints can set
  `CLAW_FETCH_ALLOW_PRIVATE=1` to bypass the check.

Note: this filter applies to `web_fetch` only. The agent has unrestricted
network access via the shell `execute` tool by design — see § 3.

### Subagents

`clawagents.subagent` (and the TS equivalent) launches child agents in their
own `RunContext`. Children inherit the parent's tool registry but receive an
isolated trajectory recorder and run-context state. The library does not
enforce a depth limit or "no recursive delegation" policy by default; if you
need that, add it via a `RunHooks` guard.

---

## 3. Out of Scope (Non-Vulnerabilities)

The following scenarios are **not** considered security breaches:

- **Prompt injection** — unless it produces a concrete bypass of the
  permission mode, MCP env scrubbing, redaction, or SSRF filter.
- **Public exposure** — deploying the gateway to the public internet without
  external authentication or network protection. The fail-closed default is
  designed to make this hard, but it is the operator's responsibility to keep
  it that way.
- **Trusted state access** — reports that require pre-existing write access
  to `~/.clawagents/`, `.env`, or `config.toml` (these are operator-owned
  files).
- **Host-level shell access** — the agent has unrestricted shell access via
  the `execute` / `bash` tool by design. Reports that a specific tool can
  reach a resource are not vulnerabilities if the same access is available
  through the shell.
- **Configuration trade-offs** — intentional break-glass settings such as
  `permission_mode: "bypassPermissions"`, `CLAW_REDACT=off`,
  `CLAW_MCP_INHERIT_ALL_ENV=1`, or `CLAW_FETCH_ALLOW_PRIVATE=1`.
- **Tool-level read restrictions without matching write restrictions** —
  per § 2, tool-level deny lists are only a meaningful security boundary when
  paired with equivalent shell-side restrictions.

---

## 4. Deployment Hardening & Best Practices

### Process & filesystem

- **Production sandboxing.** Run clawagents in an unprivileged container
  (Docker, Firejail, etc.) for any workload involving untrusted prompts or
  third-party tools. The library does not provide built-in sandboxing.
- **File permissions.** `chmod 600 ~/.clawagents/.env` (or equivalent on the
  TS side); never commit credentials.
- **Profile-aware home.** `get_clawagents_home()` honours
  `CLAW_PROFILE` so multi-tenant hosts can give each operator their own state
  directory.

### Network

- **Gateway exposure.** Do not bind the gateway/WS server to `0.0.0.0`
  without a real `GATEWAY_API_KEY`. The library refuses by default; do not
  paper over the warning with `--allow-anonymous-public` unless you have
  upstream authn (VPN, Tailscale, Cloudflare Access, …).
- **Egress filtering.** `web_fetch` blocks private IPs by default. Keep this
  enabled unless you need to reach a known-trusted internal endpoint, and
  prefer scoping `CLAW_FETCH_ALLOW_PRIVATE=1` to specific runs rather than
  setting it globally.

### Supply chain

- **Pinned dependencies.** Both packages publish lockfiles
  (`requirements.txt` / `package-lock.json`) and run `npm audit` / `pip-audit`
  in CI. Review supply-chain advisories before bumping.
- **MCP servers.** Vet `npx`/`uvx`/`pipx` packages before configuring them as
  MCP servers, and use `env_allowlist` to pass through only the variables the
  server actually needs.
- **CI/CD.** GitHub Actions in both repos are pinned to commit SHAs.

### Credential storage

- API keys belong in `~/.clawagents/.env` or process environment variables —
  never in `config.toml`/`config.json` or version control.
- The output redaction layer is the last line of defence, but it is *not* a
  substitute for keeping keys out of logs in the first place.

---

## 5. Disclosure Process

- **Coordinated disclosure.** 90-day window or until a fix is released,
  whichever comes first.
- **Communication.** All updates occur via the GHSA thread on the affected
  repository.
- **Credits.** Reporters are credited in release notes unless anonymity is
  requested.
