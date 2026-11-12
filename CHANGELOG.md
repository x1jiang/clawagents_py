# Changelog

## 6.20.98

Security hardening. **Breaking for channel users:** messaging channels now refuse every sender until an allowlist is set.

- **Channels:** `clawagents serve` installs a deny-by-default sender gate. Set `TELEGRAM_ALLOWED_SENDERS`, `WHATSAPP_ALLOWED_SENDERS` or `SIGNAL_ALLOWED_SENDERS` to a comma-separated list of sender or conversation ids (`*` to accept anyone). Refused senders are logged once with their id. Debounced messages are batched per sender instead of per conversation, so a batch can no longer mix senders.
- **Gateway bind:** `clawagents serve` with channels configured no longer binds `0.0.0.0`; it uses the same loopback default and `GATEWAY_HOST` override as `start_gateway`, with the same no-auth warning.
- **Gateway origin check:** state-changing HTTP requests and `/ws` connections carrying an `Origin` header are refused unless the origin is in `GATEWAY_CORS_ORIGINS` or is the gateway's own loopback address. Requests without `Origin` are unaffected. API keys are compared in constant time.
- **CodeAct:** the AST gate rejects any underscore-prefixed attribute (previously dunders only) and bare `except:`; the tools proxy is read-only and no longer exposes the registry or `before_tool` as attributes, closing a permission-gate bypass. `timeout_s` is now enforced on the snippet's own execution time.
- **Seatbelt:** the profile is passed inline via `sandbox-exec -p` instead of being written to `<cwd>/.clawagents/seatbelt.sb`, which the sandbox itself could overwrite.
- **Bash validator:** classify by program basename (`/bin/rm`), look through `{ …; }`, `!` and `if`/`while` bodies, unwrap here-strings fed to a shell, block redirects into privileged config paths, treat `/.` and `../..` as root-like, and block or warn on a variable used as the program name.

Models (details and sources in `MODEL_CATALOG_AUDIT.md`):

- Add **Gemini 4 Carbon** (`gemini-4-carbon`), **GPT-6 Cyber** (`gpt-6-cyber`, gated) and the hosted **Muse** family on the Meta Model API (`muse-spark-1.1/1.2/1.3` and Contributor tiers, `muse-wonder-1.0`, `muse-ember-1.0`) through a new `meta-api` provider profile (`MODEL_API_KEY`). A bare Muse id selects that profile; it never falls back to another vendor's key.

Harness² (from the reference implementation, github.com/google-research/harness2):

- New `clawagents.trajectory.harness_recursion`: the **Propose–Probe–Compose** loop that the existing surface/prompt/selection/library helpers were missing. `propose_probe_compose(...)` runs `parallel` (k bases, 2k INV/DOM candidates, k compositions) or `sequential` (k rounds, each building on the last checkpoint) at 4k rollouts / 3k edit stages, with omitted-is-inherited edits, `KEEP_BASE`, one retry carrying the rejection text, fallback to the parent tree, blind seeded selection among the compositions, a content-addressed proposer cache, a post-adoption reflection prompt and `record_experience`. Rollout and proposal are caller-supplied callables.
- `validate_harness_tree` now validates the hooks file (it was declared on `SurfaceSpec` but never checked) and can reject off-surface paths (`reject_off_surface`). `accept_or_repair_tree` accepts a `repair` callback with bounded attempts. Autotrigger descriptions are capped at 600 chars. `ExperienceLibrary` gains `retain_before` (a task sees only earlier tasks' entries), `context_dir`, and a harness inventory line in `INDEX.md`. `lint_task_prompt` flags measurement words in task text.

Structure:

- `providers/llm.py` (5,100 lines) is split into one module per provider family — `base`, `stream_retry`, `tool_convert`, `openai_provider`, `gemini_provider`, `anthropic_provider`, `bedrock_provider`, `factory` — layered without import cycles. `clawagents.providers.llm` remains the import path for every public and private name, and assigning a name on it (as tests and the VS Code sidecar's Gemini shim do) still reaches the implementation. No behaviour change; classes now report their new `__module__`.

Fixes found while clearing the type checker (mypy is now clean and runs in CI):

- **`web_fetch` over HTTPS never worked:** it passed `server_hostname=` to `http.client.HTTPSConnection`, which raises `TypeError`. It now dials the pinned IP and verifies the certificate against the real hostname.
- Bedrock streaming awaited a synchronous chunk callback (`TypeError`); `ToolResultCache(persist_denylist=True)` crashed; two `None` iterations in the Responses tool conversion.
- `openai_ssl_verify=False` is no longer silently ignored when only `httpx2` is installed (newer OpenAI SDKs), and the Context Observatory package imports without `httpx`.

Packaging — **breaking for gateway users:** `fastapi`, `uvicorn` and `plotly` are no longer installed with the core package. Install `clawagents[gateway]` for `clawagents --serve` / `clawagents.gateway`, and `clawagents[observatory]` (streamlit + plotly) for the Context Observatory dashboard; `clawagents[all]` includes both. Running `--serve` without the extra exits with the install command. `from clawagents.gateway import start_gateway` now works as documented (the package had no `__init__`).

Runtime robustness:

- **`execute` / `LocalBackend.exec`:** a command that leaves a background child holding its output pipe (`server & echo started`) no longer blocks until that child exits. The process's own exit is detected directly (asyncio's `Process.wait()` also waits for the pipes), output is drained for up to 1s, and a note is appended to stderr when the pipe was still held.
- **Cancellation kills the command:** `LocalBackend.exec` now kills the process group when the task is cancelled, not only on timeout. External hooks run in their own process group and are killed on timeout or cancellation.
- **Lane commands are cancellable:** cancelling the caller of `enqueue_command_in_lane` now cancels the running command (or removes it if still queued), and a cancelled command releases its lane slot. Previously the agent kept running after an SSE client disconnected, and a cancelled command could block a lane permanently.
- **WebSocket `chat.send`:** runs as a task so the socket keeps reading; a disconnect or a failed send cancels the turn. Ending a turn can no longer block on a full event queue.
- **Retried streams:** a stream retried after a mid-stream error no longer re-sends text already delivered. If the retry produces different text, an `assistant_reset` stream event tells consumers to discard the partial answer first.
- **bwrap:** the wrapper no longer creates an empty `.env` (or other placeholder) in the workspace; only secret files that exist are masked.
- `test_syntax_gate_catches_duplicate_js_syntax` skips when `node` is not installed.

## 6.20.97

- Adopt Harness² (*"Recursive Agent Harnessing for an Open World"*, arXiv:2610.14224) capabilities in `clawagents.trajectory` and `clawagents.skills`:
  - Add `harness_surface` with `SurfaceSpec`, recursive accept-or-repair harness tree validation (`validate_harness_tree`, `accept_or_repair_tree`), `.clawagents/skill-triggers.md` autotrigger enforcement (`autotrigger_skills`, `ensure_workspace_skill_trigger`), causal `<edits>` attribution (`EditAttribution`, `ExperienceEntry`) with non-confounded evidence gating, and per-domain `ExperienceLibrary` (`INDEX.md` + `harnesses/<task>/EXPERIENCE.md`).
  - Harden `SkillWorkshopScanner` and `SkillWorkshopService` to enforce frontmatter `name:` directory alignment, `.py` compilation checks, `SKILL.md` helper script existence checks across existing skill files, grader-verdict leakage guards, and collision-free rollback snapshots.
  - Separate `[INV]`, `[DOM]`, and `[WORKSPACE_FACT]` reflection dimensions in `trajectory.lessons` and `lesson_promotion` so workspace-specific facts remain in workspace memory rather than leaking into portable domain skills, while raising observation context to 1200 chars and enforcing normalized "cut-to-pay" deduplication.
  - Fix `compare_samples` / `best_of_n` empty-deliverable scoring bias (`TODO`/placeholder penalty and substantive deliverable bonus) and add blind anonymized deliverable selection (`select_best_candidate_blind`, `blind_select=True`).

## 6.20.96

- Add GPT-6.1 Terra model profile, reasoning, Fast mode, and Responses tool routing.
- Recognize provider overload messages without an HTTP status as retryable server errors.

## 6.20.89

- Add full model profile, effort, and request-building support for **Gemini 4 Argon** (`gemini-4-argon`, 1,048,576 input / 65,536 synchronous output tokens, `thinking_level` `low` / `medium` / `high`, rejected sampling parameters omitted) and **Claude 5.5** (`claude-opus-5-5`, `claude-sonnet-5-5`, and `claude-haiku-5-5`, 1,000,000 input / 128,000 output tokens, plus `long_context_threshold=100_000` on Haiku 5.5).
- Support `claude-haiku-5-5`, `claude-opus-5-5`, and `claude-sonnet-5-5` across direct Anthropic API, native AWS Bedrock (`bedrock-runtime`, including `us.`, `eu.`, `apac.`, `au.`, `jp.`, and `global.` inference profiles), and AWS Bedrock Mantle (`bedrock-mantle.<region>.api.aws/anthropic`).
- Omit rejected `temperature` / `top_p` / `top_k` sampling parameters across all Claude 5+ families (`fable`, `opus`, `sonnet`, `haiku`, including dotted `haiku-5.5` aliases). Keep adaptive thinking always on for `claude-opus-5-5` and `claude-sonnet-5-5` (`effort="none"` lowers effort to `"low"` without sending rejected `thinking: {"type": "disabled"}`), while `claude-haiku-5-5` sends `thinking: {"type": "disabled"}` when `effort="none"`.
- Recognize `au.` and `jp.` Bedrock geographic inference profile prefixes in `is_bedrock_model_id` and `resolve_model_profile`.

## 6.20.88

- Bridge GPT-6 Astra and GPT-6.1 Sol tool calls to Responses even when `wire_api="chat_completions"`; the OpenAI API rejects their Chat Completions function tools at every reasoning effort. A one-time warning names the reroute. Calls without tools keep the configured wire. Azure deployments keep their explicit wire and get the existing clear error.
- When an endpoint does not serve Responses, these tool calls fail with a clear error instead of falling back to Chat Completions.
- Live-verified all four GPT-6 models (Astra, Sol, Luna, 6.1 Sol) across default / `none` / `high` effort, Fast mode and Chat Completions configuration: 20/20 tool-calling agent runs pass.

## 6.20.87

- GPT-6 Astra now fails fast with a clear error when tools are requested over Chat Completions, and never falls back from Responses to Chat Completions with tools. The live API rejects Astra tool calls on Chat Completions at every reasoning effort (same contract as GPT-6.1 Sol).
- Stream retry warnings include the underlying error type and message.
- Exclude `.clawagents/` runtime state from the sdist.
- Live-verified GPT-6 Astra, Sol, Luna and GPT-6.1 Sol: tool-calling agent runs on default, `none`, `high` effort and Fast mode; Chat Completions tool runs for Sol and Luna.

## 6.20.86

- Add explicit OpenAI Fast mode (`fast_mode=True` on `create_claw_agent`) for supported direct OpenAI API models.
- Generalize tool-summary wording in completion handling.
- Exclude SQL source from ungrounded count checks while retaining checks for unsupported query results.
- Respect SQL-only requests when correcting unsupported counts instead of requiring query execution.
