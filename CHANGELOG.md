# Changelog

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
