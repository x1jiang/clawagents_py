# Changelog

## 6.20.86

- Add explicit OpenAI Fast mode (`fast_mode=True` on `create_claw_agent`) for supported direct OpenAI API models.
- Generalize tool-summary wording in completion handling.
- Exclude SQL source from ungrounded count checks while retaining checks for unsupported query results.
- Respect SQL-only requests when correcting unsupported counts instead of requiring query execution.
