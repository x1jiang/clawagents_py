# Initial observation-projection experiment (2026-09-12)

This is a public, short-task smoke experiment, not private held-out evidence. All four live OpenAI GPT-6 Astra runs passed the exact numeric verifier. The 1-full-send candidate cost more than the existing aggressive-compression baseline:

| Arm | Successful tasks | Provider-reported total tokens | Estimated list-price cost |
|---|---:|---:|---:|
| Existing compression | 2/2 | 15556 | $0.097956 |
| One full send, then handles | 2/2 | 40262 | $0.431964 |

Both capability and cost were measured separately. The efficiency gate rejected this candidate; it remains opt-in and the default stays unchanged. The larger first observation cost dominated these short tasks. This result does not settle its usefulness on long tasks where first-read evidence might prevent retries.

Configuration: development split of smoke.json, one repeat, model gpt-6-astra, reasoning low, max output1024, max6 iterations and120 seconds per run; candidate observation_full_sends=1. Frozen OpenAI prices: input10, cached input1, cache write12.5, output50 USD per million; only requests up to272000 prompt tokens qualified for those flat rates. Runtime SHA-256: c44274a1c86abd1a0a3db0141d78be380dbf8a1f33f05dec173e02ea1bd65206. Costs derive from provider usage, not invoices; unreported failed-request charges and provider-internal retries are unknown.

Deterministic fixture runs also passed all numeric contracts but used more estimated tokens with one or two full sends. Fixtures cannot establish model quality or billing savings. Mantle routing is covered by mocked integration tests and the official AWS model card; no configured Mantle credential was available for live generation here.
