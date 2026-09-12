# Model catalog audit — 2026-09-12

Release: Python 6.20.82 / VS Code 1.0.193. Scope is tool-capable text/coding models on existing provider paths. Existing saved model selections and defaults are preserved. Curated availability is not proof of account entitlement; live discovery can further restrict/extend it.

| Provider | Included or corrected |
|---|---|
| OpenAI | Latest general flagship Astra and GPT-5.6 family already present. Correct GPT-5.4/5.5 full-model context to 1,050,000, output ceiling 128,000, and >272K price cliff. Mini/nano keep 400K. |
| Anthropic direct | Add Fable 5.1, Opus 5, Sonnet 5; also expose Opus 4.8 and Sonnet 4.6. Current Claude 5 effort and request limits are wired through the runtime. Fable 5.1 requires adaptive thinking. Preserve signed thinking and redacted blocks through tool turns and session restore. |
| Gemini | Gemini 3.8 already present; remove rejected sampling, honor low/medium/high thinking, preserve retry config and count billable thought tokens. Correct its exact 1,048,576 input/65,536 output limits. Update 3.6 Flash introductory prices to 0.75/3.75 per million through 2026-12-31. |
| xAI | Add Grok 4.6 with 500K context and low/medium/high/xhigh effort. Direct input/output/cache-read 2/6/.50; ≥200K doubles token rates. |
| Mantle | Add Opus 5, Grok 4.6, MiniMax M2.5, Devstral 2, Qwen3 Coder Next, Nemotron Super 3 and Mistral Large 3. Expand vendor filtering, including existing Moonshot models. |
| Native Bedrock | Add Claude 5 US profiles, Nova 2 Lite and Llama 4 Maverick; correct the invalid dated Opus 4.6 ID to us.anthropic.claude-opus-4-6-v1. |

## Availability and pricing details

- Mantle Grok 4.6: xai.grok-4.6, /openai/v1, us-west-2. Input/output/read 2.20/6.60/.55. Global profile pricing is 2/6/.50; region choices are never rewritten automatically.
- Mantle Opus 5/Sonnet 5: us-east-1, eu-north-1, eu-west-1, ap-southeast-4 (Melbourne), plus us-gov-west-1. Native US inference profiles are separate and not subject to this Mantle restriction.
- Fable 5.1 native requires AWS review retention opt-in configured by the account/project owner. This release does not change retention consent. Its Mantle card lists GovCloud only, so it is omitted from the normal Mantle fallback. Gated Mythos models and specialized image/audio/research APIs are not advertised as ordinary coding models.
- Direct Fable 5.1 input/output/read/write 10/50/.25/12.5; Opus 5=5/25/.5/6.25; Sonnet 5=2/10/.2/2.5. New AWS Claude prices stay unknown because current AWS rates were not verified; direct-provider prices are not substituted.
- Five new generic Mantle models use published US input/output rates. Their cache tiers are unverified, so estimates use full input rates for cache tokens instead of inventing discounts. All prices remain list-price estimates, excluding storage, tools, taxes and negotiated rates.
- For AWS cards stating only 8K/16K/32K output limits, runtime caps conservatively use 8000/16000/32000. Nova 2's documented Converse cap is 65000. This avoids assuming undocumented binary expansion.

- Claude signed tool turns defer destructive context compaction until a tool-free response. Other providers and completed turns retain inline compaction. A long uninterrupted Claude tool chain can reach its context limit before that boundary. Fable 5.1 uses the documented binding-control beta to recover from changed prefixes; signatures are never rewritten.

## Primary sources

- [OpenAI catalog](https://developers.openai.com/api/docs/models/all), [GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4), [GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5)
- [Anthropic current models](https://platform.claude.com/docs/en/models/overview), [prices](https://platform.claude.com/docs/en/about-claude/pricing), [effort](https://platform.claude.com/docs/en/build-with-claude/effort), [signed thinking and compaction](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)
- [Gemini 3.8 migration](https://ai.google.dev/gemini-api/docs/latest-model), [model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash), [pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Grok 4.6](https://docs.x.ai/developers/models/grok-4.6), [xAI rates](https://docs.x.ai/developers/pricing), [AWS Grok 4.6](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-xai-grok-4-6.html)
- [AWS Opus 5](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html), [Sonnet 5](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html), [Fable 5.1](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5-1.html), [retention](https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html)
- [MiniMax](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-minimax-minimax-m2-5.html), [Devstral 2](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-devstral-2-123b.html), [Qwen](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-coder-next.html), [Nemotron](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-nvidia-nemotron-super-3-120b.html), [Mistral Large 3](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-mistral-large-3.html)
- [Nova 2 Lite](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html), [Nova Converse cap](https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-converse-api.html), [Llama 4 Maverick](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-meta-llama-4-maverick-17b-instruct.html), [Opus 4.6 ID](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-6.html)

Verification uses mocked provider transports and real request construction, catalog/picker integration, full regression suites, typecheck/build, clean package auditing and CI. No live generation or new-model account access is claimed by this audit.
