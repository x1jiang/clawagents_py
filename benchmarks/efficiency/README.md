# Paired efficiency experiments

This runner freezes task content, split membership, model/provider endpoint,
serving limits, pricing, source hashes and acceptance thresholds **before** runs.
Each case/repeat needs exactly one baseline and one candidate. Missing arms,
duplicate retries, changed cases/models/runtimes, invalid usage and incomplete
provider accounting are rejected. Failed tasks contribute their cost to the
cost-per-successful-task denominator. Every baseline success must remain a
candidate success; an improvement elsewhere cannot hide a paired regression.

This is an opt-in experiment, not a default-policy change or a claim of SoL-Pi's
published savings. No extra dependencies are needed.

## Offline smoke run (no provider calls)

From the Python repository, use its Python environment:

```sh
python scripts/benchmark_efficiency.py freeze \
  --suite benchmarks/efficiency/smoke.json --repeats 2 --full-sends 1 \
  --output /tmp/efficiency-fixture-manifest.json
python scripts/benchmark_efficiency.py run \
  --manifest /tmp/efficiency-fixture-manifest.json \
  --suite benchmarks/efficiency/smoke.json --split heldout \
  --output /tmp/efficiency-fixture-rows.jsonl
python scripts/benchmark_efficiency.py compare \
  --manifest /tmp/efficiency-fixture-manifest.json \
  --rows /tmp/efficiency-fixture-rows.jsonl \
  --output /tmp/efficiency-fixture-report.json
```

`compare` exits **2** when data is valid but does not qualify, including **all
fixture runs**. Exit 0 means a qualifying private held-out live comparison;
exit 1 means malformed/incomplete evidence. Outputs are exclusive-create: pick
fresh filenames instead of overwriting an earlier outcome.

The fixture uses the real `ClawAgent`, filesystem tools, edit/verification fusion,
artifact storage, observation projection and exact verifier. Its scripted model
reads a long file, inspects the broken function, runs verification, writes the
reference solution with `then_run`, then reads it back. Reported fixture tokens
are **character-count estimates of actual projected requests**. They are never
provider usage, billing credits, latency predictions or model-quality evidence.
The fixed script may spend *more* estimated tokens under delayed compression.
Its purpose is to detect wiring failures; do not tune it until it looks cheaper.

## Live paired experiments

The `run` command requires `--allow-live` for real providers. Export only the
appropriate provider credential; the CLI disables automatic `.env` discovery.
Use `--provider openai --model <exact-model-id>` or `--provider mantle --model
<exact-Mantle-model-id> --base-url https://bedrock-mantle.<region>.api.aws` when
freezing. Mantle uses its existing Bedrock/Mantle credential resolver. The runner
pins Responses transport, non-streaming, temperature 0, and the frozen token and
reasoning limits. A run is bounded to 12 iterations/120 seconds by default.

Provide `--prices prices.json` with current, independently checked USD rates:

```json
{
  "input_tokens": 0,
  "cached_input_tokens": 0,
  "cache_creation_tokens": 0,
  "output_tokens": 0,
  "max_prompt_tokens": 272000,
  "source": "Replace all rates with the provider's dated pricing source"
}
```

The all-zero placeholder is intentionally **rejected**. Cache-read and
cache-write tokens have independent prices. Set `max_prompt_tokens` to the
largest individual prompt where those rates apply (the example 272000 must
also be checked for your model). Per-request usage is checked against aggregate
totals; a request exceeding the frozen pricing tier prevents a cost verdict. Cost is a reproducible estimate
from provider-reported token categories and these frozen rates, not an invoice.
Provider-internal retries are reported as `null` because the runtime does not
expose them; failed-request charges may be unavailable. Missing usage or an
exception prevents an efficiency verdict. Requested `then_run` calls and
successfully completed fusions are counted separately. Actual response-model
identities must match across each pair, including any provider snapshot aliases.

The default baseline uses `observation_full_sends=0`, candidate uses 1 (or pass
`--full-sends 2`). Keep 0 as the application default until experiments support a
change. Both arms use the same prompt, tool surface, feature configuration,
provider, model and serving limits. Independent in-memory filesystems prevent
state leakage; arm execution order alternates between repeats. Cache isolation
at the provider is not guaranteed: record cache reads/writes and repeat runs in
both orders before interpreting small differences. No auxiliary model is used.

## Development versus held-out evaluation

The bundled public numeric cases are **smoke tasks**. They cannot qualify a
release, even with live usage. For a real evaluation, prepare a private suite
with `evidence_scope: "private-heldout"`, separate `development` and `heldout`
task IDs, and no task reuse between those sets. This field is an operator
attestation, not a secrecy guarantee. Task objects follow `smoke.json`: prompt,
initial source, function name/parameters, exact numeric checks, and an offline
reference solution. The reference and check inputs are never sent to the live
model. The tool's fixed `verify` command reveals only pass/fail.

1. Freeze candidate configurations and run only `--split development` while
   selecting a configuration. Development comparisons never qualify.
2. Select one configuration and archive its immutable manifest before inspecting
   held-out outcomes. Use fresh private cases if held-out results already
   informed a code, prompt, price, model, threshold or configuration change.
3. Run the entire held-out matrix once. Retain timeouts and failures; do not
   cherry-pick repeats or rerun one failed arm and silently replace its record.
4. Compare all rows. Default gates require no paired correctness regression,
   at least 5% lower total cost per successful task, and aggregate latency no
   more than 1.25 times baseline. These thresholds are frozen and configurable.

Source hashes cover every imported package Python source file, including this
harness. Changing code after freezing requires a new manifest. To compare two
package versions, install this same harness into two isolated source trees,
obtain their `source-hash` values, and freeze using `--baseline-sha256` and
`--candidate-sha256`. Run each with `--arm baseline` / `--arm candidate` in its
matching environment, then pass both JSONL files to `compare --rows`. The runtime
check rejects an accidentally imported wrong package. A historical runtime may
run the zero-full-send baseline without the new RunContext field.

The verifier intentionally accepts only a single numeric return expression,
with bounded AST/input size, basic numeric operators and `min`, `max`, `abs`.
Generated code cannot import, perform I/O, loop, access attributes or invoke
arbitrary functions. `execute` is an in-memory stub accepting only `verify`;
no model-generated shell command runs on the host. This small task family is
useful for pipeline validation, but is not representative of general repository
engineering. Extend the case family with an appropriate isolated verifier before
making broad coding claims. Point estimates and four public smoke cases cannot
establish statistical significance or a general performance improvement.

Dated Astra pricing examples are provided in `prices-openai-astra.json` and
`prices-mantle-astra-us-west-2.json`. They were checked on 2026-09-12 against
[OpenAI](https://developers.openai.com/api/docs/models/gpt-6-astra) and
[Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html).
Recheck them before spending; the Mantle example applies to in-region US West
(Oregon) requests. Prompts above 272000 tokens are deliberately rejected by the
flat-rate cost gate because their pricing differs. A different region or
inference mode needs a separate price file and manifest.
