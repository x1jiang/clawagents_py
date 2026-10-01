# Hip-dislocation report agent

A small research classifier using the canonical `clawagents_py` source and exactly `gpt-6-luna` through the OpenAI Responses API, with low reasoning effort. It retrieves one imaging report, then submits a subtype, a review flag, an obturator screening flag, an exact evidence quote, and a brief explanation. The caller validates the quote before accepting the output.

Each report has a separate agent, tool registry, and context. The agent has no filesystem, shell, web, reference-label, rationale, patient-ID, cross-report, skill, or memory tool. Its only tools are `read_report` and `submit_label`. Source linking IDs and reference labels are joined locally after inference, for scoring. The OpenAI request uses `store=False`; this alone does not establish a particular retention or institutional compliance arrangement.

## Run

Use the existing repository environment, with `OPENAI_API_KEY` available in the process environment. No new dependencies are required.

```sh
.venv/bin/python examples/hip_dislocation/agent.py \
  --input '/absolute/path/Obturator Dislocation AI Training Excel Sheet.xlsx' \
  --output '/absolute/path/new-results-folder' \
  --concurrency 3
```

The input adapter expects Jack's original five-column XLSX layout. For a pilot, use `--limit 5` and a separate results folder. Completed report receipts are reused when restarting the same run; changed source, prompt, model, code, or report count requires a new folder.

The tool transmits only the report text and generic annotation instructions to `https://api.openai.com/v1`. It does not transmit the spreadsheet, Jack's rationale, linking IDs, or reference labels. Local results retain report text and linking IDs for audit.

## Results

- `manifest.json`: frozen prompt, schema, source/code hashes, model, settings, and scope.
- `predictions/report_NNN.json`: individual model outputs, quote, usage, and latency.
- `predictions.csv`: row-level comparison with supplied labels.
- `records.json`: linked-record aggregation; conflicting supported subtypes require review.
- `metrics.json`: agreement over all examples, coverage, selective agreement, class precision/recall/F1, confusion matrices, screening capture, and token-based estimated standard API cost.

An uncertainty flag is not a calibrated probability. An unknown label or failed result remains in the all-example denominator. Report and linked-record results must be considered separately because repeated imaging is correlated. Linked-record aggregation uses all supplied reports, not the first report in time.

This is a blinded comparison against Jack's selected training examples, not an independent validation dataset. There are no other/unknown reference cases and only one anterior-superior case. Do not estimate population incidence or broad clinical performance from this benchmark. For validation, freeze the prompt and tool, obtain independently adjudicated held-out reports, and split by linked record.

## Verify

```sh
scripts/run_tests.sh tests/test_hip_dislocation_agent.py
.venv/bin/python -m ruff check examples/hip_dislocation/agent.py tests/test_hip_dislocation_agent.py
```

Tests verify report-before-submission, literal quote validation, invalid outputs, scoring denominators, conflicting linked-record aggregation, and the actual ClawAgents two-tool loop with an offline provider. They make no external API calls.

## Local configured launcher and exports

`run_local.py` uses the existing canonical parent `.env` OpenAI key in preference to an inherited environment key. Credentials are never printed or written into run artifacts. It performs the same model-access preflight before inference:

```sh
.venv/bin/python examples/hip_dislocation/run_local.py --input '/absolute/path/source.xlsx' --output '/absolute/path/new-results'
```

The saved final development run is `auto_documents/jack_zamen_hip_dislocation/analysis_2026-09-30/luna_agent_test_v3/`. The initial conservative run is `luna_agent_test_v2/`. The clarified policy accepts supported femoral-head displacement directions without requiring an explicit subtype name. Both runs are retained with frozen prompt and code hashes; refinement used this same selected dataset.

With existing runtimes containing openpyxl and matplotlib, use `report_results.py RUN_DIR` to build the workbook and `plot_results.py ANALYSIS_DIR` to plot both runs. Recalculate the workbook, then run `verify_results.py RUN_DIR`. Verification independently checks model receipts, source preservation, hashes, cached metrics, both confusion matrices, F1, and Excel errors.

Official model/API/pricing source: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).
