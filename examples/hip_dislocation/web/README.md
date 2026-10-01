# Hip Report Review

A stateless research web app around the tested ClawAgents `gpt-6-luna` annotation agent.

Upload a values-only `.xlsx` or UTF-8 `.csv`, select a worksheet and report column, optionally select case IDs and reference labels, and run. Or paste reports separated with `---` on its own line. The dashboard shows subtype counts, obturator screening flags, review flags, errors, and expandable verbatim evidence. Reports, predictions, and reference metadata remain available in memory for review until you download the CSV. Starting the download automatically clears the entire session: uploaded files, worksheet data, pasted text, original report bodies, predictions, reference metadata, and dashboard. Manual clearing or leaving also clears the session.

Reference labels are used for browser-side agreement calculations only; they are never submitted to the model. Duplicate case IDs remain separate report rows. This app does not merge observations into patient-level diagnoses. Provided-example agreement is not independent validation.

## Run locally

From the canonical `clawagents_py` repository:

```sh
.venv/bin/python -m examples.hip_dislocation.web.run_local
```

Open `http://127.0.0.1:8096`. The launcher reads the existing parent project `.env` credential, without printing or copying it. Cloud Run uses `OPENAI_API_KEY` from Secret Manager.

## No application data storage

- Uploads are read directly into memory; no multipart spooling or temporary upload files.
- Jobs/results exist only inside the active HTTP request and browser session. No database, object-storage bucket, disk archive, session cache, analytics, or browser local/session storage.
- The agent has only report-read and validated-label tools. Its memory/persistence features are disabled. In-memory text blocks bypass file-backed tool-output archival, including for long reports.
- Framework/provider logging is disabled; HTTP access logging is disabled. Cloud Run infrastructure logs may still record request metadata such as timing and route, not report bodies through this app.
- Responses carry `Cache-Control: no-store`; starting a CSV download automatically clears browser-held reports/results; manual clearing or leaving also clears them. Explicit CSV downloads are user-controlled files.
- OpenAI Responses requests use `store=False`. This does not establish zero provider retention or any institutional compliance arrangement.
- Runtime is a non-root user; application source is read-only to that user. Local container smoke tests also use a read-only root filesystem.

Limits: 5 MB upload, 25 MB expanded workbook, 2.5 MB per XML part, two million expanded cell characters, 10 worksheets, 64 columns, 1,000 source rows per sheet, 100 reports/run, 8,000 characters/report, 200,000 characters/batch. Formula cells are rejected; export values only. Three model calls run concurrently per instance. Cancel stops queued work and cancels active tasks.

## Deployment

```sh
gcloud auth login xjiang2@uth.edu
.venv/bin/python examples/hip_dislocation/web/deploy.py
```

Target: `sbmi-jiang-ai-testing01`, `us-central1`, service `hip-report-review`.

The deployer copies only the explicitly allowlisted source/static files into a build context. Original reports, results, tests, `.env`, and unrelated repository edits are excluded. It creates a dedicated runtime identity with read access to `hip-report-openai-key`, replicates the secret only in `us-central1` to respect the project's resource-location policy, and pins the secret version. Deployment uses `xjiang2@uth.edu`; the app itself is public with IAP and the Cloud Run invoker IAM check disabled. Anyone with the link can upload reports or run inference without Google sign-in. The API key remains server-side in Secret Manager. Billing controls: zero minimum instances, two maximum instances at both service and revision levels, one request/instance, 512 MiB and one vCPU.

The installed Google Cloud CLI exposes these deployment flags through `gcloud beta`. Public access is intentional; do not re-enable sign-in on a later deployment without updating the access instructions.

## Validation

```sh
scripts/run_tests.sh tests/test_hip_dislocation_agent.py tests/test_hip_dislocation_web.py
.venv/bin/ruff check examples/hip_dislocation/agent.py examples/hip_dislocation/web tests/test_hip_dislocation_web.py
node --check examples/hip_dislocation/web/static/app.js
node --test tests/test_hip_dislocation_web_client.cjs
```

The UI includes synthetic examples. They verify the complete flow; their agreement score is not a performance estimate. The prior 42-report benchmark and its evaluation limitations are documented in the parent example folder.
