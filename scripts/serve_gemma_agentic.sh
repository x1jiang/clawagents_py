#!/usr/bin/env bash
# Use the GGUF's own template; no hand-written tool grammar or prompt rewrite.
set -euo pipefail
if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "Usage: $0 /path/to/gemma4-v2-Q4_K_M.gguf" >&2
  exit 2
fi
exec llama-server \
  --model "$1" --alias gemma4-agentic-v2 \
  --ctx-size 16384 --parallel 1 --n-gpu-layers 99 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 64 --repeat-penalty 1.1 \
  --host 127.0.0.1 --port "${GEMMA_AGENTIC_PORT:-18080}"
