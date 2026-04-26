#!/usr/bin/env bash
# Canonical test runner for clawagents_py (pattern learned from hermes-agent).
#
# Run this instead of calling `pytest` directly to guarantee your local run
# matches CI behavior.
#
# What this script enforces:
#   * Pinned xdist worker count (CI is 4-core; -n auto diverges locally)
#   * TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0 (deterministic)
#   * Credential env vars blanked (belt-and-suspenders alongside conftest.py)
#   * Proper venv activation, falling back through common locations
#
# Usage:
#   scripts/run_tests.sh                            # full suite
#   scripts/run_tests.sh tests/agent/               # one directory
#   scripts/run_tests.sh tests/agent/test_foo.py::TestClass::test_method
#   scripts/run_tests.sh --tb=long -v               # pass-through pytest args
#
# Override worker count: CLAW_TEST_WORKERS=8 scripts/run_tests.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Activate venv ───────────────────────────────────────────────────────────
VENV=""
for candidate in \
    "$REPO_ROOT/.venv" \
    "$REPO_ROOT/venv" \
    "$HOME/.clawagents/clawagents/venv"; do
    if [ -f "$candidate/bin/activate" ]; then
        VENV="$candidate"
        break
    fi
done

if [ -z "$VENV" ]; then
    if [ -z "${PYTHON:-}" ]; then
        for candidate in python3 python; do
            if command -v "$candidate" >/dev/null && "$candidate" -c "import pytest, xdist" 2>/dev/null; then
                PYTHON="$candidate"
                break
            fi
        done
        PYTHON="${PYTHON:-python3}"
    fi
    if ! command -v "$PYTHON" >/dev/null; then
        echo "error: no virtualenv found and PYTHON=$PYTHON is not on PATH" >&2
        exit 1
    fi
else
    PYTHON="$VENV/bin/python"
fi

# ── Ensure pytest-xdist is installed (required for parallel runs) ──────────
if ! "$PYTHON" -c "import xdist" 2>/dev/null; then
    echo "→ installing pytest-xdist into $PYTHON"
    "$PYTHON" -m pip install --quiet "pytest-xdist>=3,<4"
fi

# ── Hermetic environment ────────────────────────────────────────────────────
# Pin xdist to a fixed worker count so local runs match CI.
WORKERS="${CLAW_TEST_WORKERS:-4}"

# Strip any credential-shaped env var so leaked secrets cannot influence tests.
while IFS='=' read -r name _; do
    case "$name" in
        *_API_KEY|*_TOKEN|*_SECRET|*_PASSWORD|*_CREDENTIALS|*_ACCESS_KEY| \
        *_SECRET_ACCESS_KEY|*_PRIVATE_KEY|*_OAUTH_TOKEN|*_WEBHOOK_SECRET| \
        *_ENCRYPT_KEY|*_APP_SECRET|*_CLIENT_SECRET|*_AES_KEY| \
        AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN| \
        GH_TOKEN|GITHUB_TOKEN)
            unset "$name"
            ;;
    esac
done < <(env)

# Strip CLAW_* / CLAWAGENTS_* behavioral env so the suite runs against
# defaults regardless of what the developer has loaded in their shell.
for name in $(env | grep -E '^(CLAW|CLAWAGENTS)_' | cut -d= -f1 || true); do
    unset "$name" || true
done

export TZ=UTC
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONHASHSEED=0

cd "$REPO_ROOT"

ARGS=("$@")

echo "▶ running pytest with $WORKERS workers, hermetic env, in $REPO_ROOT"
echo "  (TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0; CLAW_*/credential vars unset)"

exec "$PYTHON" -m pytest \
    -o "addopts=" \
    -n "$WORKERS" \
    --ignore=tests/integration \
    --ignore=tests/e2e \
    -m "not integration" \
    ${ARGS[@]+"${ARGS[@]}"}
