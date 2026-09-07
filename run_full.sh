#!/usr/bin/env bash
set -euo pipefail

# Resolve config/data paths consistently, even when invoked from another folder.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_ROOT"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash run_full.sh [--setup-only] [config.yaml ...]

Creates/reuses .venv and installs the project with model/test dependencies.
Runs Stage 0 -> Batch A -> Batch B -> combined CSVs, figures and report.
Default configs: fp, rtn3, rtn4, gptq3, awq3 (all configured seeds).
Configure checkpoints, datasets and the original IF verifier first.
GPTQ/AWQ checkpoints must already be quantized and exported.

Optional: PYTHON=/path/to/python bash run_full.sh (Python used to create .venv)
Use --setup-only to install the environment without running experiments.
Config and data paths are relative to the repository root.
HELP
    exit 0
fi

SETUP_ONLY=false
if [[ "${1:-}" == "--setup-only" ]]; then
    SETUP_ONLY=true
    shift
fi

if [[ ! -x "$REPO_ROOT/.venv/bin/python" && ! -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    if [[ -n "${PYTHON:-}" ]]; then
        BOOTSTRAP_PYTHON="$PYTHON"
    elif command -v python3 >/dev/null 2>&1; then
        BOOTSTRAP_PYTHON=python3
    else
        BOOTSTRAP_PYTHON=python
    fi
    "$BOOTSTRAP_PYTHON" -c 'import sys; sys.version_info >= (3, 10) or sys.exit("Python 3.10+ is required")'
    printf 'Creating virtual environment: %s/.venv\n' "$REPO_ROOT"
    "$BOOTSTRAP_PYTHON" -m venv "$REPO_ROOT/.venv"
fi

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/Scripts/python.exe"
else
    printf 'Could not find Python in .venv\n' >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"

printf 'Installing project and dependencies into .venv...\n'
PIP_REQUIRE_VIRTUALENV=true "$PYTHON_BIN" -m pip install --editable '.[models,test]'
"$PYTHON_BIN" -m pip check
if [[ "$SETUP_ONLY" == true ]]; then
    printf 'Environment ready. Run experiments with: bash run_full.sh\n'
    exit 0
fi

CONFIGS=(configs/fp.yaml configs/rtn3.yaml configs/rtn4.yaml configs/gptq3.yaml configs/awq3.yaml)
if (( $# > 0 )); then
    CONFIGS=("$@")
fi

exec "$PYTHON_BIN" -u -m phase1.cli \
    batch --configs "${CONFIGS[@]}" --include-batch-b
