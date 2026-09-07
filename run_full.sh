#!/usr/bin/env bash
set -euo pipefail

# Resolve config/data paths consistently, even when invoked from another folder.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_ROOT"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash run_full.sh [config.yaml ...]

Runs Stage 0 -> Batch A -> Batch B -> combined CSVs, figures and report.
Default configs: fp, rtn3, rtn4, gptq3, awq3 (all configured seeds).
Configure checkpoints, datasets and the original IF verifier first.
GPTQ/AWQ checkpoints must already be quantized and exported.

Optional: PYTHON=/path/to/python bash run_full.sh
Config and data paths are relative to the repository root.
HELP
    exit 0
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/Scripts/python.exe"
else
    PYTHON_BIN=python
fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"

CONFIGS=(configs/fp.yaml configs/rtn3.yaml configs/rtn4.yaml configs/gptq3.yaml configs/awq3.yaml)
if (( $# > 0 )); then
    CONFIGS=("$@")
fi

# The existing script imports the local src package without requiring editable install.
exec "$PYTHON_BIN" -u -c \
    'import sys; sys.path.insert(0, "src"); from phase1.cli import main; main()' \
    batch --configs "${CONFIGS[@]}" --include-batch-b
