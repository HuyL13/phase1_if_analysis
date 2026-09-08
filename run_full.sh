#!/usr/bin/env bash
set -euo pipefail

# Resolve config/data paths consistently, even when invoked from another folder.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_ROOT"
printf 'RUNNER_MODE=existing-environment-no-install\n'

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash run_full.sh [config.yaml ...]

Uses the server's existing Python environment; install requirements.txt separately before running.
Downloads/reuses the base and IF-SFT checkpoints, validates all inputs, then runs Stage 0 -> Batch A -> Batch B -> combined CSVs, figures and report.
Default configs: fp, rtn3, rtn4, awq3 (all configured seeds).
Configure checkpoints, datasets and the original IF verifier first.
AWQ checkpoints are created with the pinned upstream backend; validation checks their metadata.

Optional: PYTHON=/path/to/python bash run_full.sh
Config and data paths are relative to the repository root.
HELP
    exit 0
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    printf 'Python 3.10+ is required but no python executable was found\n' >&2
    exit 1
fi
"$PYTHON_BIN" -c 'import sys; sys.version_info >= (3, 10) or sys.exit("Python 3.10+ is required")'

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIGS=(configs/fp.yaml configs/rtn3.yaml configs/rtn4.yaml configs/awq3.yaml)
if (( $# > 0 )); then
    CONFIGS=("$@")
fi

printf 'Validating checkpoints, data, verifier and quantized exports...\n'
"$PYTHON_BIN" scripts/00_download_checkpoints.py --config "${CONFIGS[0]}"
QUANTIZED_CONFIGS=()
for config in "${CONFIGS[@]}"; do
    case "$config" in
        *rtn*.yaml|*awq*.yaml) QUANTIZED_CONFIGS+=("$config") ;;
    esac
done
if (( ${#QUANTIZED_CONFIGS[@]} > 0 )); then
    "$PYTHON_BIN" scripts/09_prepare_upstream_quantized.py \
        --python "$PYTHON_BIN" \
        --configs "${QUANTIZED_CONFIGS[@]}"
fi
"$PYTHON_BIN" -u -m phase1.cli validate --configs "${CONFIGS[@]}"

exec "$PYTHON_BIN" -u -m phase1.cli \
    batch --configs "${CONFIGS[@]}" --include-batch-b
