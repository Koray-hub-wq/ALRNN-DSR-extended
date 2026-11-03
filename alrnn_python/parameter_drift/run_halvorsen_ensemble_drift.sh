#!/usr/bin/env bash
set -euo pipefail

# From the repository root:
# bash alrnn_python/parameter_drift/run_halvorsen_ensemble_drift.sh [GPU_ID] [PYTHON_OPTIONS ...]
ROOT_DIR="${ROOT_DIR:-$HOME}"
source "$ROOT_DIR/DynaMix-python-b-tipping/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu_id="${1:-${GPU_ID:-0}}"
if (( $# > 0 )); then shift; fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
cpu_threads="${CPU_THREADS:-4}"
export OMP_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads"
extra_args=()
if [[ -n "${REGIME_FILE:-}" ]]; then extra_args+=(--regime-file "$REGIME_FILE"); fi
exec "${PYTHON_BIN:-python}" -u "$SCRIPT_DIR/evaluate_ensemble_drift.py" \
  --config "$SCRIPT_DIR/halvorsen_ensemble_config.json" \
  --gpu-id "$gpu_id" --cpu-threads "$cpu_threads" \
  "${extra_args[@]}" "$@"
