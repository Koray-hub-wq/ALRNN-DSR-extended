#!/usr/bin/env bash
set -euo pipefail

# From the repository root:
# bash alrnn_python/lorenz_paper_test/run_lorenz_paper_test.sh [GPU_ID ...]
ROOT_DIR="${ROOT_DIR:-$HOME}"
source "$ROOT_DIR/DynaMix-python-b-tipping/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu_ids=("$@")
if (( ${#gpu_ids[@]} == 0 )); then gpu_ids=("${GPU_ID:-0}"); fi
extra_args=()
if [[ -n "${RUN_ROOT:-}" ]]; then extra_args+=(--run-root "$RUN_ROOT"); fi
if [[ -n "${TRAINING_DATA_ROOT:-}" ]]; then extra_args+=(--training-data-root "$TRAINING_DATA_ROOT"); fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then extra_args+=(--output-root "$OUTPUT_ROOT"); fi
if [[ -n "${NUM_PARAMETERS:-}" ]]; then extra_args+=(--num-parameters "$NUM_PARAMETERS"); fi
if [[ -n "${CPU_THREADS:-}" ]]; then extra_args+=(--cpu-threads "$CPU_THREADS"); fi
case "${DRY_RUN:-0}" in
  0) ;;
  1) extra_args+=(--dry-run) ;;
  *) printf 'DRY_RUN must be 0 or 1.\n' >&2; exit 2 ;;
esac
case "${PLOT_ONLY:-0}" in
  0) ;;
  1) extra_args+=(--plot-only) ;;
  *) printf 'PLOT_ONLY must be 0 or 1.\n' >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
exec "${PYTHON_BIN:-python}" -u "$SCRIPT_DIR/evaluate_lorenz.py" \
  --gpu-ids "${gpu_ids[@]}" "${extra_args[@]}"
