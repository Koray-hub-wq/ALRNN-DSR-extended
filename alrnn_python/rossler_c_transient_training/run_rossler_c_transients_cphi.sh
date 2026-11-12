#!/usr/bin/env bash
set -euo pipefail

# From ALRNN-DSR-extended:
# bash alrnn_python/rossler_c_transient_training/run_rossler_c_transients_cphi.sh [GPU_ID ...]
ROOT_DIR="${ROOT_DIR:-$HOME}"
source "$ROOT_DIR/DynaMix-python-b-tipping/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu_ids=("$@")
if (( ${#gpu_ids[@]} == 0 )); then gpu_ids=("${GPU_ID:-0}"); fi
extra_args=(--backend-script "${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}/compare_model_bifurcation_diagrams.py")
if [[ -n "${TRAINING_DATA_ROOT:-}" ]]; then extra_args+=(--training-data-root "$TRAINING_DATA_ROOT"); fi
if [[ -n "${RUN_ROOT:-}" ]]; then extra_args+=(--run-root "$RUN_ROOT"); fi
if [[ -n "${EPOCHS:-}" ]]; then extra_args+=(--epochs "$EPOCHS"); fi
if [[ -n "${CPU_THREADS:-}" ]]; then extra_args+=(--cpu-threads "$CPU_THREADS"); fi
if [[ -n "${NUM_PARAMETERS:-}" ]]; then extra_args+=(--num-parameters "$NUM_PARAMETERS"); fi
if [[ -n "${TAIL_EVALUATIONS:-}" ]]; then extra_args+=(--tail-evaluations "$TAIL_EVALUATIONS"); fi
if [[ -n "${ATTRACTOR_FRACTION:-}" ]]; then extra_args+=(--attractor-fraction "$ATTRACTOR_FRACTION"); fi
case "${ANALYSIS_ONLY:-0}" in
  0) ;;
  1) extra_args+=(--analysis-only) ;;
  *) printf 'ANALYSIS_ONLY must be 0 or 1.\n' >&2; exit 2 ;;
esac
case "${DRY_RUN:-0}" in
  0) ;;
  1) extra_args+=(--dry-run) ;;
  *) printf 'DRY_RUN must be 0 or 1.\n' >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
exec "${PYTHON_BIN:-python}" -u "$SCRIPT_DIR/run_training.py" --gpu-ids "${gpu_ids[@]}" "${extra_args[@]}"
