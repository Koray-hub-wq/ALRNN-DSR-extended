#!/usr/bin/env bash
set -euo pipefail

# bash alrnn_python/rerun_mixed_bifurcations.sh EXISTING_RUN_ROOT [GPU_ID ...]
if (( $# < 1 )); then
  printf 'Usage: bash %s EXISTING_RUN_ROOT [GPU_ID ...]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi
run_root="$1"
shift
ROOT_DIR="${ROOT_DIR:-$HOME}"
source "$ROOT_DIR/DynaMix-python-b-tipping/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu_ids=("$@")
if (( ${#gpu_ids[@]} == 0 )); then gpu_ids=("${GPU_ID:-0}"); fi
extra_args=()
if [[ -n "${BIFURCATION_DIR:-}" ]]; then
  extra_args+=(--backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py")
fi
if [[ -n "${BIFURCATION_COMPONENTS:-}" ]]; then
  read -r -a components <<< "$BIFURCATION_COMPONENTS"
  extra_args+=(--components "${components[@]}")
fi
case "${REUSE_CACHE:-0}" in
  0) ;;
  1) extra_args+=(--reuse-cache) ;;
  *) printf 'REUSE_CACHE must be 0 or 1.\n' >&2; exit 2 ;;
esac
case "${DRY_RUN:-0}" in
  0) ;;
  1) extra_args+=(--dry-run) ;;
  *) printf 'DRY_RUN must be 0 or 1.\n' >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python}" -u "$SCRIPT_DIR/rerun_mixed_bifurcations.py" \
  --run-root "$run_root" --gpu-ids "${gpu_ids[@]}" "${extra_args[@]}"
