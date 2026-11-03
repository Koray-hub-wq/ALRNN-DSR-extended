#!/usr/bin/env bash
set -euo pipefail

# Shared launcher; use the wrapper for the desired regime.
# Usage: bash run_mixed_combined_balanced.sh SYSTEM [GPU_ID ...]
system="${1:-}"
case "$system" in
  dadras) sweep_min=4.9; sweep_max=6.2 ;;
  rossler) sweep_min=0.05; sweep_max=0.35 ;;
  tsucs1) sweep_min=20.0; sweep_max=24.5 ;;
  lorenz_rho) sweep_min=1.0; sweep_max=30.0 ;;
  rossler_c) sweep_min=3.0; sweep_max=6.0 ;;
  *) printf 'Usage: bash %s {dadras|rossler|tsucs1|lorenz_rho|rossler_c} [GPU_ID ...]\n' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
shift
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
training_data_root="${TRAINING_DATA_ROOT:-/export/home/klenkeit/training_data/mixed_parameter_trajectories/$system/combined_$system}"
bifurcation_dir="${BIFURCATION_DIR:-${ROOT_DIR:-$HOME}/diffrentbutconst_prams}"
gpu_ids=("$@")
if (( ${#gpu_ids[@]} == 0 )); then
  gpu_ids=("${GPU_ID:-0}")
fi
extra_args=()
case "${DRY_RUN:-0}" in
  0) ;;
  1) extra_args+=(--dry-run) ;;
  *) printf 'DRY_RUN must be 0 or 1.\n' >&2; exit 2 ;;
esac
if [[ -n "${RUN_ROOT:-}" ]]; then extra_args+=(--run-root "$RUN_ROOT"); fi
if [[ -n "${PLOT_ROOT:-}" ]]; then extra_args+=(--plot-root "$PLOT_ROOT"); fi
read -r -a components <<< "${BIFURCATION_COMPONENTS:-z}"

exec "$python_bin" -u "$SCRIPT_DIR/run_dadras_combined_balanced.py" \
  --system "$system" \
  --training-data-root "$training_data_root" \
  --backend-script "$bifurcation_dir/compare_model_bifurcation_diagrams.py" \
  --gpu-ids "${gpu_ids[@]}" \
  --cphi-latent-dim "${CPHI_LATENT_DIM:-${LATENT_DIM:-300}}" \
  --cphi-pwl-units "${CPHI_PWL_UNITS:-${PWL_UNITS:-150}}" \
  --shifted-latent-dim "${SHIFTED_LATENT_DIM:-200}" \
  --shifted-pwl-units "${SHIFTED_PWL_UNITS:-100}" \
  --epochs "${EPOCHS:-1000}" --batch-size "${BATCH_SIZE:-80}" \
  --batches-per-epoch "${BATCHES_PER_EPOCH:-100}" \
  --sequence-length "${SEQUENCE_LENGTH:-200}" \
  --time-sampling-bins "${TIME_SAMPLING_BINS:-10}" \
  --cpu-threads "${CPU_THREADS:-4}" --seed "${SEED:-7}" \
  --parameter-min "${BIFURCATION_PARAMETER_MIN:-$sweep_min}" \
  --parameter-max "${BIFURCATION_PARAMETER_MAX:-$sweep_max}" \
  --num-parameters "${BIFURCATION_NUM_PARAMETERS:-320}" \
  --components "${components[@]}" \
  "${extra_args[@]}"
