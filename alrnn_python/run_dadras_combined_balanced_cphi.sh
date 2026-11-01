#!/usr/bin/env bash
set -euo pipefail

# Six trainings: three datasets x (Cphi or Cphi + shifted ReLU), then bifurcations.
# Activate the b-tipping venv, as in the other server launchers.
# Usage: bash alrnn_python/run_dadras_combined_balanced_cphi.sh [GPU_ID ...] (1-6 GPUs)
# Preview: DRY_RUN=1 bash alrnn_python/run_dadras_combined_balanced_cphi.sh 0 1 2 3 4 5
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
training_data_root="${TRAINING_DATA_ROOT:-/export/home/klenkeit/training_data/mixed_parameter_trajectories/dadras/combined_dadras}"
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
  --parameter-min "${BIFURCATION_PARAMETER_MIN:-4.9}" \
  --parameter-max "${BIFURCATION_PARAMETER_MAX:-6.2}" \
  --num-parameters "${BIFURCATION_NUM_PARAMETERS:-320}" \
  --components "${components[@]}" \
  "${extra_args[@]}"
