#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_dadras_rossler_cphi_shifted_relu.sh [GPU_ID ...]
# Six independent trainings on one to six GPUs, then two bifurcation comparisons.
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-$ROOT_DIR/training_data/single_new_regimes_const/trajectories}"
BIFURCATION_DIR="${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}"
python_bin="${PYTHON_BIN:-python}"
gpu_ids=("$@")
if (( ${#gpu_ids[@]} == 0 )); then
  gpu_ids=("${GPU_ID:-0}")
fi

latents_dim=200
pwl_units=20
epochs=750
batch_size=80
batches_per_epoch=100
cpu_threads=4

extra_args=()
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  extra_args+=(--dry-run)
fi

exec "$python_bin" -u "$SCRIPT_DIR/run_cphi_shifted_regimes.py" \
  --training-data-root "$TRAINING_DATA_ROOT" \
  --backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  --latent-dim "$latents_dim" \
  --pwl-units "$pwl_units" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --batches-per-epoch "$batches_per_epoch" \
  --cpu-threads "$cpu_threads" \
  --gpu-ids "${gpu_ids[@]}" \
  "${extra_args[@]}"
