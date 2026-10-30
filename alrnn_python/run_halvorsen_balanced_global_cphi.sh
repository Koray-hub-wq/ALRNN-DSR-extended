#!/usr/bin/env bash
set -euo pipefail

# Experiment A: unchanged global normalization, balanced batches across phi.
# Usage: bash run_halvorsen_balanced_global_cphi.sh [GPU_ID]
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-$ROOT_DIR/training_data/mixed_parameter_trajectories/halvorsen/controll_halvorsen}"
BIFURCATION_DIR="${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${1:-${GPU_ID:-0}}"
dry_run="${DRY_RUN:-0}"
if (( $# > 1 )) || [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  printf 'Usage: bash %s [nonnegative GPU_ID]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

latents_dim=200
pwl_units=20
epochs=1000
batch_size=80
batches_per_epoch=100
cpu_threads=4
seed="${SEED:-7}"
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  printf 'SEED must be a nonnegative integer.\n' >&2
  exit 2
fi

if [[ "$dry_run" == 1 ]]; then
  printf 'DRY RUN: no training or bifurcation computation will be started.\n'
fi

DATASETS=(halvorsen_phi_m1_to_1 halvorsen_phi_m1_to_m0p5 halvorsen_phi_m0p25_to_0p75)
DATA_DIRS=()
RUN_DIRS=()
run_id="$(date +%Y%m%d_%H%M%S)_$$_m${latents_dim}_p${pwl_units}_seed${seed}"
RUN_ROOT="$SCRIPT_DIR/runs/experiments/halvorsen/balanced_global/$run_id"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/experiments/halvorsen/balanced_global/$run_id"

for name in "${DATASETS[@]}"; do
  data_dir="$TRAINING_DATA_ROOT/$name"
  DATA_DIRS+=("$data_dir")
  RUN_DIRS+=("$RUN_ROOT/$name")
  for filename in trajectories.npy trajectory_phi.npy trajectory_lengths.npy \
    trajectory_train_lengths.npy trajectory_group_names.npy trajectory_mean.npy trajectory_std.npy metadata.json; do
    if [[ ! -f "$data_dir/$filename" ]]; then
      printf 'Missing required file: %s\n' "$data_dir/$filename" >&2
      exit 1
    fi
  done
done
for required_file in \
  "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Missing required file: %s\n' "$required_file" >&2
    exit 1
  fi
done

# Validate the global scalings before starting any long training.
"$python_bin" - "${DATA_DIRS[@]}" <<'PY'
import sys
from compare_model_bifurcation_diagrams import load_training_coordinates
for path in sys.argv[1:]:
    scaling = load_training_coordinates(path)
    print(f"Dataset: {path} | training mean={scaling.mean} | std={scaling.std}", flush=True)
PY

if [[ "$dry_run" != 1 ]]; then
  "$python_bin" - "$gpu_id" <<'PY'
import sys
import torch
gpu = int(sys.argv[1])
if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
    raise SystemExit(f"CUDA GPU {gpu} is unavailable.")
print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}", flush=True)
PY
  mkdir -p "$RUN_ROOT/logs"
fi

export OMP_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1

run_logged() {
  local log_file="$1"
  shift
  printf '%q ' "$@"
  printf '\n'
  if [[ "$dry_run" != 1 ]]; then
    "$@" 2>&1 | tee "$log_file"
  fi
}

printf 'Experiment A: Cphi, global normalization, balanced phi batches, best-Dstsp selection. GPU %s. Models: %s\n' "$gpu_id" "$RUN_ROOT"
for index in "${!DATASETS[@]}"; do
  printf '\nTraining %s\n' "${DATASETS[$index]}"
  run_logged "$RUN_ROOT/logs/${DATASETS[$index]}.log" \
    "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
    --data-dir "${DATA_DIRS[$index]}" --multi-trajectory --phi-terms additive --mixed-balanced-phi \
    --gpu-id "$gpu_id" --cpu-threads "$cpu_threads" \
    --latent-dim "$latents_dim" --pwl-units "$pwl_units" \
    --epochs "$epochs" --batch-size "$batch_size" --batches-per-epoch "$batches_per_epoch" \
    --seed "$seed" \
    --output-dir "${RUN_DIRS[$index]}"
done

printf '\nComparing three Cphi models and ground truth over phi=-1..1\n'
cd "$BIFURCATION_DIR"
run_logged "$RUN_ROOT/logs/bifurcation.log" \
  "$python_bin" "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  --backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  --use-training-normalization \
  --model-type alrnn --regime halvorsen \
  --model-runs "${RUN_DIRS[@]}" --training-data-dirs "${DATA_DIRS[@]}" \
  --model-labels "Cphi: full range [-1,1]" "Cphi: negative [-1,-0.5]" "Cphi: mixed [-0.25,0.75]" \
  --phi-min -1 --phi-max 1 --num-parameters 320 --components z \
  --output "$OUTPUT_DIR/halvorsen_control_cphi_bifurcation_compare.png" \
  --cache "$OUTPUT_DIR/halvorsen_control_cphi_bifurcation_cache.npz" \
  --force-recompute --device cuda --gpu-id "$gpu_id" --torch-threads "$cpu_threads"

printf '\nFinished. Models: %s\nBifurcation diagram/cache: %s\n' "$RUN_ROOT" "$OUTPUT_DIR"
