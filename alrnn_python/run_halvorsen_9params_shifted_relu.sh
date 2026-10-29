#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_halvorsen_9params_shifted_relu.sh [GPU_ID]
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories}"
BIFURCATION_DIR="${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${1:-${GPU_ID:-0}}"
if (( $# > 1 )) || [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  printf 'Usage: bash %s [nonnegative GPU_ID]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

latents_dim=200
epochs=1000
pwl_units=20
batch_size=80
batches_per_epoch=100
cpu_threads=4
run_id="$(date +%Y%m%d_%H%M%S)_$$_m${latents_dim}_p${pwl_units}"
RUN_ROOT="$SCRIPT_DIR/runs/term_ablations/halvorsen/shifted_relu/$run_id"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/term_ablations/halvorsen/shifted_relu/$run_id"
RUN_NEGATIVE="$RUN_ROOT/negative"
RUN_MIXED="$RUN_ROOT/mixed"
RUN_FIVE="$RUN_ROOT/five_params"

for required_file in \
  "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  "$DATA_DIR/trajectories.npy" \
  "$DATA_DIR/trajectory_phi.npy" \
  "$DATA_DIR/trajectory_phi_constants.npy" \
  "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Missing required file: %s\n' "$required_file" >&2
    exit 1
  fi
done

export OMP_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --multi-trajectory
  --phi-terms shifted-relu
  --gpu-id "$gpu_id"
  --cpu-threads "$cpu_threads"
  --latent-dim "$latents_dim"
  --pwl-units "$pwl_units"
  --batch-size "$batch_size"
  --epochs "$epochs"
  --batches-per-epoch "$batches_per_epoch"
)

printf 'Shifted-ReLU only, no Cphi. GPU %s. Models: %s\n' "$gpu_id" "$RUN_ROOT"
"$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -1 -0.75 -0.5 \
  --output-dir "$RUN_NEGATIVE"

"$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -0.25 0.25 0.75 \
  --output-dir "$RUN_MIXED"

"$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -1 -0.5 0 0.5 1 \
  --output-dir "$RUN_FIVE"

printf '\nComparing three shifted-ReLU models and ground truth\n'
cd "$BIFURCATION_DIR"
"$python_bin" "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  --backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  --model-type alrnn --regime halvorsen \
  --model-runs "$RUN_NEGATIVE" "$RUN_MIXED" "$RUN_FIVE" \
  --model-labels \
    "Shifted-ReLU: phi {-1, -0.75, -0.5}" \
    "Shifted-ReLU: phi {-0.25, 0.25, 0.75}" \
    "Shifted-ReLU: phi {-1, -0.5, 0, 0.5, 1}" \
  --parameter-min 1.4 --parameter-max 2.09 \
  --components z --num-parameters 320 \
  --output "$OUTPUT_DIR/halvorsen_shifted_relu_bifurcation_compare.png" \
  --cache "$OUTPUT_DIR/halvorsen_shifted_relu_bifurcation_cache.npz" \
  --force-recompute --device cuda --gpu-id "$gpu_id" --torch-threads "$cpu_threads"

printf '\nFinished. Models: %s\nBifurcation diagram/cache: %s\n' "$RUN_ROOT" "$OUTPUT_DIR"
