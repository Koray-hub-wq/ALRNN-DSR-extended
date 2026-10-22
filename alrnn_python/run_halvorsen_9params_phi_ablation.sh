#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories}"
BIFURCATION_DIR="${BIFURCATION_DIR:-${ROOT_DIR:-$HOME}/diffrentbutconst_prams}"
python_bin="${PYTHON_BIN:-python}"
# Usage: bash run_halvorsen_9params_phi_ablation.sh [GPU_ID]
gpu_id="${1:-${GPU_ID:-0}}"
latents_dim=200
epochs=1000
pwl_units=20
batch_size=80
batches_per_epoch=100
cpu_threads=4

RUN_NEGATIVE="$SCRIPT_DIR/runs/halvorsen_9params_train_m1_m0p75_m0p5_m${latents_dim}_p${pwl_units}_cphi"
RUN_MIXED="$SCRIPT_DIR/runs/halvorsen_9params_train_m0p25_p0p25_p0p75_m${latents_dim}_p${pwl_units}_cphi"
RUN_VANILLA="$SCRIPT_DIR/runs/halvorsen_9params_all_m${latents_dim}_p${pwl_units}_vanilla"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/alrnn/halvorsen_9params_phi_ablation_m${latents_dim}_p${pwl_units}"

# Fail before training if the data or the subsequent analysis is unavailable.
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

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --multi-trajectory
  --use-phi
  --gpu-id "$gpu_id"
  --cpu-threads "$cpu_threads"
  --latent-dim "$latents_dim"
  --pwl-units "$pwl_units"
  --batch-size "$batch_size"
  --epochs "$epochs"
  --batches-per-epoch "$batches_per_epoch"
)

"$python_bin" train_halvorsen_plain.py \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -1 -0.75 -0.5 \
  --output-dir "$RUN_NEGATIVE"

"$python_bin" train_halvorsen_plain.py \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -0.25 0.25 0.75 \
  --output-dir "$RUN_MIXED"

VANILLA_ARGS=(
  --data-dir "$DATA_DIR"
  --multi-trajectory
  --gpu-id "$gpu_id"
  --cpu-threads "$cpu_threads"
  --latent-dim "$latents_dim"
  --pwl-units "$pwl_units"
  --batch-size "$batch_size"
  --epochs "$epochs"
  --batches-per-epoch "$batches_per_epoch"
)

"$python_bin" train_halvorsen_plain.py \
  "${VANILLA_ARGS[@]}" \
  --output-dir "$RUN_VANILLA"

printf '\nComparing bifurcation diagrams for both Cphi models, vanilla, and ground truth\n'
cd "$BIFURCATION_DIR"
"$python_bin" "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  --backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  --model-type alrnn \
  --regime halvorsen \
  --model-runs "$RUN_NEGATIVE" "$RUN_MIXED" "$RUN_VANILLA" \
  --model-labels \
    "Cphi: phi {-1, -0.75, -0.5}" \
    "Cphi: phi {-0.25, 0.25, 0.75}" \
    "Vanilla: all 9 phi values" \
  --parameter-min 1.4 \
  --parameter-max 2.09 \
  --components z \
  --output "$OUTPUT_DIR/halvorsen_phi_ablation_bifurcation_compare.png" \
  --cache "$OUTPUT_DIR/halvorsen_phi_ablation_bifurcation_cache.npz" \
  --force-recompute \
  --device cuda \
  --gpu-id "$gpu_id" \
  --torch-threads "$cpu_threads" \
  --num-parameters 320

printf '\nFinished. Bifurcation diagram: %s\n' \
  "$OUTPUT_DIR/halvorsen_phi_ablation_bifurcation_compare.png"
