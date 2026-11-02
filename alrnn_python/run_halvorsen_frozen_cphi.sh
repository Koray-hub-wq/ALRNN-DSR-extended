#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_halvorsen_frozen_cphi.sh [GPU_ID]
# Optional: LATENT_DIM, PWL_UNITS, BASE_EPOCHS, CPHI_EPOCHS, SEED, BASE_RUN.
ROOT_DIR="${ROOT_DIR:-$HOME}"
source "$ROOT_DIR/DynaMix-python-b-tipping/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories}"
BIFURCATION_DIR="${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${1:-${GPU_ID:-0}}"
dry_run="${DRY_RUN:-0}"
if (( $# > 1 )) || [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  printf 'Usage: bash %s [nonnegative GPU_ID]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

latent_dim="${LATENT_DIM:-200}"
pwl_units="${PWL_UNITS:-20}"
base_epochs="${BASE_EPOCHS:-1000}"
cphi_epochs="${CPHI_EPOCHS:-1000}"
seed="${SEED:-7}"
batch_size=80
batches_per_epoch=100
cpu_threads=4
run_id="$(date +%Y%m%d_%H%M%S)_$$_m${latent_dim}_p${pwl_units}_seed${seed}"
RUN_ROOT="$SCRIPT_DIR/runs/experiments/halvorsen/frozen_cphi/$run_id"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/experiments/halvorsen/frozen_cphi/$run_id"
base_run="${BASE_RUN:-$RUN_ROOT/base_phi_m0p75_vanilla}"
RUN_NEGATIVE="$RUN_ROOT/cphi_negative"
RUN_MIXED="$RUN_ROOT/cphi_mixed"
RUN_FIVE="$RUN_ROOT/cphi_five_params"

for required in "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  "$DATA_DIR/trajectories.npy" "$DATA_DIR/trajectory_phi.npy" \
  "$DATA_DIR/trajectory_phi_constants.npy"; do
  [[ -f "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
if [[ -n "${BASE_RUN:-}" ]]; then
  for name in model.pt metrics.json; do
    [[ -f "$base_run/$name" ]] || { printf 'Missing base file: %s/%s\n' "$base_run" "$name" >&2; exit 1; }
  done
fi

"$python_bin" - "$DATA_DIR" "$latent_dim" "$pwl_units" "$base_epochs" "$cphi_epochs" "$seed" "${BASE_RUN:-}" <<'PY'
import sys
from pathlib import Path
import numpy as np
_, data_dir, latent, pwl, base_epochs, cphi_epochs, seed, base_run = sys.argv
latent, pwl, base_epochs, cphi_epochs, seed = map(int, (latent, pwl, base_epochs, cphi_epochs, seed))
if latent < 3 or not 0 <= pwl <= latent or min(base_epochs, cphi_epochs) < 1 or seed < 0:
    raise SystemExit("Invalid architecture, epoch count or seed")
data_dir = Path(data_dir)
if (data_dir / "trajectory_lengths.npy").exists():
    raise SystemExit("This experiment supports legacy constant-parameter data only")
values = np.load(data_dir / "trajectory_phi_constants.npy", allow_pickle=False)
for value in [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]:
    if not np.any(np.isclose(values, value, atol=1e-5, rtol=0)):
        raise SystemExit(f"Dataset is missing phi={value}")
if base_run:
    from alrnn_model import load_saved_model
    base, metrics, _ = load_saved_model(base_run)
    reference = metrics.get("args", {}).get("train_phi_values")
    if base.phi_terms or (base.M, base.P, base.N) != (latent, pwl, 3):
        raise SystemExit("BASE_RUN must be vanilla and match LATENT_DIM/PWL_UNITS")
    if reference is None or len(reference) != 1 or not np.isclose(reference[0], -.75, atol=1e-5, rtol=0):
        raise SystemExit("BASE_RUN must have been trained only at phi=-0.75")
print("Parameter values and experiment settings validated.", flush=True)
PY
if [[ "$dry_run" != 1 ]]; then
  "$python_bin" - "$gpu_id" <<'PY'
import sys
import torch
gpu = int(sys.argv[1])
if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
    raise SystemExit(f"CUDA GPU {gpu} is unavailable")
print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}", flush=True)
PY
  mkdir -p "$RUN_ROOT/logs"
else
  printf 'DRY RUN: no training, plots or output directories will be created.\n'
fi

export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1
COMMON_ARGS=(--data-dir "$DATA_DIR" --multi-trajectory --gpu-id "$gpu_id"
  --cpu-threads "$cpu_threads" --latent-dim "$latent_dim" --pwl-units "$pwl_units"
  --batch-size "$batch_size" --batches-per-epoch "$batches_per_epoch" --seed "$seed")

run_logged() {
  local logfile="$1"
  shift
  printf '%q ' "$@"
  printf '\n'
  if [[ "$dry_run" != 1 ]]; then "$@" 2>&1 | tee "$logfile"; fi
}

if [[ -z "${BASE_RUN:-}" ]]; then
  printf '\nPhase 1: vanilla at phi=-0.75\n'
  run_logged "$RUN_ROOT/logs/base.log" "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
    "${COMMON_ARGS[@]}" --phi-terms none --train-phi-values -0.75 \
    --epochs "$base_epochs" --output-dir "$base_run"
else
  printf '\nReusing vanilla base: %s\n' "$base_run"
fi

printf '\nPhase 2: three independent C-only adaptations; reference phi=-0.75\n'
run_logged "$RUN_ROOT/logs/cphi_negative.log" "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" --phi-terms additive --frozen-base-run "$base_run" \
  --train-phi-values -1 -0.75 -0.5 --epochs "$cphi_epochs" --output-dir "$RUN_NEGATIVE"
run_logged "$RUN_ROOT/logs/cphi_mixed.log" "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" --phi-terms additive --frozen-base-run "$base_run" \
  --train-phi-values -0.25 0.25 0.75 --epochs "$cphi_epochs" --output-dir "$RUN_MIXED"
run_logged "$RUN_ROOT/logs/cphi_five_params.log" "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" --phi-terms additive --frozen-base-run "$base_run" \
  --train-phi-values -1 -0.5 0 0.5 1 --epochs "$cphi_epochs" --output-dir "$RUN_FIVE"

printf '\nBifurcation comparison: vanilla base, three adaptations and ground truth\n'
cd "$BIFURCATION_DIR"
run_logged "$RUN_ROOT/logs/bifurcation.log" "$python_bin" "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  --backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  --model-type alrnn --regime halvorsen \
  --model-runs "$base_run" "$RUN_NEGATIVE" "$RUN_MIXED" "$RUN_FIVE" \
  --model-labels "Vanilla base: phi=-0.75" "Frozen base + C: negative" \
    "Frozen base + C: mixed" "Frozen base + C: five values" \
  --phi-min -1 --phi-max 1 --num-parameters 320 --components z \
  --output "$OUTPUT_DIR/halvorsen_frozen_cphi_bifurcation_compare.png" \
  --cache "$OUTPUT_DIR/halvorsen_frozen_cphi_bifurcation_cache.npz" \
  --force-recompute --device cuda --gpu-id "$gpu_id" --torch-threads "$cpu_threads"
printf '\nFinished. Models: %s\nDiagrams/cache: %s\n' "$RUN_ROOT" "$OUTPUT_DIR"
