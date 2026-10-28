#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_halvorsen_mixed_groups_cphi.sh GPU_ID [GPU_ID ...] (1-4 GPUs)
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python_bin="${PYTHON_BIN:-python}"
TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-$ROOT_DIR/training_data/mixed_parameter_trajectories/halvorsen}"
BIFURCATION_DIR="${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}"
dry_run="${DRY_RUN:-0}"

if (( $# < 1 || $# > 4 )); then
  printf 'Usage: bash %s GPU_ID [GPU_ID ...] (1-4 distinct GPUs)\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi
GPU_IDS=("$@")
declare -A seen=()
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! "$gpu" =~ ^(0|[1-9][0-9]*)$ ]] || [[ -n "${seen[$gpu]:-}" ]]; then
    printf 'Invalid or duplicate GPU ID: %s\n' "$gpu" >&2
    exit 2
  fi
  seen[$gpu]=1
done

GROUPS_TO_TRAIN=(combined_halvorsen halvorsen_no_burnin_short halvorsen_no_burnin_veryshort timevarying_halvorsen)
DATASETS=(halvorsen_phi_m1_to_1 halvorsen_phi_m1_to_m0p5 halvorsen_phi_m0p25_to_0p75)
latents_dim=200
pwl_units=20
epochs=1000
batch_size=80
batches_per_epoch=100
cpu_threads=4
run_id="$(date +%Y%m%d_%H%M%S)_$$_m${latents_dim}_p${pwl_units}"
LOG_ROOT="$SCRIPT_DIR/runs/term_ablations/halvorsen/_queues/mixed_groups_cphi/$run_id"

export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1

for required in "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py"; do
  [[ -f "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
ALL_DATA_DIRS=()
for group in "${GROUPS_TO_TRAIN[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    data_dir="$TRAINING_DATA_ROOT/$group/$dataset"
    ALL_DATA_DIRS+=("$data_dir")
    for filename in trajectories.npy trajectory_phi.npy trajectory_lengths.npy \
      trajectory_train_lengths.npy trajectory_group_names.npy trajectory_mean.npy trajectory_std.npy metadata.json; do
      [[ -f "$data_dir/$filename" ]] || { printf 'Missing required file: %s\n' "$data_dir/$filename" >&2; exit 1; }
    done
  done
done

# Check every dataset before any worker starts a long training.
"$python_bin" - "${ALL_DATA_DIRS[@]}" <<'PY'
import sys
from compare_model_bifurcation_diagrams import load_training_coordinates
for path in sys.argv[1:]:
    scaling = load_training_coordinates(path)
    print(f"Dataset: {path} | training mean={scaling.mean} | std={scaling.std}", flush=True)
PY
if [[ "$dry_run" != 1 ]]; then
  "$python_bin" - "${GPU_IDS[@]}" <<'PY'
import sys
import torch
for value in sys.argv[1:]:
    gpu = int(value)
    if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
        raise SystemExit(f"CUDA GPU {gpu} is unavailable.")
    print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}", flush=True)
PY
  mkdir -p "$LOG_ROOT"
fi

run_logged() {
  local logfile="$1"
  shift
  printf '%q ' "$@"
  printf '\n'
  if [[ "$dry_run" != 1 ]]; then
    "$@" > "$logfile" 2>&1 &
    child_pid=$!
    local status=0
    wait "$child_pid" || status=$?
    child_pid=""
    return "$status"
  fi
}

train_group() {
  local group="$1" gpu="$2" dataset index
  local run_root="$SCRIPT_DIR/runs/term_ablations/halvorsen/${group}_cphi/$run_id"
  local output_dir="$SCRIPT_DIR/bifurcation_diagrams/term_ablations/halvorsen/${group}_cphi/$run_id"
  local -a data_dirs=() run_dirs=()
  [[ "$dry_run" == 1 ]] || mkdir -p "$run_root/logs"
  printf '[%s] GPU %s | models: %s\n' "$group" "$gpu" "$run_root"
  for dataset in "${DATASETS[@]}"; do
    data_dirs+=("$TRAINING_DATA_ROOT/$group/$dataset")
    run_dirs+=("$run_root/$dataset")
  done
  for index in "${!DATASETS[@]}"; do
    printf '[%s] GPU %s | training %s\n' "$group" "$gpu" "${DATASETS[$index]}"
    run_logged "$run_root/logs/${DATASETS[$index]}.log" \
      "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
      --data-dir "${data_dirs[$index]}" --multi-trajectory --phi-terms additive \
      --gpu-id "$gpu" --cpu-threads "$cpu_threads" \
      --latent-dim "$latents_dim" --pwl-units "$pwl_units" \
      --epochs "$epochs" --batch-size "$batch_size" --batches-per-epoch "$batches_per_epoch" \
      --output-dir "${run_dirs[$index]}" || return $?
  done
  printf '[%s] GPU %s | bifurcation comparison\n' "$group" "$gpu"
  # The external backend expects to run from its own directory.
  cd "$BIFURCATION_DIR"
  run_logged "$run_root/logs/bifurcation.log" \
    "$python_bin" "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
    --backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
    --use-training-normalization --model-type alrnn --regime halvorsen \
    --model-runs "${run_dirs[@]}" --training-data-dirs "${data_dirs[@]}" \
    --model-labels "Cphi: full range [-1,1]" "Cphi: negative [-1,-0.5]" "Cphi: mixed [-0.25,0.75]" \
    --phi-min -1 --phi-max 1 --num-parameters 320 --components z \
    --output "$output_dir/halvorsen_bifurcation_compare.png" \
    --cache "$output_dir/halvorsen_bifurcation_cache.npz" \
    --force-recompute --device cuda --gpu-id "$gpu" --torch-threads "$cpu_threads" || return $?
  cd "$SCRIPT_DIR"
  printf '[%s] Finished. Diagrams/cache: %s\n' "$group" "$output_dir"
}

worker() {
  local slot="$1" gpu="${GPU_IDS[$1]}" index failed=0 child_pid=""
  trap 'if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; fi; exit 130' INT TERM
  for (( index=slot; index<${#GROUPS_TO_TRAIN[@]}; index+=${#GPU_IDS[@]} )); do
    if ! train_group "${GROUPS_TO_TRAIN[$index]}" "$gpu"; then
      printf '[%s] FAILED on GPU %s. See group logs.\n' "${GROUPS_TO_TRAIN[$index]}" "$gpu" >&2
      failed=1
      cd "$SCRIPT_DIR"
    fi
  done
  return "$failed"
}

if [[ "$dry_run" == 1 ]]; then
  printf 'DRY RUN: no computations or output directories will be created.\n'
  for slot in "${!GPU_IDS[@]}"; do worker "$slot"; done
  exit 0
fi

PIDS=()
stop_workers() {
  trap '' INT TERM
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 130
}
trap stop_workers INT TERM
for slot in "${!GPU_IDS[@]}"; do
  worker "$slot" > "$LOG_ROOT/gpu_${GPU_IDS[$slot]}.log" 2>&1 &
  PIDS+=("$!")
  printf 'GPU %s: worker PID %s | log: %s/gpu_%s.log\n' "${GPU_IDS[$slot]}" "$!" "$LOG_ROOT" "${GPU_IDS[$slot]}"
done
printf 'Training progress: tail -f %q/*_cphi/%q/logs/*.log\n' "$SCRIPT_DIR/runs/term_ablations/halvorsen" "$run_id"
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
  printf 'At least one group failed. Worker logs: %s\n' "$LOG_ROOT" >&2
else
  printf 'All four groups finished (12 models and four bifurcation comparisons).\n'
fi
exit "$failed"
