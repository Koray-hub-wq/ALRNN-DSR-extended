#!/usr/bin/env bash
set -euo pipefail

# Activate the desired Python environment before calling this script.
# Usage: bash run_dadras_combined_cphi.sh [GPU_ID]
# Preview: DRY_RUN=1 bash run_dadras_combined_cphi.sh 0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-/export/home/klenkeit/training_data/mixed_parameter_trajectories/dadras/combined_dadras}"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${1:-${GPU_ID:-0}}"
dry_run="${DRY_RUN:-0}"

if (( $# > 1 )) || [[ ! "$gpu_id" =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'Usage: bash %s [nonnegative GPU_ID]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi
if [[ "$dry_run" != 0 && "$dry_run" != 1 ]]; then
  printf 'DRY_RUN must be 0 or 1.\n' >&2
  exit 2
fi

latent_dim="${LATENT_DIM:-300}"
pwl_units="${PWL_UNITS:-150}"
epochs="${EPOCHS:-1000}"
batch_size="${BATCH_SIZE:-80}"
batches_per_epoch="${BATCHES_PER_EPOCH:-100}"
sequence_length="${SEQUENCE_LENGTH:-200}"
cpu_threads="${CPU_THREADS:-4}"
seed="${SEED:-7}"
mixed_balanced_sampling="${MIXED_BALANCED_SAMPLING:-0}"
time_sampling_bins="${TIME_SAMPLING_BINS:-10}"
if [[ "$mixed_balanced_sampling" != 0 && "$mixed_balanced_sampling" != 1 ]]; then
  printf 'MIXED_BALANCED_SAMPLING must be 0 or 1.\n' >&2
  exit 2
fi
if [[ ! "$time_sampling_bins" =~ ^[1-9][0-9]*$ ]]; then
  printf 'TIME_SAMPLING_BINS must be a positive integer.\n' >&2
  exit 2
fi
sampling_args=(--trajectory-sampling length)
run_group=combined_dadras_cphi
if [[ "$mixed_balanced_sampling" == 1 ]]; then
  sampling_args+=(--mixed-balanced-sampling --time-sampling-bins "$time_sampling_bins")
  run_group=combined_dadras_balanced_cphi
fi

DATASETS=(dadras_e_5p9_to_6p2 dadras_e_5p2_to_5p7 dadras_e_5p2_to_6p2)
run_id="$(date +%Y%m%d_%H%M%S)_$$_m${latent_dim}_p${pwl_units}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/term_ablations/dadras/$run_group/$run_id}"

export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1

if [[ "$dry_run" == 1 ]]; then
  printf 'DRY RUN: printing commands only; no training or output directories.\n'
else
  # Check all three datasets before starting the first long training.
  for dataset in "${DATASETS[@]}"; do
    for filename in trajectories.npy trajectory_phi.npy trajectory_lengths.npy \
      trajectory_train_lengths.npy trajectory_group_names.npy trajectory_mean.npy \
      trajectory_std.npy metadata.json; do
      if [[ ! -f "$TRAINING_DATA_ROOT/$dataset/$filename" ]]; then
        printf 'Missing required file: %s/%s/%s\n' "$TRAINING_DATA_ROOT" "$dataset" "$filename" >&2
        exit 1
      fi
    done
    if [[ -e "$RUN_ROOT/$dataset" ]]; then
      printf 'Output already exists: %s/%s. Choose a new RUN_ROOT.\n' "$RUN_ROOT" "$dataset" >&2
      exit 1
    fi
  done
  # Prevent the trainer from silently falling back to CPU on the server.
  "$python_bin" - "$gpu_id" <<'PY'
import sys
import torch

gpu = int(sys.argv[1])
if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
    raise SystemExit(f"CUDA GPU {gpu} is unavailable in this Python environment.")
print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}", flush=True)
PY
  mkdir -p "$RUN_ROOT/logs"
fi

printf 'Cphi training on GPU %s | data: %s\nModels and logs: %s\n' "$gpu_id" "$TRAINING_DATA_ROOT" "$RUN_ROOT"
printf 'Balanced mixed sampling: %s | dynamic time sections: %s\n' "$mixed_balanced_sampling" "$time_sampling_bins"
for dataset in "${DATASETS[@]}"; do
  command=(
    "$python_bin" -u "$SCRIPT_DIR/train_halvorsen_plain.py"
    --data-dir "$TRAINING_DATA_ROOT/$dataset"
    --multi-trajectory --phi-terms additive "${sampling_args[@]}"
    --gpu-id "$gpu_id" --cpu-threads "$cpu_threads"
    --latent-dim "$latent_dim" --pwl-units "$pwl_units"
    --epochs "$epochs" --batch-size "$batch_size"
    --batches-per-epoch "$batches_per_epoch" --sequence-length "$sequence_length"
    --seed "$seed" --no-best-model
    --output-dir "$RUN_ROOT/$dataset"
  )
  printf '\nTraining %s\n' "$dataset"
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "$dry_run" != 1 ]]; then
    "${command[@]}" 2>&1 | tee "$RUN_ROOT/logs/$dataset.log"
  fi
done

if [[ "$dry_run" != 1 ]]; then
  printf '\nFinished all three Cphi trainings. Models and logs: %s\n' "$RUN_ROOT"
fi
