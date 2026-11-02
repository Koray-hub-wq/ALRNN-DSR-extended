#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory; keep this file beside train_halvorsen_plain.py.
# Usage: bash alrnn_python/run_hopf_quintic_const.sh [GPU_ID] [--dry-run]
# Uses the same venv as the existing ALRNN server launchers.
# Overrides: VENV_DIR, ALRNN_PYTHON_DIR, DATA_ROOT, RUN_ROOT, PYTHON_BIN,
# EPOCHS, LATENT_DIM, PWL_UNITS, BATCH_SIZE, BATCHES_PER_EPOCH,
# SEQUENCE_LENGTH, TEACHER_FORCING_INTERVAL, CPU_THREADS, SEED, DRY_RUN.

usage() {
    cat <<'USAGE'
Usage: bash run_hopf_quintic_const.sh [GPU_ID] [--dry-run]

Activates $HOME/DynaMix-python-b-tipping/venv and trains before, after and
before_after sequentially. Default: GPU 0, Cphi, M=64, P=16, 500 epochs.
Set VENV_DIR to use another virtual environment. No normal-form generator
scripts are needed on the server; only the three complete dataset folders.

Examples:
  bash alrnn_python/run_hopf_quintic_const.sh 0 --dry-run
  bash alrnn_python/run_hopf_quintic_const.sh 0
  VENV_DIR=/path/to/venv bash alrnn_python/run_hopf_quintic_const.sh 1
USAGE
}

gpu_id="${GPU_ID:-0}"
dry_run="${DRY_RUN:-0}"
gpu_supplied=0
for argument in "$@"; do
    case "$argument" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=1 ;;
        *)
            if [[ "$argument" =~ ^[0-9]+$ && "$gpu_supplied" == 0 ]]; then
                gpu_id="$argument"
                gpu_supplied=1
            else
                printf 'Unknown argument: %s\n' "$argument" >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$HOME}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/DynaMix-python-b-tipping/venv}"
ALRNN_PYTHON_DIR="${ALRNN_PYTHON_DIR:-$SCRIPT_DIR}"
DATA_ROOT="${DATA_ROOT:-/export/home/klenkeit/training_data/normal_forms/hopf_quintic_const}"
RUN_ROOT="${RUN_ROOT:-$ALRNN_PYTHON_DIR/runs/normal_forms/hopf_quintic_const}"
trainer="$ALRNN_PYTHON_DIR/train_halvorsen_plain.py"

epochs="${EPOCHS:-500}"
latent_dim="${LATENT_DIM:-64}"
pwl_units="${PWL_UNITS:-16}"
batch_size="${BATCH_SIZE:-16}"
batches_per_epoch="${BATCHES_PER_EPOCH:-100}"
sequence_length="${SEQUENCE_LENGTH:-200}"
teacher_forcing_interval="${TEACHER_FORCING_INTERVAL:-16}"
cpu_threads="${CPU_THREADS:-4}"
seed="${SEED:-7}"

for setting in epochs latent_dim batch_size batches_per_epoch sequence_length teacher_forcing_interval cpu_threads; do
    value="${!setting}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer; got %s\n' "$setting" "$value" >&2
        exit 2
    fi
done
if [[ ! "$gpu_id" =~ ^[0-9]+$ || ! "$seed" =~ ^[0-9]+$ || ! "$pwl_units" =~ ^[0-9]+$ ]] ||
   (( latent_dim < 3 || pwl_units > latent_dim )); then
    printf 'Require GPU_ID/SEED >= 0, LATENT_DIM >= 3 and 0 <= PWL_UNITS <= LATENT_DIM.\n' >&2
    exit 2
fi
if [[ "$dry_run" != 0 && "$dry_run" != 1 ]]; then
    printf 'DRY_RUN must be 0 or 1.\n' >&2
    exit 2
fi
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    printf 'Missing environment activation: %s/bin/activate\nSet VENV_DIR to your existing training venv.\n' "$VENV_DIR" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python_bin="${PYTHON_BIN:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python interpreter not found after activation: %s\n' "$python_bin" >&2
    exit 1
fi
if [[ ! -f "$trainer" ]]; then
    printf 'Missing trainer: %s\nPlace this launcher in alrnn_python or set ALRNN_PYTHON_DIR.\n' "$trainer" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads"

# Check all inputs before starting the first training.
cases=(before after before_after)
required_arrays=(trajectories trajectory_phi trajectory_lengths trajectory_train_lengths
                 trajectory_group_names trajectory_modes trajectory_mean trajectory_std)
for split in "${cases[@]}"; do
    for name in "${required_arrays[@]}"; do
        if [[ ! -f "$DATA_ROOT/$split/$name.npy" ]]; then
            printf 'Missing dataset file: %s/%s/%s.npy\n' "$DATA_ROOT" "$split" "$name" >&2
            exit 1
        fi
    done
    if [[ ! -f "$DATA_ROOT/$split/metadata.json" ]]; then
        printf 'Missing dataset metadata: %s/%s/metadata.json\n' "$DATA_ROOT" "$split" >&2
        exit 1
    fi
    if [[ "$dry_run" == 0 && ( -e "$RUN_ROOT/$split/model.pt" || -e "$RUN_ROOT/$split/metrics.json" ) ]]; then
        printf 'Existing model/results in %s/%s; set RUN_ROOT to a new directory.\n' "$RUN_ROOT" "$split" >&2
        exit 1
    fi
done

trainer_help="$("$python_bin" "$trainer" --help)"
if [[ "$trainer_help" != *"--mixed-balanced-sampling"* || "$trainer_help" != *"--phi-terms"* ]]; then
    printf 'The server trainer is missing the current mixed-dataset/phi options. Update ALRNN-DSR-extended.\n' >&2
    exit 1
fi

printf 'Environment: %s\nPython: %s\nData: %s\nResults: %s\nGPU: %s\n' \
    "$VENV_DIR" "$python_bin" "$DATA_ROOT" "$RUN_ROOT" "$gpu_id"
for split in "${cases[@]}"; do
    run_dir="$RUN_ROOT/$split"
    command=("$python_bin" "$trainer"
        --data-dir "$DATA_ROOT/$split"
        --multi-trajectory --phi-terms additive
        --latent-dim "$latent_dim" --pwl-units "$pwl_units"
        --sequence-length "$sequence_length" --batch-size "$batch_size"
        --epochs "$epochs" --batches-per-epoch "$batches_per_epoch"
        --teacher-forcing-interval "$teacher_forcing_interval"
        --mixed-balanced-sampling --no-best-model --t-transient 0
        --seed "$seed" --gpu-id "$gpu_id" --cpu-threads "$cpu_threads"
        --output-dir "$run_dir")
    printf '\nTraining %s\n' "$split"
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$dry_run" == 1 ]]; then
        continue
    fi
    mkdir -p -- "$run_dir"
    log_file="$run_dir/train_$(date -u +%Y%m%dT%H%M%SZ).log"
    # pipefail stops the queue on a failed training, even when tee succeeds.
    "${command[@]}" 2>&1 | tee "$log_file"
done

if [[ "$dry_run" == 1 ]]; then
    printf '\nDry run complete; no training or result directories created.\n'
else
    printf '\nAll three trainings completed. Results: %s\n' "$RUN_ROOT"
fi
