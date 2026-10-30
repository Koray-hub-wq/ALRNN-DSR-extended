#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_dadras_cphi_without_phi1.sh [GPU_ID ...] (one to four GPUs)
# Cphi only: all/negative/positive training and bifurcations.
# One variant runs its three trainings sequentially on the first supplied GPU.
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

latents_dim=300
epochs=750
pwl_units=150
batch_size=80
batches_per_epoch=100
cpu_threads=4

extra_args=()
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  extra_args+=(--dry-run)
fi

# Reuse the runner without changing its defaults for any other launcher.
exec "$python_bin" -u - "$SCRIPT_DIR" \
  --regime dadras_e_49_to_64 \
  --training-data-root "$TRAINING_DATA_ROOT" \
  --backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  --latent-dim "$latents_dim" \
  --epochs "$epochs" \
  --pwl-units "$pwl_units" \
  --batch-size "$batch_size" \
  --batches-per-epoch "$batches_per_epoch" \
  --cpu-threads "$cpu_threads" \
  --gpu-ids "${gpu_ids[@]}" \
  "${extra_args[@]}" <<'PY'
from pathlib import Path
import sys

script_dir = Path(sys.argv.pop(1))
sys.path.insert(0, str(script_dir))
import run_term_ablations as runner

# Other variants disabled for this launcher only:
# "diagonal": ("diagonal",),
# "diagonal_relu": ("diagonal-relu",),
# "shifted_relu": ("shifted-relu",),
# "additive_diagonal": ("additive", "diagonal"),
runner.VARIANTS = {"cphi": ("additive",)}
raise SystemExit(runner.main(sys.argv[1:]))
PY
