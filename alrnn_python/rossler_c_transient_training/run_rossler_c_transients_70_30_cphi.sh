#!/usr/bin/env bash
set -euo pipefail

# New wide-IC experiment; the original launcher's default remains 50/50.
# bash alrnn_python/rossler_c_transient_training/run_rossler_c_transients_70_30_cphi.sh [GPU_ID ...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-/export/home/klenkeit/training_data/mixed_parameter_trajectories/rossler_c/rossler_c_transients_70_30_wide_ic}"
export ATTRACTOR_FRACTION="${ATTRACTOR_FRACTION:-0.7}"
exec bash "$SCRIPT_DIR/run_rossler_c_transients_cphi.sh" "$@"
