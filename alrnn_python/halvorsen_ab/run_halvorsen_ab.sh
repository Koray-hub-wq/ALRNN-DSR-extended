#!/usr/bin/env bash
set -euo pipefail
# Activate your Python environment first, or set PYTHON_BIN.
# bash alrnn_python/halvorsen_ab/run_halvorsen_ab.sh [GPU_ID] [extra arguments ...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu_id="${1:-${GPU_ID:-0}}"
if (( $# > 0 )); then shift; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
exec "${PYTHON_BIN:-python}" -u "$SCRIPT_DIR/run_experiment.py" --gpu-id "$gpu_id" "$@"
