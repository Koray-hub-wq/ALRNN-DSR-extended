#!/usr/bin/env bash
set -euo pipefail

# Six trainings (Cphi 300/150 and Cphi + shifted ReLU 200/100), then bifurcations.
# The shared launcher activates the b-tipping venv and accepts one to six GPUs.
# Usage: bash alrnn_python/run_dadras_combined_balanced_cphi.sh [GPU_ID ...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_mixed_combined_balanced.sh" dadras "$@"
