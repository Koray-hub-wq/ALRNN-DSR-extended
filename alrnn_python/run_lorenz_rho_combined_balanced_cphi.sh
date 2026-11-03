#!/usr/bin/env bash
set -euo pipefail

# Four trainings: two datasets x (Cphi 300/150 or Cphi + shifted ReLU 200/100).
# The shared launcher activates b-tipping's venv and queues jobs on selected GPUs.
# Usage: bash alrnn_python/run_lorenz_rho_combined_balanced_cphi.sh [GPU_ID ...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_mixed_combined_balanced.sh" lorenz_rho "$@"
