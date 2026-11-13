#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_halvorsen_ab_server.sh [GPU_ID [CPU_THREADS]]
# Paths can be overridden with VENV_DIR, SIMULATOR_ROOT and PYTHON_BIN.
gpu_id="${1:-${GPU_ID:-0}}"
cpu_threads="${2:-${CPU_THREADS:-4}}"
if (( $# > 2 )) || [[ ! "$gpu_id" =~ ^[0-9]+$ ]] || [[ ! "$cpu_threads" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Usage: bash %s [nonnegative GPU_ID [positive CPU_THREADS]]\n' "$0" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
venv_dir="${VENV_DIR:-$HOME/DynaMix-python-b-tipping/venv}"
simulator_root="${SIMULATOR_ROOT:-$HOME/diffrentbutconst_prams}"
python_bin="${PYTHON_BIN:-python}"

# Set limits before environment activation or any Python/package imports.
export OMP_NUM_THREADS="$cpu_threads" OMP_THREAD_LIMIT="$cpu_threads"
export OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1
export MKL_NUM_THREADS="$cpu_threads" MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS="$cpu_threads" GOTO_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads" NUMEXPR_MAX_THREADS="$cpu_threads"
export VECLIB_MAXIMUM_THREADS="$cpu_threads" BLIS_NUM_THREADS="$cpu_threads"
export RAYON_NUM_THREADS="$cpu_threads"
export MAX_JOBS="$cpu_threads" CMAKE_BUILD_PARALLEL_LEVEL="$cpu_threads"
export MAKEFLAGS="-j$cpu_threads"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1

printf '[1/4] Activate environment and install dependencies (GPU=%s, CPU threads=%s).\n' "$gpu_id" "$cpu_threads"
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  printf 'Environment not found: %s/bin/activate. Set VENV_DIR to your CUDA environment.\n' "$venv_dir" >&2
  exit 1
fi
source "$venv_dir/bin/activate"
cd -- "$repo_dir"
command -v "$python_bin" >/dev/null

# On Linux also restrict every Python process and its descendants to at most
# CPU_THREADS CPUs from the current allowed set (respect Slurm/container masks).
cpu_prefix=()
if command -v taskset >/dev/null 2>&1; then
  affinity_line="$(LC_ALL=C taskset -pc "$$")"
  allowed_cpus="${affinity_line##*: }"
  IFS=',' read -r -a cpu_ranges <<< "$allowed_cpus"
  selected_cpus=""
  selected_count=0
  for cpu_range in "${cpu_ranges[@]}"; do
    if [[ ! "$cpu_range" =~ ^([0-9]+)(-([0-9]+))?$ ]]; then
      printf 'Cannot parse allowed CPU list: %s\n' "$allowed_cpus" >&2
      exit 1
    fi
    first_cpu="${BASH_REMATCH[1]}"
    last_cpu="${BASH_REMATCH[3]:-$first_cpu}"
    for ((cpu=first_cpu; cpu<=last_cpu && selected_count<cpu_threads; cpu++)); do
      selected_cpus+="${selected_cpus:+,}$cpu"
      selected_count=$((selected_count + 1))
    done
    if (( selected_count >= cpu_threads )); then break; fi
  done
  if (( selected_count == 0 )); then
    printf 'No allowed CPUs found.\n' >&2
    exit 1
  fi
  cpu_prefix=(taskset -c "$selected_cpus")
  printf 'CPU affinity for all Python steps: %s\n' "$selected_cpus"
else
  printf 'taskset unavailable; numerical thread pools and build jobs are limited to %s.\n' "$cpu_threads"
fi

run_python() {
  "${cpu_prefix[@]}" "$python_bin" "$@"
}

run_python -m pip install --no-compile -r "$script_dir/requirements.txt"

printf '[2/4] Check/update the existing simulator.\n'
run_python "$script_dir/sync_simulator.py" --simulator-root "$simulator_root" --apply

printf '[3/4] Generate the training dataset (or verify/reuse completed data).\n'
run_python -u "$script_dir/run_experiment.py" --stage generate \
  --simulator-root "$simulator_root" --cpu-threads "$cpu_threads" --gpu-id "$gpu_id"

printf '[4/4] Train on GPU %s, then generate trajectory comparisons and the interactive HTML.\n' "$gpu_id"
exec "${cpu_prefix[@]}" "$python_bin" -u "$script_dir/run_experiment.py" \
  --simulator-root "$simulator_root" --cpu-threads "$cpu_threads" --gpu-id "$gpu_id"
