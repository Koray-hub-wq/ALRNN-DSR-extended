# Ensemble experiment under parameter drift

This experiment evaluates existing checkpoints. All new code, configuration and
default results stay in `alrnn_python/parameter_drift`; the older single-trajectory
evaluation is unchanged. The existing model loader, ODE definitions and
single-trajectory integration are imported without modifying their files.

## Start on the Linux server

From `/export/home/klenkeit/ALRNN-DSR-extended`:

```bash
bash alrnn_python/parameter_drift/run_halvorsen_ensemble_drift.sh 0
```

The script activates `${ROOT_DIR:-$HOME}/DynaMix-python-b-tipping/venv` and uses
GPU 0. Select another GPU by changing the final number. ODE integration runs on
the CPU; model trajectories run in batches on the selected GPU. No training is
performed. Default model:

```text
/export/home/klenkeit/ALRNN-DSR-extended/alrnn_python/runs/halvorsen_9params_train_m1_m0p75_m0p5_m200_p20_cphi
```

`halvorsen_ensemble_config.json` specifies:

| Setting | Default |
| --- | --- |
| Ensemble size | 200 independent initial-state draws, paired one-to-one with model rollouts |
| Regime | `halvorsen_a_chaotic_to_periodic` |
| Raw parameter a | 1.4 to 2.09, linear increasing |
| Stored states | 4000, including the post-burn-in initial state |
| Fixed-parameter burn-in | 5000 integration steps at a = 1.4 |
| Sampling interval | 0.01 |
| Random seed | 8 |
| Model | Cphi, M=200, P=20, read from the checkpoint |
| Ensemble component | z |
| Plot coordinates | Physical observations |

The drift lasts `(4000 - 1) * 0.01 = 39.99` time units. Index 0 is the final
burn-in state and the initial state of both trajectories. No drifting states
are discarded. The true ODE sees continuously varying a(t); `phi[t]` drives the
model transition from stored state t to t+1. Latent initialization follows the
existing implementation: `latent = x0 @ B`, with the first three latent
coordinates overwritten by x0. No context observations, model burn-in or teacher
forcing follow this initialization.

Regime definitions are found in `$HOME/diffrentbutconst_prams/` on the server or
in the neighboring generator repository locally. Override with
`REGIME_FILE=/path/to/generate_dynamic_regime_trajectories.py` or Python option
`--regime-file`. Standard dependencies are NumPy, SciPy, PyTorch and Matplotlib.
There are no new Python package requirements.

## Results

Default directory, relative to the repository root:

```text
alrnn_python/parameter_drift/results/halvorsen_ensemble_linear_4000/
```

- `ensemble_z.png`: all ensemble z values at every stored parameter value;
  ground truth above the model, in separate panels with shared axes. No samples
  are removed for plotting.
- `maxima_z.png`: local maxima pooled over all trajectories, again in separate
  ground-truth/model panels. No smoothing, detrending or prominence filtering is
  applied by default.
- `peak_intervals_z.png`: time to the next local maximum of the **same**
  trajectory, plotted at the parameter value of the earlier maximum. The
  interval is measured in time units, not steps. The CSV also stores steps.
- `first_trajectory_components.png`: x, y and z of the first paired trajectory,
  with truth and model overlaid within each component panel, plus a(t) below.
- `first_trajectory_3d.png`: first ground-truth/model trajectories side by side,
  using common spatial bounds and a shared time color scale. The black dot is
  the initial observation.
- `first_trajectory_3d.html`: rotatable counterpart with separate 3D panels.
  It uses Plotly JavaScript from a CDN, without the Python Plotly package. Pass
  `--plotly-js /path/to/plotly.min.js` to embed an existing bundle for offline use.
- `maxima_z.csv`: each detected maximum, with source, trajectory index, sample
  index, time, raw parameter, phi, value and next-peak interval. Indices start at
  zero. The value uses the selected plot coordinates. Source IDs map to the
  checkpoint entries in `ensemble_sources.json`.
- `ground_truth.npz`: physical observations (`observed`), standardized/model
  observations (`x`), per-trajectory `mean`/`std`, common `times`/`parameter`/
  `phi`, pre-burn-in initial states, retry counts and metadata. Observations have
  shape `(ensemble, stored states, 3)`; mean/std have shape `(ensemble, 3)`.
- `model_01_*.npz`: aligned model-coordinate `prediction` and `failed_at` per
  trajectory (`-1` means no numerical failure). Physical predictions are
  `prediction * std[:, None, :] + mean[:, None, :]` from the paired truth cache.
- `ensemble_sources.json`: one provenance file for all plots, including
  checkpoint hashes, ground-truth hash, simulation/scaling conventions,
  integration retries, peak settings, counts and numerical failures. Each entry
  in `plots` points to an `analysis_id` in `analyses`; replotting another component
  preserves the settings/model identity of the earlier component's images.

`scipy.signal.find_peaks` is applied separately to each trajectory and each
finite segment. A flat peak uses its middle sample. Endpoints cannot be peaks;
the last peak in each segment has no observable next-peak interval. Maxima are
never paired across trajectories or across non-finite gaps. A numerically
diverging model trajectory keeps its finite prefix; its remaining values are
NaN and its failure is reported. Ground-truth integration failures retry fresh
initial states, up to 10 attempts. Different trajectories have independent seed
streams, so a retry does not change later trajectories' initial states.

These are finite-speed drift diagnostics. Initial conditions, phase dispersion,
transients and delayed response to the drifting parameter can all affect the
clouds. They do not represent equilibrated constant-parameter bifurcation data.
Peak heights and inter-peak times provide complementary views of oscillation
amplitude and timing; neither alone proves chaos.

## Scaling

The selected model was trained on independently standardized constant-parameter
trajectories. The default follows that existing convention: one mean and std
per component of each **complete recorded ground-truth drift trajectory**, held
fixed throughout its paired model rollout. The statistics therefore include
future truth observations. This is an offline choice of coordinates, not an
online estimator, and does not feed subsequent truth states to the model.

Plotting defaults to physical observations so the ensemble distribution is not
altered visually by different per-trajectory scalings. Model predictions are
inverted with the same mean/std as their paired truth. Drift reconstruction
errors can reflect this normalization choice as well as the learned dynamics.

For controlled fixed scaling, pass `--normalization fixed --normalization-stats
/path/to/stats.npz`, with length-three mean/std vectors, and select a new output
directory. Use `--normalization none` only for a model trained in physical
coordinates. Models trained with global mixed-data scaling should use that
training scaling rather than the preset's per-trajectory convention.

Phi always uses the original regime's fixed reference:
`phi = (a - 1.75) / 0.35`. The default endpoints are phi=-1 and approximately
0.97143; the raw endpoint 2.09 is intentional. A restricted drift does not
renormalize its endpoints to [-1, 1].

## Configure, reuse and replot

Edit the JSON or pass Python options after the GPU ID. Paths in JSON resolve
relative to the JSON's directory; paths passed on the command line resolve
relative to the shell's working directory. For example:

```bash
# Validate and print the resolved experiment without simulation or CUDA checks.
bash alrnn_python/parameter_drift/run_halvorsen_ensemble_drift.sh 0 --dry-run

# Smaller experiment: use another directory to keep the default cache intact.
bash alrnn_python/parameter_drift/run_halvorsen_ensemble_drift.sh 0 \
  --n-trajectories 20 --output-dir alrnn_python/parameter_drift/results/ensemble_20

# Regenerate plots and peak tables with another component; no inference/ODEs.
bash alrnn_python/parameter_drift/run_halvorsen_ensemble_drift.sh 0 \
  --plot-only --component x

# Compare another checkpoint on exactly the same cached truth.
bash alrnn_python/parameter_drift/run_halvorsen_ensemble_drift.sh 0 \
  --model-runs alrnn_python/runs/ANOTHER_MODEL
```

Existing truth caches are reused only if the simulation settings, ensemble size,
regime source and fixed scaling source agree. A mismatch raises an error. Use a
different `--output-dir` (or `--ground-truth`) for another experiment; explicitly
`--regenerate-ground-truth` replaces an existing cache. Normal evaluation repeats
model inference; `--plot-only` uses saved predictions and needs neither CUDA nor
the checkpoint files. The regime definition must still be available to validate
the cache. Replotting preserves the model identity recorded with those arrays.

To produce only ground truth use `--ground-truth-only`. To run without a GPU use
`--device cpu`. Additional peak filtering is optional:
`--peak-prominence 0.05 --peak-distance-steps 5`; prominence uses the selected plot
coordinates. The default retains every detected local maximum.

Tests, from `alrnn_python/parameter_drift`, with the environment active:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest test_ensemble_drift test_parameter_drift
```
