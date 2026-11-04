# Lorenz: short transients from two exact initial states

This isolated experiment adds no changes to the shared generator, trainer or
previous experiment launchers. It imports their simulator, compact data format,
balanced sampler, checkpoint loader and GPU queue. `lorenz_paper_test` provides
the separate two-initial-state bifurcation comparison.

## Data recipe

`config.json` generates the same two ranges, rho in [5,23] and [1,30], with
sigma=10, beta=8/3, dt=0.01. Each dataset contains:

| Mode | Distinct parameter schedules | Trajectories | Samples per trajectory |
| --- | ---: | ---: | ---: |
| Constant | 128 equally spaced values | 256 | 500 |
| Sinusoidal | 8 different upper bounds, one period | 16 | 500 |
| Linear increasing | 8 different upper bounds | 16 | 500 |
| Linear decreasing | 8 different upper bounds | 16 | 500 |

Every schedule is integrated once from (1,1,1) and once from (-1,-1,1), with no
perturbations, observation noise or burn-in. There are 304 trajectories, 152,000
samples per dataset, about 84.2% constant and 15.8% dynamic. Constant trajectories
cover the whole range. Dynamic schedules share the lower bound and vary their
upper bound from 25% to 100% of the range; full-range schedules are included.
No duplicate repetitions of the same deterministic initial-value problem are
generated. Trajectories cover t=0..4.99, rather than long attractor tails. Near a
bifurcation, this is not enough to guarantee convergence to a fixed point; the
two initial states represent both symmetry-related sides, not preconverged
fixed-point samples.

The two signs are paired within every schedule, including every constant rho.
The exact initial states and their indices are stored in separate arrays and in
metadata. This is 50/50 in the data and in expected sampling weight, not an
enforced 50/50 count in every minibatch. The unchanged balanced sampler balances
constant rho values and stratifies valid window starts for dynamic trajectories.
All trajectories have equal length, so the 84.2/15.8 mixture is preserved in
expectation. Window starts remain random; not every window includes t=0.

The training prefix is 495 samples; the last five (1%) are retained solely to
satisfy the trainer's existing train/test contract. Scaling is fitted globally
on training prefixes only. Storage is compact float32, with no padded trajectories
and no duplicate `trajectories_clean.npy` when noise is zero.

Run locally from `ALRNN-DSR-extended` using your data-generation conda environment:

```powershell
python alrnn_python/lorenz_transient_training/generate_data.py --dry-run
python alrnn_python/lorenz_transient_training/generate_data.py --skip-completed --restart-incomplete
```

The config defaults to the `config.json` next to the script, independent of the
working directory. `--config PATH` selects another recipe; this is the isolated
recipe schema, not the old `generate_batch_datasets.py` schema. Relative
`output_root` in the recipe is interpreted below the existing generator's
`datasets` directory. `--output-root PATH` overrides the full destination.
Default local output:

```text
generate_trainingdata/mixed_parameter_trajectories/datasets/lorenz_transients_two_initials/
  lorenz_rho_5_to_23/
  lorenz_rho_1_to_30/
```

`--skip-completed` checks configuration/code identity and required files.
`--restart-incomplete` preserves an incomplete dataset under `.incomplete` and
starts that dataset again; it cannot resume individual trajectories. Completed
datasets are never overwritten. Simulation failures abort without changing the
specified initial conditions. The sibling `generate_trainingdata` source tree
is required locally for generation, not on the training server.

## Server training and final analysis

Copy the generated `lorenz_transients_two_initials` folder to:

```text
/export/home/klenkeit/training_data/mixed_parameter_trajectories/lorenz_rho/lorenz_transients_two_initials
```

Copy/pull the updated ALRNN repository, then run from its root:

```bash
bash alrnn_python/lorenz_transient_training/run_lorenz_transients.sh 0 1 2 3
```

One GPU also works. Four independent trainings are queued: each dataset with
Cphi (300 latent / 150 PWL) and Cphi + shifted ReLU (200 / 100). Defaults remain
1000 epochs, 100 batches/epoch, batch size 80, sequence length 200, training seed
7, length weighting plus mixed balanced sampling. The shell activates
`$HOME/DynaMix-python-b-tipping/venv/bin/activate` like the existing launchers.
`ROOT_DIR`, `PYTHON_BIN`, `TRAINING_DATA_ROOT`, `RUN_ROOT`, `EPOCHS`, `CPU_THREADS`
and `NUM_PARAMETERS` can override defaults via environment variables. Use
`DRY_RUN=1` to print commands without training or writing outputs (the venv is
still activated). Python also accepts `--device cpu` for small smoke checks.

Results live only under this experiment's `results/<timestamp>/`:

- `models/<variant>/<dataset>/`: final checkpoint, metrics, training losses.
- `analysis/bifurcations/`: comparison and individual plots, with both exact
  initial states, x minima/maxima and stationary points, model's own 200-time-unit
  warm-up and a 50-unit observation window over 290 rho values in [1,30].
- `analysis/trajectories/<variant>/<dataset>/`: separate early/late x,y,z trace
  comparisons and time-colored 3D ground-truth/model panels for both ICs at rho
  5,14,23,28, from independent 50-time-unit free rollouts. Rho=28 is an
  extrapolation check for the [5,23] model. Arrays and divergence diagnostics are
  retained. Normalization is preserved, using the [5,23] dataset's coordinates
  for comparisons, as in the separate Lorenz paper test.
- `plan.json`, `logs/`, `gpu_ids.json`: settings, progress and live GPU queue.

Each analysis folder has a single provenance JSON for its plots. Models and
ground truth are not overlaid in bifurcation or 3D panels; time traces use two
lines for direct comparison. Non-finite model trajectories are recorded. No
additional packages are required beyond the existing project environments.

Progress and adding GPUs follow the existing queue mechanism:

```bash
tail -f <printed-run-folder>/logs/<timestamp>/*.log
printf '[0,1,2,3]\n' > <printed-run-folder>/gpu_ids.json
```

At most four trainings, then two variant analysis pipelines, run concurrently.
Training and analysis share the same live GPU queue. To repeat only the final
analyses for this new experiment (including reusing matching bifurcation caches):

```bash
RUN_ROOT=/absolute/path/to/lorenz_transient_training/results/<run> \
ANALYSIS_ONLY=1 bash alrnn_python/lorenz_transient_training/run_lorenz_transients.sh 0 1
```

## What does the withheld tail do?

In the existing trainer, gradients come exclusively from training-prefix windows.
The periodic control metrics even use a training prefix. The launcher selects
`--no-best-model`, so the final training state is saved without metric-based
checkpoint selection. The tiny test tail is deliberately skipped for final
in-trainer evaluation; it is too short to provide useful attractor metrics.
The independent analyses above replace it. Removing the tail completely would
require changing the shared data/trainer contract, so this isolated experiment
keeps 1%. Increasing the training fraction also changes the data used to fit
normalization, so this recipe is a new experiment, not an exact single-factor
comparison with the previous noisy dataset.
