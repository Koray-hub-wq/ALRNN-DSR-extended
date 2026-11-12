# Rossler-c: 50% transient windows, 50% attractor windows

This folder contains only the sampler, training launchers and trained-model
analyses. Both datasets train additive Cphi models with 300 latent dimensions /
150 PWL units by default. The shared ALRNN trainer remains unchanged.

## Input datasets

The new wide-initial-condition profile uses 30 attractor trajectories of 50,000
steps at 10 c values and 500 transient trajectories of 500 steps, with 5%
observation noise. Its data-generation config and instructions are in
[the simulation folder](../../../generate_trainingdata/mixed_parameter_trajectories/README_rossler_c_transients_70_30_wide_ic.md).
Start that profile with:

```bash
bash alrnn_python/rossler_c_transient_training/run_rossler_c_transients_70_30_cphi.sh 2
```

It defaults to the `rossler_c_transients_70_30_wide_ic` data folder and 70/30
sampling, exactly 56 attractor / 24 transient windows per batch of 80. Parameter
balancing is performed separately inside each population. `--attractor-fraction`
or shell `ATTRACTOR_FRACTION` selects the proportion; batch size must permit
integer counts in both populations. The actual counts and proportions are saved
in plan.json and metrics.json. The following sections describe the original
50/50 profile and its defaults; its existing launcher keeps those defaults.

Data-generation code, JSON configuration and simulation tests live outside this
repository in `generate_trainingdata/mixed_parameter_trajectories`, as for the
existing experiments. See the local
[simulation instructions](../../../generate_trainingdata/mixed_parameter_trajectories/README_rossler_c_transients_50_50.md).
No generator source tree is needed on the training server.

Expected datasets cover c in [4.5,5.5] and [3,6], with 30 long attractor trajectories
first (100,000 steps, 85/15 split), 700 short transients (500 steps, 99/1 split),
and 5% observation noise. The compact arrays must include
`trajectory_population.npy`, `trajectory_lengths.npy` and
`trajectory_train_lengths.npy`; training uses those saved labels and lengths.

## Exact batch composition

`HalfHalfDataset` replaces the sampler only in the isolated `train_cphi.py`
process. Every batch of 80 contains **exactly 40 attractor windows and 40 transient
windows**, even though the raw counts and training lengths are very different.
There is no length-weighted choice between populations.

Within each population, distribute slots as equally as possible among that
population's distinct parameter values, assigning any remainder to a random
subset. For the defaults:

- Attractor: all 30 c values appear in every batch; ten randomly selected values
  receive a second window, giving 40 windows.
- Transient: 40 of the 140 c values are chosen without repetition in the batch.
  For each selected value, choose uniformly among its five trajectories.

The choice of surplus/subset is random each batch; there is no global cycle
over parameter values. A c value shared between populations belongs to one pool
in each population, not a globally merged pool. This prevents the overlapping
endpoints or differing grids from changing the 50/50 mixture. Within the chosen
trajectory, the valid start is uniform and all shifted targets remain inside
its training prefix. For a 200-step window, the short trajectory's valid starts
are 0..294; the long trajectory's are 0..84,799. Windows may overlap; the first
transient point is not explicitly oversampled.

The 50/50 mixture is exact for every batch and therefore every epoch and complete
run. Each population's parameter values have equal expected frequency across
batches. Effective selection probabilities and actual per-trajectory/population
counts are saved in `metrics.json` under `mixed_sampling`.

## Server training

Copy the generated group folder to:

```text
/export/home/klenkeit/training_data/mixed_parameter_trajectories/rossler_c/rossler_c_transients_50_50
```

With the updated ALRNN repository on the server, run from its root:

```bash
bash alrnn_python/rossler_c_transient_training/run_rossler_c_transients_cphi.sh 0 1
```

One GPU also works: pass only `0`. There are two independent training jobs, one
per parameter range, followed by a comparison bifurcation job and two independent
free-trajectory jobs. The shared GPU queue can use additional GPUs during the
analysis phase and reads a live `gpu_ids.json` throughout both phases.

The shell activates `$HOME/DynaMix-python-b-tipping/venv/bin/activate`. The existing
external bifurcation backend must be available at
`$HOME/diffrentbutconst_prams/compare_model_bifurcation_diagrams.py`, as for previous
extended runs. Override its folder with `BIFURCATION_DIR` if necessary.

Training defaults: 1000 epochs, 100 batches/epoch, batch size 80, window length
200, seed 7, teacher forcing every 16 steps and the shared optimizer/learning-rate
schedule. Every batch produces one optimizer update. `--no-best-model` preserves
the final model instead of selecting a checkpoint using the control metrics.

Periodic Dstsp / power-spectrum metrics use the first 10,000 **training** samples
of the first long trajectory. Merely putting a long trajectory first does not
switch periodic monitoring to the held-out test section. The default monitor
interval is 20 epochs (epochs 1,21,41,... in the shared trainer).

At the end, only **four** evenly spaced long parameter groups are evaluated on
their held-out tails. Each can produce a time-series and a 3D plot, plus metrics.
The default tail evaluation uses 1000 settling steps followed by up to 10,000
metric steps, fitting inside the 15,000-sample tail. Short tails are not included.
To disable this extra evaluation entirely:

```bash
TAIL_EVALUATIONS=0 bash alrnn_python/rossler_c_transient_training/run_rossler_c_transients_cphi.sh 0 1
```

The skipped mode retains one explicitly skipped entry to satisfy the unchanged
trainer's metrics schema. This never changes gradients or periodic monitoring.
The independent free comparisons described below always run.

Other shell overrides are `TRAINING_DATA_ROOT`, `RUN_ROOT`, `ROOT_DIR`, `PYTHON_BIN`,
`EPOCHS`, `CPU_THREADS`, `NUM_PARAMETERS`, and `DRY_RUN=1`. Direct Python CLI options
also allow small CPU checks and explicit analysis durations. No new packages are
needed beyond the existing training/data environments.

## Final bifurcations and free trajectories

Outputs stay under this experiment's `results/<timestamp>/`:

- `models/cphi/<dataset>/`: checkpoint, metrics, loss curve and at most four
  held-out-tail comparisons.
- `analysis/bifurcations/cphi/`: the existing extended comparison over 320 c
  values spanning [3,6], with z maxima, fixed training normalization, ground truth
  and both models in separate panels. It uses the standard true-system burn-in
  of 5000 steps, context of 500 steps and 15,000-step forecast, rather than the
  special Lorenz minima/maxima/stationarity procedure. Valid existing caches are
  reusable. Provenance JSON is shared across plots in the folder.
- `analysis/trajectories/<dataset>/`: independent free simulations at constant
  c=3,4,4.5,5,5.5,6 for two deterministic Gaussian initial draws (seed 8), shared
  across parameter values, models and their ground-truth counterparts. They
  cover t=0..50 with dt=0.01, no burn-in, no teacher forcing and no observation
  noise in the newly simulated truth. Each model is initialized from the same
  physical IC in its own training coordinates. Truth/model are converted to the
  first dataset's common standardized coordinates for plotting.
- `plan.json`, `gpu_ids.json`, `logs/<timestamp>/`: settings, GPU queue and logs.

At each of the six test c values, one time-series figure shows x/y/z and both
ICs at t=0..5 and t=40..50; late segments come from the same uninterrupted free
rollout. A second figure shows full time-colored 3D trajectories with truth/model
side by side. This gives 12 free-comparison PNGs per checkpoint, 24 total.
For the [4.5,5.5] model, c=3,4,6 are marked as extrapolation. For [3,6], all test
values are within the training range. `orbits.npz` retains the arrays and
divergence indices; one `trajectory_sources.json` per dataset records settings,
initial states, checkpoint/code hashes and all its plots.

View progress or expand the GPU queue using the printed run folder:

```bash
tail -f <run>/logs/<timestamp>/*.log
printf '[0,1]\n' > <run>/gpu_ids.json
```

To repeat only the final bifurcation and independent free-trajectory analyses:

```bash
RUN_ROOT=/absolute/path/to/rossler_c_transient_training/results/<run> \
ANALYSIS_ONLY=1 bash alrnn_python/rossler_c_transient_training/run_rossler_c_transients_cphi.sh 0 1
```

This does not retrain or regenerate the small selection of in-trainer tail plots.
