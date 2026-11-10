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

## How balanced sampling works within and across batches

The defaults below apply separately to each of the four trained models. Each
batch contains 80 windows of 200 steps drawn from the 495-sample training
prefixes. Inputs use indices `start..start+199`; their one-step-shifted targets
use `start+1..start+200`. Valid starts are therefore 0..294, inclusive. Test
samples never enter these windows.

1. First draw 80 trajectory indices with replacement, weighted by their training
   lengths. All 304 trajectories have the same training length here. The expected
   batch composition is `80*256/304 = 67.37` constant windows and `80*48/304 =
   12.63` dynamic windows. These counts fluctuate; there is no fixed per-mode
   quota. Each dynamic mode contributes about 4.21 windows per batch on average.
2. Keep the number and positions of the constant slots, but redistribute those
   slots over the 128 distinct constant rho values. Each value gets the same
   integer number of slots; any remainder is assigned to a random subset of
   values. Since even a fully constant batch has at most 80 slots, this recipe
   selects distinct constant values within a batch. For example, 67 constant
   slots mean 67 randomly selected values with one window each. Across batches,
   all values have equal expected frequency, but there is no shuffled global
   cycle that guarantees all 128 values before a value can recur.
3. Within each selected constant value, choose between its two trajectories
   with equal probability, because they have equal training lengths. The exact
   initial states are paired 50/50 in the dataset and have equal expected sampling
   weight. Neither batches nor epochs enforce an exact 50/50 initial-state count.
   Dynamic trajectory choices from step 1 are left unchanged.
4. For a constant trajectory, choose a valid start uniformly at random. For a
   dynamic trajectory, split the 295 valid starts into ten contiguous sections
   of 29 or 30 starts. Each trajectory maintains its own randomly shuffled
   ordering of those ten sections. Each time it is selected, take the next
   section and choose a start uniformly within it. After all ten sections have
   been used, reshuffle and begin the next cycle.

The dynamic section counters persist across batches and epochs; they are not
reset at an epoch boundary. Within every completed ten-selection cycle of one
dynamic trajectory, each section is used exactly once. This balances window
start times, not individual sample coverage or parameter values. A sinusoid
spends more time near its bounds, and linear sweeps with different spans have
different speeds, so temporal balance does not guarantee a uniform rho histogram.
Overlapping windows also mean that some states appear in more windows than
others. In particular, the first point is not explicitly oversampled. Transients
receive more emphasis through the short trajectory length itself.

One optimizer update uses the mean squared error of all 80 windows, followed by
one backward pass and one optimizer step. The observed coordinates are injected
every 16 rollout steps using the existing teacher-forcing scheme. There are 100
updates per epoch and 1000 epochs by default. Here an epoch is a bookkeeping unit
of 100 newly sampled batches, not a complete traversal of the dataset. The
learning-rate scheduler steps after every epoch; optimizer state, model state
and dynamic sampling cycles continue across epoch boundaries.

## How the bifurcation analysis differs from earlier extended runs

Bifurcation analysis generates new constant-parameter simulations; it does not
reuse training windows or concatenate the dynamic training schedules. The new
Lorenz analysis evaluates 290 rho values in [1,30]. The comparison below describes
the existing standard mixed-run launcher/backend defaults; command-line overrides
can change the older settings.

| Property | Earlier standard extended comparison | This Lorenz experiment |
| --- | --- | --- |
| Physical initial states | One draw per parameter from the regime's Gaussian initial distribution | Both exact states (1,1,1) and (-1,-1,1) at every parameter |
| True-system burn-in | 5000 steps, then 500 context steps by default | Integrate from t=0; retain t=200..250 |
| ALRNN initialization | Seed from the last ground-truth context state, already after the true-system burn-in | Seed at t=0 from each exact physical initial state |
| Model settling before extracting points | No separate long model warm-up; extract maxima over the forecast | Model independently evolves to t=200 before extracting points |
| Extraction window | 15000 model steps; true window defaults to the same length | 50 time units, or 5000 steps at dt=0.01, including the starting observation at t=200 |
| Values plotted | Strict local maxima of the selected component; the standard launcher defaults to z, with x selectable | Local minima and maxima of x, or one stationary x per converged initial state |
| Stable branches | A constant series has no strict local maximum and can disappear | Explicit stationary-observation detection retains its x value |

Both true dynamics and the ALRNN therefore have their own full evolution from
the specified initial state in the new test. The model receives no ground-truth
state after initialization, and there is no teacher forcing during analysis.
This checks whether it reaches the appropriate long-term behavior from both
sides, rather than receiving an already settled true-system state as its start.
Each parameter is simulated independently, without continuation from the previous
parameter's terminal state.

Stationarity requires all three standardized observations to have sufficiently
small ranges over the whole observation window, and a small final rate. Defaults
are `stationary_atol=1e-5`, `stationary_rtol=1e-5`, and `stationary_rate=1e-3` per
time unit. If stationary, plot the mean x of that tail; otherwise extract x
minima/maxima. No extrema without stationarity does not automatically become a
fixed-point marker. This detects stationary observations, not a mathematical
proof of a stationary latent state. Non-finite model rollouts are excluded from
the bifurcation points and recorded in diagnostics.

Normalization remains fixed to training statistics, with every model converted
to the [5,23] dataset's reference coordinates for comparison. Ground truth and
each trained model are shown in separate vertically arranged panels.

## How the trajectory test plots are generated

`plot_trajectories.py` creates the files under
`results/<run>/analysis/trajectories/<variant>/<dataset>/` after training and
the variant's bifurcation comparison. These are independent diagnostic
simulations, not plots of the withheld five-sample tails. They reuse the physical
initial states used in training, and mainly probe familiar parameter ranges;
they should not be interpreted as a wholly unseen held-out dataset.

With the launcher's defaults, each trained model is evaluated at constant
`rho = 5,14,23,28`, from both exact physical initial states. This gives eight
model trajectories per checkpoint and 32 over all four checkpoints. Rho=28 is
outside the [5,23] model's training range, but inside the [1,30] model's range.
There is no parameter drift in these diagnostic trajectories.

The simulations proceed as follows:

1. Simulate the true Lorenz equations with sigma=10 and beta=8/3 using SciPy's
   DOP853 solver (`rtol=1e-7`, `atol=1e-9`, maximum internal step `5*dt`). Sample
   from t=0 to t=50 at the training interval dt=0.01: 5001 states including t=0.
   There is no initial perturbation, observation noise or burn-in.
2. Load the trained checkpoint. Normalize the exact physical initial state using
   that model's own training mean/std. Initialize its latent state as `x0 @ B`,
   then overwrite the first three latent coordinates with the normalized observed
   x0, matching the existing ALRNN initialization convention. Remaining latent
   coordinates retain their values from `x0 @ B`.
3. Supply the fixed normalized parameter `phi = (rho-boundary)/scale` on every
   model step (Lorenz defaults: boundary=24.74, scale=23.74). Free-run the model
   for 5000 steps. No observed states are injected after t=0: no teacher forcing,
   reinitialization, alignment to truth or parameter updates. The t=0 state is
   included in the saved result. Although the reused rollout helper receives a
   retention start of one step, that first model step is retained and the exact
   t=0 state is prepended; no physical warm-up is discarded here.
4. Convert the model output back through its own physical coordinates, then
   normalize both truth and model using the [5,23] dataset's fixed statistics.
   This gives a common coordinate system across models and matches the
   bifurcation analysis. There is no per-test-trajectory rescaling.

For every rho, two PNGs are saved:

- `rho_<value>_time_series.png`: three rows for x,y,z and four columns. Left to
  right: positive IC at t=0..5, positive IC at t=40..50, negative IC at t=0..5,
  negative IC at t=40..50. Each panel overlays the ground-truth and ALRNN curves.
  The late panels are slices of the same uninterrupted t=0..50 simulation, not
  new rollouts initialized at t=40. Thus they also reflect accumulated error.
- `rho_<value>_3d.png`: two rows for the two ICs and two columns for ground truth
  and model. It uses the entire t=0..50 trajectory, with a shared time colorbar
  and shared coordinate limits between truth/model within each row.

There are eight PNGs per checkpoint and 32 trajectory PNGs for the complete
default experiment. Different freely evolving chaotic trajectories can separate
pointwise even when their long-term geometry is similar; the late time traces
are therefore complementary to the bifurcation and 3D comparisons.

`orbits.npz` in each dataset folder stores time, rho, initial-state index,
standardized ground-truth and model arrays, and the first non-finite model step
(-1 means none). The default truth/model array shapes are `(8,5001,3)`; rows are
ordered by rho, with the positive IC then the negative IC for each rho.
Non-finite model trajectories retain their finite prefixes for plotting and have
NaN suffixes. `trajectory_sources.json`, one per variant folder, records all
plots, actual parameters/duration, coordinates, checkpoint/code hashes and failure
indices. The rollout files and this JSON allow inspection without retraining;
these trajectory plots do not compute an additional quantitative test loss or
affect the saved checkpoint.

The default duration can be changed with `run_training.py --orbit-duration`;
standalone `plot_trajectories.py` also accepts `--duration` and `--parameters`.
Early panels show t=0..min(5,duration), and late panels show the last ten time
units (or the full interval if shorter). For a particular server run, consult
its `plan.json`, analysis logs and `analysis/trajectories/<variant>/trajectory_sources.json`
for the actual settings rather than assuming the defaults above.
