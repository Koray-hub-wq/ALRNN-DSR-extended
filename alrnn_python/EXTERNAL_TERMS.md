# Optional Parameter Terms

The Python training entry point `train_halvorsen_plain.py` supports

```text
z[t+1] = A * z[t] + W @ relu_P(z[t]) + h
         + C @ phi[t]
         + (D_phi @ phi[t]) * z[t]
         + (D_relu_phi @ phi[t]) * relu_P(z[t])
         + V_phi @ relu_P(z[t] + Q_phi @ phi[t])
```

The equation above uses column vectors; batches in the implementation use rows.
`A` is a diagonal vector and `*` is elementwise multiplication. `relu_P` applies
ReLU only to the last P latent dimensions. The `diagonal` term uses the full
state **before** ReLU, matching the experts in `DynaMix_diagonal_coupling`.
There is no full phi-dependent M-by-M matrix.

`diagonal-relu` applies a second, independently learned diagonal coupling to
`relu_P(z)`. Thus the last P dimensions can have different parameter-dependent
slopes on their positive and negative branches. D_relu_phi is separate from
D_phi; the two couplings never share weights.

`shifted-relu` adds a separate nonlinear branch with phi inside the activation.
Q_phi shifts the activation input, so its last P switching boundaries become
`z_i = -(Q_phi @ phi)_i`. V_phi mixes the resulting M-dimensional vector into
the next state. The original W branch remains present. Q_phi has shape
(M, phi_dim), and V_phi has shape (M, M). Q_phi is not the encoder B, which
continues to initialize the latent state from the first observation.

## Selecting Terms

| Training option | C phi | Raw diagonal | ReLU diagonal |
| --- | --- | --- | --- |
| `--phi-terms none` | Off | Off | Off |
| `--phi-terms additive` | On | Off | Off |
| `--phi-terms diagonal` | Off | On | Off |
| `--phi-terms diagonal-relu` | Off | Off | On |
| `--phi-terms additive diagonal` | On | On | Off |
| `--phi-terms additive diagonal-relu` | On | Off | On |
| `--phi-terms diagonal diagonal-relu` | Off | On | On |
| `--phi-terms additive diagonal diagonal-relu` | On | On | On |

The fourth term can be selected alone with `--phi-terms shifted-relu`, combined
with C phi using `--phi-terms additive shifted-relu`, or combined with all existing
terms using `--phi-terms additive diagonal diagonal-relu shifted-relu`. Any subset
of the four terms is supported, giving 16 possible selections including vanilla.

Without either `--phi-terms` or the legacy `--use-phi` flag, training stays
vanilla. `--use-phi` retains its previous meaning: additive C phi only.
Explicit `--phi-terms` takes precedence, even over `--use-phi`. Terms are
independent: the diagonal-only mode still loads phi but creates no C weights.
With no active terms, phi is not loaded as an external model input. Parameter
labels can still be used for dataset filtering and balanced sampling.

From `ALRNN-DSR-extended/alrnn_python`, for example:

```bash
python train_halvorsen_plain.py \
  --data-dir /export/home/klenkeit/training_data/single_new_regimes_const/trajectories/rossler_a_005_to_035 \
  --multi-trajectory \
  --phi-terms additive diagonal \
  --latent-dim 200 --pwl-units 20 \
  --batch-size 80 --epochs 800 --batches-per-epoch 100 \
  --gpu-id 2 --cpu-threads 4 \
  --output-dir runs/rossler_additive_diagonal_m200_p20
```

Change only the `--phi-terms` selection and output directory for other ablations.
Keep the existing optimizer, scheduler, teacher forcing, sampling and evaluation
settings. New C, D_phi and D_relu_phi weights start at zero. Each enabled term adds
M * phi_dim parameters; disabled terms have no trainable weights or checkpoint
tensors. A term is not considered disabled just because its weights are zero:
zero initialization is the starting point from which it learns.

`shifted-relu` instead adds M * phi_dim + M * M parameters. It initializes Q_phi
to zero and V_phi with Gaussian standard deviation 0.01, like W. A nonzero
projection lets the shift weights receive gradients immediately. Consequently
the branch initially adds V_phi @ relu_P(z), rather than starting at zero. Its
projection and shift are both trainable and independent of W and the encoder B.

The full extra projection increases capacity substantially compared with the
diagonal terms. There are also P additional rectified coordinates in this new
branch, alongside the P in the W branch; the CLI's P configures each branch.
At phi=0 its contribution overlaps with W @ relu_P(z), so W and
V_phi are not separately identifiable there. The first M-P activation dimensions
remain linear even after shifting. For P=0 this whole branch can be absorbed
into the existing linear dynamics and an additive phi input; the new nonlinear
expressivity is in phi-dependent activation boundaries when P>0.

The first M-P dimensions of relu_P(z) are unchanged. In these dimensions the raw
and ReLU coupling contributions therefore only identify the sum D_phi +
D_relu_phi when both are enabled. For P=0 they are equivalent everywhere. The
extra expressivity is in the last P dimensions; the separate weights need not
be uniquely identifiable in the linear block. This exact formulation is kept
for ablations rather than masking those first M-P dimensions.

For `--eval-only`, pass the same architecture and term flags used for training;
strict checkpoint loading rejects a different selection. Drift and bifurcation
evaluation instead infer active terms from the saved checkpoint automatically.

## Shared Model And Checkpoints

`alrnn_model.py` provides the shared recurrence, the trainable `AL_RNN`, and the
inference-only `SavedALRNN`. Training, parameter-drift evaluation and the local
bifurcation adapter all use this implementation. The original Julia code and
tutorial notebooks retain their separate implementations.

The checkpoint keeps the existing flat tensor names A, W, h, B and optional C,
adding optional D_phi, D_relu_phi and the pair Q_phi/V_phi. Old vanilla, C-only and raw-diagonal
checkpoints remain loadable.
`metrics.json` records the resolved term selection in `args.phi_terms` and the
complete architecture in `model_config`. Loading validates metadata against the
saved tensors so a coupling term cannot silently disappear during evaluation.

Training also stores `parameter_counts` alongside `model_config` in
`metrics.json`: `total` counts all model parameters, `trainable` counts those
with `requires_grad=True`, and `by_tensor` lists the count for each named tensor.
These counts include the encoder B and all active term matrices; disabled terms
contribute no parameters. Existing result files are not modified automatically.

The existing shell files still request their previous C-only or vanilla models.
Their bifurcation commands now use the local adapter. To analyze a coupling
model manually, call the adapter in **this** repository:

```bash
python compare_model_bifurcation_diagrams.py \
  --backend-script /export/home/klenkeit/diffrentbutconst_prams/compare_model_bifurcation_diagrams.py \
  --model-type alrnn --regime rossler_a_005_to_035 \
  --model-runs runs/rossler_additive_diagonal_m200_p20 \
  --model-labels additive_diagonal \
  --components z --num-parameters 320 --device cuda --gpu-id 2 \
  --output bifurcation_diagrams/rossler_additive_diagonal.png
```

The external script's simulation, caching and plot layouts are reused. This
repository supplies its AL-RNN loader and the training-parameter annotations.
No changes to the data-generation repository are required. Calling that external
script directly still uses its old model implementation and does not support
the coupling checkpoints or these extended annotations.

For each ALRNN model, the adapter reads the actual training portions of
`trajectory_phi_raw.npy` (or `trajectory_phi.npy` as a fallback):

- Constant trajectories only: keep the existing blue lines at the selected
  training parameter values, including constant-only datasets in the new format.
- Any time-varying training trajectory, including a mix with constant ones:
  shade one pale blue band from the minimum to the maximum training parameter,
  with no additional constant-value lines for that model.

Saved `trajectory_train_lengths.npy` takes precedence over `--split-idx` from
the training run. Test tails and NaN padding are excluded. Model-side constant
parameter filters and single-trajectory selection are respected. Bounds are
shown in the plot's raw parameter units; normalized phi arrays are converted
using the dataset's saved reference. The ground-truth panel has no training band.

Annotations are also refreshed with `--plot-only`, so existing bifurcation
caches can be redrawn without retraining or recomputing the sweep. If datasets
have moved, pass `--training-data-dirs /new/path/to/dataset` (one directory per
model run). `--no-training-markers` disables both the lines and the bands.

## Adding A Term Later

`parameter_terms.py` holds a registry of term names, checkpoint tensor names and
contribution functions. A contribution receives `(selected_z, phi, *weights)`
and returns a batch-by-M tensor. Every entry has a primary M-by-phi_dim matrix;
`extra_weights` declares any further matrices using `TermWeight`, including
their axis sizes and initialization standard deviation. The CLI choices, model
parameters, forward calculation and checkpoint inference all follow this registry.

The registry entry's `use_activated_state` flag selects relu_P(z) when true and
the unactivated z otherwise. This uses the same activation as the W contribution.
`needs_activation=True` also passes the shared `activate` function as a keyword
argument, allowing a term to shift its own input before applying relu_P.

A future contribution using these declared shapes therefore only needs a new
function and registry entry. No higher-order phi contribution is implemented now.
Checkpoint loading requires every matrix of an active multi-weight term and
validates each declared shape; a missing Q_phi or V_phi is rejected.

## Tests

From the repository root:

```bash
python -m unittest discover -s alrnn_python -p test_alrnn_model.py -v
python -m unittest discover -s alrnn_python/parameter_drift -p test_parameter_drift.py -v
```
