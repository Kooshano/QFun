# Notebook 10 Architecture: MNIST With Learned Superposition Activations

## Purpose

`10_qfun_superposition_activations_mnist.ipynb` is the MNIST-scale classification benchmark for the repository's learned superposition activation idea.

At a high level, the notebook does four things:

1. Loads the full MNIST dataset.
2. Preprocesses it into a lower-dimensional feature space that is practical for the classifier.
3. Runs conventional baselines for context.
4. Trains and evaluates three versions of the quantum-activation classifier:
   - `standard`
   - `mode_a`
   - `mode_b`

This notebook is the largest classification example in the repo. It takes the same modeling idea introduced on IRIS, then expanded to Wine, Breast Cancer, and Digits, and stress-tests it on PCA-compressed MNIST.

## What The Notebook Is Actually Proving

Notebook 10 is not trying to build a production-grade CNN for raw 28x28 image classification.

Instead, it is designed to answer a research question:

Can a small classifier whose hidden activations are learned from quantum-style superposition profiles remain competitive when scaled from small datasets to a much larger multiclass problem like MNIST?

Because of that goal, the notebook intentionally:

- keeps the model compact,
- compresses input features with PCA,
- compares against simple classical baselines,
- measures learned activation profiles explicitly,
- visualizes how training changes those activation curves over time.

## End-to-End Pipeline

The notebook executes this pipeline:

```text
MNIST (784 pixels)
  -> stratified train/test split
  -> feature standardization
  -> PCA compression to 32 components
  -> baseline models
  -> quantum-activation classifier in 3 modes
  -> accuracy / macro-F1 / confusion matrices
  -> learned activation-profile diagnostics
  -> final side-by-side comparison
```

## Main Notebook Stages

### 1. Environment bootstrap

The first code cell makes sure the repository root is importable even when the notebook is launched from the `notebooks` directory. It walks through:

- `Path.cwd()`
- `Path.cwd().parent`

and inserts the first directory containing a `qfun/` package into `sys.path`.

This keeps the notebook runnable without needing an editable install first.

### 2. Imports

The notebook imports two groups of helpers:

- dataset preparation:
  - `load_classification_dataset`
  - `prepare_classification_split`
- notebook-facing benchmark helpers:
  - `run_default_baseline_suite`
  - `run_quantum_experiment`
  - plotting and reporting utilities

Those helpers live in:

- `qfun/datasets.py`
- `qfun/qfan/_classification_benchmarks.py`

### 3. Configuration

Default configuration in notebook 10:

| Parameter | Value | Meaning |
| --- | --- | --- |
| `data_seed` | `7` | Reproducible split and model initialization seed |
| `test_size` | `0.25` | 75/25 train-test split |
| `pca_components` | `32` | Compressed input dimension |
| `hidden_units` | `6` | Size of hidden activation bank |
| `n_qubits` | `3` | Activation resolution uses `2^3 = 8` grid points |
| `steps` | `60` | Training epochs |
| `learning_rate` | `0.04` | Adam/Optax learning rate |
| `log_every` | `5` | Console logging interval |
| `snapshot_interval` | `5` | How often activation snapshots are stored |
| `eval_shots` | `3000` | Sampling budget for measured activation overlays |
| `batch_size` | `1024` | Minibatch size for JAX training |

The notebook attempts `import jax`. If that succeeds, it sets `use_jax=True`; otherwise it falls back to the pure PennyLane/autograd path.

That choice matters because MNIST is large enough that full-batch autograd can be slow, while the JAX path uses minibatch Optax training and can run on CPU or GPU.

## Dataset And Preprocessing Architecture

### Source

MNIST is loaded through:

- `qfun.datasets.load_classification_dataset("mnist")`

Internally, that calls:

- `sklearn.datasets.fetch_openml("mnist_784", version=1, as_frame=False)`

Important details:

- the first run may download the dataset,
- later runs use sklearn's local cache,
- features are 784-dimensional grayscale pixel values,
- labels are coerced to integer class ids `0` through `9`.

### Split preparation

The notebook then calls:

```python
prepare_classification_split(
    mnist_dataset,
    test_size=0.25,
    seed=7,
    standardize=True,
    pca_components=32,
)
```

This helper does the following, in order:

1. Performs a reproducible stratified train/test split.
2. Fits `StandardScaler` on the training set only.
3. Applies the scaler to both train and test features.
4. Fits PCA on the scaled training set only.
5. Projects both train and test sets into a 32-dimensional feature space.

### Why standardization and PCA are used

MNIST starts with 784 features. The classifier in this repo is intentionally small and centered on learned activation profiles, not on deep convolutional feature extraction. PCA makes the benchmark practical while preserving the spirit of the experiment:

- standardization prevents large-scale feature imbalance,
- PCA reduces compute and memory cost,
- a 32D input keeps the model focused on activation design rather than raw pixel handling.

The final shape entering the classifier is:

- `x_train`: `(52500, 32)`
- `x_test`: `(17500, 32)`

assuming the standard 70,000-sample MNIST split.

## Baseline Architecture

Before any quantum-style model is trained, the notebook runs a classical reference suite via:

```python
run_default_baseline_suite(
    mnist_split,
    seed=data_seed,
    mlp_hidden_layer_sizes=(64,),
)
```

This produces two baselines:

1. `LogisticRegression`
2. `MLPClassifier(hidden_layer_sizes=(64,))`

The purpose of these baselines is not to dominate MNIST leaderboards. They serve as local reference points for the same PCA-compressed input space used by the quantum-activation model.

Each baseline produces:

- test accuracy,
- macro-F1,
- per-class classification report,
- confusion matrix.

## Core Model Architecture

The notebook's main model is `QuantumActivationClassifier` from `qfun/qfan/quantum_activation_classifier.py`.

### Big picture

This is a small multiclass neural classifier with a custom hidden layer:

- inputs are projected into hidden pre-activations,
- each hidden unit applies a learned activation function,
- those activation functions are represented as learned superposition-derived profiles on a 1D grid,
- the hidden features are linearly mapped to class logits.

### Parameter shapes for notebook 10

With the notebook defaults:

- input dimension `D = 32`
- hidden units `H = 6`
- classes `C = 10`
- qubits `Q = 3`
- grid points `G = 2^Q = 8`

The main shared parameters are:

- `W_in`: shape `(6, 32)`
- `b_in`: shape `(6,)`
- `W_out`: shape `(10, 6)`
- `b_out`: shape `(10,)`

The activation-specific parameters depend on the mode:

- `standard`: `raw_profiles` with shape `(6, 8)`
- `mode_a`: `raw_profiles` with shape `(6, 16)`
- `mode_b`:
  - `raw_plus` with shape `(6, 8)`
  - `raw_minus` with shape `(6, 8)`
  - `raw_channel_logits` with shape `(6, 2)`

### Forward pass

For a single input vector `x`, each hidden unit computes:

1. affine projection:
   - `z_h = W_in[h] dot x + b_in[h]`
2. bounded latent coordinate:
   - `z_h = tanh(z_h)`
3. hidden activation lookup:
   - evaluate a learned profile on the fixed grid `[-1, 1]`
   - use linear interpolation to read the profile at `z_h`
4. collect hidden features across all units
5. compute logits:
   - `logits = W_out @ hidden + b_out`

So the unusual part is not the classifier head. The unusual part is that every hidden unit's activation function is itself learned from a quantum-style superposition parameterization.

## Activation Profile Representation

The model keeps a fixed activation grid:

```python
activation_grid = linspace(-1.0, 1.0, 2**n_qubits)
```

For notebook 10, that means 8 grid locations.

Each hidden unit owns one activation profile over that grid. During the forward pass, the model does not call a fixed function like ReLU, GELU, or tanh. Instead, it:

1. constructs the unit's profile from trainable raw parameters,
2. linearly interpolates the profile at the hidden unit's current latent coordinate.

That makes the activation family adaptive and data-driven.

## The Three Modes

Notebook 10 compares three profile-construction schemes.

### 1. `standard`

This is the nonnegative version.

For each hidden unit:

1. start from a raw real vector of length `G`,
2. normalize it into a valid real amplitude vector,
3. square amplitudes to get probabilities,
4. multiply by `G` so the resulting activation profile has a convenient scale.

Conceptually:

```text
raw profile -> normalized amplitudes -> probabilities -> activation curve
```

Because probabilities are nonnegative, the learned activation profile is nonnegative too.

### 2. `mode_a`

This is the ancilla-based signed mode.

For each hidden unit:

1. use a raw real vector of length `2G`,
2. normalize it as amplitudes over a larger state space,
3. square to get full probabilities,
4. split alternating entries into:
   - positive channel `p_pos`
   - negative channel `p_neg`
5. form the signed profile:
   - `q = p_pos - p_neg`
6. scale by `G`

This is analogous to using an ancilla/sign bit in the quantum representation.

The advantage is that the hidden activation can become signed even though the underlying measured quantities are probabilities.

### 3. `mode_b`

This is the two-channel signed mode.

For each hidden unit:

1. learn one nonnegative profile for the positive channel,
2. learn one nonnegative profile for the negative channel,
3. learn two mixture weights via a softmax,
4. subtract the weighted negative profile from the weighted positive profile:
   - `q = z_plus * p_plus - z_minus * p_minus`
5. scale by `G`

This mode still creates a signed activation, but it does so by explicitly combining two independent nonnegative learned channels.

## Why The Superposition Story Matters

The repository's main research theme is that profiles can be understood through quantum-state-style amplitude objects.

In notebook 10, the classifier itself is still a classical predictive model. However, its hidden activations are not ordinary learned vectors in the usual deep-learning sense. They are built from normalized amplitude representations that can later be:

- inspected analytically,
- reconstructed as signed or unsigned profiles,
- sampled through PennyLane simulation,
- compared against measurement-based estimates.

That is why the notebook can both train the model and then physically "measure" representative hidden activations afterward.

## Training Architecture

Notebook 10 delegates model training to:

```python
train_quantum_activation_classifier(...)
```

There are two execution paths.

### Path A: PennyLane autograd

Used when `use_jax=False`.

Characteristics:

- full training set used in each optimizer step,
- optimizer is `qml.AdamOptimizer`,
- loss is multiclass cross-entropy,
- parameters are PennyLane NumPy tensors with gradients enabled,
- useful for smaller datasets and environments without JAX.

Loss function:

1. run the model on the full batch,
2. apply softmax to logits,
3. compute mean cross-entropy against one-hot labels.

### Path B: JAX + Optax

Used when `use_jax=True`.

Characteristics:

- minibatch training,
- Optax Adam optimizer,
- optional CPU/GPU acceleration,
- JIT-compiled forward and gradient steps,
- trained JAX parameters are copied back into a PennyLane model instance after each epoch.

That final copy-back step is important. It keeps all of the notebook's measurement and visualization utilities working, because those utilities expect a `QuantumActivationClassifier` object with PennyLane-compatible arrays.

### Why JAX is preferred for MNIST

Full MNIST has around 52.5k training examples after the 75/25 split. The notebook comments correctly note that the JAX path is the practical option for this scale, especially when GPU is available.

## The `run_quantum_experiment` Wrapper

The notebook does not call the low-level training function directly. Instead it uses:

```python
run_quantum_experiment(...)
```

This helper is notebook-specific glue code that:

1. builds a `QuantumActivationConfig`,
2. trains the model,
3. records loss snapshots,
4. records periodic train/test accuracy,
5. stores intermediate activation profiles for plotting,
6. selects representative hidden units,
7. measures those units with finite sampling shots,
8. packages everything into a `QuantumExperimentResult`.

This is why the notebook code stays compact while still exposing detailed diagnostics.

## Snapshot And Diagnostics Architecture

The notebook tracks more than final accuracy.

### Training snapshots

The callback inside `run_quantum_experiment` stores:

- step number,
- current training loss,
- train accuracy,
- test accuracy,
- selected activation profiles for a few hidden units.

Snapshots are recorded:

- before training starts with `step = -1`,
- every `snapshot_interval` steps,
- on the last step.

### Representative hidden units

After training, the helper selects the most influential hidden units by ranking columns of `W_out` using output-layer norm.

This is a smart diagnostic shortcut because it focuses measurement plots on the hidden units that matter most to the final classifier.

### Measured activation overlays

For the chosen units, the notebook calls:

- `measure_activation_profile(unit_idx, shots=eval_shots)`

This uses PennyLane simulation to sample the learned amplitude-based representation and reconstruct a finite-shot estimate of the activation profile.

The notebook then overlays:

- exact learned profile,
- measurement-based estimate.

That provides a direct "model parameterization vs sampled reconstruction" comparison.

## What Each Notebook Section Does

### Section 1: Load, standardize, and compress MNIST

Outputs:

- dataset size confirmation,
- train/test summary,
- feature dimension after PCA,
- per-class counts.

### Section 2: Baselines

Outputs:

- logistic regression metrics,
- MLP metrics,
- confusion matrices for both.

### Section 3: Standard superposition activations

Outputs:

- trained classifier metrics,
- classification report,
- confusion matrix,
- final exact activation profiles,
- finite-shot measurement overlays,
- training diagnostics plot.

### Section 4: Mode A signed superposition activations

Same diagnostic structure as section 3, but with ancilla-style signed activations.

### Section 5: Mode B signed superposition activations

Same diagnostic structure as section 3, but with two-channel signed activations.

### Section 6: Final comparison

Builds a comparison table combining:

- classical baselines,
- standard mode,
- mode A,
- mode B.

This is the notebook's summary view.

## Code Map

The notebook is small because the heavy lifting is delegated to these files:

| File | Role |
| --- | --- |
| `notebooks/10_qfun_superposition_activations_mnist.ipynb` | User-facing experiment driver |
| `qfun/datasets.py` | Dataset loading, scaling, PCA, stratified split |
| `qfun/qfan/_classification_benchmarks.py` | Baseline runners, experiment wrapper, plotting/reporting |
| `qfun/qfan/quantum_activation_classifier.py` | Model definition, profiles, training entrypoint, measurement API |
| `qfun/qfan/_jax_quantum_activation.py` | Optional JAX/Optax accelerated training |
| `qfun/quantum_learning.py` | Shared amplitude normalization and measurement utilities |

## Data Flow In More Detail

```text
load_classification_dataset("mnist")
  -> ClassificationDataset(name, X, y, feature_names, target_names)

prepare_classification_split(...)
  -> PreparedClassificationSplit(
       x_train, x_test, y_train, y_test, scaler, pca
     )

run_default_baseline_suite(split)
  -> baseline metrics and confusion matrices

run_quantum_experiment(mode, split, ...)
  -> QuantumActivationConfig
  -> train_quantum_activation_classifier(...)
  -> QuantumActivationClassifier
  -> snapshots + measured activation profiles
  -> QuantumExperimentResult

build_comparison_rows(...)
  -> final summary table
```

## Architectural Constraints And Tradeoffs

### Strengths

- Clear research story from amplitude parameterization to classifier behavior.
- Small model size makes learned activations easy to inspect.
- Same interface supports unsigned and signed activation families.
- Measurement overlays connect learned parameters back to the quantum-inspired interpretation.
- JAX path makes the approach feasible on MNIST-scale data.

### Limitations

- This is not a spatial image model, so it does not exploit local pixel structure like CNNs do.
- PCA compression trades away some raw-image information for runtime practicality.
- With `n_qubits=3`, each activation profile has only 8 grid points, so activation resolution is intentionally coarse.
- Finite-shot measurements are diagnostic, not part of the training loop.

## Why Notebook 10 Is Important In This Repo

Notebook 10 is the scale test for the classification branch of the project.

The sequence across notebooks is roughly:

- notebook 06: prove the idea on IRIS,
- notebook 07: extend to more tabular datasets,
- notebook 08: move to image-derived data with Digits,
- notebook 09: study ablations,
- notebook 10: run the idea on full MNIST with the accelerated training path.

So notebook 10 is where the repo demonstrates that learned superposition activations are not only toy examples. They can be trained on a much larger multiclass benchmark while still preserving the diagnostic and measurement story that makes the approach distinctive.

## Short Summary

If you want one-sentence intuition:

Notebook 10 loads full MNIST, compresses it with PCA, compares classical baselines against a small classifier whose hidden activation functions are learned from quantum-style superposition profiles, and then evaluates those learned activations both as exact model objects and as finite-shot measured reconstructions.
