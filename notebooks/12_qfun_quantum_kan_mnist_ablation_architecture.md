# Notebook 12 Architecture: Quantum-KAN MNIST Ablations

## Purpose

`11_qfun_superposition_activations_mnist_deep.ipynb` is the pure deep-superposition
reference.

`12_qfun_quantum_kan_mnist_ablation.ipynb` adds a new ablation family that moves
the hidden units closer to the KAN pattern without abandoning the QFun quantum
activation idea.

The key comparison is:

- `pure_superposition`: each hidden unit output comes only from a learned quantum
  superposition profile
- `kan_quantum_hybrid`: each hidden unit output is a sum of a classical base path
  and a learned quantum correction path

So notebook 12 is not a full edge-wise KAN reimplementation. It is a targeted
quantum-KAN hybrid ablation on top of the existing MNIST classifier.

## End-to-End Pipeline

```text
MNIST (784 pixels)
  -> stratified train/test split
  -> standardization
  -> PCA to 32 features
  -> classical baselines
  -> 6 quantum model runs
  -> accuracy / macro-F1 / confusion matrices
  -> activation-profile diagnostics
  -> base-vs-quantum component plots
  -> comparison table grouped by family and mode
```

## Default Notebook 12 Settings

Notebook 12 uses:

- `test_size=0.2`
- `pca_components=32`
- `hidden_layers=(6, 6)`
- `n_qubits=5`
- `steps=100`
- `learning_rate=0.01`
- `batch_size=1024`
- `hidden_preactivation="superposition"`
- `hidden_base_activation="silu"`
- `profile_smoothness_reg=1e-3`

That means the learned quantum profiles live on a `2^5 = 32` point activation grid.

## The Two Hidden-Function Families

### 1. Pure superposition

This is the notebook-11 baseline:

`hidden = quantum_profile(z_q)`

where:

- `z = W x + b`
- `z_q = z` when `hidden_preactivation="superposition"`
- `z_q = tanh(z)` when `hidden_preactivation="tanh"`

Each hidden unit learns its own quantum/superposition-defined activation curve.

### 2. KAN-quantum hybrid

This is the notebook-12 addition:

`hidden = base_mix * SiLU(z) + quantum_mix * quantum_profile(z_q)`

So each hidden unit now has:

- a classical base path
- a quantum correction path
- two learnable branch scales: `base_mix` and `quantum_mix`

This mirrors the useful idea from KAN layers:

- a simple base response
- plus a learned function correction

But the correction term here is not a spline basis. It is still the QFun
superposition profile learned from amplitudes or signed quantum-style channels.

## Hidden-Layer Architecture

With the default deep stack:

```python
QuantumActivationConfig(
    input_dim=32,
    hidden_layers=(6, 6),
    n_qubits=5,
    n_classes=10,
    hidden_function_family="pure_superposition" | "kan_quantum_hybrid",
    hidden_base_activation="silu",
    hidden_preactivation="superposition",
)
```

The trainable structure is:

- hidden layer 0:
  - weights `(6, 32)`
  - bias `(6,)`
  - one quantum activation profile per unit on a 32-point grid
  - for the hybrid family: `base_mix (6,)` and `quantum_mix (6,)`
- hidden layer 1:
  - weights `(6, 6)`
  - bias `(6,)`
  - another quantum activation bank on the same grid
  - for the hybrid family: another pair of branch-scale vectors
- output layer:
  - weights `(10, 6)`
  - bias `(10,)`

## Quantum Activation Modes

Notebook 12 keeps the same three quantum-profile constructions:

### `standard`

- learn a real amplitude vector
- normalize it
- square it into probabilities
- scale by grid size

This gives a nonnegative learned profile.

### `mode_a`

- learn amplitudes on an ancilla-extended state
- interpret alternating bins as positive and negative channels
- subtract them

This gives a signed profile.

### `mode_b`

- learn separate positive and negative channels
- learn a 2-way softmax over those channels
- subtract the weighted channels

This also gives a signed profile, but with explicit two-channel decomposition.

The `mode` choice affects only the quantum branch. The classical base path is
always `SiLU(z)` in notebook 12.

## What KAN Ideas Are Borrowed

Notebook 12 borrows three ideas from KAN-style layers:

- a base path plus learned correction path
- explicit interpretable hidden-unit functions
- mild function regularization

It does **not** borrow:

- per-edge spline functions
- adaptive grid updates
- raw-pixel input processing

The model remains node-wise, PCA-based, and quantum-profile-driven.

## Smoothness Regularization

For the hybrid family, notebook 12 adds:

`profile_smoothness_reg * smoothness_penalty`

The penalty is computed from first differences of the **effective quantum
branch**, meaning the learned quantum profile after multiplying by the
unit-specific `quantum_mix`.

This encourages the quantum correction term to stay reasonably smooth instead of
using unnecessarily jagged corrections on the activation grid.

## Diagnostics

Notebook 12 extends the benchmark helpers with layer-aware component tracking.

For each representative unit `(layer_idx, unit_idx)`, the model can now expose:

- `base`
- `quantum`
- `combined`
- the raw measured `quantum_profile`
- `base_scale`
- `quantum_scale`

This supports three kinds of plots:

1. final combined hidden activation profiles
2. exact-vs-measured quantum-branch overlays
3. base-vs-quantum-vs-combined component plots

Representative units are still selected using outgoing-weight norms:

- hidden-to-hidden norms for intermediate layers
- output-layer norms for the last hidden layer

## Ablation Matrix

Notebook 12 runs exactly six quantum variants:

1. `pure_superposition + standard`
2. `pure_superposition + mode_a`
3. `pure_superposition + mode_b`
4. `kan_quantum_hybrid + standard`
5. `kan_quantum_hybrid + mode_a`
6. `kan_quantum_hybrid + mode_b`

This isolates two questions:

- does adding a KAN-like base path help?
- does that answer depend on how the quantum profile is constructed?

## Outputs and Artifacts

Notebook 12 writes to:

`notebooks/note12_outputs/<run_id>/`

Artifacts include:

- `console.log`
- saved figures `fig_*.png`
- baseline summaries
- per-run losses and confusion matrices
- representative component curves
- representative quantum measurements
- `comparison.csv`
- `comparison.json`

## Why Notebook 12 Matters

Notebook 11 answers:

Can a deep classifier use only learned superposition activations?

Notebook 12 answers a more structured question:

Does the model improve when each hidden unit keeps a simple classical base
response and uses the quantum profile as a learned correction term, in the same
spirit that KAN layers combine a base function with a learned adjustment?

That makes notebook 12 the repo's first explicit bridge between:

- QFun superposition activations
- deep MNIST classification
- KAN-inspired hidden-function design
