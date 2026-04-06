# Deep MNIST Architecture: Learned Superposition Activations

## Purpose

`experiments/run_mnist_deep.py` is the deep follow-up to the single-layer MNIST experiment.

It keeps the same MNIST preprocessing and baseline workflow, but replaces the
single hidden superposition layer with a stacked classifier that uses:

- `hidden_layers=(6, 6)`
- one shared activation mode across all hidden layers
- learnable superposition activations at every hidden layer
- no fixed `tanh` squash in the classifier path

So notebook 11 answers a slightly different question from notebook 10:

Can the learned superposition-activation idea remain interpretable and trainable
when we stack multiple hidden layers instead of using a single hidden bank?

## End-to-End Pipeline

```text
MNIST (784 pixels)
  -> stratified train/test split
  -> standardization
  -> PCA to 32 components
  -> classical baselines
  -> deep quantum-activation classifier in 3 modes
  -> accuracy / macro-F1 / confusion matrices
  -> layer-aware activation diagnostics
  -> final comparison table
```

## What Changed Relative To Notebook 10

Notebook 10 uses one hidden layer with learned superposition activations.

Notebook 11 uses two hidden layers:

`input -> layer 0 superposition activations -> layer 1 superposition activations -> logits`

The important architectural change is that every hidden nonlinearity is now
learned from a superposition profile. The old fixed `tanh` squeeze that used to
map pre-activations into `[-1, 1]` is no longer part of the classifier.

Instead:

1. each unit computes an affine pre-activation,
2. that scalar is clipped/interpolated on the activation grid,
3. the learned superposition profile produces the unit output,
4. the next hidden layer consumes those learned outputs directly.

## Model Architecture

Notebook 11 still uses `QuantumActivationClassifier`, but with the new config:

```python
QuantumActivationConfig(
    input_dim=32,
    hidden_layers=(6, 6),
    n_qubits=5,
    n_classes=10,
    mode="standard" | "mode_a" | "mode_b",
)
```

With those defaults:

- input dimension `D = 32`
- hidden layer widths `(6, 6)`
- classes `C = 10`
- qubits `Q = 5`
- grid points `G = 2^Q = 32`

The learned structure is:

- hidden layer 0:
  - weights `(6, 32)`
  - bias `(6,)`
  - per-unit activation profiles on a 32-point grid
- hidden layer 1:
  - weights `(6, 6)`
  - bias `(6,)`
  - another bank of per-unit activation profiles on the same 32-point grid
- output layer:
  - weights `(10, 6)`
  - bias `(10,)`

## Activation Modes

Notebook 11 keeps the same three activation constructions as notebook 10, but
applies them at every hidden layer.

### `standard`

- learn real amplitudes
- square them into probabilities
- scale by the grid size
- use the resulting nonnegative profile as the activation curve

### `mode_a`

- learn amplitudes over an ancilla-extended state
- split probabilities into positive and negative channels
- subtract them to get a signed activation profile

### `mode_b`

- learn separate positive and negative channels
- combine them with learned softmax weights
- subtract the weighted channels to get a signed activation profile

## Training Architecture

Notebook 11 uses the same training entrypoint as notebook 10:

```python
train_quantum_activation_classifier(...)
```

There are still two execution paths:

- PennyLane/autograd full-batch training
- JAX + Optax minibatch training

The difference is that both paths now unroll an arbitrary hidden stack rather
than a single hidden layer. For MNIST, the JAX path is still the practical
default because the dataset is large and notebook 11 adds another hidden layer.

## Diagnostics

The notebook-facing benchmark helpers are now layer-aware.

That means notebook 11 can:

- track activation snapshots from every hidden layer during training,
- choose representative units as `(layer_idx, unit_idx)` pairs,
- measure learned activations from both hidden layers,
- plot exact versus measured activation profiles for those layer/unit pairs.

Representative units are selected using outgoing-weight norms:

- hidden-to-hidden norms for intermediate layers
- output-layer norms for the final hidden layer

This keeps the plots focused on the units that matter most to the trained
network's downstream behavior.

## Notebook Sections

Notebook 11 follows the same high-level structure as notebook 10:

1. load and preprocess MNIST
2. run baselines
3. train deep `standard`
4. train deep `mode_a`
5. train deep `mode_b`
6. compare all results

The difference is that the training diagnostics now visualize a deeper model,
so activation-evolution plots and measured overlays refer to explicit
`(layer_idx, unit_idx)` locations rather than a single hidden layer.

## Why Notebook 11 Matters

Notebook 10 shows that superposition-defined activations can work on a larger
MNIST benchmark with one hidden layer.

Notebook 11 pushes the idea one step further: it tests whether those same
learned activation families remain useful when stacked into a deeper network
without falling back to a fixed classical nonlinearity between layers.

In short:

Notebook 11 is the repo's deep MNIST benchmark for fully learned
superposition-based hidden activations.
