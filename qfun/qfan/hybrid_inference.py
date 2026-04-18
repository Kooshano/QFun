"""Hybrid quantum-classical inference for QuantumActivationClassifier.

After classical training, this module provides two inference modes:

1. **Classical inference** (default): Uses learned profiles as lookup tables
   with interpolation. Fast, deterministic, no quantum hardware needed.

2. **Quantum inference**: Prepares each activation profile as a quantum state
   via MottonenStatePreparation, measures with finite shots, and uses the
   sampled distribution as the activation profile. Demonstrates quantum
   realizability and enables noise-resilience analysis.

The key result: quantum inference converges to classical inference as the
number of shots increases, providing concrete evidence for quantum realizability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._utils import EPS
from ._profile_interp import interp_profile_np


def _interp_profile_scalar(
    z: Any,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    mode: str,
) -> float:
    """``interp_profile_np`` may return a length-1 ndarray; this always returns ``float``."""
    out = interp_profile_np(np.atleast_1d(z), x_grid, y_grid, mode, EPS)
    flat = np.asarray(out, dtype=np.float64).reshape(-1)
    if flat.size != 1:
        raise ValueError(f"expected a single interpolated value, got shape {np.asarray(out).shape}")
    return float(flat[0])


@dataclass(frozen=True)
class InferenceComparison:
    """Comparison between classical and quantum inference."""
    shots: int
    classical_accuracy: float
    quantum_accuracy: float
    accuracy_gap: float
    classical_predictions: np.ndarray
    quantum_predictions: np.ndarray
    agreement_rate: float


@dataclass
class ConvergenceResult:
    """Shot-budget convergence analysis."""
    shot_counts: list[int] = field(default_factory=list)
    classical_accuracy: float = 0.0
    quantum_accuracies: list[float] = field(default_factory=list)
    agreement_rates: list[float] = field(default_factory=list)
    comparisons: list[InferenceComparison] = field(default_factory=list)


def _sample_profile(model: Any, layer_idx: int, unit_idx: int, shots: int) -> np.ndarray:
    """Sample a quantum profile via measurement."""
    measurement = model.measure_activation_profile(layer_idx, unit_idx, shots=shots)
    return np.asarray(measurement.profile, dtype=float)


def quantum_forward_single(
    model: Any,
    x: np.ndarray,
    *,
    shots: int = 5000,
    seed: int | None = None,
) -> np.ndarray:
    """Forward pass using quantum-measured activation profiles.

    For each hidden unit, the activation profile is obtained by preparing the
    learned amplitude vector as a quantum state and measuring it, rather than
    using the exact Born-rule probabilities.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)

    x_grid = np.asarray(model.activation_grid, dtype=float)
    hidden = x.copy()

    for layer_idx in range(model.num_hidden_layers):
        w = np.asarray(model.hidden_weights[layer_idx], dtype=float)
        b = np.asarray(model.hidden_biases[layer_idx], dtype=float)
        z_pre = w @ hidden + b

        if model.hidden_preactivation == "tanh":
            z_quantum = np.tanh(z_pre)
        else:
            z_quantum = z_pre

        width = model.hidden_layer_sizes[layer_idx]
        layer_out = np.zeros(width, dtype=float)

        for unit_idx in range(width):
            measured_profile = _sample_profile(model, layer_idx, unit_idx, shots)
            activation = _interp_profile_scalar(
                z_quantum[unit_idx],
                x_grid,
                measured_profile,
                model.profile_interp,
            )

            if model.hidden_function_family == "kan_quantum_hybrid":
                base_scale = float(np.asarray(model.base_mix_layers[layer_idx][unit_idx]))
                quantum_scale = float(np.asarray(model.quantum_mix_layers[layer_idx][unit_idx]))
                z_i = float(np.asarray(z_pre[unit_idx], dtype=np.float64).reshape(-1)[0])
                silu_val = z_i / (1.0 + np.exp(-z_i))
                layer_out[unit_idx] = base_scale * silu_val + quantum_scale * activation
            else:
                layer_out[unit_idx] = activation

        hidden = layer_out

    w_out = np.asarray(model.W_out, dtype=float)
    b_out = np.asarray(model.b_out, dtype=float)
    return w_out @ hidden + b_out


def quantum_predict_proba(
    model: Any,
    x_batch: np.ndarray,
    *,
    shots: int = 5000,
) -> np.ndarray:
    """Predict class probabilities using quantum-measured profiles."""
    x_batch = np.asarray(x_batch, dtype=float)
    if x_batch.ndim == 1:
        x_batch = x_batch.reshape(1, -1)

    logits_list = []
    for xi in x_batch:
        logits = quantum_forward_single(model, xi, shots=shots)
        logits_list.append(logits)

    logits = np.stack(logits_list)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def quantum_predict(
    model: Any,
    x_batch: np.ndarray,
    *,
    shots: int = 5000,
) -> np.ndarray:
    """Predict classes using quantum-measured profiles."""
    probs = quantum_predict_proba(model, x_batch, shots=shots)
    return np.argmax(probs, axis=1)


def compare_inference_modes(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    shots: int = 5000,
    max_samples: int | None = None,
) -> InferenceComparison:
    """Compare classical and quantum inference on a test set."""
    x = np.asarray(x_test, dtype=float)
    y = np.asarray(y_test, dtype=int)

    if max_samples is not None and len(x) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), size=max_samples, replace=False)
        x = x[idx]
        y = y[idx]

    classical_preds = model.predict(x)
    classical_acc = float(np.mean(classical_preds == y))

    quantum_preds = quantum_predict(model, x, shots=shots)
    quantum_acc = float(np.mean(quantum_preds == y))

    agreement = float(np.mean(classical_preds == quantum_preds))

    return InferenceComparison(
        shots=shots,
        classical_accuracy=classical_acc,
        quantum_accuracy=quantum_acc,
        accuracy_gap=classical_acc - quantum_acc,
        classical_predictions=classical_preds,
        quantum_predictions=quantum_preds,
        agreement_rate=agreement,
    )


def shot_convergence_analysis(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    shot_counts: list[int] | None = None,
    max_samples: int = 50,
) -> ConvergenceResult:
    """Analyze how quantum inference converges to classical as shots increase.

    This is the key experiment for the quantum realizability narrative:
    it demonstrates that the classically-trained profiles can be faithfully
    reproduced via quantum measurement.
    """
    if shot_counts is None:
        shot_counts = [100, 500, 1000, 5000, 10000]

    result = ConvergenceResult()
    result.shot_counts = list(shot_counts)

    x = np.asarray(x_test, dtype=float)
    y = np.asarray(y_test, dtype=int)
    if len(x) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), size=max_samples, replace=False)
        x = x[idx]
        y = y[idx]

    classical_preds = model.predict(x)
    result.classical_accuracy = float(np.mean(classical_preds == y))

    for shots in shot_counts:
        comparison = compare_inference_modes(
            model, x, y, shots=shots,
        )
        result.quantum_accuracies.append(comparison.quantum_accuracy)
        result.agreement_rates.append(comparison.agreement_rate)
        result.comparisons.append(comparison)

    return result


def print_convergence_summary(result: ConvergenceResult) -> None:
    """Print shot-convergence analysis results."""
    print("\nShot-Budget Convergence Analysis")
    print("=" * 60)
    print(f"Classical accuracy: {result.classical_accuracy:.4f}")
    print()

    header = f"{'Shots':>8} {'Q-Accuracy':>12} {'Agreement':>12} {'Gap':>10}"
    print(header)
    print("-" * len(header))
    for shots, q_acc, agree in zip(
        result.shot_counts,
        result.quantum_accuracies,
        result.agreement_rates,
    ):
        gap = result.classical_accuracy - q_acc
        print(f"{shots:>8} {q_acc:>12.4f} {agree:>12.4f} {gap:>10.4f}")

    if len(result.quantum_accuracies) >= 2:
        final_gap = result.classical_accuracy - result.quantum_accuracies[-1]
        print(f"\nFinal gap at {result.shot_counts[-1]} shots: {final_gap:.4f}")
        if final_gap < 0.02:
            print("  -> Quantum inference closely matches classical (gap < 2%)")
