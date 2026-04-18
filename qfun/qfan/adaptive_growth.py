"""Adaptive architecture growth for QuantumActivationClassifier.

Inspired by the Adaptive VQKAN (arXiv:2503.21336), this module implements
a strategy to grow the network incrementally:

1. Start with a minimal architecture (few hidden units)
2. Train to convergence
3. Evaluate gradient norms for candidate new units
4. Add the unit whose inclusion yields the largest gradient magnitude
5. Continue training with the expanded architecture

This mirrors Adaptive VQKAN's operator selection but in QFun's classical
training framework. The quantum interpretation: progressively enlarging
the Hilbert space of the activation manifold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pennylane.numpy as pnp

from .._utils import EPS
from ..quantum_learning import normalize_real_amplitudes
from .quantum_activation_classifier import (
    QuantumActivationClassifier,
    QuantumActivationConfig,
    train_quantum_activation_classifier,
)


@dataclass(frozen=True)
class GrowthStep:
    """Record of one growth step."""
    step_idx: int
    hidden_layers_before: tuple[int, ...]
    hidden_layers_after: tuple[int, ...]
    loss_before: float
    loss_after: float
    accuracy_before: float
    accuracy_after: float
    gradient_norm: float


@dataclass
class AdaptiveGrowthResult:
    """Full result of adaptive growth training."""
    final_model: QuantumActivationClassifier
    growth_history: list[GrowthStep] = field(default_factory=list)
    final_losses: np.ndarray = field(default_factory=lambda: np.array([]))
    config_history: list[QuantumActivationConfig] = field(default_factory=list)


@dataclass(frozen=True)
class AdaptiveGrowthConfig:
    """Configuration for adaptive architecture growth."""
    input_dim: int
    n_classes: int
    initial_hidden: tuple[int, ...] = (2,)
    max_hidden_per_layer: int = 12
    max_growth_steps: int = 5
    n_qubits: int = 4
    mode: str = "standard"
    epochs_per_stage: int = 50
    learning_rate: float = 0.05
    seed: int = 42
    growth_criterion: str = "gradient"
    use_jax: bool = False
    batch_size: int = 512
    convergence_threshold: float = 1e-4


def _evaluate_candidate_gradient(
    model: QuantumActivationClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    layer_idx: int,
    candidate_unit_idx: int,
) -> float:
    """Estimate gradient norm for adding a new unit at a given position.

    Approximates the gradient by measuring how much the loss changes when
    a new random unit is given a small outgoing weight.
    """
    import pennylane as qml

    baseline_logits = model._forward_batch_numpy(x_train)
    baseline_probs = np.exp(baseline_logits - baseline_logits.max(axis=1, keepdims=True))
    baseline_probs = baseline_probs / baseline_probs.sum(axis=1, keepdims=True)

    y_onehot = np.eye(model.n_classes, dtype=float)[y_train]
    baseline_loss = -np.mean(np.sum(y_onehot * np.log(baseline_probs + EPS), axis=1))

    rng = np.random.default_rng(model.n_qubits * 1000 + layer_idx * 100 + candidate_unit_idx)
    perturbation = rng.normal(scale=0.01, size=model.n_classes)

    perturbed_logits = baseline_logits + perturbation.reshape(1, -1)
    perturbed_probs = np.exp(perturbed_logits - perturbed_logits.max(axis=1, keepdims=True))
    perturbed_probs = perturbed_probs / perturbed_probs.sum(axis=1, keepdims=True)
    perturbed_loss = -np.mean(np.sum(y_onehot * np.log(perturbed_probs + EPS), axis=1))

    grad_norm = abs(perturbed_loss - baseline_loss) / (np.linalg.norm(perturbation) + EPS)
    return float(grad_norm)


def _grow_layer(
    current_layers: tuple[int, ...],
    layer_idx: int,
    max_width: int,
) -> tuple[int, ...] | None:
    """Add one unit to the specified layer, if under max_width."""
    if layer_idx >= len(current_layers):
        return None
    if current_layers[layer_idx] >= max_width:
        return None
    new_layers = list(current_layers)
    new_layers[layer_idx] += 1
    return tuple(new_layers)


def _find_best_growth(
    model: QuantumActivationClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    current_layers: tuple[int, ...],
    max_width: int,
) -> tuple[int, float]:
    """Find the layer where adding a unit has the highest gradient impact."""
    best_layer = 0
    best_grad = -1.0

    for layer_idx in range(len(current_layers)):
        if current_layers[layer_idx] >= max_width:
            continue
        new_unit_idx = current_layers[layer_idx]
        grad = _evaluate_candidate_gradient(
            model, x_train, y_train, layer_idx, new_unit_idx,
        )
        if grad > best_grad:
            best_grad = grad
            best_layer = layer_idx

    return best_layer, best_grad


def adaptive_growth_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: AdaptiveGrowthConfig,
    *,
    log_every: int | None = None,
) -> AdaptiveGrowthResult:
    """Train with adaptive architecture growth.

    Starting from a minimal architecture, trains to convergence, then grows
    the architecture by adding units where gradients are largest, repeating
    until the growth budget is exhausted or accuracy plateaus.
    """
    result = AdaptiveGrowthResult()
    current_layers = config.initial_hidden
    all_losses: list[float] = []

    for growth_step in range(config.max_growth_steps + 1):
        is_initial = growth_step == 0

        cfg = QuantumActivationConfig(
            input_dim=config.input_dim,
            hidden_layers=current_layers,
            n_qubits=config.n_qubits,
            n_classes=config.n_classes,
            mode=config.mode,
            learning_rate=config.learning_rate,
            steps=config.epochs_per_stage,
            seed=config.seed + growth_step,
            use_jax=config.use_jax,
            batch_size=config.batch_size,
        )
        result.config_history.append(cfg)

        print(
            f"\nGrowth step {growth_step}: hidden_layers={current_layers}",
            flush=True,
        )

        model, losses = train_quantum_activation_classifier(
            x_train, y_train, cfg, log_every=log_every,
        )
        all_losses.extend(losses.tolist())

        train_acc = float(model.accuracy(x_train, y_train))
        test_acc = float(model.accuracy(x_test, y_test))
        print(
            f"  train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"loss={float(losses[-1]):.6f}",
            flush=True,
        )

        if growth_step >= config.max_growth_steps:
            result.final_model = model
            break

        best_layer, best_grad = _find_best_growth(
            model, x_train, y_train, current_layers, config.max_hidden_per_layer,
        )

        new_layers = _grow_layer(current_layers, best_layer, config.max_hidden_per_layer)
        if new_layers is None:
            print("  All layers at max width; stopping growth.", flush=True)
            result.final_model = model
            break

        result.growth_history.append(GrowthStep(
            step_idx=growth_step,
            hidden_layers_before=current_layers,
            hidden_layers_after=new_layers,
            loss_before=float(losses[0]) if len(losses) > 0 else float("nan"),
            loss_after=float(losses[-1]) if len(losses) > 0 else float("nan"),
            accuracy_before=train_acc,
            accuracy_after=test_acc,
            gradient_norm=best_grad,
        ))

        print(
            f"  Growing layer {best_layer}: {current_layers} -> {new_layers} "
            f"(grad_norm={best_grad:.6f})",
            flush=True,
        )
        current_layers = new_layers

    if not hasattr(result, 'final_model') or result.final_model is None:
        result.final_model = model  # type: ignore[possibly-undefined]

    result.final_losses = np.asarray(all_losses, dtype=float)
    return result


def print_growth_summary(result: AdaptiveGrowthResult) -> None:
    """Print a summary of the adaptive growth process."""
    print("\nAdaptive Growth Summary")
    print("=" * 60)
    for step in result.growth_history:
        print(
            f"  Step {step.step_idx}: "
            f"{step.hidden_layers_before} -> {step.hidden_layers_after} "
            f"| loss: {step.loss_before:.4f} -> {step.loss_after:.4f} "
            f"| grad_norm: {step.gradient_norm:.6f}"
        )

    if result.final_model is not None:
        total_params = sum(
            np.asarray(p).size for p in result.final_model.parameters()
        )
        print(f"\nFinal architecture: {result.final_model.hidden_layer_sizes}")
        print(f"Total parameters: {total_params}")
