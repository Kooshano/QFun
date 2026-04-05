"""Classifier with learned superposition-defined activation functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .._utils import EPS, _to_numpy_float
from ..quantum_learning import (
    measure_mode_a_superposition,
    measure_mode_b_superposition,
    measure_standard_superposition,
    normalize_real_amplitudes,
    softmax_weights,
)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


@dataclass(frozen=True)
class QuantumActivationConfig:
    input_dim: int
    hidden_units: int = 6
    n_qubits: int = 4
    n_classes: int = 3
    mode: str = "standard"
    learning_rate: float = 0.05
    steps: int = 120
    seed: int = 42


@dataclass(frozen=True)
class ActivationMeasurement:
    mode: str
    profile: np.ndarray
    counts: dict[str, int] | None = None
    p_pos: np.ndarray | None = None
    p_neg: np.ndarray | None = None
    p_plus: np.ndarray | None = None
    p_minus: np.ndarray | None = None
    z_plus: float | None = None
    z_minus: float | None = None


class QuantumActivationClassifier:
    """Small multiclass classifier with learned 1D superposition activations."""

    def __init__(self, config: QuantumActivationConfig):
        if config.mode not in {"standard", "mode_a", "mode_b"}:
            raise ValueError("mode must be 'standard', 'mode_a', or 'mode_b'.")
        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if config.hidden_units <= 0:
            raise ValueError("hidden_units must be positive.")
        if config.n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        if config.n_classes <= 1:
            raise ValueError("n_classes must be at least 2.")

        self.input_dim = int(config.input_dim)
        self.hidden_units = int(config.hidden_units)
        self.n_qubits = int(config.n_qubits)
        self.n_classes = int(config.n_classes)
        self.mode = config.mode
        self.num_grid_points = 2**self.n_qubits
        self.activation_grid = pnp.array(np.linspace(-1.0, 1.0, self.num_grid_points))

        rng = np.random.default_rng(config.seed)
        self.W_in = pnp.array(
            rng.normal(scale=0.35, size=(self.hidden_units, self.input_dim)),
            requires_grad=True,
        )
        self.b_in = pnp.array(
            rng.normal(scale=0.05, size=(self.hidden_units,)),
            requires_grad=True,
        )
        self.W_out = pnp.array(
            rng.normal(scale=0.35, size=(self.n_classes, self.hidden_units)),
            requires_grad=True,
        )
        self.b_out = pnp.array(
            np.zeros(self.n_classes, dtype=float),
            requires_grad=True,
        )

        if self.mode == "standard":
            self.raw_profiles = pnp.array(
                rng.normal(scale=0.25, size=(self.hidden_units, self.num_grid_points)),
                requires_grad=True,
            )
        elif self.mode == "mode_a":
            self.raw_profiles = pnp.array(
                rng.normal(scale=0.25, size=(self.hidden_units, 2 * self.num_grid_points)),
                requires_grad=True,
            )
        else:
            self.raw_plus = pnp.array(
                rng.normal(scale=0.25, size=(self.hidden_units, self.num_grid_points)),
                requires_grad=True,
            )
            self.raw_minus = pnp.array(
                rng.normal(scale=0.25, size=(self.hidden_units, self.num_grid_points)),
                requires_grad=True,
            )
            self.raw_channel_logits = pnp.array(
                rng.normal(scale=0.05, size=(self.hidden_units, 2)),
                requires_grad=True,
            )

    def parameters(self) -> list[Any]:
        base = [self.W_in, self.b_in, self.W_out, self.b_out]
        if self.mode in {"standard", "mode_a"}:
            return base + [self.raw_profiles]
        return base + [self.raw_plus, self.raw_minus, self.raw_channel_logits]

    def set_parameters(self, *params: Any) -> None:
        self.W_in, self.b_in, self.W_out, self.b_out = params[:4]
        if self.mode in {"standard", "mode_a"}:
            self.raw_profiles = params[4]
        else:
            self.raw_plus, self.raw_minus, self.raw_channel_logits = params[4:]

    def _validate_unit_idx(self, unit_idx: int) -> None:
        if unit_idx < 0 or unit_idx >= self.hidden_units:
            raise IndexError(f"unit_idx must be in [0, {self.hidden_units - 1}], got {unit_idx}")

    def _standard_profile(self, raw_params: Any) -> Any:
        amps = normalize_real_amplitudes(raw_params)
        probs = qml.math.real(amps**2)
        return self.num_grid_points * probs

    def _mode_a_profile(self, raw_params: Any) -> Any:
        amps = normalize_real_amplitudes(raw_params)
        full_probs = qml.math.real(amps**2)
        q = full_probs[0::2] - full_probs[1::2]
        return self.num_grid_points * q

    def _mode_b_profile(self, raw_plus: Any, raw_minus: Any, raw_logits: Any) -> Any:
        plus_amps = normalize_real_amplitudes(raw_plus)
        minus_amps = normalize_real_amplitudes(raw_minus)
        p_plus = qml.math.real(plus_amps**2)
        p_minus = qml.math.real(minus_amps**2)
        z = softmax_weights(raw_logits)
        q = z[0] * p_plus - z[1] * p_minus
        return self.num_grid_points * q

    def _profile_expr(self, unit_idx: int) -> Any:
        self._validate_unit_idx(unit_idx)
        if self.mode == "standard":
            return self._standard_profile(self.raw_profiles[unit_idx])
        if self.mode == "mode_a":
            return self._mode_a_profile(self.raw_profiles[unit_idx])
        return self._mode_b_profile(
            self.raw_plus[unit_idx],
            self.raw_minus[unit_idx],
            self.raw_channel_logits[unit_idx],
        )

    def _interp_value(self, y_grid: Any, z: Any) -> Any:
        z = pnp.clip(z, self.activation_grid[0], self.activation_grid[-1])
        dx = self.activation_grid[1] - self.activation_grid[0]
        idx_float = (z - self.activation_grid[0]) / dx
        idx0 = pnp.floor(idx_float)
        idx0 = pnp.clip(idx0, 0, self.num_grid_points - 1)
        idx1 = pnp.clip(idx0 + 1, 0, self.num_grid_points - 1)
        i0 = idx0.astype(int)
        i1 = idx1.astype(int)
        x0 = self.activation_grid[i0]
        x1 = self.activation_grid[i1]
        y0 = y_grid[i0]
        y1 = y_grid[i1]
        denom = pnp.where(pnp.abs(x1 - x0) < EPS, 1.0, x1 - x0)
        t = (z - x0) / denom
        return (1.0 - t) * y0 + t * y1

    def hidden_features(self, x: Any) -> Any:
        x_vec = pnp.array(x, dtype=float)
        if x_vec.ndim == 0:
            x_vec = x_vec.reshape(1)

        features = []
        for unit_idx in range(self.hidden_units):
            z = pnp.tanh(pnp.dot(self.W_in[unit_idx], x_vec) + self.b_in[unit_idx])
            features.append(self._interp_value(self._profile_expr(unit_idx), z))
        return pnp.array(features)

    def forward_logits(self, x: Any) -> Any:
        hidden = self.hidden_features(x)
        return pnp.dot(self.W_out, hidden) + self.b_out

    def forward_batch(self, x_batch: Any) -> Any:
        xb = pnp.array(x_batch, dtype=float)
        return pnp.array([self.forward_logits(x_i) for x_i in xb])

    def predict_proba(self, x_batch: Any) -> np.ndarray:
        logits = _to_numpy_float(self.forward_batch(x_batch))
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        return _softmax_np(logits)

    def predict(self, x_batch: Any) -> np.ndarray:
        return np.argmax(self.predict_proba(x_batch), axis=1)

    def accuracy(self, x_batch: Any, y_true: np.ndarray) -> float:
        y = np.asarray(y_true, dtype=int)
        return float(np.mean(self.predict(x_batch) == y))

    def get_activation_profile(self, unit_idx: int) -> np.ndarray:
        return _to_numpy_float(self._profile_expr(unit_idx))

    def measure_activation_profile(self, unit_idx: int, shots: int = 5000) -> ActivationMeasurement:
        self._validate_unit_idx(unit_idx)

        if self.mode == "standard":
            amplitudes = _to_numpy_float(normalize_real_amplitudes(self.raw_profiles[unit_idx]))
            measurement = measure_standard_superposition(amplitudes, self.n_qubits, shots=shots)
            return ActivationMeasurement(
                mode=self.mode,
                profile=self.num_grid_points * measurement.probs,
                counts=measurement.counts,
            )

        if self.mode == "mode_a":
            amplitudes = _to_numpy_float(normalize_real_amplitudes(self.raw_profiles[unit_idx]))
            measurement = measure_mode_a_superposition(amplitudes, self.n_qubits, shots=shots)
            return ActivationMeasurement(
                mode=self.mode,
                profile=self.num_grid_points * measurement.q,
                counts=measurement.counts,
                p_pos=measurement.p_pos,
                p_neg=measurement.p_neg,
            )

        p_plus_amplitudes = _to_numpy_float(normalize_real_amplitudes(self.raw_plus[unit_idx]))
        p_minus_amplitudes = _to_numpy_float(normalize_real_amplitudes(self.raw_minus[unit_idx]))
        z = _to_numpy_float(softmax_weights(self.raw_channel_logits[unit_idx]))
        measurement = measure_mode_b_superposition(
            p_plus_amplitudes,
            p_minus_amplitudes,
            float(z[0]),
            float(z[1]),
            self.n_qubits,
            shots=shots,
        )
        return ActivationMeasurement(
            mode=self.mode,
            profile=self.num_grid_points * measurement.q_hat,
            p_plus=measurement.p_plus_hat,
            p_minus=measurement.p_minus_hat,
            z_plus=measurement.z_plus,
            z_minus=measurement.z_minus,
        )


def train_quantum_activation_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: QuantumActivationConfig,
    *,
    after_step: Callable[[int, float, QuantumActivationClassifier], None] | None = None,
    log_every: int | None = None,
) -> tuple[QuantumActivationClassifier, np.ndarray]:
    """Train a multiclass classifier with learned superposition activations."""
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    if x.ndim != 2:
        raise ValueError(f"x_train must be 2D, got shape {x.shape}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y_train must be shape (n_samples,) and aligned with x_train.")
    if x.shape[1] != config.input_dim:
        raise ValueError(f"x_train has input_dim {x.shape[1]}, expected {config.input_dim}")
    if np.any(y < 0) or np.any(y >= config.n_classes):
        raise ValueError("y_train labels must be in [0, n_classes - 1].")

    model = QuantumActivationClassifier(config)
    x_p = pnp.array(x)
    y_onehot = pnp.array(np.eye(config.n_classes, dtype=float)[y])
    opt = qml.AdamOptimizer(stepsize=config.learning_rate)
    params = model.parameters()
    losses: list[float] = []

    def softmax(logits: Any) -> Any:
        shifted = logits - pnp.max(logits, axis=1, keepdims=True)
        exp_shifted = pnp.exp(shifted)
        return exp_shifted / (pnp.sum(exp_shifted, axis=1, keepdims=True) + EPS)

    def loss_fn(*current_params: Any) -> Any:
        model.set_parameters(*current_params)
        logits = model.forward_batch(x_p)
        probs = softmax(logits)
        return -pnp.mean(pnp.sum(y_onehot * pnp.log(probs + EPS), axis=1))

    if after_step is not None:
        model.set_parameters(*params)
        loss0 = float(loss_fn(*params))
        after_step(-1, loss0, model)

    log_n = log_every if log_every is not None and log_every > 0 else 0
    if log_n:
        print(
            f"Training {config.steps} epochs (logging every {log_n})…",
            flush=True,
        )

    for step in range(config.steps):
        params, loss_val = opt.step_and_cost(loss_fn, *params)
        loss_f = float(loss_val)
        losses.append(loss_f)
        model.set_parameters(*params)
        if after_step is not None:
            after_step(step, loss_f, model)
        if log_n and ((step + 1) % log_n == 0 or step == config.steps - 1):
            print(
                f"  epoch {step + 1}/{config.steps}  train_loss={loss_f:.6f}",
                flush=True,
            )

    return model, np.asarray(losses, dtype=float)

