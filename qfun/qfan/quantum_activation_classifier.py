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


def _softmax2_np(raw: np.ndarray) -> np.ndarray:
    """Two-class softmax (Mode B channel weights), NumPy only."""
    raw = np.asarray(raw, dtype=float)
    shifted = raw - np.max(raw)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted)


def _norm_amp_np(raw: np.ndarray, eps: float = EPS) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    return raw / np.sqrt(np.sum(raw**2) + eps)


def _interp_values_np(z: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Linear interpolation of ``y_grid`` on ``x_grid``, evaluated at each ``z`` (vectorized)."""
    z = np.clip(np.asarray(z, dtype=float), x_grid[0], x_grid[-1])
    dx = x_grid[1] - x_grid[0]
    idx_float = (z - x_grid[0]) / dx
    idx0 = np.floor(idx_float).astype(np.int64)
    idx0 = np.clip(idx0, 0, len(x_grid) - 1)
    idx1 = np.clip(idx0 + 1, 0, len(x_grid) - 1)
    x0 = x_grid[idx0]
    x1 = x_grid[idx1]
    y0 = y_grid[idx0]
    y1 = y_grid[idx1]
    denom = np.where(np.abs(x1 - x0) < eps, 1.0, x1 - x0)
    t = (z - x0) / denom
    return (1.0 - t) * y0 + t * y1


def _silu_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class QuantumActivationConfig:
    input_dim: int
    hidden_units: int = 6
    hidden_layers: tuple[int, ...] | None = None
    n_qubits: int = 4
    n_classes: int = 3
    mode: str = "standard"
    learning_rate: float = 0.05
    steps: int = 120
    seed: int = 42
    #: If ``True``, train with JAX + Optax (install ``qfun[gpu]`` or ``jax`` + ``optax``).
    #: Uses minibatches of ``batch_size``; good for large datasets and optional GPU.
    use_jax: bool = False
    batch_size: int = 512
    #: If ``True``, show a tqdm progress bar during training when ``tqdm`` is installed.
    show_training_progress: bool = False
    #: ``"superposition"``: linear pre-activation, then interpolation on the learned profile
    #: (values clipped to the grid). ``"tanh"``: apply ``tanh`` before interpolation so
    #: inputs stay in ``(-1, 1)`` (legacy bounded path, often easier to optimize).
    hidden_preactivation: str = "superposition"
    #: ``"pure_superposition"`` keeps the original learned quantum activation only.
    #: ``"kan_quantum_hybrid"`` adds a KAN-like classical base path plus quantum correction.
    hidden_function_family: str = "pure_superposition"
    #: Base-path activation for the hybrid family. Only ``"silu"`` is currently supported.
    hidden_base_activation: str = "silu"
    #: Optional smoothness regularization applied to the effective quantum branch.
    profile_smoothness_reg: float = 0.0

    def resolved_hidden_layers(self) -> tuple[int, ...]:
        """Return the effective hidden-layer widths.

        ``hidden_units`` remains as a single-layer compatibility alias. When
        ``hidden_layers`` is provided it takes precedence.
        """
        if self.hidden_layers is None:
            layers = (int(self.hidden_units),)
        else:
            layers = tuple(int(width) for width in self.hidden_layers)
        if not layers:
            raise ValueError("hidden_layers must contain at least one hidden layer.")
        if any(width <= 0 for width in layers):
            raise ValueError("All hidden layer widths must be positive.")
        return layers


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


@dataclass(frozen=True)
class ActivationComponents:
    base: np.ndarray
    quantum: np.ndarray
    combined: np.ndarray
    quantum_profile: np.ndarray
    base_scale: float
    quantum_scale: float


class QuantumActivationClassifier:
    """Small multiclass classifier with learned 1D superposition activations."""

    def __init__(self, config: QuantumActivationConfig):
        if config.mode not in {"standard", "mode_a", "mode_b"}:
            raise ValueError("mode must be 'standard', 'mode_a', or 'mode_b'.")
        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        hidden_layers = config.resolved_hidden_layers()
        if config.n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        if config.n_classes <= 1:
            raise ValueError("n_classes must be at least 2.")
        if config.hidden_preactivation not in {"superposition", "tanh"}:
            raise ValueError(
                'hidden_preactivation must be "superposition" or "tanh", '
                f"got {config.hidden_preactivation!r}."
            )
        if config.hidden_function_family not in {"pure_superposition", "kan_quantum_hybrid"}:
            raise ValueError(
                'hidden_function_family must be "pure_superposition" or "kan_quantum_hybrid", '
                f"got {config.hidden_function_family!r}."
            )
        if config.hidden_base_activation not in {"silu"}:
            raise ValueError(
                f"hidden_base_activation must be 'silu', got {config.hidden_base_activation!r}."
            )
        if config.profile_smoothness_reg < 0.0:
            raise ValueError("profile_smoothness_reg must be nonnegative.")

        self.input_dim = int(config.input_dim)
        self.hidden_layer_sizes = hidden_layers
        self.num_hidden_layers = len(hidden_layers)
        self.hidden_units = hidden_layers[-1]
        self.n_qubits = int(config.n_qubits)
        self.n_classes = int(config.n_classes)
        self.mode = config.mode
        self.hidden_preactivation = config.hidden_preactivation
        self.hidden_function_family = config.hidden_function_family
        self.hidden_base_activation = config.hidden_base_activation
        self.profile_smoothness_reg = float(config.profile_smoothness_reg)
        self.num_grid_points = 2**self.n_qubits
        self.activation_grid = pnp.array(np.linspace(-1.0, 1.0, self.num_grid_points))

        rng = np.random.default_rng(config.seed)

        self.hidden_weights: list[Any] = []
        self.hidden_biases: list[Any] = []
        prev_dim = self.input_dim
        for width in self.hidden_layer_sizes:
            self.hidden_weights.append(
                pnp.array(
                    rng.normal(scale=0.35, size=(width, prev_dim)),
                    requires_grad=True,
                )
            )
            self.hidden_biases.append(
                pnp.array(
                    rng.normal(scale=0.05, size=(width,)),
                    requires_grad=True,
                )
            )
            prev_dim = width

        self.W_out = pnp.array(
            rng.normal(scale=0.35, size=(self.n_classes, self.hidden_layer_sizes[-1])),
            requires_grad=True,
        )
        self.b_out = pnp.array(
            np.zeros(self.n_classes, dtype=float),
            requires_grad=True,
        )

        if self.mode in {"standard", "mode_a"}:
            profile_width = self.num_grid_points if self.mode == "standard" else 2 * self.num_grid_points
            self.raw_profiles_layers = [
                pnp.array(
                    rng.normal(scale=0.25, size=(width, profile_width)),
                    requires_grad=True,
                )
                for width in self.hidden_layer_sizes
            ]
        else:
            self.raw_plus_layers = [
                pnp.array(
                    rng.normal(scale=0.25, size=(width, self.num_grid_points)),
                    requires_grad=True,
                )
                for width in self.hidden_layer_sizes
            ]
            self.raw_minus_layers = [
                pnp.array(
                    rng.normal(scale=0.25, size=(width, self.num_grid_points)),
                    requires_grad=True,
                )
                for width in self.hidden_layer_sizes
            ]
            self.raw_channel_logits_layers = [
                pnp.array(
                    rng.normal(scale=0.05, size=(width, 2)),
                    requires_grad=True,
                )
                for width in self.hidden_layer_sizes
            ]

        self.base_mix_layers = [
            pnp.array(
                np.ones(width, dtype=float) if self.hidden_function_family == "kan_quantum_hybrid" else np.zeros(width, dtype=float),
                requires_grad=self.hidden_function_family == "kan_quantum_hybrid",
            )
            for width in self.hidden_layer_sizes
        ]
        self.quantum_mix_layers = [
            pnp.array(
                np.ones(width, dtype=float),
                requires_grad=self.hidden_function_family == "kan_quantum_hybrid",
            )
            for width in self.hidden_layer_sizes
        ]

        self._sync_legacy_aliases()

    def _sync_legacy_aliases(self) -> None:
        """Keep first-layer aliases available for backward compatibility."""
        self.W_in = self.hidden_weights[0]
        self.b_in = self.hidden_biases[0]
        self.base_mix = self.base_mix_layers[0]
        self.quantum_mix = self.quantum_mix_layers[0]
        if self.mode in {"standard", "mode_a"}:
            self.raw_profiles = self.raw_profiles_layers[0]
        else:
            self.raw_plus = self.raw_plus_layers[0]
            self.raw_minus = self.raw_minus_layers[0]
            self.raw_channel_logits = self.raw_channel_logits_layers[0]

    def parameters(self) -> list[Any]:
        base: list[Any] = [
            *self.hidden_weights,
            *self.hidden_biases,
            self.W_out,
            self.b_out,
        ]
        if self.mode in {"standard", "mode_a"}:
            base += list(self.raw_profiles_layers)
        else:
            base += list(self.raw_plus_layers) + list(self.raw_minus_layers) + list(self.raw_channel_logits_layers)
        if self.hidden_function_family == "kan_quantum_hybrid":
            base += list(self.base_mix_layers) + list(self.quantum_mix_layers)
        return base

    def set_parameters(self, *params: Any) -> None:
        n_layers = self.num_hidden_layers
        base_count = 2 * n_layers + 2
        if len(params) < base_count:
            raise ValueError(f"Expected at least {base_count} parameters, got {len(params)}.")

        self.hidden_weights = list(params[:n_layers])
        self.hidden_biases = list(params[n_layers : 2 * n_layers])
        self.W_out = params[2 * n_layers]
        self.b_out = params[2 * n_layers + 1]

        remaining = list(params[base_count:])
        if self.mode in {"standard", "mode_a"}:
            if len(remaining) < n_layers:
                raise ValueError(f"Expected at least {n_layers} activation tensors, got {len(remaining)}.")
            self.raw_profiles_layers = remaining[:n_layers]
            remaining = remaining[n_layers:]
        else:
            expected = 3 * n_layers
            if len(remaining) < expected:
                raise ValueError(f"Expected at least {expected} activation tensors, got {len(remaining)}.")
            self.raw_plus_layers = remaining[:n_layers]
            self.raw_minus_layers = remaining[n_layers : 2 * n_layers]
            self.raw_channel_logits_layers = remaining[2 * n_layers : expected]
            remaining = remaining[expected:]

        if self.hidden_function_family == "kan_quantum_hybrid":
            expected_scales = 2 * n_layers
            if len(remaining) != expected_scales:
                raise ValueError(f"Expected {expected_scales} branch-scale tensors, got {len(remaining)}.")
            self.base_mix_layers = remaining[:n_layers]
            self.quantum_mix_layers = remaining[n_layers:]
        elif remaining:
            raise ValueError(f"Unexpected extra parameters for pure_superposition model: {len(remaining)}")

        self._sync_legacy_aliases()

    def _parse_layer_unit_args(self, layer_or_unit: int, unit_idx: int | None) -> tuple[int, int]:
        if unit_idx is None:
            if self.num_hidden_layers != 1:
                raise ValueError(
                    "Multi-layer models require both layer_idx and unit_idx. "
                    "Use get_activation_profile(layer_idx, unit_idx)."
                )
            return 0, int(layer_or_unit)
        return int(layer_or_unit), int(unit_idx)

    def _validate_layer_unit_idx(self, layer_idx: int, unit_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_hidden_layers:
            raise IndexError(f"layer_idx must be in [0, {self.num_hidden_layers - 1}], got {layer_idx}")
        layer_width = self.hidden_layer_sizes[layer_idx]
        if unit_idx < 0 or unit_idx >= layer_width:
            raise IndexError(f"unit_idx must be in [0, {layer_width - 1}], got {unit_idx}")

    def _silu_expr(self, x: Any) -> Any:
        return x / (1.0 + pnp.exp(-x))

    def _quantum_branch_input(self, z: Any) -> Any:
        if self.hidden_preactivation == "tanh":
            return pnp.tanh(z)
        return z

    def _quantum_branch_input_np(self, z: np.ndarray) -> np.ndarray:
        if self.hidden_preactivation == "tanh":
            return np.tanh(z)
        return np.asarray(z, dtype=float)

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

    def _profile_expr(self, layer_idx: int, unit_idx: int) -> Any:
        self._validate_layer_unit_idx(layer_idx, unit_idx)
        if self.mode == "standard":
            return self._standard_profile(self.raw_profiles_layers[layer_idx][unit_idx])
        if self.mode == "mode_a":
            return self._mode_a_profile(self.raw_profiles_layers[layer_idx][unit_idx])
        return self._mode_b_profile(
            self.raw_plus_layers[layer_idx][unit_idx],
            self.raw_minus_layers[layer_idx][unit_idx],
            self.raw_channel_logits_layers[layer_idx][unit_idx],
        )

    def _base_scale_expr(self, layer_idx: int, unit_idx: int) -> Any:
        if self.hidden_function_family != "kan_quantum_hybrid":
            return pnp.array(0.0)
        return self.base_mix_layers[layer_idx][unit_idx]

    def _quantum_scale_expr(self, layer_idx: int, unit_idx: int) -> Any:
        return self.quantum_mix_layers[layer_idx][unit_idx]

    def _base_scale_np(self, layer_idx: int, unit_idx: int) -> float:
        if self.hidden_function_family != "kan_quantum_hybrid":
            return 0.0
        return float(np.asarray(self.base_mix_layers[layer_idx][unit_idx], dtype=float))

    def _quantum_scale_np(self, layer_idx: int, unit_idx: int) -> float:
        return float(np.asarray(self.quantum_mix_layers[layer_idx][unit_idx], dtype=float))

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

    def _apply_hidden_layer(self, inputs: Any, layer_idx: int) -> Any:
        x_vec = pnp.array(inputs, dtype=float)
        if x_vec.ndim == 0:
            x_vec = x_vec.reshape(1)

        features = []
        for unit_idx in range(self.hidden_layer_sizes[layer_idx]):
            z = pnp.dot(self.hidden_weights[layer_idx][unit_idx], x_vec) + self.hidden_biases[layer_idx][unit_idx]
            quantum = self._interp_value(self._profile_expr(layer_idx, unit_idx), self._quantum_branch_input(z))
            if self.hidden_function_family == "kan_quantum_hybrid":
                base = self._silu_expr(z)
                out = self._base_scale_expr(layer_idx, unit_idx) * base
                out = out + self._quantum_scale_expr(layer_idx, unit_idx) * quantum
                features.append(out)
            else:
                features.append(quantum)
        return pnp.array(features)

    def hidden_features(self, x: Any) -> Any:
        hidden = pnp.array(x, dtype=float)
        if hidden.ndim == 0:
            hidden = hidden.reshape(1)
        for layer_idx in range(self.num_hidden_layers):
            hidden = self._apply_hidden_layer(hidden, layer_idx)
        return hidden

    def forward_logits(self, x: Any) -> Any:
        hidden = self.hidden_features(x)
        return pnp.dot(self.W_out, hidden) + self.b_out

    def forward_batch(self, x_batch: Any) -> Any:
        xb = pnp.array(x_batch, dtype=float)
        return pnp.array([self.forward_logits(x_i) for x_i in xb])

    def _quantum_profile_np(self, layer_idx: int, unit_idx: int) -> np.ndarray:
        """Same quantum profile as ``_profile_expr``, purely NumPy."""
        self._validate_layer_unit_idx(layer_idx, unit_idx)
        g = float(self.num_grid_points)
        if self.mode == "standard":
            raw = np.asarray(self.raw_profiles_layers[layer_idx][unit_idx], dtype=float)
            amps = _norm_amp_np(raw)
            return g * (amps**2)
        if self.mode == "mode_a":
            raw = np.asarray(self.raw_profiles_layers[layer_idx][unit_idx], dtype=float)
            amps = _norm_amp_np(raw)
            fp = amps**2
            q = fp[0::2] - fp[1::2]
            return g * q
        rp = np.asarray(self.raw_plus_layers[layer_idx][unit_idx], dtype=float)
        rm = np.asarray(self.raw_minus_layers[layer_idx][unit_idx], dtype=float)
        lg = np.asarray(self.raw_channel_logits_layers[layer_idx][unit_idx], dtype=float)
        pp = _norm_amp_np(rp) ** 2
        pm = _norm_amp_np(rm) ** 2
        z = _softmax2_np(lg)
        return g * (z[0] * pp - z[1] * pm)

    def _activation_components_np(self, layer_idx: int, unit_idx: int) -> ActivationComponents:
        self._validate_layer_unit_idx(layer_idx, unit_idx)
        x_grid = np.asarray(self.activation_grid, dtype=float)
        quantum_profile = self._quantum_profile_np(layer_idx, unit_idx)
        quantum_scale = self._quantum_scale_np(layer_idx, unit_idx)
        quantum_inputs = self._quantum_branch_input_np(x_grid)
        quantum_curve = quantum_scale * _interp_values_np(quantum_inputs, x_grid, quantum_profile)
        base_scale = self._base_scale_np(layer_idx, unit_idx)
        base_curve = base_scale * _silu_np(x_grid)
        combined = base_curve + quantum_curve
        return ActivationComponents(
            base=base_curve,
            quantum=quantum_curve,
            combined=combined,
            quantum_profile=quantum_profile,
            base_scale=base_scale,
            quantum_scale=quantum_scale,
        )

    def _forward_batch_numpy(self, x: np.ndarray) -> np.ndarray:
        """Vectorized forward pass for metrics / prediction (no per-sample PennyLane)."""
        hidden = np.asarray(x, dtype=np.float64)
        if hidden.ndim == 1:
            hidden = hidden.reshape(1, -1)
        x_grid = np.asarray(self.activation_grid, dtype=np.float64)
        n = hidden.shape[0]
        w_out = np.asarray(self.W_out, dtype=np.float64)
        b_out = np.asarray(self.b_out, dtype=np.float64)
        for layer_idx in range(self.num_hidden_layers):
            w = np.asarray(self.hidden_weights[layer_idx], dtype=np.float64)
            b = np.asarray(self.hidden_biases[layer_idx], dtype=np.float64)
            z_pre = hidden @ w.T + b
            z_quantum = self._quantum_branch_input_np(z_pre)
            width = self.hidden_layer_sizes[layer_idx]
            quantum_out = np.empty((n, width), dtype=np.float64)
            for h in range(width):
                prof = self._quantum_profile_np(layer_idx, h)
                quantum_out[:, h] = _interp_values_np(z_quantum[:, h], x_grid, prof)
            if self.hidden_function_family == "kan_quantum_hybrid":
                base_scales = np.asarray(self.base_mix_layers[layer_idx], dtype=np.float64).reshape(1, -1)
                quantum_scales = np.asarray(self.quantum_mix_layers[layer_idx], dtype=np.float64).reshape(1, -1)
                hidden = base_scales * _silu_np(z_pre) + quantum_scales * quantum_out
            else:
                hidden = quantum_out
        return hidden @ w_out.T + b_out

    def predict_proba(self, x_batch: Any) -> np.ndarray:
        logits = self._forward_batch_numpy(np.asarray(x_batch, dtype=float))
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        return _softmax_np(logits)

    def predict(self, x_batch: Any) -> np.ndarray:
        return np.argmax(self.predict_proba(x_batch), axis=1)

    def accuracy(self, x_batch: Any, y_true: np.ndarray) -> float:
        y = np.asarray(y_true, dtype=int)
        return float(np.mean(self.predict(x_batch) == y))

    def get_activation_components(self, layer_idx: int, unit_idx: int | None = None) -> ActivationComponents:
        layer_idx, unit_idx = self._parse_layer_unit_args(layer_idx, unit_idx)
        return self._activation_components_np(layer_idx, unit_idx)

    def get_activation_profile(self, layer_idx: int, unit_idx: int | None = None) -> np.ndarray:
        layer_idx, unit_idx = self._parse_layer_unit_args(layer_idx, unit_idx)
        return self.get_activation_components(layer_idx, unit_idx).combined

    def profile_smoothness_penalty(self) -> Any:
        if self.hidden_function_family != "kan_quantum_hybrid" or self.profile_smoothness_reg <= 0.0:
            return pnp.array(0.0)
        penalties = []
        for layer_idx in range(self.num_hidden_layers):
            for unit_idx in range(self.hidden_layer_sizes[layer_idx]):
                effective_profile = self._quantum_scale_expr(layer_idx, unit_idx) * self._profile_expr(layer_idx, unit_idx)
                diffs = effective_profile[1:] - effective_profile[:-1]
                penalties.append(pnp.mean(diffs**2))
        if not penalties:
            return pnp.array(0.0)
        return pnp.mean(pnp.stack(penalties))

    def measure_activation_profile(
        self,
        layer_idx: int,
        unit_idx: int | None = None,
        shots: int = 5000,
    ) -> ActivationMeasurement:
        layer_idx, unit_idx = self._parse_layer_unit_args(layer_idx, unit_idx)
        self._validate_layer_unit_idx(layer_idx, unit_idx)

        if self.mode == "standard":
            amplitudes = _to_numpy_float(normalize_real_amplitudes(self.raw_profiles_layers[layer_idx][unit_idx]))
            measurement = measure_standard_superposition(amplitudes, self.n_qubits, shots=shots)
            return ActivationMeasurement(
                mode=self.mode,
                profile=self.num_grid_points * measurement.probs,
                counts=measurement.counts,
            )

        if self.mode == "mode_a":
            amplitudes = _to_numpy_float(normalize_real_amplitudes(self.raw_profiles_layers[layer_idx][unit_idx]))
            measurement = measure_mode_a_superposition(amplitudes, self.n_qubits, shots=shots)
            return ActivationMeasurement(
                mode=self.mode,
                profile=self.num_grid_points * measurement.q,
                counts=measurement.counts,
                p_pos=measurement.p_pos,
                p_neg=measurement.p_neg,
            )

        p_plus_amplitudes = _to_numpy_float(normalize_real_amplitudes(self.raw_plus_layers[layer_idx][unit_idx]))
        p_minus_amplitudes = _to_numpy_float(normalize_real_amplitudes(self.raw_minus_layers[layer_idx][unit_idx]))
        z = _to_numpy_float(softmax_weights(self.raw_channel_logits_layers[layer_idx][unit_idx]))
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
            counts=None,
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
    """Train a multiclass classifier with learned superposition activations.

    Set ``config.use_jax=True`` for JAX/Optax training (CPU or GPU if a CUDA jaxlib
    is installed). The trained weights are copied back into the same PennyLane-based
    model so ``measure_activation_profile`` and other diagnostics keep working.
    """
    if config.use_jax:
        from ._jax_quantum_activation import jax_ready, train_quantum_activation_classifier_jax

        if not jax_ready():
            raise ImportError(
                "config.use_jax=True requires JAX and Optax. "
                'Install with: pip install "qfun[gpu]" or pip install jax optax.'
            )
        return train_quantum_activation_classifier_jax(
            x_train,
            y_train,
            config,
            after_step=after_step,
            log_every=log_every,
        )

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
        data_loss = -pnp.mean(pnp.sum(y_onehot * pnp.log(probs + EPS), axis=1))
        return data_loss + config.profile_smoothness_reg * model.profile_smoothness_penalty()

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

    epoch_iter = range(config.steps)
    epoch_pbar = None
    if config.show_training_progress:
        try:
            from tqdm.auto import tqdm

            epoch_pbar = tqdm(epoch_iter, desc="Training (PennyLane)", unit="epoch")
            epoch_iter = epoch_pbar
        except ImportError:
            pass

    for step in epoch_iter:
        params, loss_val = opt.step_and_cost(loss_fn, *params)
        loss_f = float(loss_val)
        losses.append(loss_f)
        model.set_parameters(*params)
        if after_step is not None:
            after_step(step, loss_f, model)
        if epoch_pbar is not None:
            epoch_pbar.set_postfix(loss=f"{loss_f:.4f}", refresh=True)
        if log_n and ((step + 1) % log_n == 0 or step == config.steps - 1):
            print(
                f"  epoch {step + 1}/{config.steps}  train_loss={loss_f:.6f}",
                flush=True,
            )

    if epoch_pbar is not None:
        epoch_pbar.close()

    return model, np.asarray(losses, dtype=float)
