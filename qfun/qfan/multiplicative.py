"""Multiplicative quantum nodes (MultQFun).

Implements mixed additive-multiplicative aggregation inspired by KAN 2.0's
MultKAN and LeanKAN. Each hidden node's output is:

    z = y_mult + y_add

where:
    y_mult = prod_{j in mult_set} phi_j(x_j)
    y_add  = sum_{j in add_set} phi_j(x_j)

This is valuable for physics equations with multiplicative structure
(e.g. F = G*m1*m2/r^2) that would otherwise require deeper architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pennylane.numpy as pnp

from .._utils import EPS
from ..quantum_learning import normalize_real_amplitudes, softmax_weights
from ._profile_interp import PROFILE_INTERP_MODES, _open_uniform_knots, interp_linear_pnp


@dataclass(frozen=True)
class MultQFunConfig:
    input_dim: int
    hidden_layers: tuple[int, ...] = (6,)
    n_qubits: int = 4
    n_classes: int = 1
    mode: str = "standard"
    n_multiplicative: int = 2
    learning_rate: float = 0.05
    steps: int = 120
    seed: int = 42
    profile_interp: Literal["linear", "cubic_natural", "cubic_bspline"] = "linear"

    @property
    def num_grid_points(self) -> int:
        return 2 ** self.n_qubits


class MultQFunBlock:
    """Function approximation block with mixed multiplicative-additive aggregation.

    For each output node j, the first ``n_multiplicative`` edge activations are
    multiplied together, and the remaining are summed:

        h_j = prod_{i < n_mu} phi_{ij}(x_i) + sum_{i >= n_mu} phi_{ij}(x_i) + bias_j
    """

    def __init__(self, config: MultQFunConfig):
        if config.mode not in {"standard", "mode_a", "mode_b"}:
            raise ValueError("mode must be 'standard', 'mode_a', or 'mode_b'.")

        self.config = config
        self.input_dim = config.input_dim
        self.hidden_layer_sizes = config.hidden_layers
        self.num_hidden_layers = len(config.hidden_layers)
        self.n_qubits = config.n_qubits
        self.n_classes = config.n_classes
        self.mode = config.mode
        self.n_multiplicative = min(config.n_multiplicative, config.input_dim)
        self.num_grid_points = config.num_grid_points
        self.activation_grid = pnp.array(np.linspace(-1.0, 1.0, self.num_grid_points))

        rng = np.random.default_rng(config.seed)

        self.edge_profiles: list[Any] = []
        self.edge_scales: list[Any] = []
        self.edge_shifts: list[Any] = []
        self.edge_biases: list[Any] = []
        self.n_mult_per_layer: list[int] = []

        if config.mode == "mode_b":
            self.edge_plus: list[Any] = []
            self.edge_minus: list[Any] = []
            self.edge_logits: list[Any] = []

        prev_dim = self.input_dim
        for layer_idx, width in enumerate(self.hidden_layer_sizes):
            n_mu = min(self.n_multiplicative, prev_dim)
            self.n_mult_per_layer.append(n_mu)

            pw = self.num_grid_points if config.mode != "mode_a" else 2 * self.num_grid_points
            if config.mode != "mode_b":
                self.edge_profiles.append(
                    pnp.array(rng.normal(scale=0.25, size=(width, prev_dim, pw)), requires_grad=True)
                )
            else:
                self.edge_plus.append(
                    pnp.array(rng.normal(scale=0.25, size=(width, prev_dim, self.num_grid_points)), requires_grad=True)
                )
                self.edge_minus.append(
                    pnp.array(rng.normal(scale=0.25, size=(width, prev_dim, self.num_grid_points)), requires_grad=True)
                )
                self.edge_logits.append(
                    pnp.array(rng.normal(scale=0.05, size=(width, prev_dim, 2)), requires_grad=True)
                )
            self.edge_scales.append(
                pnp.array(rng.normal(scale=0.5, size=(width, prev_dim)) + 1.0, requires_grad=True)
            )
            self.edge_shifts.append(
                pnp.array(rng.normal(scale=0.05, size=(width, prev_dim)), requires_grad=True)
            )
            self.edge_biases.append(
                pnp.array(np.zeros(width), requires_grad=True)
            )
            prev_dim = width

        self.W_out = pnp.array(
            rng.normal(scale=0.35, size=(self.n_classes, self.hidden_layer_sizes[-1])),
            requires_grad=True,
        )
        self.b_out = pnp.array(np.zeros(self.n_classes), requires_grad=True)

    def _compute_profile(self, raw: Any) -> Any:
        if self.mode == "standard":
            amps = normalize_real_amplitudes(raw)
            return self.num_grid_points * (amps ** 2)
        if self.mode == "mode_a":
            amps = normalize_real_amplitudes(raw)
            fp = amps ** 2
            return self.num_grid_points * (fp[0::2] - fp[1::2])
        raise ValueError("Use _compute_profile_b for mode_b.")

    def _compute_profile_b(self, raw_plus: Any, raw_minus: Any, raw_logits: Any) -> Any:
        pp = normalize_real_amplitudes(raw_plus) ** 2
        pm = normalize_real_amplitudes(raw_minus) ** 2
        z = softmax_weights(raw_logits)
        return self.num_grid_points * (z[0] * pp - z[1] * pm)

    def _edge_activation(self, layer_idx: int, j: int, i: int, x_i: Any) -> Any:
        """Compute activation for edge (i -> j) at layer layer_idx."""
        z = self.edge_scales[layer_idx][j, i] * x_i + self.edge_shifts[layer_idx][j, i]
        z = pnp.clip(z, -1.0, 1.0)
        if self.mode == "mode_b":
            prof = self._compute_profile_b(
                self.edge_plus[layer_idx][j, i],
                self.edge_minus[layer_idx][j, i],
                self.edge_logits[layer_idx][j, i],
            )
        else:
            prof = self._compute_profile(self.edge_profiles[layer_idx][j, i])
        return interp_linear_pnp(z, self.activation_grid, pnp.asarray(prof, dtype=float), EPS)

    def _apply_hidden_layer(self, inputs: Any, layer_idx: int) -> Any:
        x = pnp.array(inputs, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1)
        width = self.hidden_layer_sizes[layer_idx]
        n_mu = self.n_mult_per_layer[layer_idx]
        features = []

        for j in range(width):
            y_mult = pnp.array(1.0)
            for i in range(n_mu):
                y_mult = y_mult * self._edge_activation(layer_idx, j, i, x[i])

            y_add = pnp.array(0.0)
            for i in range(n_mu, x.shape[0]):
                y_add = y_add + self._edge_activation(layer_idx, j, i, x[i])

            features.append(y_mult + y_add + self.edge_biases[layer_idx][j])
        return pnp.array(features)

    def forward(self, x: Any) -> Any:
        hidden = pnp.array(x, dtype=float)
        if hidden.ndim == 0:
            hidden = hidden.reshape(1)
        for layer_idx in range(self.num_hidden_layers):
            hidden = self._apply_hidden_layer(hidden, layer_idx)
        return pnp.dot(self.W_out, hidden) + self.b_out

    def forward_batch(self, x_batch: Any) -> Any:
        xb = pnp.array(x_batch, dtype=float)
        return pnp.array([self.forward(xi) for xi in xb])

    def parameters(self) -> list[Any]:
        params: list[Any] = []
        params.extend(self.edge_scales)
        params.extend(self.edge_shifts)
        params.extend(self.edge_biases)
        if self.mode == "mode_b":
            params.extend(self.edge_plus)
            params.extend(self.edge_minus)
            params.extend(self.edge_logits)
        else:
            params.extend(self.edge_profiles)
        params.append(self.W_out)
        params.append(self.b_out)
        return params

    def set_parameters(self, *params: Any) -> None:
        p = list(params)
        idx = 0
        n_layers = self.num_hidden_layers
        self.edge_scales = p[idx : idx + n_layers]
        idx += n_layers
        self.edge_shifts = p[idx : idx + n_layers]
        idx += n_layers
        self.edge_biases = p[idx : idx + n_layers]
        idx += n_layers
        if self.mode == "mode_b":
            self.edge_plus = p[idx : idx + n_layers]
            idx += n_layers
            self.edge_minus = p[idx : idx + n_layers]
            idx += n_layers
            self.edge_logits = p[idx : idx + n_layers]
            idx += n_layers
        else:
            self.edge_profiles = p[idx : idx + n_layers]
            idx += n_layers
        self.W_out = p[idx]
        self.b_out = p[idx + 1]
        if idx + 2 != len(p):
            raise ValueError(f"Expected {idx + 2} parameters, got {len(p)}.")

    def predict(self, x_batch: Any) -> np.ndarray:
        """For regression: return raw outputs."""
        return np.asarray(self.forward_batch(x_batch), dtype=float)


def train_multqfun(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: MultQFunConfig,
    *,
    log_every: int | None = None,
) -> tuple[MultQFunBlock, np.ndarray]:
    """Train a MultQFun block for regression (MSE loss)."""
    import pennylane as qml

    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    model = MultQFunBlock(config)

    x_p = pnp.array(x)
    y_p = pnp.array(y)
    opt = qml.AdamOptimizer(stepsize=config.learning_rate)
    params = model.parameters()
    losses: list[float] = []

    def loss_fn(*current_params: Any) -> Any:
        model.set_parameters(*current_params)
        preds = model.forward_batch(x_p)
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.ravel()
        return pnp.mean((preds - y_p) ** 2)

    log_n = log_every if log_every is not None and log_every > 0 else 0
    for step in range(config.steps):
        params, loss_val = opt.step_and_cost(loss_fn, *params)
        loss_f = float(loss_val)
        losses.append(loss_f)
        model.set_parameters(*params)
        if log_n and ((step + 1) % log_n == 0 or step == config.steps - 1):
            print(f"  epoch {step + 1}/{config.steps}  mse={loss_f:.6f}", flush=True)

    return model, np.asarray(losses, dtype=float)
