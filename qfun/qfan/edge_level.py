"""Edge-level quantum amplitude activation classifier.

Implements the KAN-style architecture where each input-to-hidden edge has its
own learned quantum amplitude profile, rather than one profile per hidden unit.

Two variants are provided:
  - ``"edge"``: independent profiles per edge (i -> j)
  - ``"shared_edge"``: shared parent profile per output node with learnable
    affine transforms per edge (inspired by GS-KAN / Sprecher's refinement)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pennylane.numpy as pnp

from .._utils import EPS
from ..quantum_learning import normalize_real_amplitudes, softmax_weights
from ._profile_interp import PROFILE_INTERP_MODES, _open_uniform_knots, interp_linear_pnp, interp_profile_np


@dataclass(frozen=True)
class EdgeLevelConfig:
    input_dim: int
    hidden_layers: tuple[int, ...] = (6,)
    n_qubits: int = 4
    n_classes: int = 3
    mode: str = "standard"
    activation_level: Literal["edge", "shared_edge"] = "edge"
    learning_rate: float = 0.05
    steps: int = 120
    seed: int = 42
    profile_interp: Literal["linear", "cubic_natural", "cubic_bspline"] = "linear"

    @property
    def num_grid_points(self) -> int:
        return 2 ** self.n_qubits


def _profile_width(mode: str, num_grid_points: int) -> int:
    if mode == "mode_a":
        return 2 * num_grid_points
    return num_grid_points


class EdgeLevelClassifier:
    """Classifier with per-edge learned quantum amplitude activation profiles."""

    def __init__(self, config: EdgeLevelConfig):
        if config.mode not in {"standard", "mode_a", "mode_b"}:
            raise ValueError("mode must be 'standard', 'mode_a', or 'mode_b'.")
        if config.activation_level not in {"edge", "shared_edge"}:
            raise ValueError("activation_level must be 'edge' or 'shared_edge'.")
        if config.profile_interp not in PROFILE_INTERP_MODES:
            raise ValueError(f"profile_interp must be one of {PROFILE_INTERP_MODES}.")

        self.config = config
        self.input_dim = config.input_dim
        self.hidden_layer_sizes = config.hidden_layers
        self.num_hidden_layers = len(config.hidden_layers)
        self.n_qubits = config.n_qubits
        self.n_classes = config.n_classes
        self.mode = config.mode
        self.activation_level = config.activation_level
        self.profile_interp = config.profile_interp
        self.num_grid_points = config.num_grid_points
        self.activation_grid = pnp.array(np.linspace(-1.0, 1.0, self.num_grid_points))
        self._bspline_knots_np = _open_uniform_knots(-1.0, 1.0, self.num_grid_points, 3)

        rng = np.random.default_rng(config.seed)
        pw = _profile_width(config.mode, self.num_grid_points)

        self.edge_biases: list[Any] = []
        self.output_biases = pnp.array(np.zeros(self.n_classes), requires_grad=True)

        if config.activation_level == "edge":
            self._init_edge_profiles(rng, pw)
        else:
            self._init_shared_edge_profiles(rng, pw)

        prev_dim = config.hidden_layers[-1]
        self.W_out = pnp.array(
            rng.normal(scale=0.35, size=(self.n_classes, prev_dim)),
            requires_grad=True,
        )

        if config.mode == "mode_b":
            self._init_mode_b_edge(rng)

    def _init_edge_profiles(self, rng: np.random.Generator, pw: int) -> None:
        self.edge_profiles: list[Any] = []
        self.edge_scales: list[Any] = []
        self.edge_shifts: list[Any] = []
        prev_dim = self.input_dim
        for width in self.hidden_layer_sizes:
            if self.mode != "mode_b":
                self.edge_profiles.append(
                    pnp.array(rng.normal(scale=0.25, size=(width, prev_dim, pw)), requires_grad=True)
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

    def _init_shared_edge_profiles(self, rng: np.random.Generator, pw: int) -> None:
        self.parent_profiles: list[Any] = []
        self.edge_alphas: list[Any] = []
        self.edge_a: list[Any] = []
        self.edge_b: list[Any] = []
        prev_dim = self.input_dim
        for width in self.hidden_layer_sizes:
            if self.mode != "mode_b":
                self.parent_profiles.append(
                    pnp.array(rng.normal(scale=0.25, size=(width, pw)), requires_grad=True)
                )
            self.edge_alphas.append(
                pnp.array(rng.normal(scale=0.5, size=(width, prev_dim)) + 1.0, requires_grad=True)
            )
            self.edge_a.append(
                pnp.array(rng.normal(scale=0.5, size=(width, prev_dim)) + 1.0, requires_grad=True)
            )
            self.edge_b.append(
                pnp.array(rng.normal(scale=0.05, size=(width, prev_dim)), requires_grad=True)
            )
            self.edge_biases.append(
                pnp.array(np.zeros(width), requires_grad=True)
            )
            prev_dim = width

    def _init_mode_b_edge(self, rng: np.random.Generator) -> None:
        gp = self.num_grid_points
        prev_dim = self.input_dim
        if self.activation_level == "edge":
            self.edge_plus: list[Any] = []
            self.edge_minus: list[Any] = []
            self.edge_logits: list[Any] = []
            for width in self.hidden_layer_sizes:
                self.edge_plus.append(pnp.array(rng.normal(scale=0.25, size=(width, prev_dim, gp)), requires_grad=True))
                self.edge_minus.append(pnp.array(rng.normal(scale=0.25, size=(width, prev_dim, gp)), requires_grad=True))
                self.edge_logits.append(pnp.array(rng.normal(scale=0.05, size=(width, prev_dim, 2)), requires_grad=True))
                prev_dim = width
        else:
            self.parent_plus: list[Any] = []
            self.parent_minus: list[Any] = []
            self.parent_logits: list[Any] = []
            for width in self.hidden_layer_sizes:
                self.parent_plus.append(pnp.array(rng.normal(scale=0.25, size=(width, gp)), requires_grad=True))
                self.parent_minus.append(pnp.array(rng.normal(scale=0.25, size=(width, gp)), requires_grad=True))
                self.parent_logits.append(pnp.array(rng.normal(scale=0.05, size=(width, 2)), requires_grad=True))
                prev_dim = width

    def _compute_profile(self, raw: Any) -> Any:
        """Derive activation profile from raw parameters using the Born rule."""
        if self.mode == "standard":
            amps = normalize_real_amplitudes(raw)
            return self.num_grid_points * (amps ** 2)
        if self.mode == "mode_a":
            amps = normalize_real_amplitudes(raw)
            fp = amps ** 2
            return self.num_grid_points * (fp[0::2] - fp[1::2])
        raise ValueError("Use _compute_profile_b for mode_b")

    def _compute_profile_b(self, raw_plus: Any, raw_minus: Any, raw_logits: Any) -> Any:
        pp = normalize_real_amplitudes(raw_plus) ** 2
        pm = normalize_real_amplitudes(raw_minus) ** 2
        z = softmax_weights(raw_logits)
        return self.num_grid_points * (z[0] * pp - z[1] * pm)

    def _interp_value(self, y_grid: Any, z: Any) -> Any:
        return interp_linear_pnp(z, self.activation_grid, pnp.asarray(y_grid, dtype=float), EPS)

    def _apply_hidden_layer_edge(self, inputs: Any, layer_idx: int) -> Any:
        """Forward pass for one layer with per-edge profiles."""
        x = pnp.array(inputs, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1)
        width = self.hidden_layer_sizes[layer_idx]
        features = []
        for j in range(width):
            total = self.edge_biases[layer_idx][j]
            for i in range(x.shape[0]):
                z_ij = self.edge_scales[layer_idx][j, i] * x[i] + self.edge_shifts[layer_idx][j, i]
                z_ij = pnp.clip(z_ij, -1.0, 1.0)
                if self.mode == "mode_b":
                    prof = self._compute_profile_b(
                        self.edge_plus[layer_idx][j, i],
                        self.edge_minus[layer_idx][j, i],
                        self.edge_logits[layer_idx][j, i],
                    )
                else:
                    prof = self._compute_profile(self.edge_profiles[layer_idx][j, i])
                total = total + self._interp_value(prof, z_ij)
            features.append(total)
        return pnp.array(features)

    def _apply_hidden_layer_shared(self, inputs: Any, layer_idx: int) -> Any:
        """Forward pass for one layer with shared parent profiles + edge transforms."""
        x = pnp.array(inputs, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1)
        width = self.hidden_layer_sizes[layer_idx]
        features = []
        for j in range(width):
            if self.mode == "mode_b":
                parent_prof = self._compute_profile_b(
                    self.parent_plus[layer_idx][j],
                    self.parent_minus[layer_idx][j],
                    self.parent_logits[layer_idx][j],
                )
            else:
                parent_prof = self._compute_profile(
                    self.parent_profiles[layer_idx][j]
                )
            total = self.edge_biases[layer_idx][j]
            for i in range(x.shape[0]):
                z_ij = self.edge_a[layer_idx][j, i] * x[i] + self.edge_b[layer_idx][j, i]
                z_ij = pnp.clip(z_ij, -1.0, 1.0)
                total = total + self.edge_alphas[layer_idx][j, i] * self._interp_value(parent_prof, z_ij)
            features.append(total)
        return pnp.array(features)

    def forward_logits(self, x: Any) -> Any:
        hidden = pnp.array(x, dtype=float)
        if hidden.ndim == 0:
            hidden = hidden.reshape(1)
        apply_fn = (
            self._apply_hidden_layer_edge
            if self.activation_level == "edge"
            else self._apply_hidden_layer_shared
        )
        for layer_idx in range(self.num_hidden_layers):
            hidden = apply_fn(hidden, layer_idx)
        return pnp.dot(self.W_out, hidden) + self.output_biases

    def forward_batch(self, x_batch: Any) -> Any:
        xb = pnp.array(x_batch, dtype=float)
        return pnp.array([self.forward_logits(xi) for xi in xb])

    def predict_proba(self, x_batch: Any) -> np.ndarray:
        logits = np.asarray(self.forward_batch(x_batch), dtype=float)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def predict(self, x_batch: Any) -> np.ndarray:
        return np.argmax(self.predict_proba(x_batch), axis=1)

    def accuracy(self, x_batch: Any, y_true: np.ndarray) -> float:
        return float(np.mean(self.predict(x_batch) == np.asarray(y_true, dtype=int)))

    def parameters(self) -> list[Any]:
        params: list[Any] = []
        if self.activation_level == "edge":
            params.extend(self.edge_scales)
            params.extend(self.edge_shifts)
            params.extend(self.edge_biases)
            if self.mode == "mode_b":
                params.extend(self.edge_plus)
                params.extend(self.edge_minus)
                params.extend(self.edge_logits)
            else:
                params.extend(self.edge_profiles)
        else:
            params.extend(self.edge_alphas)
            params.extend(self.edge_a)
            params.extend(self.edge_b)
            params.extend(self.edge_biases)
            if self.mode == "mode_b":
                params.extend(self.parent_plus)
                params.extend(self.parent_minus)
                params.extend(self.parent_logits)
            else:
                params.extend(self.parent_profiles)
        params.append(self.W_out)
        params.append(self.output_biases)
        return params

    def set_parameters(self, *params: Any) -> None:
        """Load trainable tensors in the same order as ``parameters()``."""
        p = list(params)
        idx = 0
        n_layers = self.num_hidden_layers

        if self.activation_level == "edge":
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
        else:
            self.edge_alphas = p[idx : idx + n_layers]
            idx += n_layers
            self.edge_a = p[idx : idx + n_layers]
            idx += n_layers
            self.edge_b = p[idx : idx + n_layers]
            idx += n_layers
            self.edge_biases = p[idx : idx + n_layers]
            idx += n_layers
            if self.mode == "mode_b":
                self.parent_plus = p[idx : idx + n_layers]
                idx += n_layers
                self.parent_minus = p[idx : idx + n_layers]
                idx += n_layers
                self.parent_logits = p[idx : idx + n_layers]
                idx += n_layers
            else:
                self.parent_profiles = p[idx : idx + n_layers]
                idx += n_layers

        self.W_out = p[idx]
        self.output_biases = p[idx + 1]
        if idx + 2 != len(p):
            raise ValueError(f"Expected {idx + 2} parameters, got {len(p)}.")

    def get_edge_profile(self, layer_idx: int, out_idx: int, in_idx: int) -> np.ndarray:
        """Return the effective activation profile for a specific edge."""
        if self.activation_level == "edge":
            if self.mode == "mode_b":
                prof = self._compute_profile_b(
                    self.edge_plus[layer_idx][out_idx, in_idx],
                    self.edge_minus[layer_idx][out_idx, in_idx],
                    self.edge_logits[layer_idx][out_idx, in_idx],
                )
            else:
                prof = self._compute_profile(self.edge_profiles[layer_idx][out_idx, in_idx])
        else:
            if self.mode == "mode_b":
                prof = self._compute_profile_b(
                    self.parent_plus[layer_idx][out_idx],
                    self.parent_minus[layer_idx][out_idx],
                    self.parent_logits[layer_idx][out_idx],
                )
            else:
                prof = self._compute_profile(self.parent_profiles[layer_idx][out_idx])
        return np.asarray(prof, dtype=float)


def train_edge_level_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: EdgeLevelConfig,
    *,
    log_every: int | None = None,
) -> tuple[EdgeLevelClassifier, np.ndarray]:
    """Train an edge-level quantum activation classifier."""
    import pennylane as qml

    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    model = EdgeLevelClassifier(config)

    x_p = pnp.array(x)
    y_onehot = pnp.array(np.eye(config.n_classes, dtype=float)[y])
    opt = qml.AdamOptimizer(stepsize=config.learning_rate)
    params = model.parameters()
    losses: list[float] = []

    def loss_fn(*current_params: Any) -> Any:
        model.set_parameters(*current_params)
        logits = model.forward_batch(x_p)
        shifted = logits - pnp.max(logits, axis=1, keepdims=True)
        exp_s = pnp.exp(shifted)
        probs = exp_s / (pnp.sum(exp_s, axis=1, keepdims=True) + EPS)
        return -pnp.mean(pnp.sum(y_onehot * pnp.log(probs + EPS), axis=1))

    log_n = log_every if log_every is not None and log_every > 0 else 0
    for step in range(config.steps):
        params, loss_val = opt.step_and_cost(loss_fn, *params)
        loss_f = float(loss_val)
        losses.append(loss_f)
        model.set_parameters(*params)
        if log_n and ((step + 1) % log_n == 0 or step == config.steps - 1):
            print(f"  epoch {step + 1}/{config.steps}  loss={loss_f:.6f}", flush=True)

    return model, np.asarray(losses, dtype=float)
