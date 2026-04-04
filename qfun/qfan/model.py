"""QFAN block implementation."""

from __future__ import annotations

import numpy as np
import pennylane.numpy as pnp

from .encoding import create_grid


class QFANBlock:
    """Quantum Function Approximation Network scalar-basis block."""

    def __init__(self, input_dim: int, num_functions: int, n_qubits: int, mode: str = "mode_a", seed: int = 42):
        if mode not in {"mode_a", "mode_b"}:
            raise ValueError("mode must be 'mode_a' or 'mode_b'.")
        self.input_dim = int(input_dim)
        self.num_functions = int(num_functions)
        self.n_qubits = int(n_qubits)
        self.mode = mode
        self.num_grid_points = 2**self.n_qubits
        self.x_grid = pnp.array(create_grid(self.n_qubits))
        rng = np.random.default_rng(seed)
        self.a_m = pnp.array(rng.normal(scale=0.25, size=(self.num_functions, self.input_dim)), requires_grad=True)
        self.b_m = pnp.array(rng.normal(scale=0.05, size=(self.num_functions,)), requires_grad=True)
        self.c_m = pnp.array(rng.normal(scale=0.5, size=(self.num_functions,)), requires_grad=True)
        base_grid = np.stack([
            np.sin(np.linspace(-np.pi, np.pi, self.num_grid_points) + i * 0.4)
            for i in range(self.num_functions)
        ])
        self.grid_values = pnp.array(base_grid, requires_grad=True)

    def parameters(self):
        return [self.a_m, self.b_m, self.c_m, self.grid_values]

    def _profile_from_grid(self, g_vals):
        if self.mode == "mode_a":
            return g_vals / (pnp.sum(pnp.abs(g_vals)) + 1e-12)
        return g_vals

    def _interp_value(self, y_grid, x):
        x = pnp.clip(x, self.x_grid[0], self.x_grid[-1])
        dx = self.x_grid[1] - self.x_grid[0]
        idx_float = (x - self.x_grid[0]) / dx
        idx0 = pnp.floor(idx_float)
        idx0 = pnp.clip(idx0, 0, self.num_grid_points - 1)
        idx1 = pnp.clip(idx0 + 1, 0, self.num_grid_points - 1)
        i0 = idx0.astype(int)
        i1 = idx1.astype(int)
        x0 = self.x_grid[i0]
        x1 = self.x_grid[i1]
        y0 = y_grid[i0]
        y1 = y_grid[i1]
        denom = pnp.where(pnp.abs(x1 - x0) < 1e-12, 1.0, x1 - x0)
        t = (x - x0) / denom
        return (1.0 - t) * y0 + t * y1

    def forward(self, x):
        x_vec = pnp.array(x, dtype=float)
        if x_vec.ndim == 0:
            x_vec = x_vec.reshape(1)
        y = 0.0
        for m in range(self.num_functions):
            z_m = pnp.dot(self.a_m[m], x_vec) + self.b_m[m]
            z_m = pnp.tanh(z_m)
            phi_grid = self._profile_from_grid(self.grid_values[m])
            phi_z = self._interp_value(phi_grid, z_m)
            y = y + self.c_m[m] * phi_z
        return y

    def forward_batch(self, x_batch):
        xb = pnp.array(x_batch, dtype=float)
        return pnp.array([self.forward(x_i) for x_i in xb])
