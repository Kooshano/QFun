"""Canonical QFAN training utilities."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .config import QFANConfig
from .model import QFANBlock


def train_qfan(x_train: np.ndarray, y_train: np.ndarray, config: QFANConfig):
    model = QFANBlock(
        input_dim=config.input_dim,
        num_functions=config.num_functions,
        n_qubits=config.n_qubits,
        mode=config.mode,
        seed=config.seed,
    )
    opt = qml.AdamOptimizer(stepsize=config.learning_rate)
    x_p = pnp.array(x_train)
    y_p = pnp.array(y_train)

    def loss_fn(a_m, b_m, c_m, grid_values):
        model.a_m = a_m
        model.b_m = b_m
        model.c_m = c_m
        model.grid_values = grid_values
        pred = model.forward_batch(x_p)
        return pnp.mean((pred - y_p) ** 2)

    params = model.parameters()
    losses = []
    for _ in range(config.steps):
        params, loss_val = opt.step_and_cost(loss_fn, *params)
        losses.append(float(loss_val))

    model.a_m, model.b_m, model.c_m, model.grid_values = params
    return model, np.asarray(losses)
