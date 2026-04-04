"""Canonical QFAN training utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .config import QFANConfig
from .model import QFANBlock


def train_qfan(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: QFANConfig,
    *,
    after_step: Callable[[int, float, QFANBlock], None] | None = None,
    log_every: int | None = None,
) -> tuple[QFANBlock, np.ndarray]:
    """Fit a ``QFANBlock`` with Adam on MSE; returns ``(model, losses)``.

    If ``after_step`` is set, it is invoked as ``after_step(step, loss, model)`` where
    ``step`` is ``-1`` once before any optimizer update (random init), then ``0`` …
    ``config.steps - 1`` after each Adam step, with ``model`` synced to current params.

    Each Adam step uses the full ``x_train`` batch, so one step corresponds to one
    **epoch** over the training set. If ``log_every`` is a positive integer, print
    training MSE after every ``log_every`` epochs (and always after the last epoch).
    """
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

    if after_step is not None:
        model.a_m, model.b_m, model.c_m, model.grid_values = params
        pred0 = model.forward_batch(x_p)
        after_step(-1, float(pnp.mean((pred0 - y_p) ** 2)), model)

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
        model.a_m, model.b_m, model.c_m, model.grid_values = params
        if after_step is not None:
            after_step(step, loss_f, model)
        if log_n and (
            (step + 1) % log_n == 0 or step == config.steps - 1
        ):
            print(
                f"  epoch {step + 1}/{config.steps}  train_mse={loss_f:.6f}",
                flush=True,
            )

    return model, np.asarray(losses)
