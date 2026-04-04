"""Feynman dataset adapters for QFAN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import feynman_dataset

from .config import QFANConfig
from .encoding import normalize_from_domain
from .model import QFANBlock
from .training import train_qfan


@dataclass(frozen=True)
class FeynmanBatch:
    x_raw: np.ndarray
    x_norm: np.ndarray
    y: np.ndarray


def sample_equation(eq_id: str, n_samples: int, seed: int = 0) -> FeynmanBatch:
    eq = feynman_dataset.get_equation(eq_id)
    rng = np.random.default_rng(seed)
    cols_raw = []
    cols_norm = []
    for var in eq.variables:
        lo, hi = eq.domains[var]
        values = rng.uniform(lo, hi, size=n_samples)
        cols_raw.append(values)
        cols_norm.append(normalize_from_domain(values, lo, hi))
    x_raw = np.column_stack(cols_raw)
    x_norm = np.column_stack(cols_norm)
    y = np.asarray(eq.func(*cols_raw), dtype=float)
    return FeynmanBatch(x_raw=x_raw, x_norm=x_norm, y=y)


@dataclass(frozen=True)
class FeynmanQFANResult:
    """QFAN fit on one Feynman equation (inputs in normalized ``[-1,1]`` coordinates)."""

    model: QFANBlock
    eq_id: str
    formula: str
    variables: tuple[str, ...]
    domains: dict[str, tuple[float, float]]
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    losses: np.ndarray
    train_mse: float
    test_mse: float


def train_feynman_equation(
    eq_id: str,
    *,
    n_samples: int = 512,
    test_split: float = 0.2,
    data_seed: int = 0,
    num_functions: int = 3,
    n_qubits: int = 5,
    mode: str = "mode_a",
    steps: int = 150,
    lr: float = 0.05,
    model_seed: int = 42,
) -> FeynmanQFANResult:
    """Train QFAN on a single Feynman equation; same split logic as ``run_feynman_benchmark``."""
    eq = feynman_dataset.get_equation(eq_id)
    batch = sample_equation(eq_id, n_samples=n_samples, seed=data_seed)
    split = int((1.0 - test_split) * len(batch.y))
    split = max(1, min(split, len(batch.y) - 1))

    x_train, x_test = batch.x_norm[:split], batch.x_norm[split:]
    y_train, y_test = batch.y[:split], batch.y[split:]

    cfg = QFANConfig(
        input_dim=len(eq.variables),
        num_functions=num_functions,
        n_qubits=n_qubits,
        mode=mode,
        learning_rate=lr,
        steps=steps,
        seed=model_seed,
    )
    model, losses = train_qfan(x_train, y_train, cfg)
    pred_train = np.asarray(model.forward_batch(x_train))
    pred_test = np.asarray(model.forward_batch(x_test))
    train_mse = float(np.mean((pred_train - y_train) ** 2))
    test_mse = float(np.mean((pred_test - y_test) ** 2))

    return FeynmanQFANResult(
        model=model,
        eq_id=eq.eq_id,
        formula=eq.formula,
        variables=tuple(eq.variables),
        domains=dict(eq.domains),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        losses=np.asarray(losses, dtype=float),
        train_mse=train_mse,
        test_mse=test_mse,
    )
