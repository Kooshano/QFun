"""Canonical Feynman benchmark runner for QFAN."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import feynman_dataset

from .config import BenchmarkConfig, QFANConfig
from .feynman import sample_equation
from .training import train_qfan


@dataclass
class EquationMetrics:
    eq_id: str
    input_dim: int
    train_mse: float
    test_mse: float
    final_loss: float


def run_feynman_benchmark(output_dir: str | Path, config: BenchmarkConfig, model_template: QFANConfig) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    equations = feynman_dataset.list_equations()
    if config.quick_mode:
        equations = equations[: config.quick_limit]

    records: list[EquationMetrics] = []
    for i, eq in enumerate(equations):
        batch = sample_equation(eq.eq_id, n_samples=config.samples_per_equation, seed=100 + i)
        split = int((1.0 - config.test_split) * len(batch.y))
        x_train, x_test = batch.x_norm[:split], batch.x_norm[split:]
        y_train, y_test = batch.y[:split], batch.y[split:]

        cfg = QFANConfig(
            input_dim=len(eq.variables),
            num_functions=model_template.num_functions,
            n_qubits=model_template.n_qubits,
            mode=model_template.mode,
            learning_rate=model_template.learning_rate,
            steps=model_template.steps,
            seed=model_template.seed,
        )
        model, losses = train_qfan(x_train, y_train, cfg)
        pred_train = np.asarray(model.forward_batch(x_train))
        pred_test = np.asarray(model.forward_batch(x_test))
        records.append(
            EquationMetrics(
                eq_id=eq.eq_id,
                input_dim=len(eq.variables),
                train_mse=float(np.mean((pred_train - y_train) ** 2)),
                test_mse=float(np.mean((pred_test - y_test) ** 2)),
                final_loss=float(losses[-1]),
            )
        )

    summary = {
        "num_equations": len(records),
        "quick_mode": config.quick_mode,
        "avg_test_mse": float(np.mean([r.test_mse for r in records])),
        "records": [asdict(r) for r in records],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
