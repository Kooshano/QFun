from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_paper_suite_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "run_mnist_paper_suite.py"
    spec = importlib.util.spec_from_file_location("run_mnist_paper_suite", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_variant_grid_skips_unsupported_edge_splines():
    paper_suite = _load_paper_suite_module()

    variants = paper_suite.build_variant_grid(
        families=["node", "edge", "shared_edge"],
        modes=["standard"],
        n_qubits_values=[2],
        hidden_units_values=[3],
        profile_interp_values=["linear", "cubic_natural"],
    )

    labels = [variant.label for variant in variants]

    assert labels == [
        "node/standard/q2/h3/linear",
        "node/standard/q2/h3/cubic_natural",
        "edge/standard/q2/h3/linear",
        "shared_edge/standard/q2/h3/linear",
    ]


def test_aggregate_rows_reports_mean_and_population_std():
    paper_suite = _load_paper_suite_module()
    rows = [
        {
            "dataset": "digits",
            "category": "quantum",
            "family": "node",
            "model_name": "node/standard/q2/h3/linear",
            "seed": 1,
            "mode": "standard",
            "n_qubits": 2,
            "hidden_units": 3,
            "profile_interp": "linear",
            "train_accuracy": 0.5,
            "test_accuracy": 0.4,
            "macro_f1": 0.3,
            "final_loss": 2.0,
            "num_params": 10,
            "use_jax": False,
        },
        {
            "dataset": "digits",
            "category": "quantum",
            "family": "node",
            "model_name": "node/standard/q2/h3/linear",
            "seed": 2,
            "mode": "standard",
            "n_qubits": 2,
            "hidden_units": 3,
            "profile_interp": "linear",
            "train_accuracy": 0.7,
            "test_accuracy": 0.8,
            "macro_f1": 0.5,
            "final_loss": 1.0,
            "num_params": 10,
            "use_jax": False,
        },
    ]

    aggregate = paper_suite.aggregate_rows(rows)

    assert len(aggregate) == 1
    assert aggregate[0]["num_runs"] == 2
    assert aggregate[0]["train_accuracy_mean"] == 0.6
    assert aggregate[0]["test_accuracy_mean"] == 0.6000000000000001
    assert aggregate[0]["macro_f1_mean"] == 0.4
    assert aggregate[0]["final_loss_mean"] == 1.5
    assert aggregate[0]["test_accuracy_std"] == 0.2
