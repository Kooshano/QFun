import numpy as np

from qfun.qfan._classification_benchmarks import (
    representative_units,
    run_default_baseline_suite,
    run_quantum_experiment,
)
from qfun.qfan.quantum_activation_classifier import (
    QuantumActivationClassifier,
    QuantumActivationConfig,
)


def test_default_baseline_suite_returns_finite_metrics_and_confusions(iris_split):
    results = run_default_baseline_suite(iris_split, seed=7, mlp_max_iter=300)

    assert set(results) == {"LogisticRegression", "MLPClassifier"}
    for result in results.values():
        assert np.isfinite(result.accuracy)
        assert np.isfinite(result.macro_f1)
        assert result.confusion_matrix.shape == (3, 3)


def test_representative_units_are_stable_for_same_seed():
    cfg = QuantumActivationConfig(input_dim=4, hidden_units=4, n_qubits=3, seed=11)
    model_a = QuantumActivationClassifier(cfg)
    model_b = QuantumActivationClassifier(cfg)

    units_a = representative_units(model_a)
    units_b = representative_units(model_b)

    assert units_a == units_b
    assert all(0 <= idx < cfg.hidden_units for idx in units_a)


def test_quantum_experiment_records_epoch_snapshots_and_every_five_curve_snapshots(iris_split):
    result = run_quantum_experiment(
        "standard",
        label="Standard test",
        split=iris_split,
        hidden_units=4,
        n_qubits=3,
        steps=6,
        learning_rate=0.05,
        seed=7,
        log_every=None,
        snapshot_interval=5,
        eval_shots=200,
    )

    assert len(result.training_snapshots) == 7
    assert result.training_snapshots[0].step == -1
    assert result.training_snapshots[-1].step == 5
    assert [snap.step for snap in result.training_curve_snapshots] == [-1, 0, 5]
    assert [snap.step for snap in result.accuracy_history] == [-1, 0, 5]
    assert result.confusion_matrix.shape == (3, 3)


def test_quantum_experiment_supports_all_modes(iris_split):
    for mode in ["standard", "mode_a", "mode_b"]:
        result = run_quantum_experiment(
            mode,
            label=f"{mode} test",
            split=iris_split,
            hidden_units=4,
            n_qubits=3,
            steps=3,
            learning_rate=0.05,
            seed=7,
            log_every=None,
            snapshot_interval=5,
            eval_shots=150,
        )

        assert np.isfinite(result.test_accuracy)
        assert np.isfinite(result.macro_f1)
        assert len(result.measurements) == len(result.representative_units)
