import numpy as np
import pytest

from qfun.qfan.quantum_activation_classifier import (
    QuantumActivationClassifier,
    QuantumActivationConfig,
    train_quantum_activation_classifier,
)


def test_hidden_layers_validation_and_legacy_hidden_units_alias():
    legacy_cfg = QuantumActivationConfig(input_dim=4, hidden_units=5)
    assert legacy_cfg.resolved_hidden_layers() == (5,)

    with pytest.raises(ValueError):
        QuantumActivationConfig(input_dim=4, hidden_layers=()).resolved_hidden_layers()
    with pytest.raises(ValueError):
        QuantumActivationConfig(input_dim=4, hidden_layers=(4, 0)).resolved_hidden_layers()
    with pytest.raises(ValueError):
        QuantumActivationConfig(input_dim=4, hidden_layers=(4, -2)).resolved_hidden_layers()


def test_forward_batch_returns_three_class_probabilities(iris_split_arrays):
    x_train, _, _, _ = iris_split_arrays
    cfg = QuantumActivationConfig(input_dim=4, hidden_units=4, n_qubits=3, n_classes=3)
    model = QuantumActivationClassifier(cfg)

    probs = model.predict_proba(x_train[:5])

    assert probs.shape == (5, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)


def test_activation_profiles_and_measurements_have_expected_shapes():
    for mode in ["standard", "mode_a", "mode_b"]:
        single_cfg = QuantumActivationConfig(input_dim=4, hidden_units=4, n_qubits=3, mode=mode)
        single_model = QuantumActivationClassifier(single_cfg)

        single_profile = single_model.get_activation_profile(0)
        single_measurement = single_model.measure_activation_profile(0, shots=400)

        assert single_profile.shape == (8,)
        assert single_measurement.profile.shape == (8,)
        assert np.all(np.isfinite(single_profile))
        assert np.all(np.isfinite(single_measurement.profile))

        deep_cfg = QuantumActivationConfig(input_dim=4, hidden_layers=(3, 3), n_qubits=3, mode=mode)
        deep_model = QuantumActivationClassifier(deep_cfg)

        deep_profile = deep_model.get_activation_profile(1, 2)
        deep_measurement = deep_model.measure_activation_profile(1, 2, shots=400)

        assert deep_profile.shape == (8,)
        assert deep_measurement.profile.shape == (8,)
        assert np.all(np.isfinite(deep_profile))
        assert np.all(np.isfinite(deep_measurement.profile))


def test_multi_layer_models_require_explicit_layer_indices():
    cfg = QuantumActivationConfig(input_dim=4, hidden_layers=(3, 3), n_qubits=3)
    model = QuantumActivationClassifier(cfg)

    with pytest.raises(ValueError):
        model.get_activation_profile(0)
    with pytest.raises(ValueError):
        model.measure_activation_profile(0, shots=200)


def test_standard_classifier_training_reduces_loss_and_beats_random_guessing(iris_split_arrays):
    x_train, x_test, y_train, y_test = iris_split_arrays
    cfg = QuantumActivationConfig(
        input_dim=4,
        hidden_layers=(4, 4),
        n_qubits=3,
        n_classes=3,
        mode="standard",
        learning_rate=0.05,
        steps=18,
        seed=7,
    )

    model, losses = train_quantum_activation_classifier(x_train, y_train, cfg)

    assert losses[-1] < losses[0]
    assert model.accuracy(x_test, y_test) > 0.5


def test_signed_modes_train_with_finite_metrics(iris_split_arrays):
    x_train, x_test, y_train, y_test = iris_split_arrays

    for mode in ["mode_a", "mode_b"]:
        cfg = QuantumActivationConfig(
            input_dim=4,
            hidden_layers=(4, 4),
            n_qubits=3,
            n_classes=3,
            mode=mode,
            learning_rate=0.05,
            steps=18,
            seed=7,
        )
        model, losses = train_quantum_activation_classifier(x_train, y_train, cfg)

        assert losses[-1] < losses[0]
        assert np.isfinite(model.accuracy(x_test, y_test))


def test_after_step_records_initial_snapshot_and_every_epoch(iris_split_arrays):
    x_train, _, y_train, _ = iris_split_arrays
    cfg = QuantumActivationConfig(
        input_dim=4,
        hidden_units=3,
        n_qubits=3,
        n_classes=3,
        mode="standard",
        learning_rate=0.05,
        steps=6,
        seed=7,
    )

    seen_steps: list[int] = []
    unit0_profiles: list[np.ndarray] = []

    def after_step(step: int, loss: float, model: QuantumActivationClassifier) -> None:
        assert isinstance(loss, float)
        seen_steps.append(step)
        unit0_profiles.append(model.get_activation_profile(0))

    train_quantum_activation_classifier(
        x_train,
        y_train,
        cfg,
        after_step=after_step,
    )

    assert seen_steps[0] == -1
    assert seen_steps[-1] == 5
    assert len(seen_steps) == 7
    assert all(profile.shape == (8,) for profile in unit0_profiles)


def test_jax_training_supported_for_each_mode(iris_split_arrays):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    x_train, x_test, y_train, y_test = iris_split_arrays

    for mode in ("standard", "mode_a", "mode_b"):
        cfg = QuantumActivationConfig(
            input_dim=4,
            hidden_layers=(3, 3),
            n_qubits=3,
            n_classes=3,
            mode=mode,
            steps=2,
            seed=3,
            use_jax=True,
            batch_size=32,
        )
        model, losses = train_quantum_activation_classifier(x_train, y_train, cfg)
        assert len(losses) == 2
        assert losses[-1] < float("inf")
        assert 0.0 <= model.accuracy(x_test, y_test) <= 1.0
