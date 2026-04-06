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
        single_components = single_model.get_activation_components(0)
        single_measurement = single_model.measure_activation_profile(0, shots=400)

        assert single_profile.shape == (8,)
        assert single_components.base.shape == (8,)
        assert single_components.quantum.shape == (8,)
        assert single_components.combined.shape == (8,)
        assert single_components.quantum_profile.shape == (8,)
        assert single_measurement.profile.shape == (8,)
        assert np.all(np.isfinite(single_profile))
        assert np.all(np.isfinite(single_components.base))
        assert np.all(np.isfinite(single_components.quantum))
        assert np.all(np.isfinite(single_components.combined))
        assert np.all(np.isfinite(single_components.quantum_profile))
        assert np.all(np.isfinite(single_measurement.profile))

        deep_cfg = QuantumActivationConfig(input_dim=4, hidden_layers=(3, 3), n_qubits=3, mode=mode)
        deep_model = QuantumActivationClassifier(deep_cfg)

        deep_profile = deep_model.get_activation_profile(1, 2)
        deep_components = deep_model.get_activation_components(1, 2)
        deep_measurement = deep_model.measure_activation_profile(1, 2, shots=400)

        assert deep_profile.shape == (8,)
        assert deep_components.base.shape == (8,)
        assert deep_components.quantum.shape == (8,)
        assert deep_components.combined.shape == (8,)
        assert deep_components.quantum_profile.shape == (8,)
        assert deep_measurement.profile.shape == (8,)
        assert np.all(np.isfinite(deep_profile))
        assert np.all(np.isfinite(deep_components.base))
        assert np.all(np.isfinite(deep_components.quantum))
        assert np.all(np.isfinite(deep_components.combined))
        assert np.all(np.isfinite(deep_components.quantum_profile))
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


def test_invalid_hidden_preactivation_raises():
    cfg = QuantumActivationConfig(
        input_dim=4,
        hidden_units=3,
        n_qubits=3,
        n_classes=3,
        hidden_preactivation="relu",
    )
    with pytest.raises(ValueError, match="hidden_preactivation"):
        QuantumActivationClassifier(cfg)


def test_invalid_hybrid_config_fields_raise():
    with pytest.raises(ValueError, match="hidden_function_family"):
        QuantumActivationClassifier(
            QuantumActivationConfig(
                input_dim=4,
                hidden_units=3,
                hidden_function_family="unknown",
            )
        )
    with pytest.raises(ValueError, match="hidden_base_activation"):
        QuantumActivationClassifier(
            QuantumActivationConfig(
                input_dim=4,
                hidden_units=3,
                hidden_function_family="kan_quantum_hybrid",
                hidden_base_activation="relu",
            )
        )
    with pytest.raises(ValueError, match="profile_smoothness_reg"):
        QuantumActivationClassifier(
            QuantumActivationConfig(
                input_dim=4,
                hidden_units=3,
                profile_smoothness_reg=-1e-3,
            )
        )


def test_hybrid_components_include_nonzero_branch_scales():
    cfg = QuantumActivationConfig(
        input_dim=4,
        hidden_layers=(3, 3),
        n_qubits=3,
        hidden_function_family="kan_quantum_hybrid",
        hidden_base_activation="silu",
    )
    model = QuantumActivationClassifier(cfg)

    components = model.get_activation_components(1, 1)

    assert np.isfinite(components.base_scale)
    assert np.isfinite(components.quantum_scale)
    assert components.quantum_scale != 0.0
    assert np.all(np.isfinite(components.base))
    assert np.all(np.isfinite(components.quantum))
    assert np.all(np.isfinite(components.combined))


def test_jax_tanh_preactivation_runs(iris_split_arrays):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    x_train, x_test, y_train, y_test = iris_split_arrays
    cfg = QuantumActivationConfig(
        input_dim=4,
        hidden_units=4,
        n_qubits=3,
        n_classes=3,
        mode="standard",
        steps=2,
        seed=1,
        use_jax=True,
        batch_size=32,
        hidden_preactivation="tanh",
    )
    model, losses = train_quantum_activation_classifier(x_train, y_train, cfg)
    assert len(losses) == 2
    assert model.hidden_preactivation == "tanh"
    assert 0.0 <= model.accuracy(x_test, y_test) <= 1.0


def test_hybrid_family_trains_with_finite_metrics_for_all_modes(iris_split_arrays):
    x_train, x_test, y_train, y_test = iris_split_arrays

    for mode in ("standard", "mode_a", "mode_b"):
        cfg = QuantumActivationConfig(
            input_dim=4,
            hidden_layers=(4, 4),
            n_qubits=3,
            n_classes=3,
            mode=mode,
            learning_rate=0.05,
            steps=12,
            seed=7,
            hidden_function_family="kan_quantum_hybrid",
            hidden_base_activation="silu",
            profile_smoothness_reg=1e-3,
        )

        model, losses = train_quantum_activation_classifier(x_train, y_train, cfg)

        assert len(losses) == 12
        assert np.all(np.isfinite(losses))
        assert np.isfinite(model.accuracy(x_test, y_test))


def test_jax_hybrid_training_supported_for_each_mode(iris_split_arrays):
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
            seed=5,
            use_jax=True,
            batch_size=32,
            hidden_function_family="kan_quantum_hybrid",
            hidden_base_activation="silu",
            profile_smoothness_reg=1e-3,
        )

        model, losses = train_quantum_activation_classifier(x_train, y_train, cfg)
        assert len(losses) == 2
        assert np.all(np.isfinite(losses))
        assert 0.0 <= model.accuracy(x_test, y_test) <= 1.0


def test_profile_interp_cubic_natural_matches_scipy():
    from scipy.interpolate import CubicSpline

    from qfun.qfan._profile_interp import interp_profile_np

    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 12)
    y = rng.standard_normal(12)
    z = np.linspace(-1, 1, 200)
    a = interp_profile_np(z, x, y, "cubic_natural")
    b = CubicSpline(x, y, bc_type="natural")(np.clip(z, x[0], x[-1]))
    assert np.allclose(a, b, rtol=1e-12, atol=1e-11)


def test_profile_interp_cubic_bspline_matches_scipy():
    from scipy.interpolate import BSpline

    from qfun.qfan._profile_interp import _open_uniform_knots, interp_profile_np

    rng = np.random.default_rng(1)
    x = np.linspace(-1, 1, 16)
    c = rng.standard_normal(16)
    t = _open_uniform_knots(-1.0, 1.0, 16, 3)
    z = np.linspace(-0.99, 0.99, 100)
    a = interp_profile_np(z, x, c, "cubic_bspline")
    b = BSpline(t, c, 3)(z)
    assert np.allclose(a, b, rtol=1e-12, atol=1e-11)


def test_invalid_profile_interp_raises_on_model_init():
    with pytest.raises(ValueError, match="profile_interp"):
        QuantumActivationClassifier(
            QuantumActivationConfig(input_dim=4, hidden_units=2, profile_interp="not_a_mode")  # type: ignore[arg-type]
        )


def test_eval_activation_components_matches_grid_samples(iris_split_arrays):
    x_train, _, _, _ = iris_split_arrays
    cfg = QuantumActivationConfig(
        input_dim=4,
        hidden_units=4,
        n_qubits=3,
        n_classes=3,
        profile_interp="cubic_natural",  # type: ignore[arg-type]
    )
    model = QuantumActivationClassifier(cfg)
    x_grid = np.asarray(model.activation_grid, dtype=float)
    for layer_idx in range(model.num_hidden_layers):
        for unit_idx in range(model.hidden_layer_sizes[layer_idx]):
            c_grid = model.get_activation_components(layer_idx, unit_idx)
            c_dense = model.eval_activation_components(layer_idx, unit_idx, x_coords=x_grid)
            assert np.allclose(c_grid.combined, c_dense.combined, rtol=1e-10, atol=1e-9)
            assert np.allclose(c_grid.quantum, c_dense.quantum, rtol=1e-10, atol=1e-9)


def test_predict_proba_each_profile_interp(iris_split_arrays):
    x_train, _, _, _ = iris_split_arrays
    for interp in ("linear", "cubic_natural", "cubic_bspline"):
        cfg = QuantumActivationConfig(
            input_dim=4,
            hidden_units=4,
            n_qubits=3,
            n_classes=3,
            profile_interp=interp,  # type: ignore[arg-type]
        )
        model = QuantumActivationClassifier(cfg)
        probs = model.predict_proba(x_train[:5])
        assert probs.shape == (5, 3)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)


def test_jax_training_cubic_natural_and_bspline(iris_split_arrays):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    x_train, x_test, y_train, y_test = iris_split_arrays

    for interp in ("cubic_natural", "cubic_bspline"):
        cfg = QuantumActivationConfig(
            input_dim=4,
            hidden_units=4,
            n_qubits=3,
            n_classes=3,
            steps=2,
            seed=9,
            use_jax=True,
            batch_size=32,
            profile_interp=interp,  # type: ignore[arg-type]
        )
        model, losses = train_quantum_activation_classifier(x_train, y_train, cfg)
        assert len(losses) == 2
        assert np.all(np.isfinite(losses))
        assert 0.0 <= model.accuracy(x_test, y_test) <= 1.0
