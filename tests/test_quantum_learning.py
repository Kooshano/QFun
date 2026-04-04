import numpy as np

from qfun.quantum_learning import (
    measure_mode_a_superposition,
    measure_mode_b_superposition,
    measure_standard_superposition,
    train_mode_a_superposition,
    train_mode_b_superposition,
    train_standard_superposition,
)


def test_train_standard_superposition_reduces_loss():
    target = np.array([0.62, 0.2, 0.1, 0.08], dtype=float)
    res = train_standard_superposition(
        target,
        n_qubits=2,
        steps=18,
        learning_rate=0.15,
        seed=0,
    )
    meas = measure_standard_superposition(res.amplitudes, 2, shots=4000)

    assert res.losses[-1] < res.losses[0]
    assert np.isclose(np.sum(res.probs), 1.0, atol=1e-6)
    assert np.isclose(np.sum(meas.probs), 1.0, atol=1e-6)


def test_train_mode_a_superposition_reduces_loss_and_preserves_signed_mass():
    target_q = np.array([0.4, -0.1, 0.2, -0.3], dtype=float)
    target_q /= np.sum(np.abs(target_q))
    res = train_mode_a_superposition(
        target_q,
        n_qubits=2,
        steps=20,
        learning_rate=0.18,
        seed=1,
    )
    meas = measure_mode_a_superposition(res.amplitudes, 2, shots=4000)

    assert res.losses[-1] < res.losses[0]
    assert np.isclose(np.sum(res.p_pos) + np.sum(res.p_neg), 1.0, atol=1e-6)
    assert np.isclose(np.sum(meas.p_pos) + np.sum(meas.p_neg), 1.0, atol=2e-2)
    assert np.any(meas.q < 0)


def test_train_mode_b_superposition_reduces_loss_and_preserves_channel_weights():
    target_q = np.array([0.2, -0.35, 0.4, -0.05], dtype=float)
    target_q /= np.sum(np.abs(target_q))
    res = train_mode_b_superposition(
        target_q,
        n_qubits=2,
        steps=20,
        learning_rate=0.18,
        seed=2,
    )
    meas = measure_mode_b_superposition(
        res.p_plus_amplitudes,
        res.p_minus_amplitudes,
        res.z_plus,
        res.z_minus,
        2,
        shots=4000,
    )

    assert res.losses[-1] < res.losses[0]
    assert np.isclose(res.z_plus + res.z_minus, 1.0, atol=1e-6)
    assert np.isclose(meas.z_plus + meas.z_minus, 1.0, atol=1e-6)
    assert np.any(meas.q_hat < 0)


def test_standard_superposition_supports_finite_shot_training():
    target = np.array([0.5, 0.2, 0.2, 0.1], dtype=float)
    res = train_standard_superposition(
        target,
        n_qubits=2,
        steps=10,
        learning_rate=0.12,
        seed=3,
        training_shots=300,
    )

    assert res.losses.shape == (10,)
    assert np.isclose(np.sum(res.probs), 1.0, atol=1e-6)


def test_standard_superposition_after_step_records_initial_and_epoch_snapshots():
    target = np.array([0.5, 0.25, 0.15, 0.1], dtype=float)
    seen_steps: list[int] = []
    profiles: list[np.ndarray] = []

    def after_step(step: int, loss: float, profile: np.ndarray) -> None:
        assert isinstance(loss, float)
        seen_steps.append(step)
        profiles.append(np.asarray(profile))

    train_standard_superposition(
        target,
        n_qubits=2,
        steps=6,
        learning_rate=0.12,
        seed=4,
        after_step=after_step,
    )

    assert seen_steps[0] == -1
    assert seen_steps[-1] == 5
    assert len(seen_steps) == 7
    assert all(profile.shape == (4,) for profile in profiles)
