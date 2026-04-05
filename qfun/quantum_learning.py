"""Quantum-state learning helpers for superposition-first notebooks.

The training utilities in this module optimize real-valued amplitude vectors
through PennyLane QNodes that explicitly prepare quantum states with
``qml.MottonenStatePreparation``. This keeps the learned object aligned with
the amplitude-encoding story used in the QFun notebooks.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from ._utils import EPS, _samples_to_counts, _to_numpy_float
from .simulate import counts_to_distribution, counts_to_signed_distribution, run_two_channel_signed


@contextlib.contextmanager
def _suppress_autograd_complex_cast_warning():
    """Autograd may cast tiny imaginary noise to float during PL parameter-shift grads."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Casting complex values to real discards the imaginary part",
        )
        yield


@dataclass(frozen=True)
class HistogramMeasurement:
    counts: dict[str, int]
    probs: np.ndarray


@dataclass(frozen=True)
class SignedMeasurement:
    counts: dict[str, int]
    p_pos: np.ndarray
    p_neg: np.ndarray
    q: np.ndarray


@dataclass(frozen=True)
class StandardTrainingResult:
    amplitudes: np.ndarray
    probs: np.ndarray
    losses: np.ndarray


@dataclass(frozen=True)
class ModeATrainingResult:
    amplitudes: np.ndarray
    full_probs: np.ndarray
    p_pos: np.ndarray
    p_neg: np.ndarray
    q: np.ndarray
    losses: np.ndarray


@dataclass(frozen=True)
class ModeBTrainingResult:
    p_plus_amplitudes: np.ndarray
    p_minus_amplitudes: np.ndarray
    p_plus: np.ndarray
    p_minus: np.ndarray
    z_plus: float
    z_minus: float
    q: np.ndarray
    losses: np.ndarray


def target_probability_from_function(
    f, x: np.ndarray, *, eps: float = EPS
) -> np.ndarray:
    """Return a normalized nonnegative target profile on ``x``."""
    y = np.asarray(f(x), dtype=float)
    if np.any(y < -eps):
        raise ValueError(
            "Standard amplitude learning expects a nonnegative target on the grid. "
            f"Min value found: {y.min():.6g}"
        )
    y = np.clip(y, 0.0, None)
    total = float(np.sum(y))
    if total < eps:
        raise ValueError("Target function is zero everywhere on the grid.")
    return y / total


def target_signed_profile_from_function(
    f, x: np.ndarray, *, eps: float = EPS
) -> np.ndarray:
    """Return the signed target profile normalized by its L1 mass."""
    y = np.asarray(f(x), dtype=float)
    scale = float(np.sum(np.abs(y)))
    if scale < eps:
        raise ValueError("Signed target function is zero everywhere on the grid.")
    return y / scale


def normalize_real_amplitudes(raw, *, eps: float = EPS):
    """Normalize a real vector into a valid amplitude vector."""
    out = raw / qml.math.sqrt(qml.math.sum(raw**2) + eps)
    return qml.math.real(out)


def softmax_weights(raw):
    """Two-way softmax used for Mode B channel weights."""
    shifted = raw - qml.math.max(raw)
    exp_shifted = qml.math.exp(shifted)
    return exp_shifted / qml.math.sum(exp_shifted)


def make_standard_prob_qnode(n_qubits: int, shots: int | None = None):
    """Return a QNode that maps a raw amplitude vector to measurement probabilities."""
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface="autograd")
    def qnode(raw_params):
        amps = normalize_real_amplitudes(raw_params)
        qml.MottonenStatePreparation(amps, wires=wires)
        return qml.probs(wires=wires)

    return qnode


def make_mode_a_prob_qnode(n_qubits: int, shots: int | None = None):
    """Return a QNode for ancilla-based signed superposition learning."""
    total_wires = n_qubits + 1
    wires = list(range(total_wires))
    dev = qml.device("default.qubit", wires=total_wires, shots=shots)

    @qml.qnode(dev, interface="autograd")
    def qnode(raw_params):
        amps = normalize_real_amplitudes(raw_params)
        qml.MottonenStatePreparation(amps, wires=wires)
        return qml.probs(wires=wires)

    return qnode


def _sample_state(amplitudes: np.ndarray, n_wires: int, shots: int) -> dict[str, int]:
    wires = list(range(n_wires))
    dev = qml.device("default.qubit", wires=n_wires, shots=shots)

    @qml.qnode(dev, interface="auto")
    def sampler():
        qml.MottonenStatePreparation(amplitudes, wires=wires)
        return qml.sample(wires=wires)

    return _samples_to_counts(np.asarray(sampler()))


def measure_standard_superposition(
    amplitudes: np.ndarray, n_qubits: int, *, shots: int
) -> HistogramMeasurement:
    """Sample a learned nonnegative superposition and return histogram data."""
    counts = _sample_state(np.asarray(amplitudes, dtype=float), n_qubits, shots)
    probs = counts_to_distribution(counts, n_qubits)
    return HistogramMeasurement(counts=counts, probs=probs)


def measure_mode_a_superposition(
    amplitudes: np.ndarray, n_qubits: int, *, shots: int
) -> SignedMeasurement:
    """Sample a learned ancilla-extended superposition and reconstruct ``q``."""
    counts = _sample_state(np.asarray(amplitudes, dtype=float), n_qubits + 1, shots)
    signed = counts_to_signed_distribution(counts, n_qubits)
    return SignedMeasurement(
        counts=counts,
        p_pos=signed.p_pos,
        p_neg=signed.p_neg,
        q=signed.q,
    )


def train_standard_superposition(
    target_prob: np.ndarray,
    n_qubits: int,
    *,
    steps: int = 150,
    learning_rate: float = 0.05,
    seed: int = 42,
    training_shots: int | None = None,
    log_every: int | None = None,
    after_step: Callable[[int, float, np.ndarray], None] | None = None,
) -> StandardTrainingResult:
    """Learn a nonnegative target profile by directly optimizing a quantum state."""
    target = np.asarray(target_prob, dtype=float)
    expected_size = 2**n_qubits
    if target.shape != (expected_size,):
        raise ValueError(f"target_prob must have shape ({expected_size},), got {target.shape}")
    if np.any(target < -EPS):
        raise ValueError("target_prob must be nonnegative.")
    total = float(np.sum(target))
    if total < EPS:
        raise ValueError("target_prob sums to zero.")
    target = target / total

    qnode_train = make_standard_prob_qnode(n_qubits, shots=training_shots)
    qnode_exact = make_standard_prob_qnode(n_qubits, shots=None)

    rng = np.random.default_rng(seed)
    raw_params = pnp.array(rng.normal(scale=0.25, size=expected_size), requires_grad=True)
    target_p = pnp.array(target)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    losses: list[float] = []

    def loss_fn(params):
        probs = qml.math.real(qnode_train(params))
        return pnp.mean((probs - target_p) ** 2)

    def current_profile(params) -> np.ndarray:
        return _to_numpy_float(qml.math.real(qnode_exact(params)))

    log_n = log_every if log_every is not None and log_every > 0 else 0
    with _suppress_autograd_complex_cast_warning():
        if after_step is not None:
            probs0 = current_profile(raw_params)
            after_step(-1, float(np.mean((probs0 - target) ** 2)), probs0)
        for step in range(steps):
            raw_params, loss_val = opt.step_and_cost(loss_fn, raw_params)
            loss_f = float(loss_val)
            losses.append(loss_f)
            if after_step is not None:
                after_step(step, loss_f, current_profile(raw_params))
            if log_n and ((step + 1) % log_n == 0 or step == steps - 1):
                print(
                    f"  epoch {step + 1}/{steps}  train_mse={loss_f:.6f}",
                    flush=True,
                )

    amplitudes = _to_numpy_float(normalize_real_amplitudes(raw_params))
    probs = _to_numpy_float(qml.math.real(qnode_exact(raw_params)))
    return StandardTrainingResult(
        amplitudes=amplitudes,
        probs=probs,
        losses=np.asarray(losses, dtype=float),
    )


def train_mode_a_superposition(
    target_q: np.ndarray,
    n_qubits: int,
    *,
    steps: int = 150,
    learning_rate: float = 0.05,
    seed: int = 42,
    training_shots: int | None = None,
    log_every: int | None = None,
    after_step: Callable[[int, float, np.ndarray], None] | None = None,
) -> ModeATrainingResult:
    """Learn a signed target profile with a data-register plus ancilla state."""
    target = np.asarray(target_q, dtype=float)
    expected_size = 2**n_qubits
    if target.shape != (expected_size,):
        raise ValueError(f"target_q must have shape ({expected_size},), got {target.shape}")
    scale = float(np.sum(np.abs(target)))
    if scale < EPS:
        raise ValueError("target_q is zero everywhere.")
    target = target / scale

    qnode_train = make_mode_a_prob_qnode(n_qubits, shots=training_shots)
    qnode_exact = make_mode_a_prob_qnode(n_qubits, shots=None)

    rng = np.random.default_rng(seed)
    raw_params = pnp.array(rng.normal(scale=0.25, size=2 * expected_size), requires_grad=True)
    target_p = pnp.array(target)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    losses: list[float] = []

    def loss_fn(params):
        full_probs = qml.math.real(qnode_train(params))
        q_hat = full_probs[0::2] - full_probs[1::2]
        return pnp.mean((q_hat - target_p) ** 2)

    def current_profile(params) -> np.ndarray:
        full_probs = _to_numpy_float(qml.math.real(qnode_exact(params)))
        return full_probs[0::2] - full_probs[1::2]

    log_n = log_every if log_every is not None and log_every > 0 else 0
    with _suppress_autograd_complex_cast_warning():
        if after_step is not None:
            q0 = current_profile(raw_params)
            after_step(-1, float(np.mean((q0 - target) ** 2)), q0)
        for step in range(steps):
            raw_params, loss_val = opt.step_and_cost(loss_fn, raw_params)
            loss_f = float(loss_val)
            losses.append(loss_f)
            if after_step is not None:
                after_step(step, loss_f, current_profile(raw_params))
            if log_n and ((step + 1) % log_n == 0 or step == steps - 1):
                print(
                    f"  epoch {step + 1}/{steps}  train_mse={loss_f:.6f}",
                    flush=True,
                )

    amplitudes = _to_numpy_float(normalize_real_amplitudes(raw_params))
    full_probs = _to_numpy_float(qml.math.real(qnode_exact(raw_params)))
    p_pos = full_probs[0::2]
    p_neg = full_probs[1::2]
    return ModeATrainingResult(
        amplitudes=amplitudes,
        full_probs=full_probs,
        p_pos=p_pos,
        p_neg=p_neg,
        q=p_pos - p_neg,
        losses=np.asarray(losses, dtype=float),
    )


def train_mode_b_superposition(
    target_q: np.ndarray,
    n_qubits: int,
    *,
    steps: int = 150,
    learning_rate: float = 0.05,
    seed: int = 42,
    training_shots: int | None = None,
    log_every: int | None = None,
    after_step: Callable[[int, float, np.ndarray], None] | None = None,
) -> ModeBTrainingResult:
    """Learn a signed target profile with two nonnegative quantum channels."""
    target = np.asarray(target_q, dtype=float)
    expected_size = 2**n_qubits
    if target.shape != (expected_size,):
        raise ValueError(f"target_q must have shape ({expected_size},), got {target.shape}")
    scale = float(np.sum(np.abs(target)))
    if scale < EPS:
        raise ValueError("target_q is zero everywhere.")
    target = target / scale

    qnode_train = make_standard_prob_qnode(n_qubits, shots=training_shots)
    qnode_exact = make_standard_prob_qnode(n_qubits, shots=None)

    rng = np.random.default_rng(seed)
    raw_plus = pnp.array(rng.normal(scale=0.25, size=expected_size), requires_grad=True)
    raw_minus = pnp.array(rng.normal(scale=0.25, size=expected_size), requires_grad=True)
    raw_z = pnp.array(rng.normal(scale=0.05, size=2), requires_grad=True)
    target_p = pnp.array(target)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    losses: list[float] = []

    def loss_fn(params_plus, params_minus, params_z):
        p_plus = qml.math.real(qnode_train(params_plus))
        p_minus = qml.math.real(qnode_train(params_minus))
        z = softmax_weights(params_z)
        q_hat = z[0] * p_plus - z[1] * p_minus
        return pnp.mean((q_hat - target_p) ** 2)

    def current_profile(params_plus, params_minus, params_z) -> np.ndarray:
        p_plus = _to_numpy_float(qml.math.real(qnode_exact(params_plus)))
        p_minus = _to_numpy_float(qml.math.real(qnode_exact(params_minus)))
        z = _to_numpy_float(softmax_weights(params_z))
        return z[0] * p_plus - z[1] * p_minus

    log_n = log_every if log_every is not None and log_every > 0 else 0
    with _suppress_autograd_complex_cast_warning():
        if after_step is not None:
            q0 = current_profile(raw_plus, raw_minus, raw_z)
            after_step(-1, float(np.mean((q0 - target) ** 2)), q0)
        for step in range(steps):
            (raw_plus, raw_minus, raw_z), loss_val = opt.step_and_cost(
                loss_fn, raw_plus, raw_minus, raw_z
            )
            loss_f = float(loss_val)
            losses.append(loss_f)
            if after_step is not None:
                after_step(step, loss_f, current_profile(raw_plus, raw_minus, raw_z))
            if log_n and ((step + 1) % log_n == 0 or step == steps - 1):
                print(
                    f"  epoch {step + 1}/{steps}  train_mse={loss_f:.6f}",
                    flush=True,
                )

    p_plus_amplitudes = _to_numpy_float(normalize_real_amplitudes(raw_plus))
    p_minus_amplitudes = _to_numpy_float(normalize_real_amplitudes(raw_minus))
    p_plus = _to_numpy_float(qml.math.real(qnode_exact(raw_plus)))
    p_minus = _to_numpy_float(qml.math.real(qnode_exact(raw_minus)))
    z = _to_numpy_float(softmax_weights(raw_z))
    return ModeBTrainingResult(
        p_plus_amplitudes=p_plus_amplitudes,
        p_minus_amplitudes=p_minus_amplitudes,
        p_plus=p_plus,
        p_minus=p_minus,
        z_plus=float(z[0]),
        z_minus=float(z[1]),
        q=z[0] * p_plus - z[1] * p_minus,
        losses=np.asarray(losses, dtype=float),
    )


def measure_mode_b_superposition(
    p_plus_amplitudes: np.ndarray,
    p_minus_amplitudes: np.ndarray,
    z_plus: float,
    z_minus: float,
    n_qubits: int,
    *,
    shots: int,
):
    """Sample the two-channel signed model with the existing QFun utilities."""
    return run_two_channel_signed(
        np.asarray(p_plus_amplitudes, dtype=float),
        np.asarray(p_minus_amplitudes, dtype=float),
        float(z_plus),
        float(z_minus),
        n_qubits,
        shots=shots,
    )


__all__ = [
    "HistogramMeasurement",
    "ModeATrainingResult",
    "ModeBTrainingResult",
    "SignedMeasurement",
    "StandardTrainingResult",
    "make_mode_a_prob_qnode",
    "make_standard_prob_qnode",
    "measure_mode_a_superposition",
    "measure_mode_b_superposition",
    "measure_standard_superposition",
    "normalize_real_amplitudes",
    "softmax_weights",
    "target_probability_from_function",
    "target_signed_profile_from_function",
    "train_mode_a_superposition",
    "train_mode_b_superposition",
    "train_standard_superposition",
]
