"""Build PennyLane circuits, sample, and convert to distributions.

Supports both standard (nonneg) amplitude encoding and a signed-function
mode that uses an extra ancilla qubit to carry the sign of each grid point.

Statevector simulation uses PennyLane's NumPy interface on CPU.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pennylane as qml


# ── helpers ────────────────────────────────────────────────────────────────

def _samples_to_counts(samples: np.ndarray) -> dict[str, int]:
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    counts: dict[str, int] = {}
    for row in samples:
        key = "".join(str(int(b)) for b in row)
        counts[key] = counts.get(key, 0) + 1
    return counts


# ── standard (nonneg) API ──────────────────────────────────────────────────

def build_circuit(amplitudes: np.ndarray, n_qubits: int) -> qml.QNode:
    """Return a QNode that prepares *amplitudes* and measures in the computational basis."""
    dev = qml.device("default.qubit", wires=n_qubits, shots=1)

    @qml.qnode(dev, interface="auto")
    def circuit():
        qml.AmplitudeEmbedding(features=amplitudes, wires=range(n_qubits), normalize=False)
        return qml.sample(wires=range(n_qubits))

    return circuit


def run_shots(
    amplitudes: np.ndarray,
    n_qubits: int,
    shots: int = 1000,
) -> dict[str, int]:
    """Run *shots* measurements and return a bitstring -> count mapping."""
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface="auto")
    def circuit():
        qml.AmplitudeEmbedding(features=amplitudes, wires=range(n_qubits), normalize=False)
        return qml.sample(wires=range(n_qubits))

    samples = np.asarray(circuit())
    return _samples_to_counts(samples)


def counts_to_distribution(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """Convert a counts dict to a probability array of length 2^n, indexed by bitstring value."""
    m = 2**n_qubits
    dist = np.zeros(m)
    total = sum(counts.values())
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        dist[idx] = count / total
    return dist


# ── Mode A: signed-function encoding (ancilla carries the sign) ───────────

def _build_signed_state(
    amplitudes: np.ndarray,
    sign_mask: np.ndarray,
) -> np.ndarray:
    """Build a (2 * len(amplitudes))-dim state vector with an ancilla sign bit.

    Wire ordering: data qubits are most-significant; the ancilla (sign) is
    the least-significant bit.  Index ``2*i`` → sign=0 (positive),
    ``2*i + 1`` → sign=1 (negative).
    """
    m = len(amplitudes)
    extended = np.zeros(2 * m)
    for i, amp in enumerate(amplitudes):
        if sign_mask[i]:
            extended[2 * i + 1] = amp
        else:
            extended[2 * i] = amp
    return extended


def run_shots_signed(
    amplitudes: np.ndarray,
    sign_mask: np.ndarray,
    n_qubits: int,
    shots: int = 1000,
) -> dict[str, int]:
    """Sample from the signed-function circuit (n_qubits data + 1 ancilla).

    Each returned bitstring has ``n_qubits + 1`` characters.  The last
    character is the sign bit: ``'0'`` for positive, ``'1'`` for negative.
    """
    total_wires = n_qubits + 1
    state = _build_signed_state(amplitudes, sign_mask)
    dev = qml.device("default.qubit", wires=total_wires, shots=shots)

    @qml.qnode(dev, interface="auto")
    def circuit():
        qml.AmplitudeEmbedding(features=state, wires=range(total_wires), normalize=False)
        return qml.sample(wires=range(total_wires))

    samples = np.asarray(circuit())
    return _samples_to_counts(samples)


class SignedDistribution(NamedTuple):
    """Empirical signed distribution recovered from ancilla-based measurement."""
    p_pos: np.ndarray    # unnormalised positive-part counts (length 2^n)
    p_neg: np.ndarray    # unnormalised negative-part counts (length 2^n)
    q: np.ndarray        # signed quasi-probability  q = p_pos − p_neg


def counts_to_signed_distribution(
    counts: dict[str, int],
    n_qubits: int,
) -> SignedDistribution:
    """Split ancilla-tagged counts into positive / negative empirical distributions.

    Returns arrays of length ``2**n_qubits``.  ``q`` is the signed
    quasi-probability estimate: ``q(x) = p_pos(x) − p_neg(x)`` where each
    component is normalised by the total shot count.
    """
    m = 2**n_qubits
    p_pos = np.zeros(m)
    p_neg = np.zeros(m)
    total = sum(counts.values())
    for bitstring, count in counts.items():
        x_bits = bitstring[:-1]
        sign_bit = bitstring[-1]
        idx = int(x_bits, 2)
        if sign_bit == "0":
            p_pos[idx] += count / total
        else:
            p_neg[idx] += count / total
    return SignedDistribution(p_pos=p_pos, p_neg=p_neg, q=p_pos - p_neg)


# ── Mode B: two-channel quasi-probability estimation ──────────────────────

class TwoChannelResult(NamedTuple):
    """Output of the two-channel signed estimator."""
    q_hat: np.ndarray         # reconstructed signed distribution
    p_plus_hat: np.ndarray    # empirical p+ distribution
    p_minus_hat: np.ndarray   # empirical p− distribution
    z_plus: float
    z_minus: float


def run_two_channel_signed(
    p_plus_amps: np.ndarray,
    p_minus_amps: np.ndarray,
    z_plus: float,
    z_minus: float,
    n_qubits: int,
    shots: int = 1000,
) -> TwoChannelResult:
    r"""Run two independent QFun circuits for :math:`p_+` and :math:`p_-`.

    Reconstructs :math:`\hat q(x) = Z_+ \hat p_+(x) - Z_- \hat p_-(x)`.

    Parameters
    ----------
    p_plus_amps, p_minus_amps : array
        L2-normalised amplitude vectors for :math:`p_+` and :math:`p_-`.
    z_plus, z_minus : float
        Partition sums from :func:`decompose_signed_distribution`.
    """
    counts_plus = run_shots(p_plus_amps, n_qubits, shots=shots)
    p_plus_hat = counts_to_distribution(counts_plus, n_qubits)

    if z_minus > 0:
        counts_minus = run_shots(p_minus_amps, n_qubits, shots=shots)
        p_minus_hat = counts_to_distribution(counts_minus, n_qubits)
    else:
        p_minus_hat = np.zeros(2**n_qubits)

    q_hat = z_plus * p_plus_hat - z_minus * p_minus_hat
    return TwoChannelResult(
        q_hat=q_hat,
        p_plus_hat=p_plus_hat,
        p_minus_hat=p_minus_hat,
        z_plus=z_plus,
        z_minus=z_minus,
    )


def estimate_expectation_signed(
    g_values: np.ndarray,
    p_plus_hat: np.ndarray,
    p_minus_hat: np.ndarray,
    z_plus: float,
    z_minus: float,
) -> float:
    r"""Estimate :math:`\mathbb{E}_q[g]` from two-channel empirical distributions.

    .. math::

        \mathbb{E}_q[g] = Z_+ \sum_x g(x)\,\hat p_+(x)
                         - Z_- \sum_x g(x)\,\hat p_-(x)
    """
    return float(
        z_plus * np.dot(g_values, p_plus_hat)
        - z_minus * np.dot(g_values, p_minus_hat)
    )
