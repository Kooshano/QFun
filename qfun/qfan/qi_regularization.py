"""Quantum information-theoretic regularization for activation profiles.

These regularizers exploit the quantum state interpretation of the learned
amplitude profiles, providing physically motivated constraints that standard
spline-based regularization (as in classical KAN) cannot express.

Three regularizers are provided:

1. **Von Neumann entropy** -- controls information content of profiles,
   penalizing both overly peaked (memorizing) and overly uniform (uninformative)
   distributions.

2. **Purity** -- for mode_a signed profiles derived from an ancilla state,
   regularizes Tr(rho^2) of the reduced density matrix, connecting to
   quantum coherence.

3. **Fidelity-based diversity** -- penalizes high fidelity between profiles of
   different units/edges, encouraging diverse activations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pennylane.numpy as pnp

from .._utils import EPS
from ..quantum_learning import normalize_real_amplitudes, softmax_weights


def von_neumann_entropy(raw_params: Any, *, eps: float = EPS) -> Any:
    """Shannon entropy H(p) = -sum p_i log(p_i) of the Born-rule distribution.

    For a standard-mode profile p = |psi|^2, this equals the von Neumann entropy
    of the corresponding pure state's diagonal in the computational basis.

    Returns a differentiable scalar.
    """
    amps = normalize_real_amplitudes(raw_params)
    probs = amps ** 2
    probs = pnp.clip(probs, eps, None)
    return -pnp.sum(probs * pnp.log(probs))


def entropy_regularization(
    raw_params: Any,
    *,
    target_entropy: float | None = None,
    eps: float = EPS,
) -> Any:
    """Penalize deviation from a target entropy level.

    If ``target_entropy`` is None, uses the midpoint between minimum (0) and
    maximum (log(dim)) entropy as the target.
    """
    H = von_neumann_entropy(raw_params, eps=eps)
    dim = raw_params.shape[-1]
    if target_entropy is None:
        target_entropy = 0.5 * float(np.log(dim))
    return (H - target_entropy) ** 2


def purity_regularization(raw_params: Any, *, eps: float = EPS) -> Any:
    """Purity Tr(rho^2) of the diagonal density matrix.

    For a standard-mode amplitude vector, purity = sum(p_i^2).
    Values close to 1 mean a highly peaked distribution (one basis state
    dominates), values close to 1/dim mean a uniform distribution.

    Regularizing purity toward an intermediate value encourages activation
    profiles with controlled complexity.
    """
    amps = normalize_real_amplitudes(raw_params)
    probs = amps ** 2
    return pnp.sum(probs ** 2)


def purity_penalty(
    raw_params: Any,
    *,
    target_purity: float | None = None,
    eps: float = EPS,
) -> Any:
    """Penalize deviation from a target purity level.

    If ``target_purity`` is None, uses the midpoint between minimum (1/dim)
    and maximum (1) purity.
    """
    pur = purity_regularization(raw_params, eps=eps)
    if target_purity is None:
        dim = raw_params.shape[-1]
        target_purity = 0.5 * (1.0 / dim + 1.0)
    return (pur - target_purity) ** 2


def quantum_fidelity(raw_a: Any, raw_b: Any, *, eps: float = EPS) -> Any:
    """Quantum fidelity F(p_a, p_b) = (sum sqrt(p_a * p_b))^2.

    Measures the overlap between two Born-rule distributions. F=1 means
    identical profiles, F=0 means orthogonal.
    """
    amps_a = normalize_real_amplitudes(raw_a)
    amps_b = normalize_real_amplitudes(raw_b)
    probs_a = amps_a ** 2
    probs_b = amps_b ** 2
    overlap = pnp.sum(pnp.sqrt(pnp.clip(probs_a * probs_b, eps, None)))
    return overlap ** 2


def fidelity_diversity_penalty(
    raw_profiles_list: list[Any],
    *,
    eps: float = EPS,
) -> Any:
    """Penalize high fidelity between all pairs of profiles.

    Encourages diverse activation functions across units/edges by penalizing
    profiles that are too similar according to quantum fidelity.

    Parameters
    ----------
    raw_profiles_list : list of arrays
        Each element is a raw parameter vector for one profile.

    Returns a scalar penalty (mean pairwise fidelity).
    """
    n = len(raw_profiles_list)
    if n < 2:
        return pnp.array(0.0)

    total = pnp.array(0.0)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total = total + quantum_fidelity(raw_profiles_list[i], raw_profiles_list[j], eps=eps)
            count += 1
    return total / max(count, 1)


def mode_a_reduced_purity(raw_params: Any, *, eps: float = EPS) -> Any:
    """Purity of the reduced density matrix for a mode_a ancilla state.

    For a mode_a profile, the full state lives in a 2*G-dimensional space
    (G data qubits + 1 ancilla). The reduced state on the data register has
    rho_data = Tr_anc(|psi><psi|), and its purity Tr(rho_data^2) measures
    the entanglement between the data and ancilla registers.

    Low purity (close to 1/G) means the ancilla is highly entangled with the
    data register, enabling richer signed profiles.
    """
    amps = normalize_real_amplitudes(raw_params)
    probs = amps ** 2
    n = probs.shape[0] // 2
    p_even = probs[0::2]
    p_odd = probs[1::2]
    rho_diag = p_even + p_odd
    return pnp.sum(rho_diag ** 2)


def compute_qi_regularization(
    model: Any,
    *,
    entropy_weight: float = 0.0,
    purity_weight: float = 0.0,
    diversity_weight: float = 0.0,
    target_entropy: float | None = None,
    target_purity: float | None = None,
    eps: float = EPS,
) -> Any:
    """Compute combined QI regularization for a QuantumActivationClassifier.

    This is the main entry point for adding quantum information-theoretic
    regularization to the training loss.
    """
    penalty = pnp.array(0.0)

    if entropy_weight <= 0.0 and purity_weight <= 0.0 and diversity_weight <= 0.0:
        return penalty

    for layer_idx in range(model.num_hidden_layers):
        width = model.hidden_layer_sizes[layer_idx]
        profiles_for_diversity: list[Any] = []

        for unit_idx in range(width):
            if model.mode in {"standard", "mode_a"}:
                raw = model.raw_profiles_layers[layer_idx][unit_idx]
            else:
                raw = model.raw_plus_layers[layer_idx][unit_idx]

            if entropy_weight > 0:
                penalty = penalty + entropy_weight * entropy_regularization(
                    raw, target_entropy=target_entropy, eps=eps
                )

            if purity_weight > 0:
                if model.mode == "mode_a":
                    penalty = penalty + purity_weight * (
                        mode_a_reduced_purity(raw, eps=eps) - 0.5
                    ) ** 2
                else:
                    penalty = penalty + purity_weight * purity_penalty(
                        raw, target_purity=target_purity, eps=eps
                    )

            profiles_for_diversity.append(raw)

        if diversity_weight > 0 and len(profiles_for_diversity) > 1:
            penalty = penalty + diversity_weight * fidelity_diversity_penalty(
                profiles_for_diversity, eps=eps
            )

    return penalty
