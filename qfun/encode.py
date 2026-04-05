"""Discretize a classical function and build normalized quantum amplitudes.

Supports both nonnegative functions (standard amplitude encoding) and
signed functions / quasi-probability distributions via decomposition helpers.
Includes multivariate (n-d) grid construction for multi-qubit encoding.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np


def grid_x(a: float, b: float, n_qubits: int) -> np.ndarray:
    """Return 2^n evenly-spaced sample points on [a, b]."""
    m = 2**n_qubits
    return np.linspace(a, b, m)


def amplitudes_from_function(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute L2-normalized amplitudes α_i ∝ sqrt(f(x_i)).

    ``f`` must be nonnegative on the grid; negative values raise ValueError.
    A small ``eps`` is added before the sqrt for numerical stability, then
    the result is renormalized so Σ|α_i|² = 1.
    """
    y = np.asarray(f(x), dtype=np.float64)
    if np.any(y < -eps):
        raise ValueError(
            "f(x) must be nonnegative on the grid for amplitude encoding. "
            f"Min value found: {y.min():.6g}"
        )
    y = np.clip(y, 0.0, None)
    amplitudes = np.sqrt(y + eps)
    norm = np.linalg.norm(amplitudes)
    if norm < eps:
        raise ValueError("f(x) is zero everywhere on the grid — cannot normalize.")
    return amplitudes / norm


# ---------------------------------------------------------------------------
# Mode A – signed function encoding (ancilla carries the sign)
# ---------------------------------------------------------------------------

class SignedAmplitudes(NamedTuple):
    """Result of decomposing a signed function into magnitude + sign."""
    amplitudes: np.ndarray   # L2-normalised |f(x)|^{1/2}
    sign_mask: np.ndarray    # bool array, True where f(x) < 0
    norm: float              # L2 norm before normalisation


def signed_amplitudes_from_function(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    eps: float = 1e-12,
) -> SignedAmplitudes:
    """Decompose a possibly-negative f into magnitude amplitudes and a sign mask.

    Returns a :class:`SignedAmplitudes` with:
    * ``amplitudes`` – L2-normalised sqrt(|f(x)|), suitable for amplitude
      embedding of the magnitude.
    * ``sign_mask`` – boolean array that is ``True`` wherever ``f(x) < 0``.
    * ``norm`` – the L2 norm of the raw sqrt(|f|) vector (useful for
      un-normalising later).
    """
    y = np.asarray(f(x), dtype=np.float64)
    sign_mask = y < 0
    mag = np.abs(y)
    raw = np.sqrt(mag + eps)
    norm = np.linalg.norm(raw)
    if norm < eps:
        raise ValueError("f(x) is zero everywhere on the grid — cannot normalize.")
    return SignedAmplitudes(amplitudes=raw / norm, sign_mask=sign_mask, norm=norm)


# ---------------------------------------------------------------------------
# Mode B – quasi-probability decomposition  q = q⁺ − q⁻
# ---------------------------------------------------------------------------

class SignedDecomposition(NamedTuple):
    """Positive / negative decomposition of a signed distribution."""
    p_plus: np.ndarray    # normalised positive part (proper distribution)
    p_minus: np.ndarray   # normalised negative part (proper distribution)
    z_plus: float         # sum of the positive entries of q
    z_minus: float        # sum of the absolute negative entries of q


def decompose_signed_distribution(q: np.ndarray) -> SignedDecomposition:
    r"""Decompose a signed distribution *q* into two proper distributions.

    Given :math:`q(x)` with :math:`\sum q(x)=1` (but some entries negative),
    define

    .. math::

        q^+(x) = \max(q(x),\,0),\quad q^-(x) = \max(-q(x),\,0)

    and return normalised versions :math:`p_\pm = q^\pm / Z_\pm` together
    with the partition sums :math:`Z_\pm`.  This allows reconstruction via
    :math:`q(x) = Z_+ p_+(x) - Z_- p_-(x)`.
    """
    q = np.asarray(q, dtype=np.float64)
    q_pos = np.clip(q, 0.0, None)
    q_neg = np.clip(-q, 0.0, None)
    z_plus = q_pos.sum()
    z_minus = q_neg.sum()
    if z_plus == 0:
        raise ValueError("q has no positive entries — cannot decompose.")
    p_plus = q_pos / z_plus
    p_minus = q_neg / z_minus if z_minus > 0 else np.zeros_like(q)
    return SignedDecomposition(p_plus=p_plus, p_minus=p_minus,
                              z_plus=z_plus, z_minus=z_minus)


# ---------------------------------------------------------------------------
# Multivariate (n-d) grid construction and encoding
# ---------------------------------------------------------------------------

class NDGrid(NamedTuple):
    """Result of ``grid_nd`` – everything needed to encode an n-d function."""
    flat_grid: np.ndarray       # shape (N, d) – all grid points row-major
    axes: list[np.ndarray]      # per-variable 1-D sample arrays
    var_names: list[str]        # variable names in order
    n_qubits_per_var: list[int]
    n_qubits_total: int
    shape: tuple[int, ...]      # per-variable grid sizes (2^n_k, …)


def grid_nd(
    domains: dict[str, tuple[float, float]],
    n_qubits_per_var: int | dict[str, int],
) -> NDGrid:
    """Build a multi-dimensional sample grid for multivariate encoding.

    Parameters
    ----------
    domains : dict
        ``{var_name: (lo, hi)}`` for each variable.  Iteration order is
        preserved (Python 3.7+).
    n_qubits_per_var : int or dict
        If an int, every variable gets the same number of qubits.
        If a dict, maps variable names to their qubit counts.

    Returns
    -------
    NDGrid
        Structured object with ``flat_grid`` (shape ``(N, d)``), per-variable
        axes, total qubit count, and shape tuple.
    """
    var_names = list(domains.keys())
    d = len(var_names)
    if isinstance(n_qubits_per_var, int):
        nq = [n_qubits_per_var] * d
    else:
        nq = [n_qubits_per_var[v] for v in var_names]

    axes = [np.linspace(lo, hi, 2**nq_k)
            for (lo, hi), nq_k in zip(domains.values(), nq)]
    shape = tuple(len(ax) for ax in axes)

    meshes = np.meshgrid(*axes, indexing="ij")
    flat_grid = np.column_stack([m.ravel() for m in meshes])

    return NDGrid(
        flat_grid=flat_grid,
        axes=axes,
        var_names=var_names,
        n_qubits_per_var=nq,
        n_qubits_total=sum(nq),
        shape=shape,
    )


def amplitudes_from_function_nd(
    f: Callable[..., np.ndarray],
    grid: NDGrid,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    r"""Evaluate *f* on the n-d grid and return L2-normalised amplitudes.

    ``f(*columns)`` is called with one 1-D array per variable (broadcast-ready
    columns from the flattened grid).  The resulting values are treated as
    :math:`|f|` and encoded via :math:`\alpha_i \propto \sqrt{|f(x_i)|}`.
    """
    cols = [grid.flat_grid[:, k] for k in range(grid.flat_grid.shape[1])]
    y = np.asarray(f(*cols), dtype=np.float64)
    mag = np.abs(y)
    raw = np.sqrt(mag + eps)
    norm = np.linalg.norm(raw)
    if norm < eps:
        raise ValueError("f is zero everywhere on the grid — cannot normalize.")
    return raw / norm


__all__ = [
    "NDGrid",
    "SignedAmplitudes",
    "SignedDecomposition",
    "amplitudes_from_function",
    "amplitudes_from_function_nd",
    "decompose_signed_distribution",
    "grid_nd",
    "grid_x",
    "signed_amplitudes_from_function",
]
