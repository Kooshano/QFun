"""QFAN encoding helpers."""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def normalize_from_domain(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map values from [lo, hi] to [-1, 1]."""
    vals = np.asarray(values, dtype=float)
    if hi <= lo:
        raise ValueError("Domain upper bound must be greater than lower bound.")
    return 2.0 * (vals - lo) / (hi - lo) - 1.0


def create_grid(n_qubits: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Uniform 1D grid with ``2**n_qubits`` points on ``[lo, hi]``."""
    return np.linspace(lo, hi, 2**n_qubits, dtype=float)


def encode_nonnegative(f_vals: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return L2-normalized sqrt(f) amplitudes for nonnegative ``f_vals``."""
    y = np.asarray(f_vals, dtype=float)
    if np.any(y < -eps):
        raise ValueError("encode_nonnegative expects nonnegative values")
    amp = np.sqrt(np.clip(y, 0.0, None) + eps)
    return amp / np.linalg.norm(amp)
