"""QFAN encoding helpers."""

from __future__ import annotations

import numpy as np

from ..encode import grid_x


def normalize_from_domain(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map values from [lo, hi] to [-1, 1]."""
    vals = np.asarray(values, dtype=float)
    if hi <= lo:
        raise ValueError("Domain upper bound must be greater than lower bound.")
    return 2.0 * (vals - lo) / (hi - lo) - 1.0


def create_grid(n_qubits: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Uniform 1D grid with ``2**n_qubits`` points on ``[lo, hi]``."""
    return np.asarray(grid_x(lo, hi, n_qubits), dtype=float)
