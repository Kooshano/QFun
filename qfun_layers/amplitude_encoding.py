"""Nonnegative amplitude encoding helpers."""

from __future__ import annotations

import numpy as np

from utils import interpolate_on_grid


def create_grid(a: float, b: float, n_qubits: int) -> np.ndarray:
    """Return ``2**n_qubits`` evenly spaced points on [a, b]."""
    return np.linspace(a, b, 2**n_qubits, dtype=float)


def encode_function(f_vals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Encode nonnegative values into normalized amplitudes."""
    y = np.asarray(f_vals, dtype=float)
    if np.any(y < -eps):
        raise ValueError("encode_function expects nonnegative values.")
    y = np.clip(y, 0.0, None)
    alpha = np.sqrt(y + eps)
    norm = np.linalg.norm(alpha)
    if norm < eps:
        raise ValueError("Cannot normalize an all-zero function.")
    return alpha / norm


def query_grid(alpha: np.ndarray, x_grid: np.ndarray, x: float) -> float:
    """Reconstruct encoded profile |alpha|^2 and query it by linear interpolation."""
    profile = np.abs(np.asarray(alpha)) ** 2
    return interpolate_on_grid(np.asarray(x_grid, dtype=float), profile, float(x))
