"""Signed encoding modes for QFun-KAN."""

from __future__ import annotations

import numpy as np


def mode_a_signed_encode(f_vals: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Mode A: amplitudes for |f| plus ancilla sign bits (1 where negative)."""
    f = np.asarray(f_vals, dtype=float)
    magnitudes = np.abs(f)
    alpha = np.sqrt(magnitudes + eps)
    norm = np.linalg.norm(alpha)
    if norm < eps:
        raise ValueError("Cannot encode an all-zero signed profile.")
    return alpha / norm, (f < 0).astype(np.int8)


def mode_b_signed_decompose(f_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Mode B: decompose signed vector into two normalized positive channels."""
    f = np.asarray(f_vals, dtype=float)
    f_plus = np.clip(f, 0.0, None)
    f_minus = np.clip(-f, 0.0, None)
    z_plus = float(np.sum(f_plus))
    z_minus = float(np.sum(f_minus))
    if z_plus <= 0.0 and z_minus <= 0.0:
        raise ValueError("Cannot decompose an all-zero signed profile.")
    p_plus = f_plus / z_plus if z_plus > 0.0 else np.zeros_like(f)
    p_minus = f_minus / z_minus if z_minus > 0.0 else np.zeros_like(f)
    return p_plus, p_minus, z_plus, z_minus


def reconstruct_mode_a_signed(alpha: np.ndarray, sign_bits: np.ndarray) -> np.ndarray:
    """Recover the signed quasi-profile from Mode A outputs."""
    probs = np.abs(np.asarray(alpha, dtype=float)) ** 2
    signs = np.where(np.asarray(sign_bits) > 0, -1.0, 1.0)
    return signs * probs


def reconstruct_mode_b_signed(
    p_plus: np.ndarray,
    p_minus: np.ndarray,
    z_plus: float,
    z_minus: float,
) -> np.ndarray:
    """Recover signed values from Mode B decomposition."""
    return float(z_plus) * np.asarray(p_plus, dtype=float) - float(z_minus) * np.asarray(p_minus, dtype=float)
