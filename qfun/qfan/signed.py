"""Signed encoding modes used by QFAN."""

from __future__ import annotations

import numpy as np


def mode_a_signed_encode(f_vals: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Encode a signed discrete profile as normalized amplitudes plus per-bin sign bits (Mode A)."""
    f = np.asarray(f_vals, dtype=float)
    alpha = np.sqrt(np.abs(f) + eps)
    norm = np.linalg.norm(alpha)
    if norm < eps:
        raise ValueError("Cannot encode an all-zero signed profile")
    return alpha / norm, (f < 0).astype(np.int8)


def reconstruct_mode_a_signed(alpha: np.ndarray, sign_bits: np.ndarray) -> np.ndarray:
    """Reconstruct signed profile from Mode A amplitudes and sign bits (probabilities × sign)."""
    probs = np.abs(np.asarray(alpha, dtype=float)) ** 2
    signs = np.where(np.asarray(sign_bits, dtype=int) > 0, -1.0, 1.0)
    return signs * probs


def mode_b_signed_decompose(f_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Split a signed profile into positive/negative normalized channels.

    Returns ``(p_plus, p_minus, z_plus, z_minus)`` such that the signed profile
    can be reconstructed as ``z_plus * p_plus - z_minus * p_minus``.
    """

    f = np.asarray(f_vals, dtype=float)
    f_plus = np.clip(f, 0.0, None)
    f_minus = np.clip(-f, 0.0, None)
    z_plus = float(np.sum(f_plus))
    z_minus = float(np.sum(f_minus))
    p_plus = f_plus / z_plus if z_plus > 0.0 else np.zeros_like(f)
    p_minus = f_minus / z_minus if z_minus > 0.0 else np.zeros_like(f)
    return p_plus, p_minus, z_plus, z_minus


def reconstruct_mode_b_signed(
    p_plus: np.ndarray, p_minus: np.ndarray, z_plus: float, z_minus: float
) -> np.ndarray:
    """Reconstruct signed profile from Mode B decomposition (see `mode_b_signed_decompose`)."""
    return z_plus * np.asarray(p_plus, dtype=float) - z_minus * np.asarray(p_minus, dtype=float)


__all__ = [
    "mode_a_signed_encode",
    "mode_b_signed_decompose",
    "reconstruct_mode_a_signed",
    "reconstruct_mode_b_signed",
]
