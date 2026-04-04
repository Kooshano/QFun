"""Signed encoding modes used by QFAN."""

from __future__ import annotations

import numpy as np


def mode_a_signed_encode(f_vals: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(f_vals, dtype=float)
    alpha = np.sqrt(np.abs(f) + eps)
    norm = np.linalg.norm(alpha)
    if norm < eps:
        raise ValueError("Cannot encode an all-zero signed profile")
    return alpha / norm, (f < 0).astype(np.int8)


def reconstruct_mode_a_signed(alpha: np.ndarray, sign_bits: np.ndarray) -> np.ndarray:
    probs = np.abs(np.asarray(alpha, dtype=float)) ** 2
    signs = np.where(np.asarray(sign_bits, dtype=int) > 0, -1.0, 1.0)
    return signs * probs
