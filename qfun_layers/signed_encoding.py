"""Transitional shim for QFAN signed helpers."""

from __future__ import annotations

import numpy as np

from qfun.qfan.signed import mode_a_signed_encode, reconstruct_mode_a_signed


def mode_b_signed_decompose(f_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    f = np.asarray(f_vals, dtype=float)
    f_plus = np.clip(f, 0.0, None)
    f_minus = np.clip(-f, 0.0, None)
    z_plus = float(np.sum(f_plus))
    z_minus = float(np.sum(f_minus))
    p_plus = f_plus / z_plus if z_plus > 0 else np.zeros_like(f)
    p_minus = f_minus / z_minus if z_minus > 0 else np.zeros_like(f)
    return p_plus, p_minus, z_plus, z_minus


__all__ = ["mode_a_signed_encode", "reconstruct_mode_a_signed", "mode_b_signed_decompose"]
