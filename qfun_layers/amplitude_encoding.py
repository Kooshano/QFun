"""Transitional shim for QFAN encoding helpers."""

from __future__ import annotations

import numpy as np

from qfun.qfan.encoding import create_grid, encode_nonnegative


def encode_function(f_vals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return encode_nonnegative(f_vals, eps=eps)


def query_grid(alpha: np.ndarray, x_grid: np.ndarray, x: float) -> float:
    profile = np.abs(np.asarray(alpha, dtype=float)) ** 2
    return float(np.interp(float(x), np.asarray(x_grid, dtype=float), profile))


__all__ = ["create_grid", "encode_nonnegative", "encode_function", "query_grid"]
