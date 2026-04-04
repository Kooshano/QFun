"""Shared interpolation helpers for the QFun-KAN prototype."""

from __future__ import annotations

import numpy as np


def linear_interpolation(x0: float, x1: float, y0: float, y1: float, x: float) -> float:
    """Return linearly interpolated y(x) between points (x0, y0), (x1, y1)."""
    if x1 == x0:
        return float(y0)
    t = (x - x0) / (x1 - x0)
    return float((1.0 - t) * y0 + t * y1)


def clamp_to_grid(x_grid: np.ndarray, x: float) -> float:
    """Clamp x into the grid domain [x_grid[0], x_grid[-1]]."""
    return float(np.clip(x, x_grid[0], x_grid[-1]))


def neighbor_indices(x_grid: np.ndarray, x: float) -> tuple[int, int]:
    """Return neighboring indices around x on an ordered 1D grid."""
    if x <= x_grid[0]:
        return 0, 0
    if x >= x_grid[-1]:
        last = len(x_grid) - 1
        return last, last
    hi = int(np.searchsorted(x_grid, x, side="right"))
    lo = hi - 1
    return lo, hi


def interpolate_on_grid(x_grid: np.ndarray, y_grid: np.ndarray, x: float) -> float:
    """Interpolate y_grid values at x over x_grid (both 1D arrays)."""
    x_c = clamp_to_grid(x_grid, x)
    i0, i1 = neighbor_indices(x_grid, x_c)
    if i0 == i1:
        return float(y_grid[i0])
    return linear_interpolation(x_grid[i0], x_grid[i1], y_grid[i0], y_grid[i1], x_c)
