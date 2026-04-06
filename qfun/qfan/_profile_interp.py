"""Profile readout on the activation grid: linear, natural cubic, or cubic B-spline."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pennylane.numpy as pnp

from .._utils import EPS

ProfileInterpMode = Literal["linear", "cubic_natural", "cubic_bspline"]

PROFILE_INTERP_MODES: tuple[str, ...] = ("linear", "cubic_natural", "cubic_bspline")


def _assert_uniform_grid(x_grid: np.ndarray, eps: float) -> float:
    x_grid = np.asarray(x_grid, dtype=np.float64)
    if x_grid.ndim != 1 or len(x_grid) < 2:
        raise ValueError("x_grid must be 1D with length >= 2.")
    d = np.diff(x_grid)
    h = float(d[0])
    if not np.allclose(d, h, rtol=0.0, atol=eps * max(1.0, abs(h))):
        raise ValueError("cubic spline modes require a uniform activation grid.")
    return h


def _interp_linear_np(z: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, eps: float) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), x_grid[0], x_grid[-1])
    dx = float(x_grid[1] - x_grid[0])
    idx_float = (z - x_grid[0]) / dx
    idx0 = np.floor(idx_float).astype(np.int64)
    idx0 = np.clip(idx0, 0, len(x_grid) - 1)
    idx1 = np.clip(idx0 + 1, 0, len(x_grid) - 1)
    x0 = x_grid[idx0]
    x1 = x_grid[idx1]
    y0 = y_grid[idx0]
    y1 = y_grid[idx1]
    denom = np.where(np.abs(x1 - x0) < eps, 1.0, x1 - x0)
    t = (z - x0) / denom
    return (1.0 - t) * y0 + t * y1


def interp_linear_pnp(z: Any, x_grid: Any, y_grid: Any, eps: float = EPS) -> Any:
    """PennyLane/Autograd-friendly linear interpolation on a uniform grid."""
    z = pnp.clip(z, x_grid[0], x_grid[-1])
    dx = x_grid[1] - x_grid[0]
    idx_float = (z - x_grid[0]) / dx
    idx0 = pnp.floor(idx_float)
    idx0 = pnp.clip(idx0, 0, y_grid.shape[0] - 1)
    idx1 = pnp.clip(idx0 + 1, 0, y_grid.shape[0] - 1)
    i0 = idx0.astype(int)
    i1 = idx1.astype(int)
    x0 = x_grid[i0]
    x1 = x_grid[i1]
    y0 = y_grid[i0]
    y1 = y_grid[i1]
    denom = pnp.where(pnp.abs(x1 - x0) < eps, 1.0, x1 - x0)
    t = (z - x0) / denom
    return (1.0 - t) * y0 + t * y1


def _thomas_solve_np(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system; a[0] and c[-1] ignored."""
    n = len(d)
    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x = np.zeros(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _natural_cubic_M_np(y: np.ndarray, h: float) -> np.ndarray:
    """Second derivatives M[0..n-1] at knots; natural: M[0]=M[n-1]=0."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 4:
        raise ValueError("natural cubic spline requires at least 4 grid points.")
    m = n - 2
    inv_h2 = 6.0 / (h * h)
    rhs_full = inv_h2 * (y[:-2] - 2.0 * y[1:-1] + y[2:])
    a = np.ones(m, dtype=np.float64)
    b = np.full(m, 4.0, dtype=np.float64)
    c = np.ones(m, dtype=np.float64)
    d = np.zeros(m, dtype=np.float64)
    a[0] = 0.0
    d[0] = rhs_full[0]
    for i in range(1, m - 1):
        d[i] = rhs_full[i]
    c[m - 1] = 0.0
    d[m - 1] = rhs_full[m - 1]
    interior = _thomas_solve_np(a, b, c, d)
    M = np.zeros(n, dtype=np.float64)
    M[1:-1] = interior
    return M


def _cubic_natural_eval_np(
    z: np.ndarray,
    x0: float,
    h: float,
    y: np.ndarray,
    M: np.ndarray,
    x_max: float,
    eps: float,
) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), x0, x_max)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    idx_float = (z - x0) / h
    i = np.floor(idx_float).astype(np.int64)
    i = np.clip(i, 0, n - 2)
    xi = x0 + i.astype(np.float64) * h
    A = (xi + h - z) / h
    B = (z - xi) / h
    Mi = M[i]
    Mi1 = M[i + 1]
    yi = y[i]
    yi1 = y[i + 1]
    h2 = h * h
    return (
        A * yi
        + B * yi1
        + ((A**3 - A) * Mi + (B**3 - B) * Mi1) * h2 / 6.0
    )


def _open_uniform_knots(lo: float, hi: float, n_ctrl: int, degree: int = 3) -> np.ndarray:
    """Open uniform knot vector of length n_ctrl + degree + 1 on [lo, hi]."""
    p = degree
    inner = np.linspace(lo, hi, n_ctrl - p + 1, dtype=np.float64)
    inner = inner[1:-1]
    return np.concatenate([np.full(p + 1, lo, dtype=np.float64), inner, np.full(p + 1, hi, dtype=np.float64)])


def _interp_cubic_bspline_np(z: np.ndarray, x_grid: np.ndarray, coeffs: np.ndarray, eps: float) -> np.ndarray:
    try:
        from scipy.interpolate import BSpline
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "cubic_bspline profile_interp requires SciPy (install scikit-learn or scipy)."
        ) from e

    lo, hi = float(x_grid[0]), float(x_grid[-1])
    n_ctrl = int(np.asarray(coeffs).size)
    t = _open_uniform_knots(lo, hi, n_ctrl, 3)
    zc = np.clip(np.asarray(z, dtype=np.float64), lo, hi)
    spl = BSpline(t, np.asarray(coeffs, dtype=np.float64), 3, extrapolate=False)
    out = np.zeros_like(zc, dtype=np.float64)
    m = zc.size
    if m == 0:
        return out.reshape(z.shape)
    # BSpline returns nan outside support; clip handles endpoints.
    flat = zc.ravel()
    vals = spl(flat)
    nan_mask = ~np.isfinite(vals)
    if np.any(nan_mask):
        flat2 = flat.copy()
        flat2[nan_mask] = np.clip(flat[nan_mask], t[3], t[-4])
        vals = spl(flat2)
    out.ravel()[:] = vals
    return out.reshape(z.shape)


def interp_profile_np(
    z: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    mode: ProfileInterpMode,
    eps: float = EPS,
) -> np.ndarray:
    """Interpolate profile ``y_grid`` at coordinates ``z`` (any shape; output matches ``z``)."""
    z_shape = np.asarray(z).shape
    zf = np.asarray(z, dtype=np.float64).ravel()
    x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
    y_grid = np.asarray(y_grid, dtype=np.float64).ravel()
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid and y_grid must have the same shape.")

    if mode == "linear":
        out = _interp_linear_np(zf, x_grid, y_grid, eps)
    elif mode == "cubic_natural":
        h = _assert_uniform_grid(x_grid, eps)
        if len(x_grid) < 4:
            out = _interp_linear_np(zf, x_grid, y_grid, eps)
        else:
            M = _natural_cubic_M_np(y_grid, h)
            out = _cubic_natural_eval_np(zf, float(x_grid[0]), h, y_grid, M, float(x_grid[-1]), eps)
    elif mode == "cubic_bspline":
        if len(x_grid) < 4:
            out = _interp_linear_np(zf, x_grid, y_grid, eps)
        else:
            _assert_uniform_grid(x_grid, eps)
            out = _interp_cubic_bspline_np(zf, x_grid, y_grid, eps)
    else:  # pragma: no cover
        raise ValueError(f"unknown profile_interp mode: {mode!r}")

    return out.reshape(z_shape)
