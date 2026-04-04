from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import qfun
from qfun.feynman_dataset import get_equation


FIG_DIR = Path(__file__).resolve().parent / "figures"


def _ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_2d_pair(grid: qfun.NDGrid, target: np.ndarray, measured: np.ndarray, *, title: str) -> plt.Figure:
    target_2d = np.asarray(target).reshape(grid.shape)
    measured_2d = np.asarray(measured).reshape(grid.shape)

    vmin = min(float(target_2d.min()), float(measured_2d.min()))
    vmax = max(float(target_2d.max()), float(measured_2d.max()))
    extent = [grid.axes[1][0], grid.axes[1][-1], grid.axes[0][0], grid.axes[0][-1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(target_2d, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_title("Target $|\\alpha|^2$")
    ax1.set_xlabel(grid.var_names[1])
    ax1.set_ylabel(grid.var_names[0])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(measured_2d, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_title("Measured (shots)")
    ax2.set_xlabel(grid.var_names[1])
    ax2.set_ylabel(grid.var_names[0])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_2d_pair_signed(grid: qfun.NDGrid, target: np.ndarray, measured: np.ndarray, *, title: str) -> plt.Figure:
    target_2d = np.asarray(target).reshape(grid.shape)
    measured_2d = np.asarray(measured).reshape(grid.shape)

    vmin = min(float(target_2d.min()), float(measured_2d.min()))
    vmax = max(float(target_2d.max()), float(measured_2d.max()))
    extent = [grid.axes[1][0], grid.axes[1][-1], grid.axes[0][0], grid.axes[0][-1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(target_2d, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax1.set_title("Target (signed)")
    ax1.set_xlabel(grid.var_names[1])
    ax1.set_ylabel(grid.var_names[0])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(measured_2d, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax2.set_title("Measured (Mode A)")
    ax2.set_xlabel(grid.var_names[1])
    ax2.set_ylabel(grid.var_names[0])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_1d_pair(x: np.ndarray, target: np.ndarray, measured: np.ndarray, *, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    bar_width = (x[-1] - x[0]) / len(x) * 0.8
    ax.bar(x, measured, width=bar_width, alpha=0.6, label="Measured (shots)")
    ax.plot(x, target, "o-", color="crimson", linewidth=2, markersize=5, label="Target $|\\alpha|^2$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    _ensure_dir()
    np.random.seed(0)

    shots = int(os.environ.get("QFUN_SHOTS", "100000"))

    # --- 1D examples (standard amplitude encoding) ---
    n_qubits_1d = int(os.environ.get("QFUN_1D_QUBITS", "6"))
    x = qfun.grid_x(-2.0, 2.0, n_qubits_1d)

    def _run_1d(name: str, f, x_grid: np.ndarray) -> None:
        amps = qfun.amplitudes_from_function(f, x_grid)
        target = amps**2
        counts = qfun.run_shots(amps, n_qubits_1d, shots=shots)
        measured = qfun.counts_to_distribution(counts, n_qubits_1d)
        fig = plot_1d_pair(x_grid, target, measured, title=f"1D amplitude encoding: {name}  ({n_qubits_1d} qubits)")
        _save(fig, f"1D_{name}.png")

    _run_1d("x2", lambda t: t**2, x)
    _run_1d("exp_minus_x2", lambda t: np.exp(-(t**2)), x)
    x_abs = qfun.grid_x(-1.0, 1.0, n_qubits_1d)
    _run_1d("abs_x", lambda t: np.abs(t), x_abs)

    # Figure 1: I.12.11
    eq = get_equation("I.12.11")
    grid = qfun.grid_nd(eq.domains, n_qubits_per_var=3)
    amps = qfun.amplitudes_from_function_nd(eq.func, grid)
    target = amps**2
    counts = qfun.run_shots(amps, grid.n_qubits_total, shots=shots)
    measured = qfun.counts_to_distribution(counts, grid.n_qubits_total)
    fig = plot_2d_pair(grid, target, measured, title=f"Feynman {eq.eq_id}: {eq.formula}")
    _save(fig, "I_12_11_target_vs_measured.png")

    # Figure 2: Signed I.50.26 (Mode A)
    eqs = get_equation("I.50.26")
    grid_s = qfun.grid_nd(eqs.domains, n_qubits_per_var=3)
    cols = [grid_s.flat_grid[:, k] for k in range(grid_s.flat_grid.shape[1])]
    y = np.asarray(eqs.func(*cols), dtype=np.float64)
    sign_mask = y < 0
    raw = np.sqrt(np.abs(y) + 1e-12)
    amps_s = raw / np.linalg.norm(raw)

    counts_s = qfun.run_shots_signed(amps_s, sign_mask, grid_s.n_qubits_total, shots=shots)
    sd = qfun.counts_to_signed_distribution(counts_s, grid_s.n_qubits_total)

    target_signed = y / np.abs(y).sum()
    fig = plot_2d_pair_signed(grid_s, target_signed, sd.q, title=f"Feynman {eqs.eq_id}: {eqs.formula}")
    _save(fig, "I_50_26_signed_target_vs_measured.png")


if __name__ == "__main__":
    main()

