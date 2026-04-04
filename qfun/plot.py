"""Plotting helpers for QFun: standard, signed, and 2-D distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .encode import NDGrid


def plot_comparison(
    x: np.ndarray,
    target_prob: np.ndarray,
    empirical_prob: np.ndarray,
    *,
    title: str = "Quantum sampling vs target distribution",
    save_path: str | None = None,
) -> None:
    """Bar chart of empirical probabilities overlaid with the exact target curve."""
    fig, ax = plt.subplots(figsize=(8, 4))

    bar_width = (x[-1] - x[0]) / len(x) * 0.8
    ax.bar(x, empirical_prob, width=bar_width, alpha=0.6, label="Measured (shots)")
    ax.plot(x, target_prob, "o-", color="crimson", linewidth=2, markersize=5, label="Target |α|²")

    ax.set_xlabel("x")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_signed_comparison(
    x: np.ndarray,
    target_q: np.ndarray,
    measured_q: np.ndarray,
    *,
    title: str = "Signed quasi-probability: target vs measured",
    save_path: str | None = None,
) -> None:
    """Bar chart where bars can go negative, overlaid with the target signed curve."""
    fig, ax = plt.subplots(figsize=(9, 4))

    bar_width = (x[-1] - x[0]) / len(x) * 0.8
    colors = np.where(measured_q >= 0, "steelblue", "coral")
    ax.bar(x, measured_q, width=bar_width, color=colors, alpha=0.6,
           label="Measured (signed)")
    ax.plot(x, target_q, "o-", color="crimson", linewidth=2, markersize=5,
            label="Target q(x)")
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("Quasi-probability")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 2-D heatmap comparison (for multivariate functions with 2 variables)
# ---------------------------------------------------------------------------

def plot_comparison_2d(
    grid: "NDGrid",
    target_prob: np.ndarray,
    empirical_prob: np.ndarray,
    *,
    title: str = "Target vs Measured (2-D)",
    var_labels: tuple[str, str] | None = None,
    save_path: str | None = None,
) -> None:
    """Side-by-side heatmaps comparing target and measured 2-D distributions.

    Parameters
    ----------
    grid : NDGrid
        The n-d grid (must have exactly 2 variables).
    target_prob, empirical_prob : array-like
        Flat probability vectors of length ``prod(grid.shape)``.
    var_labels : tuple of two str, optional
        Axis labels; defaults to the grid's ``var_names``.
    """
    if len(grid.shape) != 2:
        raise ValueError(
            f"plot_comparison_2d requires a 2-variable grid, got {len(grid.shape)} variables."
        )

    labels = var_labels or (grid.var_names[0], grid.var_names[1])
    target_2d = np.asarray(target_prob).reshape(grid.shape)
    empirical_2d = np.asarray(empirical_prob).reshape(grid.shape)

    vmin = min(target_2d.min(), empirical_2d.min())
    vmax = max(target_2d.max(), empirical_2d.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    extent = [
        grid.axes[1][0], grid.axes[1][-1],
        grid.axes[0][0], grid.axes[0][-1],
    ]

    im1 = ax1.imshow(
        target_2d, origin="lower", aspect="auto", extent=extent,
        vmin=vmin, vmax=vmax, cmap="viridis",
    )
    ax1.set_title("Target |α|²")
    ax1.set_xlabel(labels[1])
    ax1.set_ylabel(labels[0])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(
        empirical_2d, origin="lower", aspect="auto", extent=extent,
        vmin=vmin, vmax=vmax, cmap="viridis",
    )
    ax2.set_title("Measured (shots)")
    ax2.set_xlabel(labels[1])
    ax2.set_ylabel(labels[0])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
