"""Theoretical analysis utilities for QFun.

Provides computational tools to support the theoretical contributions:

1. Universality of amplitude-encoded activations
2. Expressivity comparison with B-splines
3. Parameter scaling analysis
4. Approximation error bounds

These tools generate empirical evidence for the theoretical claims in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._utils import EPS
from ._profile_interp import interp_profile_np


@dataclass(frozen=True)
class ApproximationErrorResult:
    """Error of approximating a target function with G grid points."""
    grid_size: int
    n_qubits: int
    l2_error: float
    linf_error: float
    relative_error: float


@dataclass(frozen=True)
class ScalingLawPoint:
    """A single point in a parameter-scaling experiment."""
    n_params: int
    test_error: float
    n_qubits: int
    model_type: str


def approximation_error_vs_grid_size(
    target_func: Any,
    *,
    n_qubits_range: list[int] | None = None,
    n_eval: int = 1000,
    interp_mode: str = "linear",
) -> list[ApproximationErrorResult]:
    """Measure how well a target function can be represented on grids of increasing size.

    This provides empirical evidence for the universality claim: as the grid
    size G = 2^n increases, the approximation error should decrease, showing
    that amplitude-encoded profiles can approximate any continuous function.

    The "best possible" approximation on each grid is computed by simply
    evaluating the target on grid points and interpolating — this represents
    the discretization error floor.
    """
    if n_qubits_range is None:
        n_qubits_range = [2, 3, 4, 5, 6, 7, 8]

    x_eval = np.linspace(-1.0, 1.0, n_eval)
    y_target = np.asarray(target_func(x_eval), dtype=float)
    y_scale = np.max(np.abs(y_target)) + EPS

    results = []
    for nq in n_qubits_range:
        g = 2 ** nq
        x_grid = np.linspace(-1.0, 1.0, g)
        y_grid = np.asarray(target_func(x_grid), dtype=float)

        y_interp = interp_profile_np(x_eval, x_grid, y_grid, interp_mode, EPS)
        errors = y_target - y_interp
        l2 = float(np.sqrt(np.mean(errors ** 2)))
        linf = float(np.max(np.abs(errors)))
        rel = l2 / y_scale

        results.append(ApproximationErrorResult(
            grid_size=g,
            n_qubits=nq,
            l2_error=l2,
            linf_error=linf,
            relative_error=rel,
        ))

    return results


def born_rule_constraint_analysis(
    n_qubits_range: list[int] | None = None,
    *,
    n_random: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze the implicit regularization from the Born rule constraint.

    The Born rule constrains profiles to satisfy sum(p_i) = 1 and p_i >= 0
    (for standard mode). This reduces the effective parameter space and
    provides implicit regularization that B-splines lack.

    Returns statistics about the constrained manifold vs unconstrained space.
    """
    if n_qubits_range is None:
        n_qubits_range = [3, 4, 5, 6]

    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {"n_qubits": [], "data": []}

    for nq in n_qubits_range:
        g = 2 ** nq
        raw_params = rng.normal(size=(n_random, g))
        norms = np.sqrt(np.sum(raw_params ** 2, axis=1, keepdims=True) + EPS)
        amplitudes = raw_params / norms
        probs = amplitudes ** 2

        smoothness_vals = []
        entropy_vals = []
        purity_vals = []

        for p in probs:
            diffs = np.diff(p)
            smoothness_vals.append(float(np.mean(diffs ** 2)))
            p_safe = np.clip(p, EPS, None)
            entropy_vals.append(float(-np.sum(p_safe * np.log(p_safe))))
            purity_vals.append(float(np.sum(p ** 2)))

        unconstrained_profiles = rng.uniform(0, 1, size=(n_random, g))
        uc_smoothness = []
        for p in unconstrained_profiles:
            p_norm = p / (p.sum() + EPS)
            diffs = np.diff(p_norm)
            uc_smoothness.append(float(np.mean(diffs ** 2)))

        entry = {
            "n_qubits": nq,
            "grid_size": g,
            "params_per_edge": g,
            "effective_dof": g - 1,
            "born_rule_mean_smoothness": float(np.mean(smoothness_vals)),
            "born_rule_std_smoothness": float(np.std(smoothness_vals)),
            "unconstrained_mean_smoothness": float(np.mean(uc_smoothness)),
            "born_rule_mean_entropy": float(np.mean(entropy_vals)),
            "born_rule_mean_purity": float(np.mean(purity_vals)),
            "max_entropy": float(np.log(g)),
        }
        results["n_qubits"].append(nq)
        results["data"].append(entry)

    return results


def parameter_count_comparison(
    input_dim: int,
    output_dim: int,
    hidden_widths: list[int],
    n_qubits: int,
) -> dict[str, dict[str, int]]:
    """Compare parameter counts across MLP, KAN, and QFun architectures.

    For fair comparison, all architectures use the same layer structure.

    MLP: W (d_in x d_out) + bias (d_out) per layer, fixed activation
    KAN: G control points per edge + no separate weights
    QFun (node-level): W + bias + G profile params per node
    QFun (edge-level): G profile params + scale + shift per edge
    """
    g = 2 ** n_qubits

    mlp_params = 0
    kan_params = 0
    qfun_node_params = 0
    qfun_edge_params = 0

    prev_dim = input_dim
    for width in hidden_widths:
        mlp_params += prev_dim * width + width
        kan_params += prev_dim * width * g
        qfun_node_params += prev_dim * width + width + width * g
        qfun_edge_params += prev_dim * width * (g + 2) + width
        prev_dim = width

    mlp_params += prev_dim * output_dim + output_dim
    kan_params += prev_dim * output_dim * g
    qfun_node_params += prev_dim * output_dim + output_dim
    qfun_edge_params += prev_dim * output_dim + output_dim

    return {
        "MLP": {"total": mlp_params, "per_layer_type": "weights + bias"},
        "KAN": {"total": kan_params, "per_layer_type": "G control points per edge"},
        "QFun_node": {"total": qfun_node_params, "per_layer_type": "weights + bias + G per node"},
        "QFun_edge": {"total": qfun_edge_params, "per_layer_type": "G + scale + shift per edge"},
    }


def convergence_rate_analysis(
    target_func: Any,
    *,
    n_qubits_range: list[int] | None = None,
) -> dict[str, Any]:
    """Estimate the convergence rate of approximation error vs grid size.

    For smooth functions, the error should decay as O(G^{-k}) for some k
    depending on the smoothness. This fits a power law to the empirical
    errors to estimate k.
    """
    if n_qubits_range is None:
        n_qubits_range = [2, 3, 4, 5, 6, 7, 8]

    for interp_mode in ["linear", "cubic_natural"]:
        errors = approximation_error_vs_grid_size(
            target_func,
            n_qubits_range=n_qubits_range,
            interp_mode=interp_mode,
        )
        grid_sizes = np.array([e.grid_size for e in errors], dtype=float)
        l2_errors = np.array([e.l2_error for e in errors], dtype=float)

        mask = l2_errors > EPS
        if mask.sum() >= 2:
            log_g = np.log(grid_sizes[mask])
            log_e = np.log(l2_errors[mask])
            slope, intercept = np.polyfit(log_g, log_e, 1)
        else:
            slope, intercept = 0.0, 0.0

    result = {
        "convergence_results": {},
    }

    for interp_mode in ["linear", "cubic_natural"]:
        errors = approximation_error_vs_grid_size(
            target_func,
            n_qubits_range=n_qubits_range,
            interp_mode=interp_mode,
        )
        grid_sizes = np.array([e.grid_size for e in errors], dtype=float)
        l2_errors = np.array([e.l2_error for e in errors], dtype=float)

        mask = l2_errors > EPS
        if mask.sum() >= 2:
            log_g = np.log(grid_sizes[mask])
            log_e = np.log(l2_errors[mask])
            slope, intercept = np.polyfit(log_g, log_e, 1)
        else:
            slope, intercept = 0.0, 0.0

        result["convergence_results"][interp_mode] = {
            "power_law_exponent": float(-slope),
            "grid_sizes": grid_sizes.tolist(),
            "l2_errors": l2_errors.tolist(),
        }

    return result


def print_approximation_error_table(results: list[ApproximationErrorResult]) -> None:
    """Print approximation error results in a table."""
    print("\nApproximation Error vs Grid Size")
    print("-" * 55)
    header = f"{'NQ':>3} {'G':>5} {'L2 Error':>12} {'Linf Error':>12} {'Rel Error':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n_qubits:>3} {r.grid_size:>5} "
            f"{r.l2_error:>12.6f} {r.linf_error:>12.6f} "
            f"{r.relative_error:>12.6f}"
        )


def print_parameter_comparison(comparison: dict[str, dict[str, Any]]) -> None:
    """Print parameter count comparison table."""
    print("\nParameter Count Comparison")
    print("-" * 50)
    header = f"{'Model':>15} {'Total Params':>12}"
    print(header)
    print("-" * len(header))
    for name, info in comparison.items():
        print(f"{name:>15} {info['total']:>12}")


def print_born_rule_analysis(analysis: dict[str, Any]) -> None:
    """Print Born rule constraint analysis."""
    print("\nBorn Rule Constraint Analysis")
    print("-" * 70)
    header = (
        f"{'NQ':>3} {'G':>5} {'DoF':>5} "
        f"{'Born Smooth':>12} {'Unconstr.':>12} "
        f"{'Mean H':>8} {'Max H':>8}"
    )
    print(header)
    print("-" * len(header))
    for entry in analysis["data"]:
        print(
            f"{entry['n_qubits']:>3} {entry['grid_size']:>5} "
            f"{entry['effective_dof']:>5} "
            f"{entry['born_rule_mean_smoothness']:>12.6f} "
            f"{entry['unconstrained_mean_smoothness']:>12.6f} "
            f"{entry['born_rule_mean_entropy']:>8.4f} "
            f"{entry['max_entropy']:>8.4f}"
        )
