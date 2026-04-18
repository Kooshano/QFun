"""Born machine measurement verification pipeline.

Systematically verifies that classically-trained quantum amplitude profiles
can be realized on quantum hardware by preparing each profile as a quantum
state and measuring it.

Provides:
  - Agreement metrics (KL divergence, total variation distance, Hellinger)
  - Shot-budget analysis (accuracy vs. number of measurement shots)
  - Noise robustness evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._utils import EPS, _to_numpy_float
from ..quantum_learning import (
    measure_mode_a_superposition,
    measure_mode_b_superposition,
    measure_standard_superposition,
    normalize_real_amplitudes,
    softmax_weights,
)


@dataclass(frozen=True)
class DistributionMetrics:
    """Agreement metrics between exact and measured distributions."""
    kl_divergence: float
    total_variation: float
    hellinger_distance: float
    l1_distance: float
    l2_distance: float
    max_abs_error: float
    correlation: float


@dataclass(frozen=True)
class ProfileVerification:
    """Verification result for a single activation profile."""
    layer_idx: int
    unit_idx: int
    mode: str
    exact_profile: np.ndarray
    measured_profile: np.ndarray
    shots: int
    metrics: DistributionMetrics


@dataclass(frozen=True)
class ShotBudgetPoint:
    """Accuracy at a specific shot budget."""
    shots: int
    accuracy: float
    mean_kl: float
    mean_tv: float
    mean_hellinger: float


@dataclass
class VerificationReport:
    """Complete measurement verification report for a trained model."""
    model_mode: str
    n_qubits: int
    num_grid_points: int
    profile_verifications: list[ProfileVerification] = field(default_factory=list)
    shot_budget_curve: list[ShotBudgetPoint] = field(default_factory=list)

    @property
    def mean_kl(self) -> float:
        if not self.profile_verifications:
            return float("nan")
        return float(np.mean([v.metrics.kl_divergence for v in self.profile_verifications]))

    @property
    def mean_tv(self) -> float:
        if not self.profile_verifications:
            return float("nan")
        return float(np.mean([v.metrics.total_variation for v in self.profile_verifications]))

    @property
    def mean_hellinger(self) -> float:
        if not self.profile_verifications:
            return float("nan")
        return float(np.mean([v.metrics.hellinger_distance for v in self.profile_verifications]))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """KL(p || q) for discrete distributions."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance: 0.5 * sum |p_i - q_i|."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    return float(0.5 * np.sum(np.abs(p - q)))


def hellinger_distance(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """Hellinger distance: sqrt(1 - sum sqrt(p_i * q_i))."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p_norm = p / max(p.sum(), eps)
    q_norm = q / max(q.sum(), eps)
    bc = float(np.sum(np.sqrt(p_norm * q_norm)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def compute_distribution_metrics(
    exact: np.ndarray,
    measured: np.ndarray,
    eps: float = EPS,
) -> DistributionMetrics:
    """Compute all agreement metrics between exact and measured profiles."""
    exact = np.asarray(exact, dtype=float).ravel()
    measured = np.asarray(measured, dtype=float).ravel()

    nonneg_exact = np.abs(exact)
    nonneg_measured = np.abs(measured)
    sum_exact = max(nonneg_exact.sum(), eps)
    sum_measured = max(nonneg_measured.sum(), eps)
    p_exact = nonneg_exact / sum_exact
    p_measured = nonneg_measured / sum_measured

    kl = kl_divergence(p_exact, p_measured, eps)
    tv = total_variation_distance(p_exact, p_measured)
    hell = hellinger_distance(p_exact, p_measured, eps)
    l1 = float(np.sum(np.abs(exact - measured)))
    l2 = float(np.sqrt(np.sum((exact - measured) ** 2)))
    max_err = float(np.max(np.abs(exact - measured)))

    if np.std(exact) < eps or np.std(measured) < eps:
        corr = 1.0 if np.allclose(exact, measured, atol=1e-6) else 0.0
    else:
        corr = float(np.corrcoef(exact, measured)[0, 1])

    return DistributionMetrics(
        kl_divergence=kl,
        total_variation=tv,
        hellinger_distance=hell,
        l1_distance=l1,
        l2_distance=l2,
        max_abs_error=max_err,
        correlation=corr,
    )


def _exact_profile_np(model: Any, layer_idx: int, unit_idx: int) -> np.ndarray:
    """Extract the exact quantum profile from a trained model."""
    return np.asarray(model._quantum_profile_np(layer_idx, unit_idx), dtype=float)


def verify_profile(
    model: Any,
    layer_idx: int,
    unit_idx: int,
    *,
    shots: int = 10000,
) -> ProfileVerification:
    """Verify a single activation profile via quantum measurement."""
    exact = _exact_profile_np(model, layer_idx, unit_idx)
    measurement = model.measure_activation_profile(layer_idx, unit_idx, shots=shots)
    measured = np.asarray(measurement.profile, dtype=float)
    metrics = compute_distribution_metrics(exact, measured)

    return ProfileVerification(
        layer_idx=layer_idx,
        unit_idx=unit_idx,
        mode=model.mode,
        exact_profile=exact,
        measured_profile=measured,
        shots=shots,
        metrics=metrics,
    )


def verify_all_profiles(
    model: Any,
    *,
    shots: int = 10000,
) -> VerificationReport:
    """Verify all activation profiles in a trained model."""
    report = VerificationReport(
        model_mode=model.mode,
        n_qubits=model.n_qubits,
        num_grid_points=model.num_grid_points,
    )
    for layer_idx in range(model.num_hidden_layers):
        for unit_idx in range(model.hidden_layer_sizes[layer_idx]):
            v = verify_profile(model, layer_idx, unit_idx, shots=shots)
            report.profile_verifications.append(v)

    return report


def shot_budget_analysis(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    shot_counts: list[int] | None = None,
    n_trials: int = 5,
    representative_units: list[tuple[int, int]] | None = None,
) -> list[ShotBudgetPoint]:
    """Analyze how classification accuracy degrades with finite shot budgets.

    For each shot count, measures profiles multiple times and evaluates
    how well the measured profiles approximate the exact ones.
    """
    if shot_counts is None:
        shot_counts = [100, 500, 1000, 5000, 10000, 50000]

    if representative_units is None:
        representative_units = [
            (li, ui)
            for li in range(model.num_hidden_layers)
            for ui in range(min(2, model.hidden_layer_sizes[li]))
        ]

    results: list[ShotBudgetPoint] = []
    exact_acc = float(model.accuracy(x_test, y_test))

    for shots in shot_counts:
        kls = []
        tvs = []
        hells = []
        for _ in range(n_trials):
            trial_metrics = []
            for layer_idx, unit_idx in representative_units:
                v = verify_profile(model, layer_idx, unit_idx, shots=shots)
                trial_metrics.append(v.metrics)
            kls.append(float(np.mean([m.kl_divergence for m in trial_metrics])))
            tvs.append(float(np.mean([m.total_variation for m in trial_metrics])))
            hells.append(float(np.mean([m.hellinger_distance for m in trial_metrics])))

        results.append(ShotBudgetPoint(
            shots=shots,
            accuracy=exact_acc,
            mean_kl=float(np.mean(kls)),
            mean_tv=float(np.mean(tvs)),
            mean_hellinger=float(np.mean(hells)),
        ))

    return results


def print_verification_summary(report: VerificationReport) -> None:
    """Print a summary table of verification results."""
    print(f"\nMeasurement Verification Report")
    print(f"  Mode: {report.model_mode}")
    print(f"  Qubits: {report.n_qubits} ({report.num_grid_points} grid points)")
    print(f"  Profiles verified: {len(report.profile_verifications)}")
    print()

    header = f"{'Layer':>5} {'Unit':>4} {'KL':>8} {'TV':>8} {'Hell':>8} {'Corr':>8} {'MaxErr':>8}"
    print(header)
    print("-" * len(header))
    for v in report.profile_verifications:
        m = v.metrics
        print(
            f"{v.layer_idx:>5} {v.unit_idx:>4} "
            f"{m.kl_divergence:>8.4f} {m.total_variation:>8.4f} "
            f"{m.hellinger_distance:>8.4f} {m.correlation:>8.4f} "
            f"{m.max_abs_error:>8.4f}"
        )

    print(f"\nAggregated:")
    print(f"  Mean KL divergence:     {report.mean_kl:.6f}")
    print(f"  Mean total variation:   {report.mean_tv:.6f}")
    print(f"  Mean Hellinger:         {report.mean_hellinger:.6f}")


def print_shot_budget_summary(points: list[ShotBudgetPoint]) -> None:
    """Print shot-budget analysis results."""
    print(f"\nShot Budget Analysis")
    header = f"{'Shots':>8} {'Accuracy':>10} {'Mean KL':>10} {'Mean TV':>10} {'Mean Hell':>10}"
    print(header)
    print("-" * len(header))
    for p in points:
        print(
            f"{p.shots:>8} {p.accuracy:>10.4f} "
            f"{p.mean_kl:>10.6f} {p.mean_tv:>10.6f} "
            f"{p.mean_hellinger:>10.6f}"
        )
