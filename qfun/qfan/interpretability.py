"""Interpretability pipeline for learned quantum activation profiles.

Provides tools for:
  - Profile taxonomy: classify learned profiles by shape similarity to known
    activation functions (ReLU, sigmoid, Gaussian, SiLU, etc.)
  - Importance scoring: rank profiles by outgoing weight norms
  - Pruning: remove low-importance profiles and evaluate accuracy retention
  - Profile clustering: group similar activation shapes using L2 distance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._utils import EPS
from ._profile_interp import interp_profile_np


REFERENCE_ACTIVATIONS = {
    "relu": lambda x: np.maximum(0.0, x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-5.0 * x)),
    "tanh": lambda x: np.tanh(x),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "gaussian": lambda x: np.exp(-2.0 * x ** 2),
    "identity": lambda x: x,
    "quadratic": lambda x: x ** 2,
    "abs": lambda x: np.abs(x),
    "step": lambda x: np.where(x > 0, 1.0, 0.0),
    "soft_step": lambda x: 0.5 * (1.0 + np.tanh(3.0 * x)),
}


@dataclass(frozen=True)
class ProfileClassification:
    """Classification of a learned profile against reference shapes."""
    layer_idx: int
    unit_idx: int
    best_match: str
    similarity: float
    all_similarities: dict[str, float]
    is_novel: bool


@dataclass(frozen=True)
class ImportanceScore:
    """Importance score for a hidden unit."""
    layer_idx: int
    unit_idx: int
    weight_norm: float
    relative_importance: float


@dataclass(frozen=True)
class PruningResult:
    """Result of pruning low-importance units."""
    original_accuracy: float
    pruned_accuracy: float
    units_pruned: list[tuple[int, int]]
    units_retained: list[tuple[int, int]]
    accuracy_drop: float
    fraction_pruned: float


@dataclass(frozen=True)
class ProfileCluster:
    """A cluster of similar activation profiles."""
    cluster_id: int
    members: list[tuple[int, int]]
    centroid: np.ndarray
    mean_intra_distance: float


def _normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Normalize a profile to unit L2 norm for shape comparison."""
    p = np.asarray(profile, dtype=float)
    norm = np.linalg.norm(p)
    if norm < EPS:
        return p
    return p / norm


def classify_profile(
    profile: np.ndarray,
    x_grid: np.ndarray,
    *,
    novelty_threshold: float = 0.85,
) -> tuple[str, float, dict[str, float]]:
    """Classify a learned profile by similarity to reference activation shapes.

    Returns (best_match_name, best_similarity, all_similarities).
    Similarity is the absolute Pearson correlation coefficient.
    """
    profile = np.asarray(profile, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    p_norm = _normalize_profile(profile)

    similarities: dict[str, float] = {}
    for name, func in REFERENCE_ACTIVATIONS.items():
        ref = func(x_grid)
        r_norm = _normalize_profile(ref)
        if np.std(p_norm) < EPS or np.std(r_norm) < EPS:
            sim = 1.0 if np.allclose(p_norm, r_norm, atol=1e-6) else 0.0
        else:
            sim = float(np.abs(np.corrcoef(p_norm, r_norm)[0, 1]))
        similarities[name] = sim

    best_name = max(similarities, key=similarities.get)  # type: ignore[arg-type]
    best_sim = similarities[best_name]
    return best_name, best_sim, similarities


def classify_all_profiles(
    model: Any,
    *,
    novelty_threshold: float = 0.85,
) -> list[ProfileClassification]:
    """Classify all activation profiles in a trained model."""
    x_grid = np.asarray(model.activation_grid, dtype=float)
    results = []

    for layer_idx in range(model.num_hidden_layers):
        for unit_idx in range(model.hidden_layer_sizes[layer_idx]):
            profile = np.asarray(
                model.get_activation_profile(layer_idx, unit_idx), dtype=float
            )
            best_match, similarity, all_sims = classify_profile(
                profile, x_grid, novelty_threshold=novelty_threshold
            )
            results.append(ProfileClassification(
                layer_idx=layer_idx,
                unit_idx=unit_idx,
                best_match=best_match,
                similarity=similarity,
                all_similarities=all_sims,
                is_novel=similarity < novelty_threshold,
            ))

    return results


def compute_importance_scores(model: Any) -> list[ImportanceScore]:
    """Score each hidden unit by the L2 norm of its outgoing weights."""
    scores: list[ImportanceScore] = []
    for layer_idx in range(model.num_hidden_layers):
        if layer_idx < model.num_hidden_layers - 1:
            outgoing = np.asarray(model.hidden_weights[layer_idx + 1], dtype=float)
        else:
            outgoing = np.asarray(model.W_out, dtype=float)

        norms = np.linalg.norm(outgoing, axis=0)
        max_norm = float(norms.max()) if norms.max() > EPS else 1.0

        for unit_idx in range(model.hidden_layer_sizes[layer_idx]):
            weight_norm = float(norms[unit_idx])
            scores.append(ImportanceScore(
                layer_idx=layer_idx,
                unit_idx=unit_idx,
                weight_norm=weight_norm,
                relative_importance=weight_norm / max_norm,
            ))

    return scores


def rank_units_by_importance(model: Any) -> list[tuple[int, int, float]]:
    """Return (layer_idx, unit_idx, importance) sorted by importance descending."""
    scores = compute_importance_scores(model)
    ranked = sorted(scores, key=lambda s: s.weight_norm, reverse=True)
    return [(s.layer_idx, s.unit_idx, s.relative_importance) for s in ranked]


def evaluate_pruning(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    prune_fraction: float = 0.3,
) -> PruningResult:
    """Evaluate the effect of pruning low-importance units.

    Rather than physically removing units, this zeros out the outgoing weights
    of the least important units and measures accuracy change.
    """
    original_acc = float(model.accuracy(x_test, y_test))
    scores = compute_importance_scores(model)
    sorted_scores = sorted(scores, key=lambda s: s.weight_norm)

    n_total = len(sorted_scores)
    n_prune = max(1, int(n_total * prune_fraction))

    to_prune = sorted_scores[:n_prune]
    to_retain = sorted_scores[n_prune:]

    saved_weights = {}
    for s in to_prune:
        li, ui = s.layer_idx, s.unit_idx
        if li < model.num_hidden_layers - 1:
            w = model.hidden_weights[li + 1]
            key = ("hidden", li + 1)
        else:
            w = model.W_out
            key = ("output",)

        w_np = np.asarray(w, dtype=float)
        if key not in saved_weights:
            saved_weights[key] = w_np.copy()
        w_np[:, ui] = 0.0
        if key == ("output",):
            model.W_out = type(w)(w_np, requires_grad=True)
        else:
            model.hidden_weights[key[1]] = type(w)(w_np, requires_grad=True)

    pruned_acc = float(model.accuracy(x_test, y_test))

    for key, orig_w in saved_weights.items():
        if key == ("output",):
            model.W_out = type(model.W_out)(orig_w, requires_grad=True)
        else:
            li = key[1]
            model.hidden_weights[li] = type(model.hidden_weights[li])(orig_w, requires_grad=True)

    return PruningResult(
        original_accuracy=original_acc,
        pruned_accuracy=pruned_acc,
        units_pruned=[(s.layer_idx, s.unit_idx) for s in to_prune],
        units_retained=[(s.layer_idx, s.unit_idx) for s in to_retain],
        accuracy_drop=original_acc - pruned_acc,
        fraction_pruned=n_prune / n_total,
    )


def cluster_profiles(
    model: Any,
    *,
    n_clusters: int = 4,
) -> list[ProfileCluster]:
    """Cluster activation profiles using k-means on L2-normalized shapes.

    Uses a simple iterative k-means implementation to avoid hard scipy
    dependency in the clustering path.
    """
    all_profiles = []
    all_indices = []
    for layer_idx in range(model.num_hidden_layers):
        for unit_idx in range(model.hidden_layer_sizes[layer_idx]):
            prof = np.asarray(
                model.get_activation_profile(layer_idx, unit_idx), dtype=float
            )
            all_profiles.append(_normalize_profile(prof))
            all_indices.append((layer_idx, unit_idx))

    if len(all_profiles) < n_clusters:
        n_clusters = len(all_profiles)

    profiles_matrix = np.stack(all_profiles)
    n = profiles_matrix.shape[0]

    rng = np.random.default_rng(42)
    indices = rng.choice(n, size=n_clusters, replace=False)
    centroids = profiles_matrix[indices].copy()

    for _ in range(50):
        dists = np.array([
            np.linalg.norm(profiles_matrix - c, axis=1)
            for c in centroids
        ]).T
        assignments = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = profiles_matrix[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    dists = np.array([
        np.linalg.norm(profiles_matrix - c, axis=1)
        for c in centroids
    ]).T
    assignments = np.argmin(dists, axis=1)

    clusters = []
    for k in range(n_clusters):
        mask = assignments == k
        members = [all_indices[i] for i in range(n) if mask[i]]
        if not members:
            continue
        member_profiles = profiles_matrix[mask]
        intra_dists = np.linalg.norm(member_profiles - centroids[k], axis=1)
        clusters.append(ProfileCluster(
            cluster_id=k,
            members=members,
            centroid=centroids[k],
            mean_intra_distance=float(intra_dists.mean()),
        ))

    return clusters


def print_taxonomy_summary(classifications: list[ProfileClassification]) -> None:
    """Print a summary of profile classifications."""
    print("\nActivation Profile Taxonomy")
    print("-" * 60)

    type_counts: dict[str, int] = {}
    novel_count = 0
    for c in classifications:
        type_counts[c.best_match] = type_counts.get(c.best_match, 0) + 1
        if c.is_novel:
            novel_count += 1

    header = f"{'Layer':>5} {'Unit':>4} {'Best Match':>12} {'Similarity':>10} {'Novel':>6}"
    print(header)
    print("-" * len(header))
    for c in classifications:
        print(
            f"{c.layer_idx:>5} {c.unit_idx:>4} {c.best_match:>12} "
            f"{c.similarity:>10.4f} {'*' if c.is_novel else '':>6}"
        )

    print(f"\nType distribution:")
    for name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    print(f"  Novel profiles: {novel_count}/{len(classifications)}")


def print_importance_summary(scores: list[ImportanceScore]) -> None:
    """Print importance scores sorted by importance."""
    print("\nUnit Importance Scores")
    print("-" * 50)
    sorted_scores = sorted(scores, key=lambda s: s.weight_norm, reverse=True)
    header = f"{'Layer':>5} {'Unit':>4} {'Weight Norm':>12} {'Rel. Imp.':>10}"
    print(header)
    print("-" * len(header))
    for s in sorted_scores:
        print(
            f"{s.layer_idx:>5} {s.unit_idx:>4} "
            f"{s.weight_norm:>12.4f} {s.relative_importance:>10.4f}"
        )


def print_pruning_summary(result: PruningResult) -> None:
    """Print pruning evaluation results."""
    print(f"\nPruning Analysis")
    print(f"  Fraction pruned:    {result.fraction_pruned:.1%}")
    print(f"  Units pruned:       {len(result.units_pruned)}")
    print(f"  Units retained:     {len(result.units_retained)}")
    print(f"  Original accuracy:  {result.original_accuracy:.4f}")
    print(f"  Pruned accuracy:    {result.pruned_accuracy:.4f}")
    print(f"  Accuracy drop:      {result.accuracy_drop:.4f}")


def print_cluster_summary(clusters: list[ProfileCluster]) -> None:
    """Print profile clustering results."""
    print(f"\nProfile Clusters ({len(clusters)} clusters)")
    print("-" * 50)
    for c in clusters:
        members_str = ", ".join(f"L{li}U{ui}" for li, ui in c.members)
        print(f"  Cluster {c.cluster_id}: {len(c.members)} members "
              f"(mean intra-dist={c.mean_intra_distance:.4f})")
        print(f"    Members: {members_str}")


def full_interpretability_report(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_clusters: int = 4,
    prune_fraction: float = 0.3,
    novelty_threshold: float = 0.85,
) -> dict[str, Any]:
    """Run the full interpretability pipeline and print results."""
    classifications = classify_all_profiles(model, novelty_threshold=novelty_threshold)
    print_taxonomy_summary(classifications)

    scores = compute_importance_scores(model)
    print_importance_summary(scores)

    pruning = evaluate_pruning(model, x_test, y_test, prune_fraction=prune_fraction)
    print_pruning_summary(pruning)

    clusters = cluster_profiles(model, n_clusters=n_clusters)
    print_cluster_summary(clusters)

    return {
        "classifications": classifications,
        "importance_scores": scores,
        "pruning_result": pruning,
        "clusters": clusters,
    }
