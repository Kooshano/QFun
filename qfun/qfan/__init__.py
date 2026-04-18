"""Canonical QFAN API under qfun."""

from __future__ import annotations

from .benchmark import run_feynman_benchmark
from .config import BenchmarkConfig, QFANConfig
from .feynman import FeynmanBatch, FeynmanQFANResult, sample_equation, train_feynman_equation
from .model import QFANBlock
from .quantum_activation_classifier import (
    QuantumActivationClassifier,
    QuantumActivationConfig,
    train_quantum_activation_classifier,
)
from .signed import (
    mode_a_signed_encode,
    mode_b_signed_decompose,
    reconstruct_mode_a_signed,
    reconstruct_mode_b_signed,
)
from .training import train_qfan

from .edge_level import EdgeLevelClassifier, EdgeLevelConfig, train_edge_level_classifier
from .grid_refinement import (
    ProgressiveGridSchedule,
    interpolate_profile,
    refine_classifier_grid,
)
from .multiplicative import MultQFunBlock, MultQFunConfig, train_multqfun
from .qi_regularization import (
    compute_qi_regularization,
    entropy_regularization,
    fidelity_diversity_penalty,
    purity_penalty,
    quantum_fidelity,
    von_neumann_entropy,
)
from .measurement_pipeline import (
    VerificationReport,
    compute_distribution_metrics,
    print_shot_budget_summary,
    print_verification_summary,
    shot_budget_analysis,
    verify_all_profiles,
    verify_profile,
)
from .interpretability import (
    classify_all_profiles,
    cluster_profiles,
    compute_importance_scores,
    evaluate_pruning,
    full_interpretability_report,
    print_cluster_summary,
    print_importance_summary,
    print_pruning_summary,
    print_taxonomy_summary,
)
from .comprehensive_benchmarks import (
    BenchmarkSuite,
    print_ablation_table,
    run_classification_benchmark_suite,
    run_feynman_benchmark_suite,
    run_multi_seed_ablation,
    run_single_classification,
    save_benchmark_results,
)
from .adaptive_growth import (
    AdaptiveGrowthConfig,
    AdaptiveGrowthResult,
    adaptive_growth_train,
    print_growth_summary,
)
from .hybrid_inference import (
    compare_inference_modes,
    print_convergence_summary,
    quantum_predict,
    quantum_predict_proba,
    shot_convergence_analysis,
)
from .theoretical_analysis import (
    approximation_error_vs_grid_size,
    born_rule_constraint_analysis,
    convergence_rate_analysis,
    parameter_count_comparison,
    print_approximation_error_table,
    print_born_rule_analysis,
    print_parameter_comparison,
)

__all__ = [
    "AdaptiveGrowthConfig",
    "AdaptiveGrowthResult",
    "BenchmarkConfig",
    "BenchmarkSuite",
    "EdgeLevelClassifier",
    "EdgeLevelConfig",
    "FeynmanBatch",
    "FeynmanQFANResult",
    "MultQFunBlock",
    "MultQFunConfig",
    "ProgressiveGridSchedule",
    "QFANBlock",
    "QFANConfig",
    "QuantumActivationClassifier",
    "QuantumActivationConfig",
    "VerificationReport",
    "adaptive_growth_train",
    "approximation_error_vs_grid_size",
    "born_rule_constraint_analysis",
    "classify_all_profiles",
    "cluster_profiles",
    "compare_inference_modes",
    "compute_distribution_metrics",
    "compute_importance_scores",
    "compute_qi_regularization",
    "convergence_rate_analysis",
    "entropy_regularization",
    "evaluate_pruning",
    "fidelity_diversity_penalty",
    "full_interpretability_report",
    "interpolate_profile",
    "mode_a_signed_encode",
    "mode_b_signed_decompose",
    "parameter_count_comparison",
    "print_ablation_table",
    "print_approximation_error_table",
    "print_born_rule_analysis",
    "print_cluster_summary",
    "print_convergence_summary",
    "print_growth_summary",
    "print_importance_summary",
    "print_parameter_comparison",
    "print_pruning_summary",
    "print_shot_budget_summary",
    "print_taxonomy_summary",
    "print_verification_summary",
    "purity_penalty",
    "quantum_fidelity",
    "quantum_predict",
    "quantum_predict_proba",
    "reconstruct_mode_a_signed",
    "reconstruct_mode_b_signed",
    "refine_classifier_grid",
    "run_classification_benchmark_suite",
    "run_feynman_benchmark",
    "run_feynman_benchmark_suite",
    "run_multi_seed_ablation",
    "run_single_classification",
    "sample_equation",
    "save_benchmark_results",
    "shot_budget_analysis",
    "shot_convergence_analysis",
    "train_edge_level_classifier",
    "train_multqfun",
    "train_quantum_activation_classifier",
    "train_feynman_equation",
    "train_qfan",
    "verify_all_profiles",
    "verify_profile",
    "von_neumann_entropy",
]
