"""QFun: quantum-inspired function encoding, simulation, and QFAN."""

from __future__ import annotations

from . import datasets, feynman_dataset, qfan
from .datasets import (
    ClassificationDataset,
    PreparedClassificationSplit,
    load_classification_dataset,
    prepare_classification_split,
)
from .encode import (
    NDGrid,
    SignedAmplitudes,
    SignedDecomposition,
    amplitudes_from_function,
    amplitudes_from_function_nd,
    decompose_signed_distribution,
    grid_nd,
    grid_x,
    signed_amplitudes_from_function,
)
from .plot import plot_comparison, plot_comparison_2d, plot_signed_comparison
from .qfan import (
    BenchmarkConfig,
    FeynmanQFANResult,
    QFANBlock,
    QFANConfig,
    QuantumActivationClassifier,
    QuantumActivationConfig,
    run_feynman_benchmark,
    train_feynman_equation,
    train_qfan,
    train_quantum_activation_classifier,
)
from .simulate import (
    SignedDistribution,
    TwoChannelResult,
    counts_to_distribution,
    counts_to_signed_distribution,
    estimate_expectation_signed,
    run_shots,
    run_shots_signed,
    run_two_channel_signed,
)

__all__ = [
    "BenchmarkConfig",
    "ClassificationDataset",
    "FeynmanQFANResult",
    "NDGrid",
    "PreparedClassificationSplit",
    "QFANBlock",
    "QFANConfig",
    "QuantumActivationClassifier",
    "QuantumActivationConfig",
    "SignedAmplitudes",
    "SignedDecomposition",
    "SignedDistribution",
    "TwoChannelResult",
    "amplitudes_from_function",
    "amplitudes_from_function_nd",
    "counts_to_distribution",
    "counts_to_signed_distribution",
    "datasets",
    "decompose_signed_distribution",
    "estimate_expectation_signed",
    "feynman_dataset",
    "grid_nd",
    "grid_x",
    "load_classification_dataset",
    "prepare_classification_split",
    "plot_comparison",
    "plot_comparison_2d",
    "plot_signed_comparison",
    "qfan",
    "run_feynman_benchmark",
    "run_shots",
    "run_shots_signed",
    "run_two_channel_signed",
    "signed_amplitudes_from_function",
    "train_feynman_equation",
    "train_qfan",
    "train_quantum_activation_classifier",
]
