"""Canonical QFAN API under qfun."""

from __future__ import annotations

import warnings

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

__all__ = [
    "BenchmarkConfig",
    "FeynmanBatch",
    "FeynmanQFANResult",
    "QFANBlock",
    "QFANConfig",
    "QKANBlock",
    "QuantumActivationClassifier",
    "QuantumActivationConfig",
    "mode_a_signed_encode",
    "mode_b_signed_decompose",
    "reconstruct_mode_a_signed",
    "reconstruct_mode_b_signed",
    "run_feynman_benchmark",
    "sample_equation",
    "train_quantum_activation_classifier",
    "train_feynman_equation",
    "train_qfan",
]


def __getattr__(name: str):
    if name == "QKANBlock":
        warnings.warn(
            (
                "QKANBlock is deprecated and will be removed in a future release; "
                "use QFANBlock instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return QFANBlock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
