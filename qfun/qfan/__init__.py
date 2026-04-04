"""Canonical QFAN API under qfun."""

from .benchmark import run_feynman_benchmark
from .config import BenchmarkConfig, QFANConfig
from .feynman import FeynmanBatch, FeynmanQFANResult, sample_equation, train_feynman_equation
from .model import QFANBlock
from .signed import (
    mode_a_signed_encode,
    mode_b_signed_decompose,
    reconstruct_mode_a_signed,
    reconstruct_mode_b_signed,
)
from .training import train_qfan

# transitional alias
QKANBlock = QFANBlock

__all__ = [
    "BenchmarkConfig",
    "FeynmanBatch",
    "FeynmanQFANResult",
    "QFANBlock",
    "QFANConfig",
    "QKANBlock",
    "mode_a_signed_encode",
    "mode_b_signed_decompose",
    "reconstruct_mode_a_signed",
    "reconstruct_mode_b_signed",
    "run_feynman_benchmark",
    "sample_equation",
    "train_feynman_equation",
    "train_qfan",
]
