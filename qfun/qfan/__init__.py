"""Canonical QFAN API under qfun."""

from .benchmark import run_feynman_benchmark
from .config import BenchmarkConfig, QFANConfig
from .feynman import FeynmanBatch, sample_equation
from .model import QFANBlock
from .training import train_qfan

# transitional alias
QKANBlock = QFANBlock

__all__ = [
    "BenchmarkConfig",
    "FeynmanBatch",
    "QFANBlock",
    "QFANConfig",
    "QKANBlock",
    "run_feynman_benchmark",
    "sample_equation",
    "train_qfan",
]
