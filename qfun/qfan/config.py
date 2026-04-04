"""Structured configuration for QFAN training and benchmarking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QFANConfig:
    input_dim: int
    num_functions: int = 3
    n_qubits: int = 5
    mode: str = "mode_a"
    learning_rate: float = 0.05
    steps: int = 150
    seed: int = 42


@dataclass(frozen=True)
class BenchmarkConfig:
    samples_per_equation: int = 512
    test_split: float = 0.2
    shots: int = 2000
    quick_mode: bool = False
    quick_limit: int = 5
