"""Feynman dataset adapters for QFAN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qfun import feynman_dataset
from .encoding import normalize_from_domain


@dataclass(frozen=True)
class FeynmanBatch:
    x_raw: np.ndarray
    x_norm: np.ndarray
    y: np.ndarray


def sample_equation(eq_id: str, n_samples: int, seed: int = 0) -> FeynmanBatch:
    eq = feynman_dataset.get_equation(eq_id)
    rng = np.random.default_rng(seed)
    cols_raw = []
    cols_norm = []
    for var in eq.variables:
        lo, hi = eq.domains[var]
        values = rng.uniform(lo, hi, size=n_samples)
        cols_raw.append(values)
        cols_norm.append(normalize_from_domain(values, lo, hi))
    x_raw = np.column_stack(cols_raw)
    x_norm = np.column_stack(cols_norm)
    y = np.asarray(eq.func(*cols_raw), dtype=float)
    return FeynmanBatch(x_raw=x_raw, x_norm=x_norm, y=y)
