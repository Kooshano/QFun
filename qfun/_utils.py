"""Internal helpers shared across qfun modules (not part of the public API)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pennylane as qml

EPS = 1e-12


def _to_numpy_float(x: Any) -> np.ndarray:
    """Real float64 ndarray from PennyLane/autograd outputs (avoids ComplexWarning on cast)."""
    arr = np.asarray(qml.math.unwrap(x))
    if np.iscomplexobj(arr):
        arr = arr.real
    return arr.astype(np.float64, copy=False)


def _samples_to_counts(samples: np.ndarray) -> dict[str, int]:
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    counts: dict[str, int] = {}
    for row in samples:
        key = "".join(str(int(b)) for b in row)
        counts[key] = counts.get(key, 0) + 1
    return counts
