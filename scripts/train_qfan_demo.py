"""Canonical toy/demo QFAN training entry point."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from qfun.qfan import QFANConfig, train_qfan


def target_function(x: np.ndarray) -> np.ndarray:
    return np.sin(x) + 0.3 * x


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(128, 1))
    y = target_function(x[:, 0])
    cfg = QFANConfig(input_dim=1, steps=80)
    _, losses = train_qfan(x, y, cfg)
    print(f"QFAN demo done. initial_loss={losses[0]:.6f}, final_loss={losses[-1]:.6f}")


if __name__ == "__main__":
    main()
