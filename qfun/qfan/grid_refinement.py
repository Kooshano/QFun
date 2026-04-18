"""Progressive grid refinement via qubit scaling.

Implements the quantum analogue of KAN's grid extension: starting with a coarse
activation grid (few qubits) and progressively refining to finer grids (more
qubits) by interpolating learned profiles into the larger Hilbert space.

Adding a qubit doubles the grid resolution (2^n -> 2^(n+1)), which can be
framed as embedding a learned quantum state into a larger Hilbert space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pennylane.numpy as pnp

from .._utils import EPS
from ._profile_interp import _open_uniform_knots, interp_profile_np


def interpolate_profile(
    old_profile: np.ndarray,
    old_n_qubits: int,
    new_n_qubits: int,
) -> np.ndarray:
    """Interpolate a 1D activation profile from a coarse grid to a finer grid.

    Uses natural cubic spline interpolation (or linear if fewer than 4 points)
    to map learned grid values from ``2^old_n_qubits`` points to
    ``2^new_n_qubits`` points on [-1, 1].
    """
    if new_n_qubits <= old_n_qubits:
        raise ValueError("new_n_qubits must be greater than old_n_qubits.")

    old_grid = np.linspace(-1.0, 1.0, 2 ** old_n_qubits)
    new_grid = np.linspace(-1.0, 1.0, 2 ** new_n_qubits)
    profile = np.asarray(old_profile, dtype=np.float64)

    interp_mode = "cubic_natural" if len(old_grid) >= 4 else "linear"
    return interp_profile_np(new_grid, old_grid, profile, interp_mode, EPS)


def interpolate_profile_batch(
    profiles: np.ndarray,
    old_n_qubits: int,
    new_n_qubits: int,
) -> np.ndarray:
    """Interpolate a batch of profiles (shape ``(..., 2^old_n_qubits)``)."""
    profiles = np.asarray(profiles, dtype=np.float64)
    original_shape = profiles.shape[:-1]
    old_g = profiles.shape[-1]
    flat = profiles.reshape(-1, old_g)

    results = np.stack([
        interpolate_profile(row, old_n_qubits, new_n_qubits)
        for row in flat
    ])
    new_g = 2 ** new_n_qubits
    return results.reshape(*original_shape, new_g)


def refine_classifier_grid(model: Any, new_n_qubits: int) -> None:
    """Refine the activation grid of a QuantumActivationClassifier in-place.

    Interpolates all learned profiles from the current resolution to the new
    (finer) resolution. Updates ``n_qubits``, ``num_grid_points``,
    ``activation_grid``, and all profile tensors.

    This is the quantum analogue of KAN grid extension: each qubit addition
    doubles the Hilbert space dimension, providing finer activation resolution
    while preserving the learned function shape.
    """
    old_n_qubits = model.n_qubits
    if new_n_qubits <= old_n_qubits:
        raise ValueError(
            f"new_n_qubits ({new_n_qubits}) must exceed current "
            f"n_qubits ({old_n_qubits})."
        )

    old_g = model.num_grid_points
    new_g = 2 ** new_n_qubits

    if model.mode in {"standard", "mode_a"}:
        new_profile_layers = []
        for layer_profiles in model.raw_profiles_layers:
            old_arr = np.asarray(layer_profiles, dtype=np.float64)
            if model.mode == "mode_a":
                pw = 2 * new_g
                new_arr = interpolate_profile_batch(old_arr, old_n_qubits + 1, new_n_qubits + 1)
            else:
                pw = new_g
                new_arr = interpolate_profile_batch(old_arr, old_n_qubits, new_n_qubits)
            new_profile_layers.append(
                pnp.array(new_arr, requires_grad=True)
            )
        model.raw_profiles_layers = new_profile_layers
    else:
        model.raw_plus_layers = [
            pnp.array(
                interpolate_profile_batch(np.asarray(p), old_n_qubits, new_n_qubits),
                requires_grad=True,
            )
            for p in model.raw_plus_layers
        ]
        model.raw_minus_layers = [
            pnp.array(
                interpolate_profile_batch(np.asarray(p), old_n_qubits, new_n_qubits),
                requires_grad=True,
            )
            for p in model.raw_minus_layers
        ]

    model.n_qubits = new_n_qubits
    model.num_grid_points = new_g
    model.activation_grid = pnp.array(np.linspace(-1.0, 1.0, new_g))
    model._bspline_knots_np = _open_uniform_knots(-1.0, 1.0, new_g, 3)
    model._sync_legacy_aliases()


class ProgressiveGridSchedule:
    """Schedule for progressive grid refinement during training.

    Example usage::

        schedule = ProgressiveGridSchedule(
            stages=[(3, 40), (4, 40), (5, 40)],
        )
        # Train 40 epochs at n_qubits=3, then refine to 4 for 40 more,
        # then refine to 5 for the final 40 epochs.
    """

    def __init__(self, stages: list[tuple[int, int]]):
        """
        Parameters
        ----------
        stages : list of (n_qubits, epochs)
            Each entry specifies the grid resolution and how many epochs
            to train at that resolution.
        """
        if not stages:
            raise ValueError("stages must be non-empty.")
        for nq, ep in stages:
            if nq < 1 or ep < 1:
                raise ValueError("Each stage must have n_qubits >= 1 and epochs >= 1.")
        for i in range(1, len(stages)):
            if stages[i][0] <= stages[i - 1][0]:
                raise ValueError("n_qubits must strictly increase across stages.")
        self.stages = list(stages)

    @property
    def total_epochs(self) -> int:
        return sum(ep for _, ep in self.stages)

    @property
    def initial_n_qubits(self) -> int:
        return self.stages[0][0]

    def refinement_epochs(self) -> list[int]:
        """Return epoch numbers (0-based) at which grid refinement should occur."""
        epochs = []
        cumulative = 0
        for i, (_, ep) in enumerate(self.stages[:-1]):
            cumulative += ep
            epochs.append(cumulative)
        return epochs

    def n_qubits_at_epoch(self, epoch: int) -> int:
        """Return the n_qubits value that should be active at a given epoch."""
        cumulative = 0
        for nq, ep in self.stages:
            cumulative += ep
            if epoch < cumulative:
                return nq
        return self.stages[-1][0]
