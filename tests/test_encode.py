import numpy as np
import pytest

from qfun.encode import (
    amplitudes_from_function,
    decompose_signed_distribution,
    grid_nd,
    grid_x,
    signed_amplitudes_from_function,
)
from qfun.qfan.encoding import create_grid
from qfun.qfan.signed import mode_b_signed_decompose


def test_grid_x_matches_qfan_create_grid():
    grid = grid_x(-1.0, 1.0, n_qubits=3)

    assert grid.shape == (8,)
    assert np.allclose(grid, create_grid(3))


def test_grid_nd_builds_expected_shape():
    grid = grid_nd({"x": (-1.0, 1.0), "y": (0.0, 2.0)}, {"x": 2, "y": 1})

    assert grid.shape == (4, 2)
    assert grid.flat_grid.shape == (8, 2)
    assert grid.n_qubits_total == 3


def test_amplitudes_from_function_normalizes_nonnegative_values():
    x = grid_x(-1.0, 1.0, n_qubits=2)
    amplitudes = amplitudes_from_function(lambda t: t**2 + 0.5, x)

    assert amplitudes.shape == (4,)
    assert np.isclose(np.sum(amplitudes**2), 1.0, atol=1e-8)


def test_amplitudes_from_function_rejects_negative_values():
    x = grid_x(-1.0, 1.0, n_qubits=2)

    with pytest.raises(ValueError, match="nonnegative"):
        amplitudes_from_function(lambda t: t, x)


def test_signed_amplitudes_track_sign_mask_and_norm():
    x = grid_x(-1.0, 1.0, n_qubits=2)
    encoded = signed_amplitudes_from_function(lambda t: np.array([1.0, -4.0, 0.0, 9.0]), x)

    assert encoded.sign_mask.tolist() == [False, True, False, False]
    assert np.isclose(np.sum(encoded.amplitudes**2), 1.0, atol=1e-8)
    assert encoded.norm > 0.0


def test_mode_b_signed_decompose_uses_canonical_distribution_split():
    signed = np.array([0.6, -0.1, 0.3, -0.2], dtype=float)

    qfun_dec = decompose_signed_distribution(signed)
    qfan_dec = mode_b_signed_decompose(signed)

    assert np.allclose(qfan_dec[0], qfun_dec.p_plus)
    assert np.allclose(qfan_dec[1], qfun_dec.p_minus)
    assert qfan_dec[2:] == (qfun_dec.z_plus, qfun_dec.z_minus)
