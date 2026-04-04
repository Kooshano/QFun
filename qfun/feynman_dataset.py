"""Feynman physics equations benchmark dataset.

Contains 27 equations from the Feynman Lectures on Physics, as curated in
the KAN paper (arXiv:2404.19756v5, Table 4 / Appendix D) and originally
from the AI Feynman benchmark (arXiv:1905.11481).

Each equation is stored in its *dimensionless* form (fewer variables, no
physical constants) so it can be directly used with QFun's amplitude encoding.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np


class FeynmanEquation(NamedTuple):
    eq_id: str
    name: str
    formula: str
    variables: list[str]
    domains: dict[str, tuple[float, float]]
    func: Callable[..., np.ndarray]


def _eq_I_6_2(theta, sigma):
    return np.exp(-theta**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


def _eq_I_6_2b(theta, theta1, sigma):
    return np.exp(-(theta - theta1)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


def _eq_I_9_18(a, b, c, d, e, f):
    return (a - c)**2 + (b - d)**2 + (e - f)**2


def _eq_I_12_11(a, theta):
    return 1 + a * np.sin(theta)


def _eq_I_13_12(a, b):
    return a * (1.0 / b - 1.0)


def _eq_I_15_3x(a, b):
    return a / np.sqrt(1 - b**2)


def _eq_I_16_6(a, b):
    return (a + b) / (1 + a * b)


def _eq_I_18_4(a, b):
    return (1 + a * b) / (1 + a)


def _eq_I_26_2(n, theta2):
    return np.arcsin(n * np.sin(theta2))


def _eq_I_27_6(a, b):
    return 1.0 / (1.0 / a + 1.0 / b)


def _eq_I_29_16(a, theta1, theta2):
    return np.sqrt(1 + a**2 - 2 * a * np.cos(theta1 - theta2))


def _eq_I_30_3(a, b):
    half_b = b / 2.0
    num = np.sin(a * half_b)**2
    den = np.sin(half_b)**2
    return num / np.where(den < 1e-30, 1e-30, den)


def _eq_I_30_5(a, b):
    return np.arcsin(a / b)


def _eq_I_37_4(a, delta):
    return 1 + a + 2 * np.sqrt(np.abs(a)) * np.cos(delta)


def _eq_I_40_1(n0, a):
    return n0 * np.exp(-a)


def _eq_I_44_4(n, a):
    return n * np.log(a)


def _eq_I_50_26(a, o):
    return np.cos(a) + o * np.cos(a)**2


def _eq_II_2_42(a, b):
    return (a - 1) * b


def _eq_II_6_15a(a, b, c):
    return (a / c**2) * np.sqrt(b**2 + c**2)


def _eq_II_11_7(n0, p, theta):
    return n0 * (1 + p * np.cos(theta))


def _eq_II_11_27(a, b):
    return a + b


def _eq_II_35_18(n0, a):
    return n0 / (np.exp(a) + np.exp(-a))


def _eq_II_36_38(a, b):
    return a + a * b


def _eq_II_38_3(a, b):
    return a * b


def _eq_III_9_52(a, b, c):
    half = (a - b) * c / 2.0
    return np.sin(half)**2 / np.where(np.abs(half) < 1e-30, 1e-30, half**2)


def _eq_III_10_19(a, b):
    return np.sqrt(1 + a**2 + b**2)


def _eq_III_17_37(beta, alpha, theta):
    return beta * (1 + alpha * np.cos(theta))


EQUATIONS: list[FeynmanEquation] = [
    FeynmanEquation(
        "I.6.2", "Gaussian distribution",
        "exp(-θ²/(2σ²)) / √(2πσ²)",
        ["theta", "sigma"],
        {"theta": (-3.0, 3.0), "sigma": (0.5, 3.0)},
        _eq_I_6_2,
    ),
    FeynmanEquation(
        "I.6.2b", "Shifted Gaussian",
        "exp(-(θ-θ₁)²/(2σ²)) / √(2πσ²)",
        ["theta", "theta1", "sigma"],
        {"theta": (-3.0, 3.0), "theta1": (-1.0, 1.0), "sigma": (0.5, 3.0)},
        _eq_I_6_2b,
    ),
    FeynmanEquation(
        "I.9.18", "Gravitational distance",
        "(a-c)² + (b-d)² + (e-f)²",
        ["a", "b", "c", "d", "e", "f"],
        {"a": (-2, 2), "b": (-2, 2), "c": (-2, 2),
         "d": (-2, 2), "e": (-2, 2), "f": (-2, 2)},
        _eq_I_9_18,
    ),
    FeynmanEquation(
        "I.12.11", "Lorentz force",
        "1 + a·sin(θ)",
        ["a", "theta"],
        {"a": (0.0, 2.0), "theta": (0.0, 2 * np.pi)},
        _eq_I_12_11,
    ),
    FeynmanEquation(
        "I.13.12", "Gravitational PE change",
        "a·(1/b − 1)",
        ["a", "b"],
        {"a": (0.5, 3.0), "b": (0.3, 3.0)},
        _eq_I_13_12,
    ),
    FeynmanEquation(
        "I.15.3x", "Lorentz contraction",
        "a / √(1 − b²)",
        ["a", "b"],
        {"a": (0.5, 2.0), "b": (0.0, 0.9)},
        _eq_I_15_3x,
    ),
    FeynmanEquation(
        "I.16.6", "Relativistic velocity addition",
        "(a + b) / (1 + a·b)",
        ["a", "b"],
        {"a": (-0.9, 0.9), "b": (-0.9, 0.9)},
        _eq_I_16_6,
    ),
    FeynmanEquation(
        "I.18.4", "Center of mass",
        "(1 + a·b) / (1 + a)",
        ["a", "b"],
        {"a": (0.1, 5.0), "b": (0.1, 5.0)},
        _eq_I_18_4,
    ),
    FeynmanEquation(
        "I.26.2", "Snell's law",
        "arcsin(n·sin(θ₂))",
        ["n", "theta2"],
        {"n": (0.1, 0.9), "theta2": (0.1, 1.0)},
        _eq_I_26_2,
    ),
    FeynmanEquation(
        "I.27.6", "Thin lens formula",
        "1 / (1/a + 1/b)",
        ["a", "b"],
        {"a": (0.5, 5.0), "b": (0.5, 5.0)},
        _eq_I_27_6,
    ),
    FeynmanEquation(
        "I.29.16", "Law of cosines",
        "√(1 + a² − 2a·cos(θ₁ − θ₂))",
        ["a", "theta1", "theta2"],
        {"a": (0.1, 3.0), "theta1": (0.0, 2 * np.pi), "theta2": (0.0, 2 * np.pi)},
        _eq_I_29_16,
    ),
    FeynmanEquation(
        "I.30.3", "Diffraction intensity",
        "sin²(a·b/2) / sin²(b/2)",
        ["a", "b"],
        {"a": (1.0, 5.0), "b": (0.3, 2.5)},
        _eq_I_30_3,
    ),
    FeynmanEquation(
        "I.30.5", "Diffraction angle",
        "arcsin(a/b)",
        ["a", "b"],
        {"a": (0.1, 1.0), "b": (1.5, 5.0)},
        _eq_I_30_5,
    ),
    FeynmanEquation(
        "I.37.4", "Two-slit intensity",
        "1 + a + 2√a·cos(δ)",
        ["a", "delta"],
        {"a": (0.1, 4.0), "delta": (0.0, 2 * np.pi)},
        _eq_I_37_4,
    ),
    FeynmanEquation(
        "I.40.1", "Boltzmann distribution",
        "n₀·exp(−a)",
        ["n0", "a"],
        {"n0": (0.5, 3.0), "a": (0.0, 5.0)},
        _eq_I_40_1,
    ),
    FeynmanEquation(
        "I.44.4", "Entropy change",
        "n·ln(a)",
        ["n", "a"],
        {"n": (0.5, 3.0), "a": (0.5, 5.0)},
        _eq_I_44_4,
    ),
    FeynmanEquation(
        "I.50.26", "Driven oscillator",
        "cos(a) + o·cos²(a)",
        ["a", "o"],
        {"a": (0.0, 2 * np.pi), "o": (0.0, 2.0)},
        _eq_I_50_26,
    ),
    FeynmanEquation(
        "II.2.42", "Pressure relation",
        "(a − 1)·b",
        ["a", "b"],
        {"a": (1.0, 5.0), "b": (0.5, 5.0)},
        _eq_II_2_42,
    ),
    FeynmanEquation(
        "II.6.15a", "Electric field energy",
        "(a/c²)·√(b² + c²)",
        ["a", "b", "c"],
        {"a": (0.5, 3.0), "b": (0.5, 3.0), "c": (0.5, 3.0)},
        _eq_II_6_15a,
    ),
    FeynmanEquation(
        "II.11.7", "Dipole radiation",
        "n₀·(1 + p·cos(θ))",
        ["n0", "p", "theta"],
        {"n0": (0.5, 3.0), "p": (0.0, 1.0), "theta": (0.0, 2 * np.pi)},
        _eq_II_11_7,
    ),
    FeynmanEquation(
        "II.11.27", "Chemical potential",
        "a + b",
        ["a", "b"],
        {"a": (0.0, 5.0), "b": (0.0, 5.0)},
        _eq_II_11_27,
    ),
    FeynmanEquation(
        "II.35.18", "Fermi–Dirac (sech type)",
        "n₀ / (exp(a) + exp(−a))",
        ["n0", "a"],
        {"n0": (0.5, 3.0), "a": (-3.0, 3.0)},
        _eq_II_35_18,
    ),
    FeynmanEquation(
        "II.36.38", "Magnetisation",
        "a + a·b",
        ["a", "b"],
        {"a": (0.5, 3.0), "b": (0.0, 3.0)},
        _eq_II_36_38,
    ),
    FeynmanEquation(
        "II.38.3", "Young's modulus (stress-strain)",
        "a·b",
        ["a", "b"],
        {"a": (0.5, 5.0), "b": (0.5, 5.0)},
        _eq_II_38_3,
    ),
    FeynmanEquation(
        "III.9.52", "Transition probability (sinc²)",
        "sin²((a−b)c/2) / ((a−b)c/2)²",
        ["a", "b", "c"],
        {"a": (1.0, 5.0), "b": (0.0, 4.0), "c": (0.5, 3.0)},
        _eq_III_9_52,
    ),
    FeynmanEquation(
        "III.10.19", "Magnetic moment energy",
        "√(1 + a² + b²)",
        ["a", "b"],
        {"a": (-3.0, 3.0), "b": (-3.0, 3.0)},
        _eq_III_10_19,
    ),
    FeynmanEquation(
        "III.17.37", "Angular distribution",
        "β·(1 + α·cos(θ))",
        ["beta", "alpha", "theta"],
        {"beta": (0.5, 3.0), "alpha": (0.0, 1.0), "theta": (0.0, 2 * np.pi)},
        _eq_III_17_37,
    ),
]

_INDEX = {eq.eq_id: eq for eq in EQUATIONS}


def list_equations() -> list[FeynmanEquation]:
    """Return the full list of Feynman equations."""
    return list(EQUATIONS)


def get_equation(eq_id: str) -> FeynmanEquation:
    """Look up an equation by its Feynman ID (e.g. ``'I.12.11'``)."""
    try:
        return _INDEX[eq_id]
    except KeyError:
        raise KeyError(
            f"Unknown equation '{eq_id}'. "
            f"Available: {sorted(_INDEX.keys())}"
        ) from None
