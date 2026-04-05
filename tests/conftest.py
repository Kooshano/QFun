"""Shared pytest fixtures for QFun tests."""

from __future__ import annotations

import pytest

from qfun.datasets import load_classification_dataset, prepare_classification_split


@pytest.fixture
def iris_split():
    """Standardized Iris train/test split via ``qfun.datasets`` (matches benchmark notebooks)."""
    dataset = load_classification_dataset("iris")
    return prepare_classification_split(
        dataset,
        test_size=0.25,
        seed=7,
        standardize=True,
    )


@pytest.fixture
def iris_split_arrays(iris_split):
    """Iris split as ``(x_train, x_test, y_train, y_test)`` arrays."""
    return (
        iris_split.x_train,
        iris_split.x_test,
        iris_split.y_train,
        iris_split.y_test,
    )
