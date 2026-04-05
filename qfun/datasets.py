"""Classification dataset helpers for notebook benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClassificationDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedClassificationSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    scaler: StandardScaler | None
    pca: PCA | None


def _coerce_feature_names(raw_names, n_features: int) -> tuple[str, ...]:
    if raw_names is None:
        return tuple(f"feature_{idx}" for idx in range(n_features))
    names = tuple(str(name) for name in raw_names)
    if len(names) != n_features:
        return tuple(f"feature_{idx}" for idx in range(n_features))
    return names


def _coerce_target_names(raw_names, y: np.ndarray) -> tuple[str, ...]:
    if raw_names is None:
        classes = np.unique(y)
        return tuple(str(cls) for cls in classes)
    return tuple(str(name) for name in raw_names)


def _dataset_from_bunch(name: str, bunch) -> ClassificationDataset:
    x = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target, dtype=int)
    feature_names = _coerce_feature_names(getattr(bunch, "feature_names", None), x.shape[1])
    target_names = _coerce_target_names(getattr(bunch, "target_names", None), y)
    return ClassificationDataset(
        name=name,
        X=x,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
    )


_DATASET_LOADERS: dict[str, Callable[[], ClassificationDataset]] = {
    "iris": lambda: _dataset_from_bunch("iris", load_iris()),
    "wine": lambda: _dataset_from_bunch("wine", load_wine()),
    "breast_cancer": lambda: _dataset_from_bunch("breast_cancer", load_breast_cancer()),
    "digits": lambda: _dataset_from_bunch("digits", load_digits()),
}


def load_classification_dataset(name: str) -> ClassificationDataset:
    """Load one of the built-in offline sklearn classification datasets."""
    key = str(name).strip().lower()
    if key not in _DATASET_LOADERS:
        supported = ", ".join(sorted(_DATASET_LOADERS))
        raise ValueError(f"Unknown dataset '{name}'. Supported datasets: {supported}.")
    return _DATASET_LOADERS[key]()


def prepare_classification_split(
    dataset: ClassificationDataset,
    *,
    test_size: float = 0.25,
    seed: int = 7,
    standardize: bool = True,
    pca_components: int | None = None,
) -> PreparedClassificationSplit:
    """Create a reproducible train/test split with optional scaling and PCA."""
    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(dataset.X, dtype=float),
        np.asarray(dataset.y, dtype=int),
        test_size=test_size,
        random_state=seed,
        stratify=np.asarray(dataset.y, dtype=int),
    )

    scaler: StandardScaler | None = None
    if standardize:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    pca: PCA | None = None
    feature_names = dataset.feature_names
    if pca_components is not None:
        if pca_components <= 0:
            raise ValueError("pca_components must be positive when provided.")
        pca = PCA(n_components=int(pca_components))
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        feature_names = tuple(f"pc{idx}" for idx in range(int(pca_components)))

    return PreparedClassificationSplit(
        x_train=np.asarray(x_train, dtype=float),
        x_test=np.asarray(x_test, dtype=float),
        y_train=np.asarray(y_train, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        feature_names=feature_names,
        target_names=dataset.target_names,
        scaler=scaler,
        pca=pca,
    )


__all__ = [
    "ClassificationDataset",
    "PreparedClassificationSplit",
    "load_classification_dataset",
    "prepare_classification_split",
]
