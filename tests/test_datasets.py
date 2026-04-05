import numpy as np
from sklearn.model_selection import train_test_split

from qfun.datasets import load_classification_dataset, prepare_classification_split


def test_builtin_dataset_shapes_and_class_counts():
    expected = {
        "iris": (4, 3),
        "wine": (13, 3),
        "breast_cancer": (30, 2),
        "digits": (64, 10),
    }

    for name, (n_features, n_classes) in expected.items():
        dataset = load_classification_dataset(name)

        assert dataset.X.ndim == 2
        assert dataset.y.ndim == 1
        assert dataset.X.shape[1] == n_features
        assert len(dataset.feature_names) == n_features
        assert len(dataset.target_names) == n_classes


def test_prepare_split_is_reproducible_and_matches_stratified_split():
    dataset = load_classification_dataset("wine")
    split_a = prepare_classification_split(
        dataset,
        test_size=0.25,
        seed=7,
        standardize=False,
    )
    split_b = prepare_classification_split(
        dataset,
        test_size=0.25,
        seed=7,
        standardize=False,
    )
    manual = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.25,
        random_state=7,
        stratify=dataset.y,
    )

    assert np.array_equal(split_a.x_train, split_b.x_train)
    assert np.array_equal(split_a.y_train, split_b.y_train)
    assert np.array_equal(split_a.x_train, manual[0])
    assert np.array_equal(split_a.x_test, manual[1])
    assert np.array_equal(split_a.y_train, manual[2])
    assert np.array_equal(split_a.y_test, manual[3])


def test_scaler_is_fit_on_train_only():
    dataset = load_classification_dataset("breast_cancer")
    split = prepare_classification_split(
        dataset,
        test_size=0.25,
        seed=7,
        standardize=True,
    )
    x_train_raw, _, _, _ = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.25,
        random_state=7,
        stratify=dataset.y,
    )

    assert split.scaler is not None
    assert np.allclose(split.scaler.mean_, np.mean(x_train_raw, axis=0))


def test_pca_path_returns_requested_dimension_and_feature_names():
    dataset = load_classification_dataset("digits")
    split = prepare_classification_split(
        dataset,
        test_size=0.25,
        seed=7,
        standardize=True,
        pca_components=16,
    )

    assert split.x_train.shape[1] == 16
    assert split.x_test.shape[1] == 16
    assert split.pca is not None
    assert split.feature_names == tuple(f"pc{idx}" for idx in range(16))
