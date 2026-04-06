"""Internal helpers for superposition-activation classification notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier

from ..datasets import PreparedClassificationSplit
from .quantum_activation_classifier import (
    ActivationMeasurement,
    QuantumActivationClassifier,
    QuantumActivationConfig,
    train_quantum_activation_classifier,
)


@dataclass(frozen=True)
class TrainingSnapshot:
    step: int
    loss: float


@dataclass(frozen=True)
class CurveSnapshot:
    step: int
    loss: float
    profiles: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class AccuracySnapshot:
    step: int
    train_accuracy: float
    test_accuracy: float


@dataclass(frozen=True)
class BaselineResult:
    name: str
    estimator: Any
    accuracy: float
    macro_f1: float
    y_pred: np.ndarray
    confusion_matrix: np.ndarray
    classification_report: str


@dataclass(frozen=True)
class QuantumExperimentResult:
    label: str
    mode: str
    model: QuantumActivationClassifier
    config: QuantumActivationConfig
    losses: np.ndarray
    training_snapshots: tuple[TrainingSnapshot, ...]
    training_curve_snapshots: tuple[CurveSnapshot, ...]
    accuracy_history: tuple[AccuracySnapshot, ...]
    train_accuracy: float
    test_accuracy: float
    macro_f1: float
    y_pred: np.ndarray
    confusion_matrix: np.ndarray
    classification_report: str
    representative_units: tuple[int, ...]
    measurements: tuple[ActivationMeasurement, ...]
    eval_shots: int
    snapshot_interval: int


def print_split_summary(dataset_name: str, split: PreparedClassificationSplit) -> None:
    """Print a compact dataset summary for a prepared split."""
    pretty_name = dataset_name.replace("_", " ").title()
    print(f"Dataset: {pretty_name}")
    print(f"Classes: {list(split.target_names)}")
    print(f"Training set: {split.x_train.shape[0]} samples")
    print(f"Test set:     {split.x_test.shape[0]} samples")
    print(f"Feature dimension: {split.x_train.shape[1]}")
    print("Train class counts:", np.bincount(split.y_train))
    print("Test class counts: ", np.bincount(split.y_test))
    if split.pca is not None:
        print(f"PCA components: {split.x_train.shape[1]}")


def run_baseline(
    name: str,
    estimator: Any,
    split: PreparedClassificationSplit,
) -> BaselineResult:
    """Fit a baseline estimator and package common metrics."""
    estimator.fit(split.x_train, split.y_train)
    y_pred = np.asarray(estimator.predict(split.x_test), dtype=int)
    return BaselineResult(
        name=name,
        estimator=estimator,
        accuracy=float(accuracy_score(split.y_test, y_pred)),
        macro_f1=float(f1_score(split.y_test, y_pred, average="macro")),
        y_pred=y_pred,
        confusion_matrix=confusion_matrix(split.y_test, y_pred),
        classification_report=classification_report(
            split.y_test,
            y_pred,
            target_names=list(split.target_names),
            digits=3,
        ),
    )


def run_default_baseline_suite(
    split: PreparedClassificationSplit,
    *,
    seed: int,
    mlp_hidden_layer_sizes: tuple[int, ...] = (16,),
    mlp_max_iter: int = 1500,
) -> dict[str, BaselineResult]:
    """Run the default LogisticRegression + MLP baseline pair."""
    return {
        "LogisticRegression": run_baseline(
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=seed),
            split,
        ),
        "MLPClassifier": run_baseline(
            "MLPClassifier",
            MLPClassifier(
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                max_iter=mlp_max_iter,
                random_state=seed,
            ),
            split,
        ),
    }


def representative_units(model: QuantumActivationClassifier, k: int = 2) -> list[int]:
    """Pick hidden units with the largest output-layer norms."""
    scores = np.linalg.norm(np.asarray(model.W_out, dtype=float), axis=0)
    order = np.argsort(scores)[::-1]
    return [int(idx) for idx in order[: min(k, model.hidden_units)]]


def run_quantum_experiment(
    mode: str,
    *,
    label: str,
    split: PreparedClassificationSplit,
    hidden_units: int,
    n_qubits: int,
    steps: int,
    learning_rate: float,
    seed: int,
    log_every: int | None,
    snapshot_interval: int,
    eval_shots: int,
    use_jax: bool = False,
    batch_size: int = 512,
    show_training_progress: bool = False,
) -> QuantumExperimentResult:
    """Train one quantum-activation classifier and collect notebook-facing diagnostics."""
    cfg = QuantumActivationConfig(
        input_dim=split.x_train.shape[1],
        hidden_units=hidden_units,
        n_qubits=n_qubits,
        n_classes=len(split.target_names),
        mode=mode,
        learning_rate=learning_rate,
        steps=steps,
        seed=seed,
        use_jax=use_jax,
        batch_size=batch_size,
        show_training_progress=show_training_progress,
    )
    snapshot_unit_count = min(3, hidden_units)
    training_snapshots: list[TrainingSnapshot] = []
    training_curve_snapshots: list[CurveSnapshot] = []
    accuracy_history: list[AccuracySnapshot] = []

    def after_step(step: int, loss: float, model: QuantumActivationClassifier) -> None:
        loss_value = float(loss)
        training_snapshots.append(TrainingSnapshot(step=step, loss=loss_value))
        if step == -1 or step % snapshot_interval == 0 or step == cfg.steps - 1:
            train_acc = model.accuracy(split.x_train, split.y_train)
            test_acc = model.accuracy(split.x_test, split.y_test)
            accuracy_history.append(
                AccuracySnapshot(
                    step=step,
                    train_accuracy=float(train_acc),
                    test_accuracy=float(test_acc),
                )
            )
            tracked_profiles = tuple(
                np.asarray(model.get_activation_profile(unit_idx), dtype=float)
                for unit_idx in range(snapshot_unit_count)
            )
            training_curve_snapshots.append(
                CurveSnapshot(
                    step=step,
                    loss=loss_value,
                    profiles=tracked_profiles,
                )
            )

    model, losses = train_quantum_activation_classifier(
        split.x_train,
        split.y_train,
        cfg,
        after_step=after_step,
        log_every=log_every,
    )
    y_pred = np.asarray(model.predict(split.x_test), dtype=int)
    chosen_units = tuple(representative_units(model, k=2))
    measurements = tuple(
        model.measure_activation_profile(unit_idx, shots=eval_shots)
        for unit_idx in chosen_units
    )
    return QuantumExperimentResult(
        label=label,
        mode=mode,
        model=model,
        config=cfg,
        losses=np.asarray(losses, dtype=float),
        training_snapshots=tuple(training_snapshots),
        training_curve_snapshots=tuple(training_curve_snapshots),
        accuracy_history=tuple(accuracy_history),
        train_accuracy=float(model.accuracy(split.x_train, split.y_train)),
        test_accuracy=float(accuracy_score(split.y_test, y_pred)),
        macro_f1=float(f1_score(split.y_test, y_pred, average="macro")),
        y_pred=y_pred,
        confusion_matrix=confusion_matrix(split.y_test, y_pred),
        classification_report=classification_report(
            split.y_test,
            y_pred,
            target_names=list(split.target_names),
            digits=3,
        ),
        representative_units=chosen_units,
        measurements=measurements,
        eval_shots=eval_shots,
        snapshot_interval=snapshot_interval,
    )


def print_metric_summary(name: str, accuracy: float, macro_f1: float) -> None:
    print(name)
    print(f"  accuracy = {accuracy:.4f}")
    print(f"  macro-F1 = {macro_f1:.4f}")


def display_baseline_suite(
    baseline_results: dict[str, BaselineResult],
    class_names: tuple[str, ...] | list[str],
) -> None:
    for result in baseline_results.values():
        print_metric_summary(result.name, result.accuracy, result.macro_f1)
        print(result.classification_report)
    plot_baseline_confusions(baseline_results, class_names)


def display_quantum_result(
    result: QuantumExperimentResult,
    class_names: tuple[str, ...] | list[str],
) -> None:
    print(result.label)
    print(f"  train accuracy = {result.train_accuracy:.4f}")
    print(f"  test accuracy  = {result.test_accuracy:.4f}")
    print(f"  macro-F1       = {result.macro_f1:.4f}")
    print(f"  tracked units  = {list(result.representative_units)}")
    print(result.classification_report)
    plot_confusion_result(
        result.confusion_matrix,
        class_names,
        title=f"{result.label} confusion matrix",
    )
    plot_final_activation_profiles(
        result.model,
        title=f"{result.label}: final learned activation profiles",
    )
    plot_measurement_overlays(
        result.model,
        result.representative_units,
        result.measurements,
        shots=result.eval_shots,
        title_prefix=f"{result.label} measured activations",
    )


def plot_confusion_result(
    cm: np.ndarray,
    class_names: tuple[str, ...] | list[str],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_baseline_confusions(
    baseline_results: dict[str, BaselineResult],
    class_names: tuple[str, ...] | list[str],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(baseline_results), figsize=(5.0 * len(baseline_results), 4.0))
    axes = np.atleast_1d(axes)
    for ax, result in zip(axes, baseline_results.values()):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=result.confusion_matrix,
            display_labels=list(class_names),
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(result.name)
    plt.tight_layout()
    plt.show()


def plot_snapshot_progress(
    losses: np.ndarray,
    training_snapshots: tuple[TrainingSnapshot, ...],
    training_curve_snapshots: tuple[CurveSnapshot, ...],
    *,
    snapshot_interval: int,
) -> None:
    import matplotlib.pyplot as plt

    if training_snapshots:
        plt.figure(figsize=(6, 3))
        plt.plot(losses, label="Cross-entropy")
        if training_curve_snapshots:
            for snap in training_curve_snapshots:
                if snap.step >= 0:
                    plt.axvline(snap.step, color="gray", alpha=0.4, linewidth=0.8)
        else:
            for snap in training_snapshots:
                if snap.step >= 0 and (snap.step % snapshot_interval == 0 or snap.step == len(losses) - 1):
                    plt.axvline(snap.step, color="gray", alpha=0.4, linewidth=0.8)
        plt.title("Training loss (vertical lines = curve snapshots)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy")
        plt.legend()
        plt.show()

        subsample = training_snapshots[:: max(1, len(training_snapshots) // 8)]
        print("Epoch | loss at snapshot")
        for snap in subsample:
            print(f"{snap.step:5d} | {snap.loss:.6g}")
    else:
        print("No snapshots recorded (rerun the training cell).")


def plot_accuracy_history(
    accuracy_history: tuple[AccuracySnapshot, ...],
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if not accuracy_history:
        print("No accuracy snapshots recorded (rerun the training cell).")
        return
    steps = [snap.step for snap in accuracy_history]
    train_acc = [snap.train_accuracy for snap in accuracy_history]
    test_acc = [snap.test_accuracy for snap in accuracy_history]
    plt.figure(figsize=(6, 3))
    plt.plot(steps, train_acc, marker="o", label="train accuracy")
    plt.plot(steps, test_acc, marker="o", label="test accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.02)
    plt.legend(loc="best")
    plt.show()


def plot_activation_evolution(
    model: QuantumActivationClassifier,
    training_curve_snapshots: tuple[CurveSnapshot, ...],
    *,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    if not training_curve_snapshots:
        print("No activation curve snapshots recorded (rerun the training cell).")
        return
    x_grid = np.asarray(model.activation_grid, dtype=float)
    n_units = len(training_curve_snapshots[0].profiles)
    fig, axes = plt.subplots(
        n_units,
        1,
        figsize=(7, 2.5 * n_units),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    cmap = plt.cm.viridis
    steps = [snap.step for snap in training_curve_snapshots]
    smin, smax = min(steps), max(steps)

    for unit_idx, ax in enumerate(axes):
        for snap in training_curve_snapshots:
            t = 0.0 if smax == smin else (snap.step - smin) / (smax - smin)
            ax.plot(
                x_grid,
                snap.profiles[unit_idx],
                color=cmap(t),
                alpha=0.85,
                linewidth=1.2,
            )
        ax.axhline(0.0, color="gray", alpha=0.3, linewidth=0.8)
        ax.set_ylabel(f"unit {unit_idx}")
        ax.set_title(f"{title_prefix}: hidden unit {unit_idx}")
    axes[-1].set_xlabel("pre-activation z")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=smin, vmax=smax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="training step (snapshot)")
    plt.show()


def plot_final_activation_profiles(
    model: QuantumActivationClassifier,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    n_units = model.hidden_units
    ncols = 2
    nrows = int(np.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    x_grid = np.asarray(model.activation_grid, dtype=float)
    for unit_idx, ax in enumerate(axes):
        if unit_idx >= n_units:
            ax.axis("off")
            continue
        profile = model.get_activation_profile(unit_idx)
        ax.plot(x_grid, profile, linewidth=2)
        ax.axhline(0.0, color="gray", alpha=0.3, linewidth=0.8)
        ax.set_title(f"Hidden unit {unit_idx}")
        ax.set_ylabel("activation")
        ax.set_xlabel("pre-activation z")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_measurement_overlays(
    model: QuantumActivationClassifier,
    unit_indices: tuple[int, ...] | list[int],
    measurements: tuple[ActivationMeasurement, ...] | list[ActivationMeasurement],
    *,
    shots: int,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(unit_indices), 1, figsize=(7, 3 * len(unit_indices)), sharex=True)
    axes = np.atleast_1d(axes)
    x_grid = np.asarray(model.activation_grid, dtype=float)
    for ax, unit_idx, measurement in zip(axes, unit_indices, measurements):
        exact = model.get_activation_profile(unit_idx)
        ax.plot(x_grid, exact, "k--", linewidth=2, label="exact activation")
        ax.plot(x_grid, measurement.profile, linewidth=1.6, label=f"measured ({shots} shots)")
        ax.axhline(0.0, color="gray", alpha=0.3, linewidth=0.8)
        ax.set_title(f"{title_prefix}: unit {unit_idx}")
        ax.set_ylabel("activation")
        ax.legend(loc="best")
        print(
            f"unit {unit_idx}: exact-vs-measured L1 = "
            f"{np.sum(np.abs(exact - measurement.profile)):.6f}"
        )
        if measurement.p_pos is not None and measurement.p_neg is not None:
            print(f"  measured p_pos + p_neg = {measurement.p_pos.sum() + measurement.p_neg.sum():.6f}")
        if measurement.z_plus is not None and measurement.z_minus is not None:
            print(f"  measured z_plus + z_minus = {measurement.z_plus + measurement.z_minus:.6f}")
    axes[-1].set_xlabel("pre-activation z")
    plt.tight_layout()
    plt.show()


def plot_training_diagnostics(result: QuantumExperimentResult) -> None:
    plot_snapshot_progress(
        result.losses,
        result.training_snapshots,
        result.training_curve_snapshots,
        snapshot_interval=result.snapshot_interval,
    )
    plot_accuracy_history(
        result.accuracy_history,
        title=f"{result.label}: train/test accuracy over snapshots",
    )
    plot_activation_evolution(
        result.model,
        result.training_curve_snapshots,
        title_prefix=f"{result.label} activation evolution",
    )


def build_comparison_rows(
    baseline_results: dict[str, BaselineResult],
    quantum_results: list[QuantumExperimentResult] | tuple[QuantumExperimentResult, ...],
) -> list[tuple[str, float, float]]:
    rows = [
        (baseline_results["LogisticRegression"].name, baseline_results["LogisticRegression"].accuracy, baseline_results["LogisticRegression"].macro_f1),
        (baseline_results["MLPClassifier"].name, baseline_results["MLPClassifier"].accuracy, baseline_results["MLPClassifier"].macro_f1),
    ]
    for result in quantum_results:
        rows.append((result.label, result.test_accuracy, result.macro_f1))
    return rows


def print_comparison_table(rows: list[tuple[str, float, float]]) -> None:
    headers = ("Model", "Test accuracy", "Macro-F1")
    widths = [
        max(len(headers[0]), max(len(row[0]) for row in rows)),
        len(headers[1]),
        len(headers[2]),
    ]
    print(f"{headers[0]:<{widths[0]}} | {headers[1]:>{widths[1]}} | {headers[2]:>{widths[2]}}")
    print(f"{'-' * widths[0]}-+-{'-' * widths[1]}-+-{'-' * widths[2]}")
    for name, acc, macro_f1 in rows:
        print(f"{name:<{widths[0]}} | {acc:>{widths[1]}.4f} | {macro_f1:>{widths[2]}.4f}")

