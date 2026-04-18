"""Comprehensive benchmark suite for QFun paper experiments.

Provides a unified runner for:
  - All 27 Feynman equations (function approximation)
  - Classification benchmarks (Iris, Wine, Breast Cancer, Digits,
    MNIST, Fashion-MNIST)
  - Multi-seed ablation tables with mean +/- std
  - Comparison against baselines (LogisticRegression, MLP)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import load_classification_dataset, prepare_classification_split
from ..feynman_dataset import list_equations
from ._classification_benchmarks import (
    BaselineResult,
    run_baseline,
    run_default_baseline_suite,
)
from .quantum_activation_classifier import (
    QuantumActivationClassifier,
    QuantumActivationConfig,
    train_quantum_activation_classifier,
)


@dataclass(frozen=True)
class SingleRunResult:
    dataset: str
    mode: str
    n_qubits: int
    hidden_units: int
    profile_interp: str
    seed: int
    test_accuracy: float
    train_accuracy: float
    macro_f1: float
    final_loss: float
    total_params: int


@dataclass(frozen=True)
class AblationCell:
    """Aggregated result for one configuration across multiple seeds."""
    dataset: str
    mode: str
    n_qubits: int
    hidden_units: int
    profile_interp: str
    n_seeds: int
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    std_f1: float
    mean_loss: float
    std_loss: float


@dataclass
class BenchmarkSuite:
    """Container for all benchmark results."""
    single_runs: list[SingleRunResult] = field(default_factory=list)
    ablation_table: list[AblationCell] = field(default_factory=list)
    baseline_results: dict[str, dict[str, float]] = field(default_factory=dict)
    feynman_results: list[dict[str, Any]] = field(default_factory=list)


def _count_params(model: QuantumActivationClassifier) -> int:
    return sum(np.asarray(p).size for p in model.parameters())


def run_single_classification(
    dataset_name: str,
    *,
    mode: str = "standard",
    n_qubits: int = 4,
    hidden_units: int = 6,
    hidden_layers: tuple[int, ...] | None = None,
    profile_interp: str = "linear",
    steps: int = 80,
    learning_rate: float = 0.05,
    seed: int = 42,
    use_jax: bool = False,
    batch_size: int = 512,
    pca_components: int | None = None,
    test_size: float = 0.3,
    log_every: int | None = None,
) -> SingleRunResult:
    """Train and evaluate a QFun classifier on a single dataset/config."""
    from sklearn.metrics import f1_score

    dataset = load_classification_dataset(dataset_name)
    split = prepare_classification_split(
        dataset,
        test_size=test_size,
        seed=seed,
        pca_components=pca_components,
    )

    cfg = QuantumActivationConfig(
        input_dim=split.x_train.shape[1],
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        n_qubits=n_qubits,
        n_classes=len(split.target_names),
        mode=mode,
        learning_rate=learning_rate,
        steps=steps,
        seed=seed,
        use_jax=use_jax,
        batch_size=batch_size,
        profile_interp=profile_interp,
    )

    model, losses = train_quantum_activation_classifier(
        split.x_train, split.y_train, cfg, log_every=log_every,
    )
    y_pred = model.predict(split.x_test)
    test_acc = float(np.mean(y_pred == split.y_test))
    train_acc = float(model.accuracy(split.x_train, split.y_train))
    macro_f1 = float(f1_score(split.y_test, y_pred, average="macro", zero_division=0))

    return SingleRunResult(
        dataset=dataset_name,
        mode=mode,
        n_qubits=n_qubits,
        hidden_units=hidden_units,
        profile_interp=profile_interp,
        seed=seed,
        test_accuracy=test_acc,
        train_accuracy=train_acc,
        macro_f1=macro_f1,
        final_loss=float(losses[-1]),
        total_params=_count_params(model),
    )


def run_multi_seed_ablation(
    dataset_name: str,
    *,
    modes: list[str] | None = None,
    n_qubits_list: list[int] | None = None,
    hidden_units_list: list[int] | None = None,
    profile_interp_list: list[str] | None = None,
    seeds: list[int] | None = None,
    steps: int = 80,
    learning_rate: float = 0.05,
    use_jax: bool = False,
    batch_size: int = 512,
    pca_components: int | None = None,
    test_size: float = 0.3,
    log_every: int | None = None,
) -> tuple[list[SingleRunResult], list[AblationCell]]:
    """Run a full multi-seed ablation grid.

    Default grid:
      - modes: [standard, mode_a, mode_b]
      - n_qubits: [3, 4, 5]
      - hidden_units: [4, 6, 8]
      - profile_interp: [linear, cubic_natural]
      - seeds: [42, 43, 44]
    """
    if modes is None:
        modes = ["standard", "mode_a", "mode_b"]
    if n_qubits_list is None:
        n_qubits_list = [3, 4, 5]
    if hidden_units_list is None:
        hidden_units_list = [4, 6, 8]
    if profile_interp_list is None:
        profile_interp_list = ["linear", "cubic_natural"]
    if seeds is None:
        seeds = [42, 43, 44]

    all_runs: list[SingleRunResult] = []
    ablation_cells: list[AblationCell] = []

    total_configs = len(modes) * len(n_qubits_list) * len(hidden_units_list) * len(profile_interp_list)
    config_idx = 0

    for mode in modes:
        for nq in n_qubits_list:
            for hu in hidden_units_list:
                for pi in profile_interp_list:
                    config_idx += 1
                    print(
                        f"\n[{config_idx}/{total_configs}] "
                        f"{dataset_name} | mode={mode} nq={nq} hu={hu} pi={pi}",
                        flush=True,
                    )
                    seed_results = []
                    for seed in seeds:
                        try:
                            result = run_single_classification(
                                dataset_name,
                                mode=mode,
                                n_qubits=nq,
                                hidden_units=hu,
                                profile_interp=pi,
                                steps=steps,
                                learning_rate=learning_rate,
                                seed=seed,
                                use_jax=use_jax,
                                batch_size=batch_size,
                                pca_components=pca_components,
                                test_size=test_size,
                                log_every=log_every,
                            )
                            seed_results.append(result)
                            all_runs.append(result)
                        except Exception as e:
                            print(f"    FAILED seed={seed}: {e}", flush=True)

                    if seed_results:
                        accs = [r.test_accuracy for r in seed_results]
                        f1s = [r.macro_f1 for r in seed_results]
                        losses = [r.final_loss for r in seed_results]
                        cell = AblationCell(
                            dataset=dataset_name,
                            mode=mode,
                            n_qubits=nq,
                            hidden_units=hu,
                            profile_interp=pi,
                            n_seeds=len(seed_results),
                            mean_accuracy=float(np.mean(accs)),
                            std_accuracy=float(np.std(accs)),
                            mean_f1=float(np.mean(f1s)),
                            std_f1=float(np.std(f1s)),
                            mean_loss=float(np.mean(losses)),
                            std_loss=float(np.std(losses)),
                        )
                        ablation_cells.append(cell)
                        print(
                            f"    acc={cell.mean_accuracy:.4f}+/-{cell.std_accuracy:.4f} "
                            f"f1={cell.mean_f1:.4f}+/-{cell.std_f1:.4f}",
                            flush=True,
                        )

    return all_runs, ablation_cells


def run_classification_benchmark_suite(
    datasets: list[str] | None = None,
    *,
    steps: int = 80,
    use_jax: bool = False,
    log_every: int | None = None,
) -> BenchmarkSuite:
    """Run the full classification benchmark suite across multiple datasets."""
    if datasets is None:
        datasets = ["iris", "wine", "breast_cancer", "digits"]

    suite = BenchmarkSuite()

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        dataset = load_classification_dataset(ds_name)
        split = prepare_classification_split(dataset, test_size=0.3, seed=42)
        baselines = run_default_baseline_suite(split, seed=42)
        suite.baseline_results[ds_name] = {
            name: {"accuracy": r.accuracy, "macro_f1": r.macro_f1}
            for name, r in baselines.items()
        }

        for name, r in baselines.items():
            print(f"  {name}: acc={r.accuracy:.4f} f1={r.macro_f1:.4f}")

        runs, cells = run_multi_seed_ablation(
            ds_name,
            modes=["standard", "mode_a", "mode_b"],
            n_qubits_list=[3, 4],
            hidden_units_list=[4, 6],
            profile_interp_list=["linear"],
            seeds=[42, 43, 44],
            steps=steps,
            use_jax=use_jax,
            log_every=log_every,
        )
        suite.single_runs.extend(runs)
        suite.ablation_table.extend(cells)

    return suite


def run_feynman_benchmark_suite(
    *,
    n_equations: int | None = None,
    steps: int = 200,
    n_qubits: int = 4,
    num_functions: int = 8,
    n_samples: int = 500,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run function approximation benchmarks on Feynman equations."""
    from .feynman import sample_equation
    from .config import QFANConfig
    from .training import train_qfan

    if seeds is None:
        seeds = [42, 43, 44]

    equations = list_equations()
    if n_equations is not None:
        equations = equations[:n_equations]

    results: list[dict[str, Any]] = []

    for eq_idx, eq in enumerate(equations):
        print(f"\n[{eq_idx + 1}/{len(equations)}] {eq.eq_id}: {eq.name}", flush=True)
        seed_mses = []

        for seed in seeds:
            batch = sample_equation(eq.eq_id, n_samples=n_samples, seed=seed)
            n = len(batch.y)
            split = int(0.8 * n)
            x_train, x_test = batch.x_norm[:split], batch.x_norm[split:]
            y_train, y_test = batch.y[:split], batch.y[split:]

            cfg = QFANConfig(
                input_dim=len(eq.variables),
                num_functions=num_functions,
                n_qubits=n_qubits,
                mode="mode_a",
                learning_rate=0.01,
                steps=steps,
                seed=seed,
            )
            model, losses = train_qfan(x_train, y_train, cfg)
            pred_test = np.asarray(model.forward_batch(x_test), dtype=float)
            test_mse = float(np.mean((pred_test.ravel() - y_test.ravel()) ** 2))
            seed_mses.append(test_mse)

        result = {
            "eq_id": eq.eq_id,
            "name": eq.name,
            "input_dim": len(eq.variables),
            "mean_test_mse": float(np.mean(seed_mses)),
            "std_test_mse": float(np.std(seed_mses)),
            "n_seeds": len(seeds),
        }
        results.append(result)
        print(
            f"  test_mse={result['mean_test_mse']:.6f} +/- {result['std_test_mse']:.6f}",
            flush=True,
        )

    return results


def print_ablation_table(cells: list[AblationCell], *, dataset: str | None = None) -> None:
    """Print a paper-ready ablation table."""
    filtered = cells if dataset is None else [c for c in cells if c.dataset == dataset]
    if not filtered:
        print("No results to display.")
        return

    datasets_shown = sorted(set(c.dataset for c in filtered))
    for ds in datasets_shown:
        ds_cells = [c for c in filtered if c.dataset == ds]
        print(f"\n{'='*70}")
        print(f"Ablation Table: {ds}")
        print(f"{'='*70}")

        header = (
            f"{'Mode':>10} {'NQ':>3} {'HU':>3} {'Interp':>14} "
            f"{'Accuracy':>16} {'Macro-F1':>16} {'Seeds':>5}"
        )
        print(header)
        print("-" * len(header))

        sorted_cells = sorted(ds_cells, key=lambda c: -c.mean_accuracy)
        for c in sorted_cells:
            acc_str = f"{c.mean_accuracy:.4f}+/-{c.std_accuracy:.4f}"
            f1_str = f"{c.mean_f1:.4f}+/-{c.std_f1:.4f}"
            print(
                f"{c.mode:>10} {c.n_qubits:>3} {c.hidden_units:>3} "
                f"{c.profile_interp:>14} {acc_str:>16} {f1_str:>16} "
                f"{c.n_seeds:>5}"
            )


def save_benchmark_results(
    suite: BenchmarkSuite,
    output_dir: str | Path,
) -> None:
    """Save benchmark results to JSON files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs_data = [asdict(r) for r in suite.single_runs]
    (out / "single_runs.json").write_text(
        json.dumps(runs_data, indent=2), encoding="utf-8"
    )

    ablation_data = [asdict(c) for c in suite.ablation_table]
    (out / "ablation_table.json").write_text(
        json.dumps(ablation_data, indent=2), encoding="utf-8"
    )

    (out / "baselines.json").write_text(
        json.dumps(suite.baseline_results, indent=2), encoding="utf-8"
    )

    if suite.feynman_results:
        (out / "feynman_results.json").write_text(
            json.dumps(suite.feynman_results, indent=2), encoding="utf-8"
        )

    print(f"\nResults saved to {out}")
