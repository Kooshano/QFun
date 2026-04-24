"""Run paper-oriented QFun/QFAN experiment suites.

The default ``smoke`` preset is intentionally small and local. Use
``--preset paper`` for the full paper grid after confirming compute budget and
OpenML dataset availability.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qfun.datasets import load_classification_dataset, prepare_classification_split
from qfun.qfan import (
    EdgeLevelConfig,
    classify_all_profiles,
    compute_importance_scores,
    evaluate_pruning,
    shot_budget_analysis,
    train_edge_level_classifier,
)
from qfun.qfan._classification_benchmarks import (
    run_default_baseline_suite,
    run_quantum_experiment,
)


PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "datasets": "digits",
        "seeds": "7",
        "modes": "standard",
        "n_qubits": "2",
        "hidden_units": "3",
        "profile_interp": "linear",
        "families": "node,hybrid,edge,shared_edge",
        "pca_components": 8,
        "steps": 2,
        "learning_rate": 0.03,
        "test_size": 0.2,
        "baseline_mlp_hidden": 16,
        "baseline_mlp_max_iter": 300,
    },
    "core": {
        "datasets": "digits",
        "seeds": "7,13,29",
        "modes": "standard,mode_a,mode_b",
        "n_qubits": "3,4",
        "hidden_units": "4,6",
        "profile_interp": "linear,cubic_natural",
        "families": "node,hybrid,edge,shared_edge",
        "pca_components": 16,
        "steps": 60,
        "learning_rate": 0.02,
        "test_size": 0.2,
        "baseline_mlp_hidden": 64,
        "baseline_mlp_max_iter": 1000,
    },
    "paper": {
        "datasets": "digits,mnist,fashion_mnist",
        "seeds": "7,13,29,43,101",
        "modes": "standard,mode_a,mode_b",
        "n_qubits": "3,4,5",
        "hidden_units": "4,6,8",
        "profile_interp": "linear,cubic_natural,cubic_bspline",
        "families": "node,hybrid,edge,shared_edge",
        "pca_components": 32,
        "steps": 100,
        "learning_rate": 0.01,
        "test_size": 0.2,
        "baseline_mlp_hidden": 64,
        "baseline_mlp_max_iter": 1500,
    },
}

EDGE_FAMILIES = {"edge", "shared_edge"}
QUANTUM_FAMILIES = {"node", "hybrid", *EDGE_FAMILIES}


@dataclass(frozen=True)
class PaperVariant:
    family: str
    mode: str
    n_qubits: int
    hidden_units: int
    profile_interp: str

    @property
    def label(self) -> str:
        return (
            f"{self.family}/{self.mode}/q{self.n_qubits}/"
            f"h{self.hidden_units}/{self.profile_interp}"
        )


def _parse_str_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(values: list[float]) -> tuple[float, float]:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return float(mean(clean)), float(pstdev(clean))


def build_variant_grid(
    *,
    families: list[str],
    modes: list[str],
    n_qubits_values: list[int],
    hidden_units_values: list[int],
    profile_interp_values: list[str],
) -> list[PaperVariant]:
    """Build the quantum-variant grid, skipping unsupported edge spline rows."""
    variants: list[PaperVariant] = []
    unknown = sorted(set(families) - QUANTUM_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown families: {', '.join(unknown)}.")

    for family, mode, n_qubits, hidden_units, profile_interp in product(
        families,
        modes,
        n_qubits_values,
        hidden_units_values,
        profile_interp_values,
    ):
        if family in EDGE_FAMILIES and profile_interp != "linear":
            continue
        variants.append(
            PaperVariant(
                family=family,
                mode=mode,
                n_qubits=n_qubits,
                hidden_units=hidden_units,
                profile_interp=profile_interp,
            )
        )
    return variants


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["category"],
            row["family"],
            row["model_name"],
            row["mode"],
            row["n_qubits"],
            row["hidden_units"],
            row["profile_interp"],
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        train_mean, train_std = _metric_summary([row["train_accuracy"] for row in group])
        test_mean, test_std = _metric_summary([row["test_accuracy"] for row in group])
        f1_mean, f1_std = _metric_summary([row["macro_f1"] for row in group])
        loss_mean, loss_std = _metric_summary([row["final_loss"] for row in group])
        summary.append(
            {
                "dataset": key[0],
                "category": key[1],
                "family": key[2],
                "model_name": key[3],
                "mode": key[4],
                "n_qubits": key[5],
                "hidden_units": key[6],
                "profile_interp": key[7],
                "num_runs": len(group),
                "train_accuracy_mean": train_mean,
                "train_accuracy_std": train_std,
                "test_accuracy_mean": test_mean,
                "test_accuracy_std": test_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "final_loss_mean": loss_mean,
                "final_loss_std": loss_std,
            }
        )
    return summary


def _resolve_use_jax(mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    try:
        from qfun.qfan._jax_quantum_activation import jax_ready
    except ImportError:
        return False
    return bool(jax_ready())


def _count_params(params: list[Any]) -> int:
    return int(sum(np.asarray(param).size for param in params))


def _baseline_rows(dataset_name: str, seed: int, split: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    results = run_default_baseline_suite(
        split,
        seed=seed,
        mlp_hidden_layer_sizes=(args.baseline_mlp_hidden,),
        mlp_max_iter=args.baseline_mlp_max_iter,
    )
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "dataset": dataset_name,
                "category": "baseline",
                "family": "baseline",
                "model_name": name,
                "seed": seed,
                "mode": "",
                "n_qubits": "",
                "hidden_units": "",
                "profile_interp": "",
                "train_accuracy": float("nan"),
                "test_accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "final_loss": float("nan"),
                "num_params": "",
                "use_jax": False,
            }
        )
    return rows


def _run_node_variant(
    variant: PaperVariant,
    dataset_name: str,
    seed: int,
    split: Any,
    args: argparse.Namespace,
    *,
    use_jax: bool,
) -> tuple[dict[str, Any], Any]:
    hidden_family = "kan_quantum_hybrid" if variant.family == "hybrid" else "pure_superposition"
    result = run_quantum_experiment(
        variant.mode,
        label=variant.label,
        split=split,
        hidden_units=variant.hidden_units,
        n_qubits=variant.n_qubits,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=seed,
        log_every=args.log_every,
        snapshot_interval=args.snapshot_interval,
        eval_shots=args.eval_shots,
        use_jax=use_jax,
        batch_size=args.batch_size,
        show_training_progress=False,
        hidden_preactivation=args.hidden_preactivation,
        hidden_function_family=hidden_family,
        profile_smoothness_reg=args.profile_smoothness_reg if variant.family == "hybrid" else 0.0,
        profile_interp=variant.profile_interp,
        collect_diagnostics=args.collect_diagnostics,
    )
    return (
        {
            "dataset": dataset_name,
            "category": "quantum",
            "family": variant.family,
            "model_name": variant.label,
            "seed": seed,
            "mode": variant.mode,
            "n_qubits": variant.n_qubits,
            "hidden_units": variant.hidden_units,
            "profile_interp": variant.profile_interp,
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "macro_f1": result.macro_f1,
            "final_loss": float(result.losses[-1]) if len(result.losses) else float("nan"),
            "num_params": _count_params(result.model.parameters()),
            "use_jax": use_jax,
        },
        result,
    )


def _run_edge_variant(
    variant: PaperVariant,
    dataset_name: str,
    seed: int,
    split: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any]:
    activation_level = "shared_edge" if variant.family == "shared_edge" else "edge"
    cfg = EdgeLevelConfig(
        input_dim=split.x_train.shape[1],
        hidden_layers=(variant.hidden_units,),
        n_qubits=variant.n_qubits,
        n_classes=len(split.target_names),
        mode=variant.mode,
        activation_level=activation_level,
        learning_rate=args.learning_rate,
        steps=args.steps,
        seed=seed,
        profile_interp=variant.profile_interp,
    )
    model, losses = train_edge_level_classifier(
        split.x_train,
        split.y_train,
        cfg,
        log_every=args.log_every,
    )
    y_pred = np.asarray(model.predict(split.x_test), dtype=int)
    return (
        {
            "dataset": dataset_name,
            "category": "quantum",
            "family": variant.family,
            "model_name": variant.label,
            "seed": seed,
            "mode": variant.mode,
            "n_qubits": variant.n_qubits,
            "hidden_units": variant.hidden_units,
            "profile_interp": variant.profile_interp,
            "train_accuracy": model.accuracy(split.x_train, split.y_train),
            "test_accuracy": float(accuracy_score(split.y_test, y_pred)),
            "macro_f1": float(f1_score(split.y_test, y_pred, average="macro", zero_division=0)),
            "final_loss": float(losses[-1]) if len(losses) else float("nan"),
            "num_params": _count_params(model.parameters()),
            "use_jax": False,
        },
        model,
    )


def _interpretability_rows(
    dataset_name: str,
    seed: int,
    variant: PaperVariant,
    result: Any,
    args: argparse.Namespace,
    split: Any,
) -> list[dict[str, Any]]:
    if variant.family in EDGE_FAMILIES:
        return []
    rows: list[dict[str, Any]] = []
    for item in classify_all_profiles(result.model, novelty_threshold=args.novelty_threshold):
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "model_name": variant.label,
                "layer_idx": item.layer_idx,
                "unit_idx": item.unit_idx,
                "kind": "profile_taxonomy",
                "metric": item.best_match,
                "value": item.similarity,
                "extra": "novel" if item.is_novel else "",
            }
        )
    for item in compute_importance_scores(result.model):
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "model_name": variant.label,
                "layer_idx": item.layer_idx,
                "unit_idx": item.unit_idx,
                "kind": "importance",
                "metric": "relative_importance",
                "value": item.relative_importance,
                "extra": f"weight_norm={item.weight_norm:.8g}",
            }
        )
    if args.prune_fraction > 0.0:
        pruning = evaluate_pruning(
            result.model,
            split.x_test,
            split.y_test,
            prune_fraction=args.prune_fraction,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "model_name": variant.label,
                "layer_idx": "",
                "unit_idx": "",
                "kind": "pruning",
                "metric": "accuracy_drop",
                "value": pruning.accuracy_drop,
                "extra": f"fraction_pruned={pruning.fraction_pruned:.8g}",
            }
        )
    return rows


def _shot_budget_rows(
    dataset_name: str,
    seed: int,
    variant: PaperVariant,
    result: Any,
    args: argparse.Namespace,
    split: Any,
) -> list[dict[str, Any]]:
    if variant.family in EDGE_FAMILIES:
        return []
    representatives = list(result.representative_units)[: args.max_shot_units]
    if not representatives:
        representatives = [(0, 0)]
    points = shot_budget_analysis(
        result.model,
        split.x_test,
        split.y_test,
        shot_counts=args.shot_counts,
        n_trials=args.shot_trials,
        representative_units=representatives,
        max_samples=args.max_shot_samples,
    )
    return [
        {
            "dataset": dataset_name,
            "seed": seed,
            "model_name": variant.label,
            "shots": point.shots,
            "accuracy": point.accuracy,
            "mean_kl": point.mean_kl,
            "mean_tv": point.mean_tv,
            "mean_hellinger": point.mean_hellinger,
        }
        for point in points
    ]


def _preset_value(args: argparse.Namespace, key: str) -> Any:
    value = getattr(args, key)
    if value not in {"", None}:
        return value
    return PRESET_DEFAULTS[args.preset][key]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESET_DEFAULTS), default="smoke")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--modes", default="")
    parser.add_argument("--n-qubits", default="")
    parser.add_argument("--hidden-units", default="")
    parser.add_argument("--profile-interp", default="")
    parser.add_argument("--families", default="")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--eval-shots", type=int, default=2000)
    parser.add_argument("--hidden-preactivation", default="superposition")
    parser.add_argument("--profile-smoothness-reg", type=float, default=1e-3)
    parser.add_argument("--baseline-mlp-hidden", type=int, default=None)
    parser.add_argument("--baseline-mlp-max-iter", type=int, default=None)
    parser.add_argument("--use-jax", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--collect-diagnostics", action="store_true")
    parser.add_argument("--with-interpretability", action="store_true")
    parser.add_argument("--novelty-threshold", type=float, default=0.85)
    parser.add_argument("--prune-fraction", type=float, default=0.0)
    parser.add_argument("--with-shot-budget", action="store_true")
    parser.add_argument("--shot-counts", default="100,500,1000,5000,10000")
    parser.add_argument("--shot-trials", type=int, default=3)
    parser.add_argument("--max-shot-samples", type=int, default=50)
    parser.add_argument("--max-shot-units", type=int, default=2)
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.datasets = _parse_str_csv(_preset_value(args, "datasets"))
    args.seeds = _parse_int_csv(_preset_value(args, "seeds"))
    args.modes = _parse_str_csv(_preset_value(args, "modes"))
    args.n_qubits = _parse_int_csv(_preset_value(args, "n_qubits"))
    args.hidden_units = _parse_int_csv(_preset_value(args, "hidden_units"))
    args.profile_interp = _parse_str_csv(_preset_value(args, "profile_interp"))
    args.families = _parse_str_csv(_preset_value(args, "families"))
    args.pca_components = int(_preset_value(args, "pca_components"))
    args.test_size = float(_preset_value(args, "test_size"))
    args.steps = int(_preset_value(args, "steps"))
    args.learning_rate = float(_preset_value(args, "learning_rate"))
    args.baseline_mlp_hidden = int(_preset_value(args, "baseline_mlp_hidden"))
    args.baseline_mlp_max_iter = int(_preset_value(args, "baseline_mlp_max_iter"))
    args.shot_counts = _parse_int_csv(args.shot_counts)
    args.collect_diagnostics = args.collect_diagnostics or args.with_shot_budget
    return args


def main() -> None:
    args = _finalize_args(_build_parser().parse_args())
    variants = build_variant_grid(
        families=args.families,
        modes=args.modes,
        n_qubits_values=args.n_qubits,
        hidden_units_values=args.hidden_units,
        profile_interp_values=args.profile_interp,
    )
    if args.limit_configs > 0:
        variants = variants[: args.limit_configs]
    if not variants:
        raise ValueError("No quantum variants selected.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else PROJECT_ROOT / "artifacts" / "paper_suite" / run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    use_jax = _resolve_use_jax(args.use_jax)

    manifest = {
        "preset": args.preset,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "variants": [asdict(variant) for variant in variants],
        "pca_components": args.pca_components,
        "test_size": args.test_size,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "use_jax": use_jax,
        "with_interpretability": args.with_interpretability,
        "with_shot_budget": args.with_shot_budget,
    }
    _save_json(output_root / "manifest.json", manifest)
    print(f"Results directory: {output_root}")

    per_run_rows: list[dict[str, Any]] = []
    interpretability_rows: list[dict[str, Any]] = []
    shot_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        dataset = load_classification_dataset(dataset_name)
        for seed in args.seeds:
            print(f"\nDataset={dataset_name} seed={seed}", flush=True)
            split = prepare_classification_split(
                dataset,
                test_size=args.test_size,
                seed=seed,
                standardize=True,
                pca_components=args.pca_components,
            )
            per_run_rows.extend(_baseline_rows(dataset_name, seed, split, args))

            for variant in variants:
                print(f"  {variant.label}", flush=True)
                try:
                    if variant.family in EDGE_FAMILIES:
                        row, result = _run_edge_variant(variant, dataset_name, seed, split, args)
                    else:
                        row, result = _run_node_variant(
                            variant,
                            dataset_name,
                            seed,
                            split,
                            args,
                            use_jax=use_jax,
                        )
                    per_run_rows.append(row)
                    if args.with_interpretability:
                        interpretability_rows.extend(
                            _interpretability_rows(dataset_name, seed, variant, result, args, split)
                        )
                    if args.with_shot_budget:
                        shot_rows.extend(
                            _shot_budget_rows(dataset_name, seed, variant, result, args, split)
                        )
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    error_rows.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "model_name": variant.label,
                            "error": repr(exc),
                        }
                    )
                    print(f"    FAILED: {exc}", flush=True)

    aggregate = aggregate_rows(per_run_rows)
    _save_csv(output_root / "per_run_results.csv", per_run_rows)
    _save_json(output_root / "per_run_results.json", per_run_rows)
    _save_csv(output_root / "aggregate_results.csv", aggregate)
    _save_json(output_root / "aggregate_results.json", aggregate)
    _save_csv(output_root / "interpretability_results.csv", interpretability_rows)
    _save_json(output_root / "interpretability_results.json", interpretability_rows)
    _save_csv(output_root / "shot_budget_results.csv", shot_rows)
    _save_json(output_root / "shot_budget_results.json", shot_rows)
    _save_csv(output_root / "errors.csv", error_rows)
    _save_json(output_root / "errors.json", error_rows)

    quantum_rows = [row for row in aggregate if row["category"] == "quantum"]
    quantum_rows.sort(key=lambda row: (row["test_accuracy_mean"], row["macro_f1_mean"]), reverse=True)
    print("\nTop quantum variants:")
    for row in quantum_rows[:10]:
        print(
            "  "
            f"{row['dataset']} {row['model_name']}: "
            f"test={row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f}, "
            f"f1={row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
