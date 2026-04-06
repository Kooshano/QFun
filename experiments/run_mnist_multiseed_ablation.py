"""Run paper-oriented multi-seed ablations for the quantum activation classifier.

This script implements the highest-priority research-roadmap item from the
project plan: repeated-seed runs over a small grid of activation configs, then
aggregation into paper-ready mean/std tables.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qfun.datasets import load_classification_dataset, prepare_classification_split
from qfun.qfan._classification_benchmarks import run_default_baseline_suite, run_quantum_experiment


@dataclass(frozen=True)
class AblationConfig:
    mode: str
    n_qubits: int
    hidden_units: int
    profile_interp: str

    @property
    def label(self) -> str:
        return (
            f"pure_superposition/{self.mode}/"
            f"q{self.n_qubits}/h{self.hidden_units}/{self.profile_interp}"
        )


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_str_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


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
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def _resolve_use_jax(mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--eval-shots", type=int, default=2000)
    parser.add_argument("--hidden-preactivation", default="superposition")
    parser.add_argument("--use-jax", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--seeds", default="7,13,29")
    parser.add_argument("--n-qubits", default="3,4,5")
    parser.add_argument("--hidden-units", default="4,6,8")
    parser.add_argument("--profile-interp", default="linear,cubic_natural")
    parser.add_argument("--modes", default="standard,mode_a,mode_b")
    parser.add_argument("--baseline-mlp-hidden", type=int, default=64)
    parser.add_argument("--baseline-mlp-max-iter", type=int, default=1500)
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--output-root", default="")
    return parser


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["category"],
            row["model_name"],
            row["mode"],
            row["n_qubits"],
            row["hidden_units"],
            row["profile_interp"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        train_mean, train_std = _metric_summary([float(row["train_accuracy"]) for row in group])
        test_mean, test_std = _metric_summary([float(row["test_accuracy"]) for row in group])
        f1_mean, f1_std = _metric_summary([float(row["macro_f1"]) for row in group])
        summary_rows.append(
            {
                "category": key[0],
                "model_name": key[1],
                "mode": key[2],
                "n_qubits": key[3],
                "hidden_units": key[4],
                "profile_interp": key[5],
                "num_runs": len(group),
                "train_accuracy_mean": train_mean,
                "train_accuracy_std": train_std,
                "test_accuracy_mean": test_mean,
                "test_accuracy_std": test_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
            }
        )
    return summary_rows


def _print_top_quantum_rows(rows: list[dict[str, Any]], *, limit: int = 10) -> None:
    quantum_rows = [row for row in rows if row["category"] == "quantum"]
    quantum_rows.sort(key=lambda row: (row["test_accuracy_mean"], row["macro_f1_mean"]), reverse=True)
    print("Top aggregated quantum configs:")
    for row in quantum_rows[:limit]:
        print(
            "  "
            f"{row['model_name']}: "
            f"test={row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f}, "
            f"macro_f1={row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}"
        )


def main() -> None:
    args = _build_parser().parse_args()

    seeds = _parse_int_csv(args.seeds)
    n_qubits_values = _parse_int_csv(args.n_qubits)
    hidden_units_values = _parse_int_csv(args.hidden_units)
    interp_values = _parse_str_csv(args.profile_interp)
    modes = _parse_str_csv(args.modes)

    ablations = [
        AblationConfig(mode=mode, n_qubits=n_qubits, hidden_units=hidden_units, profile_interp=profile_interp)
        for mode, n_qubits, hidden_units, profile_interp in product(
            modes,
            n_qubits_values,
            hidden_units_values,
            interp_values,
        )
    ]
    if args.limit_configs > 0:
        ablations = ablations[: args.limit_configs]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else PROJECT_ROOT / "artifacts" / "multiseed_ablation" / run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {output_root}")

    pca_components = args.pca_components if args.pca_components > 0 else None
    use_jax = _resolve_use_jax(args.use_jax)
    dataset = load_classification_dataset(args.dataset)

    config_payload = {
        "dataset": args.dataset,
        "test_size": args.test_size,
        "pca_components": pca_components,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "log_every": args.log_every,
        "snapshot_interval": args.snapshot_interval,
        "eval_shots": args.eval_shots,
        "hidden_preactivation": args.hidden_preactivation,
        "use_jax": use_jax,
        "seeds": seeds,
        "baseline_mlp_hidden": args.baseline_mlp_hidden,
        "baseline_mlp_max_iter": args.baseline_mlp_max_iter,
        "ablations": [asdict(cfg) for cfg in ablations],
    }
    _save_json(output_root / "config.json", config_payload)

    per_run_rows: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"Running seed {seed}...")
        split = prepare_classification_split(
            dataset,
            test_size=args.test_size,
            seed=seed,
            standardize=True,
            pca_components=pca_components,
        )

        baseline_results = run_default_baseline_suite(
            split,
            seed=seed,
            mlp_hidden_layer_sizes=(args.baseline_mlp_hidden,),
            mlp_max_iter=args.baseline_mlp_max_iter,
        )
        for baseline_name, result in baseline_results.items():
            per_run_rows.append(
                {
                    "category": "baseline",
                    "model_name": baseline_name,
                    "seed": seed,
                    "mode": "",
                    "n_qubits": "",
                    "hidden_units": "",
                    "profile_interp": "",
                    "train_accuracy": float("nan"),
                    "test_accuracy": result.accuracy,
                    "macro_f1": result.macro_f1,
                }
            )

        for cfg in ablations:
            print(f"  {cfg.label}")
            result = run_quantum_experiment(
                cfg.mode,
                label=cfg.label,
                split=split,
                hidden_units=cfg.hidden_units,
                n_qubits=cfg.n_qubits,
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
                hidden_function_family="pure_superposition",
                profile_interp=cfg.profile_interp,
                collect_diagnostics=False,
            )
            per_run_rows.append(
                {
                    "category": "quantum",
                    "model_name": cfg.label,
                    "seed": seed,
                    "mode": cfg.mode,
                    "n_qubits": cfg.n_qubits,
                    "hidden_units": cfg.hidden_units,
                    "profile_interp": cfg.profile_interp,
                    "train_accuracy": result.train_accuracy,
                    "test_accuracy": result.test_accuracy,
                    "macro_f1": result.macro_f1,
                }
            )

            _save_json(
                output_root / f"seed_{seed}_{cfg.mode}_q{cfg.n_qubits}_h{cfg.hidden_units}_{cfg.profile_interp}.json",
                {
                    "seed": seed,
                    "config": asdict(cfg),
                    "train_accuracy": result.train_accuracy,
                    "test_accuracy": result.test_accuracy,
                    "macro_f1": result.macro_f1,
                    "losses": result.losses.tolist(),
                },
            )

    aggregate_rows = _aggregate_rows(per_run_rows)
    _save_csv(output_root / "per_run_results.csv", per_run_rows)
    _save_json(output_root / "per_run_results.json", per_run_rows)
    _save_csv(output_root / "aggregate_results.csv", aggregate_rows)
    _save_json(output_root / "aggregate_results.json", aggregate_rows)

    _print_top_quantum_rows(aggregate_rows)


if __name__ == "__main__":
    main()
