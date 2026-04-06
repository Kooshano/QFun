"""Notebook 12 companion: PCA-compressed MNIST ablations for quantum KAN-like activations.

This script keeps the MNIST + PCA workflow from notebooks 10 and 11, but adds a
KAN-inspired hidden-function family:

    hidden = base_mix * SiLU(z) + quantum_mix * quantum_profile(z_q)

The quantum branch stays fully QFun-native and supports the same three activation
construction modes:

- standard
- mode_a
- mode_b

Notebook 12 compares six models on the same data split:

- pure_superposition + standard
- pure_superposition + mode_a
- pure_superposition + mode_b
- kan_quantum_hybrid + standard
- kan_quantum_hybrid + mode_a
- kan_quantum_hybrid + mode_b
"""

from __future__ import annotations

import atexit
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _p in (Path.cwd(), Path.cwd().parent):
    if (_p / "qfun").is_dir():
        _root = str(_p.resolve())
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = Path(__file__).resolve().parent / "note12_outputs" / RUN_ID
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
BASELINES_DIR = OUTPUT_ROOT / "baselines"
QUANTUM_DIR = OUTPUT_ROOT / "quantum"


class _Tee:
    """Mirror writes to multiple text streams (e.g. terminal + run log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


_console_log = open(OUTPUT_ROOT / "console.log", "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _console_log)
sys.stderr = _Tee(sys.__stderr__, _console_log)


def _note12_restore_stdio() -> None:
    """Undo Tee and close log so interpreter shutdown does not write to a closed file."""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if not _console_log.closed:
        _console_log.close()


atexit.register(_note12_restore_stdio)

import matplotlib.pyplot as plt
import numpy as np

_plt_show = plt.show
_figure_counter = 0


def _show_and_savefig(*args, **kwargs):
    global _figure_counter
    fig = plt.gcf()
    if fig.get_axes():
        _figure_counter += 1
        out = OUTPUT_ROOT / f"fig_{_figure_counter:03d}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return _plt_show(*args, **kwargs)


plt.show = _show_and_savefig

from qfun.datasets import load_classification_dataset, prepare_classification_split
from qfun.qfan._classification_benchmarks import (
    BaselineResult,
    QuantumExperimentResult,
    display_baseline_suite,
    display_quantum_result,
    print_split_summary,
    plot_training_diagnostics,
    run_default_baseline_suite,
    run_quantum_experiment,
)

print(f"Run artifacts directory: {OUTPUT_ROOT}", flush=True)


def _save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _family_label(family: str) -> str:
    if family == "pure_superposition":
        return "pure_superposition"
    return "kan_quantum_hybrid"


def _mode_label(mode: str) -> str:
    return {
        "standard": "standard",
        "mode_a": "mode_a",
        "mode_b": "mode_b",
    }[mode]


def _save_baseline_artifacts(results: dict[str, BaselineResult]) -> None:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, res in results.items():
        sub = BASELINES_DIR / key.replace(" ", "_")
        sub.mkdir(exist_ok=True)
        np.save(sub / "confusion_matrix.npy", res.confusion_matrix)
        (sub / "classification_report.txt").write_text(res.classification_report, encoding="utf-8")
        rows.append(
            {
                "key": key,
                "name": res.name,
                "accuracy": res.accuracy,
                "macro_f1": res.macro_f1,
            }
        )
    _save_json(BASELINES_DIR / "summary.json", rows)


def _save_quantum_artifacts(tag: str, result: QuantumExperimentResult) -> None:
    d = QUANTUM_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "losses.npy", result.losses)
    np.save(d / "confusion_matrix.npy", result.confusion_matrix)
    (d / "classification_report.txt").write_text(result.classification_report, encoding="utf-8")
    np.savez(
        d / "training_snapshots.npz",
        steps=np.asarray([s.step for s in result.training_snapshots], dtype=np.int32),
        losses=np.asarray([s.loss for s in result.training_snapshots], dtype=np.float64),
    )
    if result.accuracy_history:
        np.savez(
            d / "accuracy_history.npz",
            steps=np.asarray([s.step for s in result.accuracy_history], dtype=np.int32),
            train_accuracy=np.asarray([s.train_accuracy for s in result.accuracy_history], dtype=np.float64),
            test_accuracy=np.asarray([s.test_accuracy for s in result.accuracy_history], dtype=np.float64),
        )
    if result.representative_components:
        np.savez(
            d / "representative_components.npz",
            layer_idx=np.asarray([ref[0] for ref in result.representative_units], dtype=np.int32),
            unit_idx=np.asarray([ref[1] for ref in result.representative_units], dtype=np.int32),
            base=np.stack([component.base for component in result.representative_components], axis=0),
            quantum=np.stack([component.quantum for component in result.representative_components], axis=0),
            combined=np.stack([component.combined for component in result.representative_components], axis=0),
            quantum_profile=np.stack(
                [component.quantum_profile for component in result.representative_components],
                axis=0,
            ),
            base_scale=np.asarray([component.base_scale for component in result.representative_components], dtype=np.float64),
            quantum_scale=np.asarray(
                [component.quantum_scale for component in result.representative_components],
                dtype=np.float64,
            ),
        )
    if result.measurements:
        np.savez(
            d / "representative_measurements.npz",
            layer_idx=np.asarray([ref[0] for ref in result.representative_units], dtype=np.int32),
            unit_idx=np.asarray([ref[1] for ref in result.representative_units], dtype=np.int32),
            measured_profile=np.stack([measurement.profile for measurement in result.measurements], axis=0),
        )
    _save_json(
        d / "summary.json",
        {
            "label": result.label,
            "mode": result.mode,
            "hidden_function_family": result.config.hidden_function_family,
            "hidden_base_activation": result.config.hidden_base_activation,
            "profile_smoothness_reg": result.config.profile_smoothness_reg,
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "macro_f1": result.macro_f1,
            "representative_units": [list(unit) for unit in result.representative_units],
            "eval_shots": result.eval_shots,
            "snapshot_interval": result.snapshot_interval,
        },
    )


def _print_ablation_table(rows: list[dict[str, object]]) -> None:
    headers = ("Family", "Mode", "Test accuracy", "Macro-F1")
    family_w = max(len(headers[0]), max(len(str(row["hidden_function_family"])) for row in rows))
    mode_w = max(len(headers[1]), max(len(str(row["mode"])) for row in rows))
    print(f"{headers[0]:<{family_w}} | {headers[1]:<{mode_w}} | {headers[2]:>13} | {headers[3]:>8}")
    print(f"{'-' * family_w}-+-{'-' * mode_w}-+-{'-' * 13}-+-{'-' * 8}")
    for row in rows:
        print(
            f"{row['hidden_function_family']:<{family_w}} | "
            f"{row['mode']:<{mode_w}} | "
            f"{float(row['test_accuracy']):>13.4f} | "
            f"{float(row['macro_f1']):>8.4f}"
        )


# %% [markdown]
# # Quantum-KAN MNIST Ablations
#
# Notebook 12 compares the original deep superposition-only classifier against a
# KAN-like hybrid hidden-function family where each hidden unit combines:
#
# - a learnable classical base path: `base_mix * SiLU(z)`
# - a learnable quantum correction path:
#   `quantum_mix * quantum_profile(z_q)`
#
# The data path stays aligned with notebooks 10 and 11:
#
# - full MNIST
# - standardization
# - PCA to 32 components
# - JAX training when available
# - artifact saving under `notebooks/note12_outputs/<run_id>/`

# %% [markdown]
# ## Config

# %%
data_seed = 7
test_size = 0.2
pca_components = 32
hidden_layers = (6, 6)
n_qubits = 5
steps = 100
learning_rate = 0.01
log_every = 1
show_training_progress = True
snapshot_interval = 5
eval_shots = 8_000
batch_size = 1024
hidden_preactivation = "superposition"
hidden_base_activation = "silu"
profile_smoothness_reg = 1e-3

try:
    import jax  # noqa: F401

    use_jax = True
except ImportError:
    use_jax = False
    print('Tip: pip install "qfun[gpu]" for JAX training on full MNIST (much faster).')

ABLATIONS = [
    {
        "hidden_function_family": "pure_superposition",
        "mode": "standard",
        "tag": "pure_superposition_standard",
        "label": "MNIST pure-superposition standard",
        "profile_smoothness_reg": 0.0,
    },
    {
        "hidden_function_family": "pure_superposition",
        "mode": "mode_a",
        "tag": "pure_superposition_mode_a",
        "label": "MNIST pure-superposition Mode A",
        "profile_smoothness_reg": 0.0,
    },
    {
        "hidden_function_family": "pure_superposition",
        "mode": "mode_b",
        "tag": "pure_superposition_mode_b",
        "label": "MNIST pure-superposition Mode B",
        "profile_smoothness_reg": 0.0,
    },
    {
        "hidden_function_family": "kan_quantum_hybrid",
        "mode": "standard",
        "tag": "kan_quantum_hybrid_standard",
        "label": "MNIST quantum-KAN hybrid standard",
        "profile_smoothness_reg": profile_smoothness_reg,
    },
    {
        "hidden_function_family": "kan_quantum_hybrid",
        "mode": "mode_a",
        "tag": "kan_quantum_hybrid_mode_a",
        "label": "MNIST quantum-KAN hybrid Mode A",
        "profile_smoothness_reg": profile_smoothness_reg,
    },
    {
        "hidden_function_family": "kan_quantum_hybrid",
        "mode": "mode_b",
        "tag": "kan_quantum_hybrid_mode_b",
        "label": "MNIST quantum-KAN hybrid Mode B",
        "profile_smoothness_reg": profile_smoothness_reg,
    },
]

_save_json(
    OUTPUT_ROOT / "config.json",
    {
        "script": "12_qfun_quantum_kan_mnist_ablation",
        "data_seed": data_seed,
        "test_size": test_size,
        "pca_components": pca_components,
        "hidden_layers": list(hidden_layers),
        "n_qubits": n_qubits,
        "steps": steps,
        "learning_rate": learning_rate,
        "log_every": log_every,
        "show_training_progress": show_training_progress,
        "snapshot_interval": snapshot_interval,
        "eval_shots": eval_shots,
        "use_jax": use_jax,
        "batch_size": batch_size,
        "hidden_preactivation": hidden_preactivation,
        "hidden_base_activation": hidden_base_activation,
        "profile_smoothness_reg": profile_smoothness_reg,
        "ablations": ABLATIONS,
        "run_id": RUN_ID,
    },
)

# %% [markdown]
# ## 1. Load, standardize, and compress MNIST

# %%
mnist_dataset = load_classification_dataset("mnist")
mnist_split = prepare_classification_split(
    mnist_dataset,
    test_size=test_size,
    seed=data_seed,
    standardize=True,
    pca_components=pca_components,
)
class_names = mnist_split.target_names
print(f"Dataset size: {mnist_dataset.X.shape[0]} samples (full MNIST)")
print_split_summary(mnist_dataset.name, mnist_split)

_save_json(
    OUTPUT_ROOT / "dataset_split.json",
    {
        "name": mnist_dataset.name,
        "n_total": int(mnist_dataset.X.shape[0]),
        "n_train": int(mnist_split.x_train.shape[0]),
        "n_test": int(mnist_split.x_test.shape[0]),
        "n_features": int(mnist_split.x_train.shape[1]),
        "class_names": list(class_names),
        "train_class_counts": np.bincount(mnist_split.y_train).tolist(),
        "test_class_counts": np.bincount(mnist_split.y_test).tolist(),
    },
)
np.savez_compressed(
    OUTPUT_ROOT / "data_split_arrays.npz",
    x_train=mnist_split.x_train.astype(np.float64),
    y_train=mnist_split.y_train.astype(np.int32),
    x_test=mnist_split.x_test.astype(np.float64),
    y_test=mnist_split.y_test.astype(np.int32),
)

# %% [markdown]
# ## 2. Baselines

# %%
baseline_results = run_default_baseline_suite(
    mnist_split,
    seed=data_seed,
    mlp_hidden_layer_sizes=(64,),
)
display_baseline_suite(baseline_results, class_names)
_save_baseline_artifacts(baseline_results)

# %% [markdown]
# ## 3. Quantum ablation matrix

# %%
quantum_results: list[QuantumExperimentResult] = []
comparison_rows: list[dict[str, object]] = []

for spec in ABLATIONS:
    print()
    print(
        "### "
        f"{_family_label(spec['hidden_function_family'])} + {_mode_label(spec['mode'])}",
        flush=True,
    )
    result = run_quantum_experiment(
        spec["mode"],
        label=spec["label"],
        split=mnist_split,
        hidden_layers=hidden_layers,
        n_qubits=n_qubits,
        steps=steps,
        learning_rate=learning_rate,
        seed=data_seed,
        log_every=log_every,
        snapshot_interval=snapshot_interval,
        eval_shots=eval_shots,
        use_jax=use_jax,
        batch_size=batch_size,
        show_training_progress=show_training_progress,
        hidden_preactivation=hidden_preactivation,
        hidden_function_family=spec["hidden_function_family"],
        hidden_base_activation=hidden_base_activation,
        profile_smoothness_reg=spec["profile_smoothness_reg"],
    )
    quantum_results.append(result)
    display_quantum_result(result, class_names)
    plot_training_diagnostics(result)
    _save_quantum_artifacts(spec["tag"], result)
    comparison_rows.append(
        {
            "label": result.label,
            "hidden_function_family": result.config.hidden_function_family,
            "mode": result.mode,
            "test_accuracy": result.test_accuracy,
            "macro_f1": result.macro_f1,
            "train_accuracy": result.train_accuracy,
            "profile_smoothness_reg": result.config.profile_smoothness_reg,
        }
    )

# %% [markdown]
# ## 4. Final comparison

# %%
comparison_rows = sorted(
    comparison_rows,
    key=lambda row: (str(row["hidden_function_family"]), str(row["mode"])),
)
_print_ablation_table(comparison_rows)

with open(OUTPUT_ROOT / "comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label",
            "hidden_function_family",
            "mode",
            "train_accuracy",
            "test_accuracy",
            "macro_f1",
            "profile_smoothness_reg",
        ],
    )
    writer.writeheader()
    writer.writerows(comparison_rows)

_save_json(OUTPUT_ROOT / "comparison.json", comparison_rows)

print(
    f"Finished. Figures: fig_*.png, log: console.log, metrics under subfolders of {OUTPUT_ROOT}",
    flush=True,
)
