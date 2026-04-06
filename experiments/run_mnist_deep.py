# %% [markdown]
# # Deep MNIST Classification With Learned Superposition Activations
#
# Two (or more) stacked hidden layers of learned superposition activations—no fixed tanh on
# the quantum activation path (`hidden_layers=(6, 6)` by default). Otherwise this script
# mirrors ``run_mnist_single_layer.py``: full MNIST, PCA features, JAX training when available, and run artifacts
# under ``notebooks/note11_outputs/<run_id>/``.
#
# Training uses **JAX + Optax** when available (`pip install "qfun[gpu]"` + CUDA jaxlib on GPU).
# Without JAX, set ``use_jax = False`` (very slow on full MNIST with PennyLane autograd).

# %%
import atexit
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = PROJECT_ROOT / "notebooks" / "note11_outputs" / RUN_ID
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


def _note11_restore_stdio() -> None:
    """Undo Tee and close log so interpreter shutdown does not write to a closed file."""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if not _console_log.closed:
        _console_log.close()


atexit.register(_note11_restore_stdio)

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
    build_comparison_rows,
    display_baseline_suite,
    display_quantum_result,
    print_comparison_table,
    print_split_summary,
    plot_training_diagnostics,
    run_default_baseline_suite,
    run_quantum_experiment,
)

print(f"Run artifacts directory: {OUTPUT_ROOT}", flush=True)


def _save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


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
    train_steps = [s.step for s in result.training_snapshots]
    train_losses = [s.loss for s in result.training_snapshots]
    np.savez(
        d / "training_snapshots.npz",
        steps=np.asarray(train_steps),
        losses=np.asarray(train_losses),
    )
    if result.accuracy_history:
        np.savez(
            d / "accuracy_history.npz",
            steps=np.asarray([s.step for s in result.accuracy_history], dtype=np.int32),
            train_accuracy=np.asarray([s.train_accuracy for s in result.accuracy_history]),
            test_accuracy=np.asarray([s.test_accuracy for s in result.accuracy_history]),
        )
    _save_json(
        d / "summary.json",
        {
            "label": result.label,
            "mode": result.mode,
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "macro_f1": result.macro_f1,
            "representative_units": list(result.representative_units),
            "eval_shots": result.eval_shots,
            "snapshot_interval": result.snapshot_interval,
        },
    )


# %% [markdown]
# ## Config
#

# %%
data_seed = 7
test_size = 0.2
pca_components = 32

# Tuple needs a trailing comma for one layer: ``(6,)`` — ``(6)`` is just the int 6.
hidden_layers = (6, 6)
if isinstance(hidden_layers, int):
    hidden_layers = (hidden_layers,)
else:
    hidden_layers = tuple(int(w) for w in hidden_layers)

n_qubits = 5
steps = 100
learning_rate = 0.01
log_every = 1
show_training_progress = True
snapshot_interval = 5
eval_shots = 8_000

try:
    import jax  # noqa: F401

    use_jax = True
except ImportError:
    use_jax = False
    print('Tip: pip install "qfun[gpu]" for JAX training on full MNIST (much faster).')

batch_size = 1024
hidden_preactivation = "superposition"

_save_json(
    OUTPUT_ROOT / "config.json",
    {
        "script": Path(__file__).stem,
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
        "run_id": RUN_ID,
    },
)

# %% [markdown]
# ## 1. Load, standardize, and compress MNIST
#

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
#

# %%
baseline_results = run_default_baseline_suite(
    mnist_split,
    seed=data_seed,
    mlp_hidden_layer_sizes=(64,),
)
display_baseline_suite(baseline_results, class_names)
_save_baseline_artifacts(baseline_results)

# %% [markdown]
# ## 3. Deep standard superposition activations
#

# %%
standard_result = run_quantum_experiment(
    "standard",
    label="MNIST deep standard superposition activations",
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
)
display_quantum_result(standard_result, class_names)
_save_quantum_artifacts("standard", standard_result)

# %% [markdown]
# ### Standard training diagnostics
#

# %%
plot_training_diagnostics(standard_result)

# %% [markdown]
# ## 4. Deep Mode A signed superposition activations
#

# %%
mode_a_result = run_quantum_experiment(
    "mode_a",
    label="MNIST deep Mode A superposition activations",
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
)
display_quantum_result(mode_a_result, class_names)
_save_quantum_artifacts("mode_a", mode_a_result)

# %% [markdown]
# ### Mode A training diagnostics
#

# %%
plot_training_diagnostics(mode_a_result)

# %% [markdown]
# ## 5. Deep Mode B signed superposition activations
#

# %%
mode_b_result = run_quantum_experiment(
    "mode_b",
    label="MNIST deep Mode B superposition activations",
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
)
display_quantum_result(mode_b_result, class_names)
_save_quantum_artifacts("mode_b", mode_b_result)

# %% [markdown]
# ### Mode B training diagnostics
#

# %%
plot_training_diagnostics(mode_b_result)

# %% [markdown]
# ## 6. Final comparison
#

# %%
comparison_rows = build_comparison_rows(
    baseline_results,
    [standard_result, mode_a_result, mode_b_result],
)
print_comparison_table(comparison_rows)

with open(OUTPUT_ROOT / "comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Test accuracy", "Macro-F1"])
    w.writerows(comparison_rows)

_save_json(
    OUTPUT_ROOT / "comparison.json",
    [{"model": name, "test_accuracy": acc, "macro_f1": f1} for name, acc, f1 in comparison_rows],
)

print(f"Finished. Figures: fig_*.png, log: console.log, metrics under subfolders of {OUTPUT_ROOT}", flush=True)
