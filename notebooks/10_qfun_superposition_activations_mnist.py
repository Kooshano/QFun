"""Script companion for notebook 10: MNIST with learned superposition activations.

This file mirrors the notebook flow in a runnable Python module:

1. Load full MNIST through sklearn/OpenML.
2. Create a stratified train/test split.
3. Standardize features and reduce them with PCA.
4. Run classical baselines.
5. Train three quantum-activation classifier variants:
   - standard
   - mode_a
   - mode_b
6. Plot diagnostics and print a final comparison table.

The model is intentionally small. Its distinctive feature is that each hidden
unit learns its activation curve from a superposition-derived profile rather
than using a fixed activation such as ReLU.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    """Make the repo importable when this file is run from the notebooks folder."""
    script_root = Path(__file__).resolve().parent.parent
    for candidate in (Path.cwd(), Path.cwd().parent, script_root):
        if (candidate / "qfun").is_dir():
            resolved = str(candidate.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return
    raise RuntimeError("Could not find the repository root containing the qfun package.")


_ensure_repo_on_path()

from qfun.datasets import load_classification_dataset, prepare_classification_split
from qfun.qfan._classification_benchmarks import (
    build_comparison_rows,
    display_baseline_suite,
    display_quantum_result,
    print_comparison_table,
    print_split_summary,
    plot_training_diagnostics,
    run_default_baseline_suite,
    run_quantum_experiment,
)


def detect_jax() -> bool:
    """Return True when JAX is importable for accelerated MNIST training."""
    try:
        import jax  # noqa: F401
    except ImportError:
        print('Tip: pip install "qfun[gpu]" for JAX training on full MNIST (much faster).')
        return False
    return True


def main() -> None:
    # These defaults match notebook 10.
    data_seed = 7
    test_size = 0.25
    pca_components = 32

    hidden_units = 6
    n_qubits = 3
    steps = 60
    learning_rate = 0.04
    log_every = 5
    snapshot_interval = 5
    eval_shots = 3_000
    batch_size = 1024

    use_jax = detect_jax()

    print("# MNIST Classification With Learned Superposition Activations")
    print()
    print("Loading full MNIST, standardizing features, and applying PCA...")

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

    print()
    print("## Baselines")
    baseline_results = run_default_baseline_suite(
        mnist_split,
        seed=data_seed,
        mlp_hidden_layer_sizes=(64,),
    )
    display_baseline_suite(baseline_results, class_names)

    print()
    print("## Quantum Activation Models")

    standard_result = run_quantum_experiment(
        "standard",
        label="MNIST standard superposition activations",
        split=mnist_split,
        hidden_units=hidden_units,
        n_qubits=n_qubits,
        steps=steps,
        learning_rate=learning_rate,
        seed=data_seed,
        log_every=log_every,
        snapshot_interval=snapshot_interval,
        eval_shots=eval_shots,
        use_jax=use_jax,
        batch_size=batch_size,
    )
    display_quantum_result(standard_result, class_names)
    plot_training_diagnostics(standard_result)

    mode_a_result = run_quantum_experiment(
        "mode_a",
        label="MNIST Mode A superposition activations",
        split=mnist_split,
        hidden_units=hidden_units,
        n_qubits=n_qubits,
        steps=steps,
        learning_rate=learning_rate,
        seed=data_seed,
        log_every=log_every,
        snapshot_interval=snapshot_interval,
        eval_shots=eval_shots,
        use_jax=use_jax,
        batch_size=batch_size,
    )
    display_quantum_result(mode_a_result, class_names)
    plot_training_diagnostics(mode_a_result)

    mode_b_result = run_quantum_experiment(
        "mode_b",
        label="MNIST Mode B superposition activations",
        split=mnist_split,
        hidden_units=hidden_units,
        n_qubits=n_qubits,
        steps=steps,
        learning_rate=learning_rate,
        seed=data_seed,
        log_every=log_every,
        snapshot_interval=snapshot_interval,
        eval_shots=eval_shots,
        use_jax=use_jax,
        batch_size=batch_size,
    )
    display_quantum_result(mode_b_result, class_names)
    plot_training_diagnostics(mode_b_result)

    print()
    print("## Final Comparison")
    comparison_rows = build_comparison_rows(
        baseline_results,
        [standard_result, mode_a_result, mode_b_result],
    )
    print_comparison_table(comparison_rows)


if __name__ == "__main__":
    main()
