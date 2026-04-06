# QFun

QFun is a research codebase for quantum-inspired function encoding, signed superposition diagnostics, and learned superposition activations for small neural classifiers.

It has two main layers:

- `qfun`: grids, amplitude encoding, simulation helpers, plotting, datasets, and superposition-learning utilities.
- `qfun.qfan`: QFAN function fitting plus the `QuantumActivationClassifier` research path.

## Installation

Base install:

```bash
pip install -e .
```

Development tools:

```bash
pip install -e ".[dev]"
```

Optional JAX/Optax path for accelerated classifier training:

```bash
pip install -e ".[gpu]"
```

All dependency and tool configuration lives in [pyproject.toml](pyproject.toml).

## Quick start

Encode and sample a nonnegative function:

```python
import numpy as np

from qfun import amplitudes_from_function, counts_to_distribution, grid_x, run_shots

x = grid_x(-2.0, 2.0, n_qubits=5)
amps = amplitudes_from_function(lambda t: np.exp(-t**2), x)
counts = run_shots(amps, n_qubits=5, shots=20_000)

print(counts_to_distribution(counts, n_qubits=5)[:5])
```

Fit a Feynman equation with QFAN:

```python
from qfun.qfan import train_feynman_equation

result = train_feynman_equation(
    "I.12.11",
    n_samples=512,
    num_functions=3,
    n_qubits=5,
    mode="mode_a",
    steps=150,
    lr=0.05,
)

print(result.formula, result.test_mse)
```

Run the benchmark programmatically:

```python
from qfun.qfan import BenchmarkConfig, QFANConfig, run_feynman_benchmark

summary = run_feynman_benchmark(
    "artifacts/feynman_quick",
    BenchmarkConfig(quick_mode=True, quick_limit=5),
    QFANConfig(input_dim=1, num_functions=3, n_qubits=5, steps=100),
)

print(summary["avg_test_mse"])
```

## Maintained notebooks

The curated teaching and methodology notebooks live in [notebooks](notebooks):

1. [01_basics_signed_modes.ipynb](notebooks/01_basics_signed_modes.ipynb)
2. [02_feynman_encoding.ipynb](notebooks/02_feynman_encoding.ipynb)
3. [03_qfan_training.ipynb](notebooks/03_qfan_training.ipynb)
4. [04_superposition_learning.ipynb](notebooks/04_superposition_learning.ipynb)
5. [05_activations_iris.ipynb](notebooks/05_activations_iris.ipynb)
6. [06_activation_ablation.ipynb](notebooks/06_activation_ablation.ipynb)

These are the notebooks covered by the syntax smoke test in `tests/test_notebook_syntax.py`.

## Reproducible experiments

The MNIST-scale research scripts live in [experiments](experiments):

- [run_mnist_single_layer.py](experiments/run_mnist_single_layer.py)
- [run_mnist_spline.py](experiments/run_mnist_spline.py)
- [run_mnist_deep.py](experiments/run_mnist_deep.py)
- [run_mnist_kan_ablation.py](experiments/run_mnist_kan_ablation.py)
- [run_mnist_multiseed_ablation.py](experiments/run_mnist_multiseed_ablation.py)

Run artifacts are written under `notebooks/note*_outputs/<run_id>/` and are gitignored.

## Architecture notes

Long-form architecture references live in [docs](docs):

- [architecture_single_layer.md](docs/architecture_single_layer.md)
- [architecture_deep.md](docs/architecture_deep.md)
- [architecture_kan_hybrid.md](docs/architecture_kan_hybrid.md)
- [paper_roadmap.md](docs/paper_roadmap.md)

## Public API

Top-level `qfun` exposes the encoding and simulation surface:

```python
from qfun import (
    amplitudes_from_function,
    amplitudes_from_function_nd,
    counts_to_distribution,
    counts_to_signed_distribution,
    decompose_signed_distribution,
    estimate_expectation_signed,
    grid_nd,
    grid_x,
    run_shots,
    run_shots_signed,
    run_two_channel_signed,
    signed_amplitudes_from_function,
)
```

The maintained QFAN API lives under `qfun.qfan`:

```python
from qfun.qfan import (
    BenchmarkConfig,
    QFANBlock,
    QFANConfig,
    QuantumActivationClassifier,
    QuantumActivationConfig,
    run_feynman_benchmark,
    train_feynman_equation,
    train_qfan,
    train_quantum_activation_classifier,
)
```

`QuantumActivationConfig` supports both the legacy single-layer form (`hidden_units=6`) and explicit deep stacks (`hidden_layers=(6, 6)`), plus both `pure_superposition` and `kan_quantum_hybrid` hidden-function families.

Legacy advanced helper: `qfun.simulate.build_circuit` is still available for custom workflows, but most callers should use `run_shots`.

## Project layout

```text
QFun/
├── docs/
├── experiments/
├── notebooks/
├── qfun/
│   └── qfan/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Development

```bash
python -m pytest
ruff check .
```

## Notes

- Python requirement: `>=3.10`
- PennyLane powers the circuit construction and simulation path.
- Signed-function handling is a first-class part of the library surface.
