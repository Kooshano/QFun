# QFun

QFun is a small research codebase for representing functions with quantum-inspired and quantum-state-based tools.

It currently includes two closely related layers:

- `qfun`: amplitude encoding, signed/quasi-probability helpers, sampling utilities, plotting helpers, the built-in Feynman benchmark dataset, and quantum superposition learning helpers.
- `qfun.qfan`: the QFAN model and training utilities for fitting scalar functions and running the Feynman benchmark.

The repo is notebook-first, but the Python API is clean enough to use directly.

## What is in this repo

### Core `qfun` layer

- 1D and nD grid construction
- amplitude encoding for nonnegative functions
- signed function handling with:
  - Mode A: ancilla sign bit
  - Mode B: two-channel signed decomposition
- PennyLane-based sampling and reconstruction utilities
- plotting helpers for measured vs target distributions
- the 27-equation Feynman benchmark dataset
- superposition-learning helpers used by notebook 05

### `qfun.qfan` layer

- `QFANBlock` for function approximation on a fixed 1D basis grid
- training helpers built on Adam
- utilities for fitting a single Feynman equation
- a benchmark runner across the Feynman dataset

## Installation

Runtime dependencies:

```bash
pip install -r requirements.txt
```

Editable install with development extras:

```bash
pip install -e ".[dev]"
```

Requirements and tool config live in [pyproject.toml](/Users/kooshan/Desktop/QFun/pyproject.toml).

## Quick start

### 1. Encode and sample a nonnegative function

```python
import numpy as np

from qfun import grid_x, amplitudes_from_function, run_shots, counts_to_distribution

x = grid_x(-2.0, 2.0, n_qubits=5)
amps = amplitudes_from_function(lambda t: np.exp(-t**2), x)

counts = run_shots(amps, n_qubits=5, shots=20_000)
measured = counts_to_distribution(counts, n_qubits=5)

print(measured[:5])
```

### 2. Fit a Feynman equation with QFAN

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

print(result.formula)
print(result.train_mse, result.test_mse)
```

### 3. Run the benchmark programmatically

```python
from qfun.qfan import BenchmarkConfig, QFANConfig, run_feynman_benchmark

summary = run_feynman_benchmark(
    "artifacts/feynman_quick",
    BenchmarkConfig(quick_mode=True, quick_limit=5),
    QFANConfig(input_dim=1, num_functions=3, n_qubits=5, steps=100),
)

print(summary["avg_test_mse"])
```

The benchmark runner infers the true input dimension for each equation internally; `input_dim` on the template config is just a required constructor field.

## Notebooks

The maintained notebooks live in [notebooks](/Users/kooshan/Desktop/QFun/notebooks):

1. [01_qfun_basics_signed_modes.ipynb](/Users/kooshan/Desktop/QFun/notebooks/01_qfun_basics_signed_modes.ipynb)  
   Basic amplitude encoding, measurement histograms, and signed modes.
2. [02_feynman_dataset_encoding.ipynb](/Users/kooshan/Desktop/QFun/notebooks/02_feynman_dataset_encoding.ipynb)  
   Walkthrough of the built-in Feynman dataset and its encoding pipeline.
3. [03_qfan_training_feynman_benchmark.ipynb](/Users/kooshan/Desktop/QFun/notebooks/03_qfan_training_feynman_benchmark.ipynb)  
   QFAN training on benchmark equations.
4. [04_qfun_ml_demo.ipynb](/Users/kooshan/Desktop/QFun/notebooks/04_qfun_ml_demo.ipynb)  
   QFAN training on a user-defined target function.
5. [05_qfun_superposition_learning.ipynb](/Users/kooshan/Desktop/QFun/notebooks/05_qfun_superposition_learning.ipynb)  
   Learn the quantum superposition itself and recover the function from measurements.

If you installed the dev extras, open them with Jupyter:

```bash
jupyter lab
```

## Public API

Top-level `qfun` exports the core encoding and simulation surface:

```python
from qfun import (
    grid_x,
    amplitudes_from_function,
    signed_amplitudes_from_function,
    run_shots,
    run_shots_signed,
    run_two_channel_signed,
    counts_to_distribution,
    counts_to_signed_distribution,
    estimate_expectation_signed,
)
```

The maintained QFAN API is under `qfun.qfan`:

```python
from qfun.qfan import (
    QFANBlock,
    QFANConfig,
    BenchmarkConfig,
    train_qfan,
    train_feynman_equation,
    run_feynman_benchmark,
)
```

Legacy `QKANBlock` is still available as a compatibility alias.

## Project layout

```text
QFun/
├── qfun/
│   ├── encode.py
│   ├── simulate.py
│   ├── plot.py
│   ├── feynman_dataset.py
│   ├── quantum_learning.py
│   └── qfan/
├── notebooks/
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Development

Run the tests:

```bash
python -m pytest
```

Run Ruff:

```bash
ruff check .
```

## Notes

- Python requirement: `>=3.10`
- The codebase uses PennyLane for circuit construction and simulation.
- Signed-function support is a first-class part of the repo, not an afterthought.
- Notebook 05 is intentionally different from notebook 04:
  - notebook 04 learns a classical QFAN surrogate
  - notebook 05 learns the quantum superposition itself
