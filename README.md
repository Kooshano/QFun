# QFun with QFAN (Quantum Function Approximation Network)

This repository now uses **one canonical runtime surface: `qfun`**.

## Canonical structure

- `qfun/` – package code and APIs.
  - `qfun/qfan/` – maintained QFAN implementation:
    - encoding helpers
    - signed modes
    - model block
    - training utilities
    - Feynman dataset adapters
    - benchmark runner
- `scripts/` – canonical runnable entries:
  - `train_qfan_demo.py` (toy/demo training)
  - `run_feynman_benchmark.py` (Feynman benchmark: quick or full)
- `notebooks/` – supported notebooks only:
  1. QFun basics and signed modes
  2. Feynman dataset + encoding walkthrough
  3. QFAN training + benchmark walkthrough
- `archive/notebooks/` – unsupported historical notebooks
- `references/` – reference PDFs
- `paper/` – paper sources

## Naming

`QFAN` is the canonical name (**Quantum Function Approximation Network**).
Legacy `QKAN` names are retained only as tiny compatibility aliases.

## Canonical usage

Install deps:

```bash
pip install -r requirements.txt
```

Toy/demo training:

```bash
python scripts/train_qfan_demo.py
```

Quick benchmark smoke test:

```bash
python scripts/run_feynman_benchmark.py --quick
```

Full 27-equation benchmark:

```bash
python scripts/run_feynman_benchmark.py --output artifacts/benchmarks/feynman_full
```

Outputs are written as structured JSON summaries in the chosen output directory.

## Public interface

Use `qfun` imports directly:

```python
from qfun.qfan import QFANBlock, QFANConfig, BenchmarkConfig, train_qfan, run_feynman_benchmark
```

A minimal transitional shim remains at `qfun_layers/`, but it is not a second maintained implementation surface.
