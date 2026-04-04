"""Canonical Feynman benchmark runner."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from qfun.qfan import BenchmarkConfig, QFANConfig, run_feynman_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run smoke mode over a small subset")
    parser.add_argument("--output", default="artifacts/benchmarks/feynman")
    args = parser.parse_args()

    bcfg = BenchmarkConfig(quick_mode=args.quick, quick_limit=5 if args.quick else 27)
    mcfg = QFANConfig(input_dim=2, steps=60)
    summary = run_feynman_benchmark(output_dir=args.output, config=bcfg, model_template=mcfg)
    print(f"Benchmark complete: equations={summary['num_equations']} avg_test_mse={summary['avg_test_mse']:.6f}")


if __name__ == "__main__":
    main()
