# QFun Paper Blueprint

## Working Thesis

QFun learns activation functions from quantum-state amplitude profiles. The
paper should frame this as a practical bridge between KAN-style learnable
functions and Born-rule quantum parameterizations, not as a quantum-advantage or
MNIST-SOTA claim.

Proposed title:

**Born-Rule Activation Networks: Quantum-State Parameterized Functions for
Interpretable KAN-Style Learning**

## Positioning

The core novelty is the combination of:

- Born-rule activation profiles with three realizations: nonnegative
  probability profiles, ancilla-derived signed profiles, and two-channel signed
  profiles.
- Node-level, KAN-hybrid, edge-level, and shared-edge ablations in one
  reproducible codebase.
- Qubit/grid refinement as a quantum analogue of KAN grid extension.
- Finite-shot measurement verification for trained activation profiles.
- Profile-level interpretability: taxonomy, importance, clustering, and pruning
  sensitivity.

Do not claim that QFun invents learnable activations. KANs already established
learnable spline functions on graph edges. Do not claim quantum advantage unless
hardware-backed scaling evidence is produced.

## Related Work Anchors

- KANs introduce learnable edge functions and spline-based grid extension:
  <https://arxiv.org/abs/2404.19756>
- KAN 2.0 / MultKAN focuses on scientific discovery, modularity, and
  multiplication nodes: <https://journals.aps.org/prx/accepted/10.1103/4t7t-v19l>
- QKAN formulates quantum Kolmogorov-Arnold networks through quantum algorithmic
  primitives and reports no public code/datasets in the paper:
  <https://www.nature.com/articles/s41534-026-01202-5>
- QuKAN uses quantum circuit Born machines for quantum KAN residual functions on
  small classification/regression problems:
  <https://www.nature.com/articles/s41598-025-22705-9>
- Quantum Circuit Born Machines support the Born-rule generative framing:
  <https://www.nature.com/articles/s41534-019-0157-8>
- Quantum trainability work cautions against overclaiming; report optimization
  limits and barren-plateau risks honestly:
  <https://www.nature.com/articles/s42254-025-00813-9>

## Required Experiments

Use `experiments/run_mnist_paper_suite.py` as the canonical runner.

### Smoke Check

Run this before any long experiment:

```bash
python experiments/run_mnist_paper_suite.py --preset smoke --output-root artifacts/paper_suite/smoke_latest
```

Expected outputs:

- `manifest.json`
- `per_run_results.csv`
- `aggregate_results.csv`
- `interpretability_results.json`
- `shot_budget_results.json`

### Core Development Grid

Use this grid to choose winner configurations before spending full MNIST compute:

```bash
python experiments/run_mnist_paper_suite.py \
  --preset core \
  --with-interpretability \
  --prune-fraction 0.3 \
  --output-root artifacts/paper_suite/core_digits_latest
```

### Paper Grid

Use this only after the smoke and core grids are clean:

```bash
python experiments/run_mnist_paper_suite.py \
  --preset paper \
  --use-jax auto \
  --with-interpretability \
  --with-shot-budget \
  --prune-fraction 0.3 \
  --max-shot-samples 50 \
  --output-root artifacts/paper_suite/paper_latest
```

The full paper grid may download MNIST and Fashion-MNIST from OpenML on first
use and can be expensive. If the full grid is too slow, first run:

```bash
python experiments/run_mnist_paper_suite.py \
  --preset paper \
  --datasets digits,mnist \
  --families node,hybrid \
  --limit-configs 18 \
  --output-root artifacts/paper_suite/paper_pilot_latest
```

## Tables And Figures

Main paper tables:

- Dataset/configuration table: dataset, split, PCA dimension, seeds, classes.
- Accuracy table: mean/std test accuracy and macro-F1 per family/mode/grid.
- Parameter table: parameter count versus MLP/KAN/QFun node/QFun edge.
- Measurement table: shot budget, accuracy, KL, total variation, Hellinger.

Main paper figures:

- Architecture: pixel/tabular input -> PCA -> QFun activation family -> logits.
- Ablation heatmap: family x mode x qubit resolution.
- Learned profile evolution: activation curves over training checkpoints.
- Grid-refinement curve: coarse-to-fine qubit schedule versus fixed-grid run.
- Exact versus finite-shot measured profiles.
- Interpretability figure: profile taxonomy and pruning sensitivity.

## Manuscript Outline

1. **Introduction**: learned functions are useful but classical splines are not
   the only parameterization; Born-rule profiles provide a quantum-realizable
   alternative with interpretable learned shapes.
2. **Background**: KANs, learnable activations, Born-rule models, finite-shot
   measurement, and quantum trainability risks.
3. **Method**: define standard, mode-A, and mode-B profiles; describe node,
   hybrid, edge, and shared-edge variants; define interpolation and grid
   refinement.
4. **Experiments**: report smoke/core/paper grids, baselines, parameter counts,
   and dataset protocols.
5. **Measurement Realizability**: compare exact profiles to finite-shot
   measured profiles and quantify inference degradation.
6. **Interpretability**: profile taxonomy, clustering, importance, and pruning.
7. **Limitations**: no quantum advantage claim, PCA-compressed image inputs,
   compute limits, and optimizer sensitivity.
8. **Conclusion**: QFun is a reproducible empirical platform for
   quantum-realizable learned activation functions.

## Acceptance Bar

The paper is worth drafting if at least one of these holds:

- QFun matches or beats parameter-comparable MLP/KAN-style baselines on Digits
  and at least one MNIST-family dataset within reasonable variance.
- QFun underperforms but shows a convincing interpretability/measurement story:
  stable learned profiles, finite-shot convergence, and low pruning sensitivity.
- Edge/shared-edge QFun gives a clear accuracy or parameter-efficiency gain over
  node-level QFun on at least one nontrivial dataset.

If none of these holds, the right output is a negative-results technical report,
not a "groundbreaking" paper.
