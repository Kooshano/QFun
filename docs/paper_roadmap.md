# Paper Roadmap

This document captures the paper-oriented research roadmap from the project
plan and turns it into a concrete in-repo reference.

## Best Ideas For QFun

### Tier 1

- Multi-seed ablation tables: run 3-5 seeds per configuration and report mean +/- std over `mode`, `n_qubits`, `hidden_units`, and `profile_interp`. This is the minimum bar for a paper-ready table.
- Edge-level learned activations: move from one learned profile per hidden unit to one learned profile per input-to-hidden edge. This is the clearest path toward a more KAN-like architecture.
- Progressive grid refinement: train on a coarse activation grid, then interpolate onto a finer grid mid-training. This gives QFun a strong optimization story that matches KAN-style grid extension.

### Tier 2

- Multivariate activation profiles: extend 1D profiles to small 2D activation grids using the existing `grid_nd` machinery.
- Function-fitting benchmark track: reuse the Feynman benchmark path to compare direct amplitude encoding and learned superposition fitting.
- Interpretability analysis: cluster learned activation profiles, compare them with familiar activation shapes, and measure whether flat units can be pruned.

### Tier 3

- Compositional two-level architecture: stack learned profile layers while shrinking reliance on dense weight matrices in later stages.
- Hardware measurement experiments: run learned state-preparation profiles on real hardware and compare measured activations with simulator outputs.

## Priority Roadmap

### Phase 0: Cleanup

- Commit and track the maintained interpolation utilities.
- Keep generated experiment outputs ignored.
- Preserve only the maintained notebooks and canonical experiment scripts.
- Keep the docs and experiments split clean.

### Phase 1: Research Foundation

- Build a multi-seed experiment runner.
- Run the first ablation grid over `n_qubits in {3,4,5}`, `hidden_units in {4,6,8}`, `profile_interp in {linear, cubic_natural}`, and `mode in {standard, mode_a, mode_b}`.
- Produce paper-style mean/std tables.
- Keep tests in sync with the maintained notebooks and encoding surface.

### Phase 2: Architecture Improvements

- Implement edge-level activations.
- Implement progressive grid refinement.
- Re-run the best variants on Fashion-MNIST and at least one non-image benchmark.

### Phase 3: Paper Preparation

- Investigate why spline readout underperforms.
- Add interpretability analysis and pruning-style summaries.
- Produce the main figures: architecture, ablation heatmap, profile evolution, and measurement agreement.
- Draft the paper around the most stable ablation story.

## Experiments To Run Next

- Multi-seed MNIST ablation to establish reliable baselines.
- Edge-level activations on Iris or Digits before scaling to MNIST.
- Fashion-MNIST with the best Phase 1 configuration.
- Grid refinement from `n_qubits=3` to `n_qubits=5`.
- Learned-profile clustering and activation-shape comparisons.
