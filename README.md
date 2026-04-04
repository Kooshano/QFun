# QFun — Quantum Function Encoder

Encode any classical function into a quantum state and recover its shape through
measurement.  QFun goes beyond the textbook requirement that functions be
nonnegative: it implements two strategies — grounded in the physics of
**negative probabilities** — for handling functions that take negative values.

---

## Table of Contents

1. [Overview](#overview)
2. [Physical Motivation](#physical-motivation)
3. [Mathematical Framework](#mathematical-framework)
   - [Standard Amplitude Encoding](#standard-amplitude-encoding)
   - [The Negativity Problem](#the-negativity-problem)
   - [Mode A — Signed Encoding with an Ancilla Qubit](#mode-a--signed-encoding-with-an-ancilla-qubit)
   - [Mode B — Two-Channel Quasi-Probability Decomposition](#mode-b--two-channel-quasi-probability-decomposition)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [API Reference](#api-reference)
7. [Project Structure](#project-structure)
8. [Theory Deep Dive: Negative Probabilities](#theory-deep-dive-negative-probabilities)
9. [References](#references)
10. [License](#license)

---

## Overview

Given a classical function $f : [a, b] \to \mathbb{R}$ and an $n$-qubit
register, QFun:

1. **Discretises** the domain into $2^n$ evenly spaced points
   $x_0, x_1, \dots, x_{2^n - 1}$.
2. **Amplitude-encodes** the function values into a quantum state
   $|\psi\rangle = \sum_i \alpha_i |i\rangle$.
3. **Samples** the state in the computational basis to obtain an empirical
   distribution.
4. **Compares** the measured distribution against the ideal target.

For **nonnegative** functions the mapping is straightforward: set each amplitude
proportional to $\sqrt{f(x_i)}$ and the Born-rule probabilities $|\alpha_i|^2$
reproduce the shape of $f$.

For **signed** functions (those that can be negative), QFun offers two
physically motivated modes that encode the sign information and recover a
**signed quasi-probability distribution** from measurement.

---

## Physical Motivation

### Why Quantum Amplitude Encoding?

A classical vector of $N$ real numbers requires $N$ classical bits of storage.
A quantum state of $n = \log_2 N$ qubits stores the same $N$ numbers as
probability amplitudes:

$$|\psi\rangle = \sum_{i=0}^{N-1} \alpha_i\,|i\rangle, \qquad
\sum_i |\alpha_i|^2 = 1.$$

This **exponential compression** is the starting point of many quantum
algorithms (HHL linear solver, quantum machine learning, quantum Monte Carlo).
The catch is that the only information directly accessible through measurement
is the probability distribution $p(i) = |\alpha_i|^2$ — nonnegative by the
Born rule.

### The Sign Problem

Many functions of interest in physics and engineering are **signed**: wave
functions, Fourier coefficients, financial derivatives, derivatives of loss
landscapes.  The Born rule squashes all sign information:

$$p(i) = |\alpha_i|^2 \geq 0 \quad \text{always.}$$

How, then, can a quantum computer represent something that is negative?  The
answer has a long history in physics, stretching from Wigner (1932) and Dirac
(1942) through Feynman (1987) to modern quantum computing.

---

## Mathematical Framework

### Standard Amplitude Encoding

Given $f(x) \geq 0$ on the grid, define raw amplitudes and normalise:

$$\tilde\alpha_i = \sqrt{f(x_i)}, \qquad
\alpha_i = \frac{\tilde\alpha_i}{\|\tilde\alpha\|_2}.$$

Prepare the state $|\psi\rangle = \sum_i \alpha_i |i\rangle$ using PennyLane's
`AmplitudeEmbedding`.  Repeated measurement in the computational basis yields
counts $n_i$ from which we estimate:

$$\hat p(i) = \frac{n_i}{S}, \qquad S = \text{total shots.}$$

By the Born rule $\hat p(i) \approx |\alpha_i|^2 \propto f(x_i)$, so the
histogram recovers the shape of $f$.

### The Negativity Problem

If $f(x_j) < 0$ for some grid point $j$, the square-root $\sqrt{f(x_j)}$ is
imaginary, and $|\alpha_j|^2 = |f(x_j)|$ loses the sign.  We need a way to
**encode** and **retrieve** the sign through measurement.

QFun implements two complementary strategies.

---

### Mode A — Signed Encoding with an Ancilla Qubit

**Idea.**  Decompose $f$ pointwise into magnitude and sign:

$$f(x_i) = s_i \,|f(x_i)|, \qquad s_i \in \{+1, -1\}.$$

Encode $|f|$ into amplitude magnitudes as before, but attach a single
**ancilla qubit** that records the sign:

$$|\psi\rangle = \sum_{i=0}^{2^n - 1} \alpha_i\,|i\rangle\,|b_i\rangle,
\qquad b_i = \begin{cases} 0 & f(x_i) \geq 0 \\ 1 & f(x_i) < 0 \end{cases}$$

where $\alpha_i = \sqrt{|f(x_i)|}\,/\,\text{norm}$.

This is a valid quantum state in a Hilbert space of dimension $2^{n+1}$ (the
original $n$ data qubits plus 1 ancilla).  The state vector has a specific
structure: for each grid index $i$, amplitude sits at either
$|i\rangle|0\rangle$ or $|i\rangle|1\rangle$, never both.

**Measurement.**  Each shot yields an $(n+1)$-bit string.  The first $n$ bits
identify the grid point; the last bit reveals the sign.  We accumulate two
histograms:

$$p_+(x_i) = \frac{\text{counts with } b=0 \text{ at index } i}{S},
\qquad
p_-(x_i) = \frac{\text{counts with } b=1 \text{ at index } i}{S}.$$

The **signed quasi-probability** is:

$$q(x_i) = p_+(x_i) - p_-(x_i).$$

By construction $p_+ + p_-$ sums to 1 (a proper joint distribution over
$(x, b)$), but $q$ can take negative values — exactly where $f$ was negative.

**Circuit diagram** ($n = 3$, one ancilla):

```
0: ─╭|Ψ⟩─┤ ╭Sample      (data qubit, MSB)
1: ─├|Ψ⟩─┤ ├Sample      (data qubit)
2: ─├|Ψ⟩─┤ ├Sample      (data qubit, LSB)
3: ─╰|Ψ⟩─┤ ╰Sample      (ancilla / sign bit)
```

---

### Mode B — Two-Channel Quasi-Probability Decomposition

**Idea.**  Start from a target **signed distribution** $q(x)$ with
$\sum_x q(x) = 1$ but some entries negative.  Decompose it into positive and
negative parts:

$$q^+(x) = \max\bigl(q(x),\, 0\bigr), \qquad
q^-(x) = \max\bigl(-q(x),\, 0\bigr).$$

Let $Z_+ = \sum_x q^+(x)$ and $Z_- = \sum_x q^-(x)$.  Define two proper
(nonneg, normalised) probability distributions:

$$p_+(x) = \frac{q^+(x)}{Z_+}, \qquad
p_-(x) = \frac{q^-(x)}{Z_-}.$$

Then $q$ is recovered exactly:

$$q(x) = Z_+\, p_+(x) \;-\; Z_-\, p_-(x).$$

**Procedure.**

1. Amplitude-encode $p_+$ into circuit 1 and sample.
2. Amplitude-encode $p_-$ into circuit 2 and sample.
3. Recombine classically: $\hat q(x) = Z_+\,\hat p_+(x) - Z_-\,\hat p_-(x)$.

Expectations under the signed distribution follow the same linear combination:

$$\mathbb{E}_q[g]
= Z_+\,\mathbb{E}_{p_+}[g] \;-\; Z_-\,\mathbb{E}_{p_-}[g]
= Z_+ \sum_x g(x)\,\hat p_+(x) \;-\; Z_- \sum_x g(x)\,\hat p_-(x).$$

This is exactly the paradigm described by Polson and Sokolov (arXiv:2405.03043):
**extraordinary (negative) weights are latent** and only become physically
meaningful when combined with ordinary (positive) weights to yield a valid
observable.

---

## Installation

```bash
git clone <this-repo>
cd QFun
pip install -r requirements.txt
```

Dependencies (all pure-Python / CPU):

| Package      | Role                                |
|-------------|-------------------------------------|
| `numpy`     | Array operations, linear algebra    |
| `matplotlib`| Plotting                            |
| `pennylane` | Quantum circuit simulation (CPU)    |

No GPU or JAX required.

---

## Quick Start

### Nonnegative function (standard mode)

```python
import numpy as np
from qfun import grid_x, amplitudes_from_function, run_shots, counts_to_distribution

f = lambda x: np.exp(-x**2)
x = grid_x(-2, 2, n_qubits=5)                  # 32 grid points
amps = amplitudes_from_function(f, x)            # L2-normalised amplitudes
counts = run_shots(amps, n_qubits=5, shots=50000)
prob = counts_to_distribution(counts, n_qubits=5)
# prob[i] ≈ f(x[i]) / sum(f(x))
```

### Signed function (Mode A — ancilla)

```python
from qfun import (
    grid_x, signed_amplitudes_from_function,
    run_shots_signed, counts_to_signed_distribution,
)

f = lambda x: np.sin(x)
x = grid_x(0, 2 * np.pi, n_qubits=5)
sa = signed_amplitudes_from_function(f, x)

counts = run_shots_signed(sa.amplitudes, sa.sign_mask, n_qubits=5, shots=100_000)
sd = counts_to_signed_distribution(counts, n_qubits=5)

# sd.q is the signed quasi-probability: negative where sin(x) < 0
# sd.p_pos, sd.p_neg are the nonneg components
```

### Signed distribution (Mode B — two-channel)

```python
from qfun import (
    grid_x, decompose_signed_distribution,
    run_two_channel_signed, estimate_expectation_signed,
)

# Build a signed target distribution
x = grid_x(0, 2 * np.pi, n_qubits=5)
f_vals = np.sin(x)
q_target = f_vals / np.sum(np.abs(f_vals))       # normalised signed dist

dec = decompose_signed_distribution(q_target)     # p+, p-, Z+, Z-

p_plus_amps = np.sqrt(dec.p_plus + 1e-12)
p_plus_amps /= np.linalg.norm(p_plus_amps)
p_minus_amps = np.sqrt(dec.p_minus + 1e-12)
p_minus_amps /= np.linalg.norm(p_minus_amps)

res = run_two_channel_signed(
    p_plus_amps, p_minus_amps,
    dec.z_plus, dec.z_minus,
    n_qubits=5, shots=100_000,
)
# res.q_hat ≈ q_target, with negative entries intact

# Estimate E_q[x]
E = estimate_expectation_signed(
    x, res.p_plus_hat, res.p_minus_hat, dec.z_plus, dec.z_minus,
)
```

### Plotting

```python
from qfun import plot_comparison, plot_signed_comparison

# Standard mode
plot_comparison(x, target_prob, empirical_prob, title="cos(x)")

# Signed mode (bars go below zero)
plot_signed_comparison(x, target_q, measured_q, title="sin(x) signed")
```

---

## API Reference

### `qfun.encode`

| Function | Description |
|----------|-------------|
| `grid_x(a, b, n_qubits)` | Return $2^n$ evenly spaced points on $[a, b]$. |
| `amplitudes_from_function(f, x)` | $\alpha_i \propto \sqrt{f(x_i)}$, L2-normalised. Raises if $f < 0$. |
| `signed_amplitudes_from_function(f, x)` | Returns `SignedAmplitudes(amplitudes, sign_mask, norm)` for possibly-negative $f$. |
| `decompose_signed_distribution(q)` | Returns `SignedDecomposition(p_plus, p_minus, z_plus, z_minus)` for a signed distribution. |

### `qfun.simulate`

| Function | Description |
|----------|-------------|
| `build_circuit(amplitudes, n_qubits)` | Return a PennyLane `QNode` for inspection / drawing. |
| `run_shots(amplitudes, n_qubits, shots)` | Standard sampling; returns `{bitstring: count}`. |
| `counts_to_distribution(counts, n_qubits)` | Convert counts dict to probability array of length $2^n$. |
| `run_shots_signed(amplitudes, sign_mask, n_qubits, shots)` | **Mode A**: sample from $(n+1)$-qubit circuit with sign ancilla. |
| `counts_to_signed_distribution(counts, n_qubits)` | **Mode A**: split ancilla-tagged counts into `SignedDistribution(p_pos, p_neg, q)`. |
| `run_two_channel_signed(p_plus_amps, p_minus_amps, z_plus, z_minus, n_qubits, shots)` | **Mode B**: run two circuits, recombine into `TwoChannelResult`. |
| `estimate_expectation_signed(g_values, p_plus_hat, p_minus_hat, z_plus, z_minus)` | **Mode B**: compute $\mathbb{E}_q[g] = Z_+ \mathbb{E}_{p_+}[g] - Z_- \mathbb{E}_{p_-}[g]$. |

### `qfun.plot`

| Function | Description |
|----------|-------------|
| `plot_comparison(x, target, measured)` | Standard bar chart of measured vs target probabilities. |
| `plot_signed_comparison(x, target_q, measured_q)` | Signed bar chart (bars go below zero; blue = positive, coral = negative). |

---

## Project Structure

```
QFun/
├── qfun/
│   ├── __init__.py       # Public API re-exports
│   ├── encode.py         # Discretisation & amplitude construction
│   ├── simulate.py       # PennyLane circuits & sampling
│   └── plot.py           # Visualisation helpers
├── qfun_demo.ipynb       # Interactive notebook with all three modes
├── requirements.txt      # numpy, matplotlib, pennylane
└── README.md             # This file
```

---

## Theory Deep Dive: Negative Probabilities

This section summarises the key ideas from
**Polson & Sokolov, "Negative Probability" (arXiv:2405.03043v2)** that
motivate Modes A and B.

### 1. Extraordinary Random Variables (Bartlett, 1945)

A standard probability distribution satisfies $p(x) \geq 0$ and
$\sum p(x) = 1$.  Bartlett relaxed the first condition: an
**extraordinary random variable** has a signed measure $p^\bullet$ where
some values are negative, but $\sum |p^\bullet(x)| < \infty$ and the
total is still 1.

The critical constraint is that **extraordinary distributions must always be
combined with ordinary ones before physical interpretation**.  In isolation,
a negative probability has no direct physical meaning; it is an intermediate
bookkeeping device — like a negative balance in double-entry accounting.

### 2. Feynman's Conditional Table

Feynman (1987) gave a simple example: a conditional probability table where
one entry is $-0.4$ and another is $1.2$.  Despite these "impossible" values,
when marginalised (combined with the prior weights $p(A)=0.7$, $p(B)=0.3$),
every marginal probability is in $[0, 1]$:

$$p(\text{state}=1) = 0.7 \times 0.3 + 0.3 \times (-0.4) = 0.09.$$

The law of total probability still holds; the negative weight is an
unobservable latent variable.

### 3. The Wigner Distribution

Wigner (1932) showed that the joint distribution of position $x$ and momentum
$p$ of a quantum particle **cannot be nonneg everywhere** — yet its marginals
are valid probability distributions.  The Wigner function is the prototypical
quasi-probability distribution:

$$f_\psi(x, p) = \frac{1}{2\pi}
\int \psi\!\left(x + \tfrac{s}{2}\hbar\right)\,
     \psi^*\!\left(x - \tfrac{s}{2}\hbar\right)\,
     e^{isp}\,ds.$$

It can be negative in parts of phase space, but any observable expectation
value computed from it is real and physical.

### 4. Convolutions with Negative Weights

The paper's central mathematical theme is that many distributions can be
written as **mixtures** (convolutions) with signed mixing weights:

$$f_Y(y) = \int f_{Y|Z}(y \mid z)\,f_Z^\bullet(z)\,dz$$

where the mixing measure $f_Z^\bullet$ may take negative values.  The
observable $f_Y$ is an ordinary distribution.  This is precisely the structure
of Mode B: we prepare two ordinary quantum distributions $p_+$ and $p_-$
(the positive and negative mixing components) and combine them classically
with weights $Z_+$ and $-Z_-$.

### 5. Heisenberg Uncertainty and Dual Densities

The paper also discusses **dual densities** (Good, 1995; Gneiting, 1997):
given a density $p(x)$ that is a scale mixture of normals, its Fourier
transform (characteristic function) $\hat p$ is also a scale mixture of
normals, but potentially with **negative** mixing weights.  The two densities
satisfy a Heisenberg-style uncertainty relation:

$$\sigma_p \,\sigma_{\hat p} \geq 1,$$

with equality only for Gaussian $p$.  This duality — one "ordinary", one
"extraordinary" — is a recurring motif.  QFun's Mode A makes this concrete:
the positive and negative histograms are individually ordinary; only their
**difference** yields the extraordinary signed distribution.

### 6. Connection to QFun

| Paper concept | QFun implementation |
|---------------|---------------------|
| Extraordinary random variable $Z^\bullet$ | Signed quasi-probability $q(x)$ |
| Mixing weights can be negative | Bars in `plot_signed_comparison` go below zero |
| Must combine with ordinary distributions | Mode A: $q = p_+ - p_-$ from a joint measurement; Mode B: $\hat q = Z_+ \hat p_+ - Z_- \hat p_-$ from two ordinary circuits |
| Law of total probability still holds | $\sum q(x) = Z_+ - Z_- = 1$ (or whatever the original normalisation) |
| Expectation under signed measure | `estimate_expectation_signed`: $\mathbb{E}_q[g] = Z_+ \mathbb{E}_{p_+}[g] - Z_- \mathbb{E}_{p_-}[g]$ |

### 7. When to Use Which Mode

| | Mode A (ancilla) | Mode B (two-channel) |
|---|---|---|
| **Input** | A function $f(x)$ that can be negative | A pre-computed signed distribution $q(x)$ |
| **Circuits** | 1 circuit with $n + 1$ qubits | 2 circuits with $n$ qubits each |
| **Sign recovery** | From ancilla measurement | From classical recombination of two runs |
| **Best for** | Exploring / visualising a signed function | Computing expectations, integrating observables |
| **Shot efficiency** | All shots go to one circuit | Shots split across two circuits |

---

## References

1. **N. Polson and V. Sokolov**, "Negative Probability,"
   *arXiv:2405.03043v2* [quant-ph], 2024.
   — The paper motivating QFun's signed-function support.

2. **E. Wigner**, "On the Quantum Correction For Thermodynamic Equilibrium,"
   *Physical Review* **40**(5):749--759, 1932.
   — Introduction of the Wigner quasi-probability distribution.

3. **P. A. M. Dirac**, "Bakerian Lecture — The Physical Interpretation of
   Quantum Mechanics," *Proc. Royal Soc. London A* **180**(980):1--40, 1942.
   — "Negative energies and probabilities should not be considered as nonsense."

4. **R. P. Feynman**, "Negative Probability," in *Quantum Implications: Essays
   in Honour of David Bohm*, pp. 235--248, 1987.
   — The conditional-table example and diffusion-equation example.

5. **M. S. Bartlett**, "Negative Probability," *Math. Proc. Cambridge Phil.
   Soc.* **41**:71--73, 1945.
   — Formal definition of extraordinary random variables via characteristic
   functions.

6. **T. Gneiting**, "Normal Scale Mixtures and Dual Probability Densities,"
   *J. Stat. Comput. Simul.* **59**(4):375--384, 1997.
   — Dual densities as Fourier transforms of scale mixtures; Heisenberg
   uncertainty for density pairs.

7. **I. J. Good**, "Dual Density Functions," *J. Stat. Comput. Simul.*
   **52**(2):193--194, 1995.
   — Original introduction of dual densities.

---

## License

This project is provided for educational and research purposes.
