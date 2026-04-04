"""Reproducible QFun-KAN training demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from qfun_layers.qkan_block import QKANBlock
from qfun_layers.signed_encoding import mode_a_signed_encode, reconstruct_mode_a_signed


ARTIFACT_DIR = Path("artifacts/qkan_demo")


def target_function(x: np.ndarray) -> np.ndarray:
    return np.sin(x) + 0.3 * x


def make_dataset(n_samples: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    y = target_function(x[:, 0])
    return pnp.array(x), pnp.array(y)


def train_model(
    num_functions: int = 3,
    n_qubits: int = 5,
    mode: str = "mode_a",
    n_samples: int = 128,
    steps: int = 150,
    lr: float = 0.05,
):
    x_train, y_train = make_dataset(n_samples=n_samples)
    model = QKANBlock(input_dim=1, num_functions=num_functions, n_qubits=n_qubits, mode=mode)
    opt = qml.AdamOptimizer(stepsize=lr)

    def loss_fn(a_m, b_m, c_m, grid_values):
        model.a_m = a_m
        model.b_m = b_m
        model.c_m = c_m
        model.grid_values = grid_values
        pred = model.forward_batch(x_train)
        return pnp.mean((pred - y_train) ** 2)

    params = model.parameters()
    losses = []
    for _ in range(steps):
        params, loss_val = opt.step_and_cost(loss_fn, *params)
        losses.append(float(loss_val))

    model.a_m, model.b_m, model.c_m, model.grid_values = params
    return model, x_train, y_train, np.array(losses)


def _build_mode_a_state(alpha: np.ndarray, sign_bits: np.ndarray) -> np.ndarray:
    state = np.zeros(alpha.size * 2, dtype=float)
    for i, amp in enumerate(alpha):
        state[2 * i + (1 if sign_bits[i] else 0)] = amp
    return state


def mode_a_statevector_and_histograms(f_vals: np.ndarray, n_qubits: int, shots: int = 4000):
    alpha, sign_bits = mode_a_signed_encode(f_vals)
    state = _build_mode_a_state(alpha, sign_bits)
    n_wires = n_qubits + 1

    dev_sv = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev_sv)
    def state_circuit():
        qml.AmplitudeEmbedding(state, wires=range(n_wires), normalize=False)
        return qml.state()

    psi = np.asarray(state_circuit())
    probs = np.abs(psi) ** 2
    p_pos = probs[0::2]
    p_neg = probs[1::2]

    dev_shot = qml.device("default.qubit", wires=n_wires, shots=shots)

    @qml.qnode(dev_shot)
    def shot_circuit():
        qml.AmplitudeEmbedding(state, wires=range(n_wires), normalize=False)
        return qml.sample(wires=range(n_wires))

    samples = np.asarray(shot_circuit())
    hist_pos = np.zeros(2**n_qubits)
    hist_neg = np.zeros(2**n_qubits)
    for row in samples:
        bits = "".join(str(int(b)) for b in row)
        idx = int(bits[:-1], 2)
        if bits[-1] == "0":
            hist_pos[idx] += 1
        else:
            hist_neg[idx] += 1
    hist_pos /= shots
    hist_neg /= shots
    return p_pos, p_neg, hist_pos, hist_neg, reconstruct_mode_a_signed(alpha, sign_bits)


def save_artifacts(model: QKANBlock, x_train, y_train, losses: np.ndarray):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("Training loss")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    x_line = np.linspace(-1.0, 1.0, 256).reshape(-1, 1)
    y_true = target_function(x_line[:, 0])
    y_pred = np.asarray(model.forward_batch(pnp.array(x_line)))
    plt.figure(figsize=(6, 4))
    plt.scatter(np.asarray(x_train[:, 0]), np.asarray(y_train), s=12, alpha=0.4, label="train")
    plt.plot(x_line[:, 0], y_true, lw=2, label="target")
    plt.plot(x_line[:, 0], y_pred, lw=2, label="qkan")
    plt.legend()
    plt.title("Target vs prediction")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "target_vs_prediction.png", dpi=150)
    plt.close()

    for m in range(model.num_functions):
        profile = np.asarray(model.get_profile(m))
        plt.figure(figsize=(6, 3))
        plt.plot(np.asarray(model.x_grid), profile)
        plt.title(f"Learned grid profile m={m}")
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / f"learned_grid_m{m}.png", dpi=150)
        plt.close()

        p_pos, p_neg, h_pos, h_neg, q_hat = mode_a_statevector_and_histograms(np.asarray(model.grid_values[m]), model.n_qubits)
        xg = np.asarray(model.x_grid)

        plt.figure(figsize=(6, 3))
        plt.plot(xg, p_pos, label="exact p+")
        plt.plot(xg, p_neg, label="exact p-")
        plt.plot(xg, q_hat, label="q_hat")
        plt.legend()
        plt.title(f"Mode A exact profile m={m}")
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / f"mode_a_exact_m{m}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 3))
        plt.bar(xg, h_pos, width=(xg[1] - xg[0]) * 0.8, alpha=0.6, label="sample p+")
        plt.bar(xg, -h_neg, width=(xg[1] - xg[0]) * 0.8, alpha=0.6, label="sample -p-")
        plt.legend()
        plt.title(f"Ancilla histogram m={m}")
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / f"ancilla_hist_m{m}.png", dpi=150)
        plt.close()


def main():
    model, x_train, y_train, losses = train_model()
    save_artifacts(model, x_train, y_train, losses)
    print(f"Done. Initial loss={losses[0]:.6f}, final loss={losses[-1]:.6f}")
    print(f"Artifacts written to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
