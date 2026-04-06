"""Optional JAX/accelerator training for :class:`QuantumActivationClassifier`.

Install JAX (and the matching CUDA build for GPU) plus Optax, then set
``QuantumActivationConfig(use_jax=True)``. Example::

    pip install "qfun[gpu]"

For NVIDIA GPUs, follow the official JAX install guide so ``jaxlib`` matches
your CUDA version.

Without JAX, ``use_jax`` must remain ``False`` (default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._utils import EPS

if TYPE_CHECKING:
    from .quantum_activation_classifier import QuantumActivationClassifier, QuantumActivationConfig

try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    optax = None  # type: ignore[assignment]


def jax_ready() -> bool:
    return jax is not None and optax is not None


def _require_jax() -> None:
    if not jax_ready():
        raise ImportError(
            "JAX and Optax are required for GPU/accelerated training. "
            'Install with: pip install "qfun[gpu]" '
            "or pip install jax optax (and a CUDA-enabled jaxlib for GPU)."
        )


def _norm_amp(raw: Any, eps: float = EPS) -> Any:
    out = raw / jnp.sqrt(jnp.sum(raw**2) + eps)
    return jnp.real(out)


def _profiles_standard(raw_profiles: Any, num_grid_points: int, eps: float = EPS) -> Any:
    amps = jax.vmap(lambda r: _norm_amp(r, eps))(raw_profiles)
    return num_grid_points * (amps**2)


def _profiles_mode_a(raw_profiles: Any, num_grid_points: int, eps: float = EPS) -> Any:
    def one(r: Any) -> Any:
        amps = _norm_amp(r, eps)
        fp = amps**2
        q = fp[0::2] - fp[1::2]
        return num_grid_points * q

    return jax.vmap(one)(raw_profiles)


def _profiles_mode_b(
    raw_plus: Any,
    raw_minus: Any,
    raw_logits: Any,
    num_grid_points: int,
    eps: float = EPS,
) -> Any:
    def one(rp: Any, rm: Any, lg: Any) -> Any:
        pp = _norm_amp(rp, eps) ** 2
        pm = _norm_amp(rm, eps) ** 2
        z = jax.nn.softmax(lg)
        q = z[0] * pp - z[1] * pm
        return num_grid_points * q

    return jax.vmap(one)(raw_plus, raw_minus, raw_logits)


def _interp_hidden(z_bh: Any, profiles_hg: Any, x_grid: Any) -> Any:
    """Linear interp: z (B,H), profiles (H,G), x_grid (G,) -> hidden (B,H)."""

    def col_interp(z_b: Any, prof_g: Any) -> Any:
        return jnp.interp(z_b, x_grid, prof_g)

    return jax.vmap(col_interp, in_axes=(1, 0), out_axes=1)(z_bh, profiles_hg)


def _forward_standard(
    params: tuple[Any, ...],
    x: Any,
    num_grid_points: int,
    x_grid: Any,
    eps: float,
) -> Any:
    w_in, b_in, w_out, b_out, raw_profiles = params
    z = jnp.tanh(x @ w_in.T + b_in)
    prof = _profiles_standard(raw_profiles, num_grid_points, eps)
    hidden = _interp_hidden(z, prof, x_grid)
    return hidden @ w_out.T + b_out


def _forward_mode_a(
    params: tuple[Any, ...],
    x: Any,
    num_grid_points: int,
    x_grid: Any,
    eps: float,
) -> Any:
    w_in, b_in, w_out, b_out, raw_profiles = params
    z = jnp.tanh(x @ w_in.T + b_in)
    prof = _profiles_mode_a(raw_profiles, num_grid_points, eps)
    hidden = _interp_hidden(z, prof, x_grid)
    return hidden @ w_out.T + b_out


def _forward_mode_b(
    params: tuple[Any, ...],
    x: Any,
    num_grid_points: int,
    x_grid: Any,
    eps: float,
) -> Any:
    w_in, b_in, w_out, b_out, raw_plus, raw_minus, raw_logits = params
    z = jnp.tanh(x @ w_in.T + b_in)
    prof = _profiles_mode_b(raw_plus, raw_minus, raw_logits, num_grid_points, eps)
    hidden = _interp_hidden(z, prof, x_grid)
    return hidden @ w_out.T + b_out


def _make_forward(mode: str):
    if mode == "standard":
        return _forward_standard
    if mode == "mode_a":
        return _forward_mode_a
    if mode == "mode_b":
        return _forward_mode_b
    raise ValueError(f"Unknown mode {mode!r}")


def _init_params(key: Any, config: QuantumActivationConfig) -> tuple[Any, ...]:
    H = config.hidden_units
    D = config.input_dim
    C = config.n_classes
    G = 2**config.n_qubits
    keys = jax.random.split(key, 32)
    k = 0

    def ksplit(n: int = 1):
        nonlocal k
        out = keys[k : k + n]
        k += n
        return out[0] if n == 1 else out

    w_in = jax.random.normal(ksplit(), (H, D)) * 0.35
    b_in = jax.random.normal(ksplit(), (H,)) * 0.05
    w_out = jax.random.normal(ksplit(), (C, H)) * 0.35
    b_out = jnp.zeros((C,), dtype=jnp.float64)

    if config.mode == "standard":
        raw = jax.random.normal(ksplit(), (H, G)) * 0.25
        return (w_in, b_in, w_out, b_out, raw)
    if config.mode == "mode_a":
        raw = jax.random.normal(ksplit(), (H, 2 * G)) * 0.25
        return (w_in, b_in, w_out, b_out, raw)
    raw_plus = jax.random.normal(ksplit(), (H, G)) * 0.25
    raw_minus = jax.random.normal(ksplit(), (H, G)) * 0.25
    raw_logits = jax.random.normal(ksplit(), (H, 2)) * 0.05
    return (w_in, b_in, w_out, b_out, raw_plus, raw_minus, raw_logits)


def _params_to_model(
    model: QuantumActivationClassifier,
    params: tuple[Any, ...],
) -> None:
    """Copy trained JAX parameters (as numpy) into a PennyLane model."""
    import pennylane.numpy as pnp

    w_in, b_in, w_out, b_out = (np.asarray(x, dtype=float) for x in params[:4])
    model.W_in = pnp.array(w_in, requires_grad=True)
    model.b_in = pnp.array(b_in, requires_grad=True)
    model.W_out = pnp.array(w_out, requires_grad=True)
    model.b_out = pnp.array(b_out, requires_grad=True)

    if model.mode in {"standard", "mode_a"}:
        rp = np.asarray(params[4], dtype=float)
        model.raw_profiles = pnp.array(rp, requires_grad=True)
        return
    rp, rm, rl = (np.asarray(x, dtype=float) for x in params[4:7])
    model.raw_plus = pnp.array(rp, requires_grad=True)
    model.raw_minus = pnp.array(rm, requires_grad=True)
    model.raw_channel_logits = pnp.array(rl, requires_grad=True)


def train_quantum_activation_classifier_jax(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: QuantumActivationConfig,
    *,
    after_step: Any | None = None,
    log_every: int | None = None,
) -> tuple[QuantumActivationClassifier, np.ndarray]:
    """Train with Optax on JAX (CPU or GPU); return PennyLane model + loss history."""
    _require_jax()
    jax.config.update("jax_enable_x64", True)

    from .quantum_activation_classifier import QuantumActivationClassifier

    x = jnp.asarray(np.asarray(x_train, dtype=np.float64))
    y = np.asarray(y_train, dtype=np.int64)
    n = x.shape[0]
    bs = max(1, int(config.batch_size))
    y_eye = jnp.asarray(np.eye(config.n_classes, dtype=np.float64)[y])

    num_grid_points = 2**config.n_qubits
    x_grid = jnp.linspace(-1.0, 1.0, num_grid_points)
    forward = _make_forward(config.mode)

    key = jax.random.PRNGKey(config.seed)
    key, k_init = jax.random.split(key)
    params = _init_params(k_init, config)

    def loss_batch(p: Any, xb: Any, yb: Any) -> Any:
        logits = forward(p, xb, num_grid_points, x_grid, EPS)
        logp = logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)
        return -jnp.mean(jnp.sum(yb * logp, axis=1))

    loss_grad = jax.value_and_grad(loss_batch)

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(p: Any, state: Any, xb: Any, yb: Any) -> tuple[Any, Any, Any]:
        loss, g = loss_grad(p, xb, yb)
        upd, new_state = optimizer.update(g, state)
        new_p = optax.apply_updates(p, upd)
        return new_p, new_state, loss

    model = QuantumActivationClassifier(config)
    losses_out: list[float] = []

    def full_data_loss(p: Any) -> float:
        loss_acc = 0.0
        for start in range(0, n, bs):
            sl = slice(start, min(start + bs, n))
            loss_acc += float(loss_batch(p, x[sl], y_eye[sl])) * (sl.stop - sl.start)
        return loss_acc / n

    if after_step is not None:
        _params_to_model(model, params)
        after_step(-1, full_data_loss(params), model)

    log_n = log_every if log_every is not None and log_every > 0 else 0
    if log_n:
        print(
            f"Training {config.steps} epochs on JAX ({jax.default_backend()}), "
            f"batch_size={bs}…",
            flush=True,
        )

    rng = np.random.default_rng(config.seed)

    for step in range(config.steps):
        perm = rng.permutation(n)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            if idx.size == 0:
                break
            xb = x[idx]
            yb = y_eye[idx]
            if idx.size == bs:
                params, opt_state, _ = train_step(params, opt_state, xb, yb)
            else:
                _, g = loss_grad(params, xb, yb)
                updates, opt_state = optimizer.update(g, opt_state)
                params = optax.apply_updates(params, updates)

        loss_f = full_data_loss(params)
        losses_out.append(loss_f)
        _params_to_model(model, params)

        if after_step is not None:
            after_step(step, loss_f, model)
        if log_n and ((step + 1) % log_n == 0 or step == config.steps - 1):
            print(
                f"  epoch {step + 1}/{config.steps}  train_loss={loss_f:.6f}",
                flush=True,
            )

    return model, np.asarray(losses_out, dtype=float)
