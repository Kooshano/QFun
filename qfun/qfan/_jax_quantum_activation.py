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


def _silu(x: Any) -> Any:
    return x * jax.nn.sigmoid(x)


def _interp_hidden(z_bh: Any, profiles_hg: Any, x_grid: Any) -> Any:
    """Linear interp: z (B,H), profiles (H,G), x_grid (G,) -> hidden (B,H)."""

    def col_interp(z_b: Any, prof_g: Any) -> Any:
        return jnp.interp(z_b, x_grid, prof_g)

    return jax.vmap(col_interp, in_axes=(1, 0), out_axes=1)(z_bh, profiles_hg)


def _activation_offset(mode: str) -> int:
    return 5 if mode in {"standard", "mode_a"} else 7


def _layer_profiles(mode: str, params: tuple[Any, ...], layer_idx: int, num_grid_points: int, eps: float) -> Any:
    if mode == "standard":
        raw_profiles = params[4]
        return _profiles_standard(raw_profiles[layer_idx], num_grid_points, eps)
    if mode == "mode_a":
        raw_profiles = params[4]
        return _profiles_mode_a(raw_profiles[layer_idx], num_grid_points, eps)
    raw_plus, raw_minus, raw_logits = params[4:7]
    return _profiles_mode_b(raw_plus[layer_idx], raw_minus[layer_idx], raw_logits[layer_idx], num_grid_points, eps)


def _branch_scales(
    mode: str,
    family: str,
    params: tuple[Any, ...],
    layer_idx: int,
) -> tuple[Any | None, Any | None]:
    if family != "kan_quantum_hybrid":
        return None, None
    offset = _activation_offset(mode)
    base_mix_layers = params[offset]
    quantum_mix_layers = params[offset + 1]
    return base_mix_layers[layer_idx], quantum_mix_layers[layer_idx]


def _smoothness_penalty(
    params: tuple[Any, ...],
    mode: str,
    family: str,
    num_layers: int,
    num_grid_points: int,
    eps: float,
) -> Any:
    if family != "kan_quantum_hybrid":
        return jnp.array(0.0, dtype=jnp.float64)
    penalties = []
    for layer_idx in range(num_layers):
        profiles = _layer_profiles(mode, params, layer_idx, num_grid_points, eps)
        _, quantum_mix = _branch_scales(mode, family, params, layer_idx)
        assert quantum_mix is not None
        effective = quantum_mix[:, None] * profiles
        diffs = effective[:, 1:] - effective[:, :-1]
        penalties.append(jnp.mean(diffs**2))
    if not penalties:
        return jnp.array(0.0, dtype=jnp.float64)
    return jnp.mean(jnp.stack(penalties))


def _forward(
    params: tuple[Any, ...],
    x: Any,
    mode: str,
    family: str,
    num_grid_points: int,
    x_grid: Any,
    eps: float,
    *,
    tanh_preactivation: bool,
) -> Any:
    hidden_weights, hidden_biases, w_out, b_out = params[:4]
    hidden = x
    for layer_idx, (weights, biases) in enumerate(zip(hidden_weights, hidden_biases)):
        z = hidden @ weights.T + biases
        z_quantum = jnp.tanh(z) if tanh_preactivation else z
        profiles = _layer_profiles(mode, params, layer_idx, num_grid_points, eps)
        quantum_hidden = _interp_hidden(z_quantum, profiles, x_grid)
        if family == "kan_quantum_hybrid":
            base_mix, quantum_mix = _branch_scales(mode, family, params, layer_idx)
            assert base_mix is not None
            assert quantum_mix is not None
            hidden = base_mix.reshape((1, -1)) * _silu(z) + quantum_mix.reshape((1, -1)) * quantum_hidden
        else:
            hidden = quantum_hidden
    return hidden @ w_out.T + b_out


def _init_params(key: Any, config: QuantumActivationConfig) -> tuple[Any, ...]:
    hidden_layers = config.resolved_hidden_layers()
    hidden_weights: list[Any] = []
    hidden_biases: list[Any] = []
    raw_profiles: list[Any] = []
    raw_plus: list[Any] = []
    raw_minus: list[Any] = []
    raw_logits: list[Any] = []
    base_mix: list[Any] = []
    quantum_mix: list[Any] = []

    prev_dim = config.input_dim
    key_count = max(4 * len(hidden_layers) + 4, 8)
    keys = list(jax.random.split(key, key_count))
    key_idx = 0

    def take_key() -> Any:
        nonlocal key_idx
        result = keys[key_idx]
        key_idx += 1
        return result

    for width in hidden_layers:
        hidden_weights.append(jax.random.normal(take_key(), (width, prev_dim)) * 0.35)
        hidden_biases.append(jax.random.normal(take_key(), (width,)) * 0.05)
        if config.mode == "standard":
            raw_profiles.append(jax.random.normal(take_key(), (width, 2**config.n_qubits)) * 0.25)
        elif config.mode == "mode_a":
            raw_profiles.append(jax.random.normal(take_key(), (width, 2 * (2**config.n_qubits))) * 0.25)
        else:
            raw_plus.append(jax.random.normal(take_key(), (width, 2**config.n_qubits)) * 0.25)
            raw_minus.append(jax.random.normal(take_key(), (width, 2**config.n_qubits)) * 0.25)
            raw_logits.append(jax.random.normal(take_key(), (width, 2)) * 0.05)
        if config.hidden_function_family == "kan_quantum_hybrid":
            base_mix.append(jnp.ones((width,), dtype=jnp.float64))
            quantum_mix.append(jnp.ones((width,), dtype=jnp.float64))
        prev_dim = width

    w_out = jax.random.normal(take_key(), (config.n_classes, hidden_layers[-1])) * 0.35
    b_out = jnp.zeros((config.n_classes,), dtype=jnp.float64)

    if config.mode in {"standard", "mode_a"}:
        params: tuple[Any, ...] = (tuple(hidden_weights), tuple(hidden_biases), w_out, b_out, tuple(raw_profiles))
    else:
        params = (
            tuple(hidden_weights),
            tuple(hidden_biases),
            w_out,
            b_out,
            tuple(raw_plus),
            tuple(raw_minus),
            tuple(raw_logits),
        )
    if config.hidden_function_family == "kan_quantum_hybrid":
        params = params + (tuple(base_mix), tuple(quantum_mix))
    return params


def _params_to_model(
    model: QuantumActivationClassifier,
    params: tuple[Any, ...],
) -> None:
    """Copy trained JAX parameters (as numpy) into a PennyLane model."""
    import pennylane.numpy as pnp

    hidden_weights = [pnp.array(np.asarray(x, dtype=float), requires_grad=True) for x in params[0]]
    hidden_biases = [pnp.array(np.asarray(x, dtype=float), requires_grad=True) for x in params[1]]
    w_out = pnp.array(np.asarray(params[2], dtype=float), requires_grad=True)
    b_out = pnp.array(np.asarray(params[3], dtype=float), requires_grad=True)

    if model.mode in {"standard", "mode_a"}:
        raw_profiles = [pnp.array(np.asarray(x, dtype=float), requires_grad=True) for x in params[4]]
        activation_params: list[Any] = [*raw_profiles]
        next_offset = 5
    else:
        raw_plus = [pnp.array(np.asarray(x, dtype=float), requires_grad=True) for x in params[4]]
        raw_minus = [pnp.array(np.asarray(x, dtype=float), requires_grad=True) for x in params[5]]
        raw_logits = [pnp.array(np.asarray(x, dtype=float), requires_grad=True) for x in params[6]]
        activation_params = [*raw_plus, *raw_minus, *raw_logits]
        next_offset = 7

    if model.hidden_function_family == "kan_quantum_hybrid":
        base_mix = [pnp.array(np.asarray(x, dtype=float), requires_grad=True) for x in params[next_offset]]
        quantum_mix = [pnp.array(np.asarray(x, dtype=float), requires_grad=True) for x in params[next_offset + 1]]
        activation_params.extend([*base_mix, *quantum_mix])

    model.set_parameters(*hidden_weights, *hidden_biases, w_out, b_out, *activation_params)


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

    key = jax.random.PRNGKey(config.seed)
    key, k_init = jax.random.split(key)
    params = _init_params(k_init, config)
    hidden_layers = config.resolved_hidden_layers()

    _tanh_pre = config.hidden_preactivation == "tanh"

    def loss_batch(p: Any, xb: Any, yb: Any) -> Any:
        logits = _forward(
            p,
            xb,
            config.mode,
            config.hidden_function_family,
            num_grid_points,
            x_grid,
            EPS,
            tanh_preactivation=_tanh_pre,
        )
        logp = logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)
        data_loss = -jnp.mean(jnp.sum(yb * logp, axis=1))
        reg_loss = config.profile_smoothness_reg * _smoothness_penalty(
            p,
            config.mode,
            config.hidden_function_family,
            len(hidden_layers),
            num_grid_points,
            EPS,
        )
        return data_loss + reg_loss

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

    num_batches = (n + bs - 1) // bs
    total_minibatch_updates = config.steps * num_batches
    batch_pbar = None
    if config.show_training_progress:
        try:
            from tqdm.auto import tqdm

            batch_pbar = tqdm(
                total=total_minibatch_updates,
                desc="Training (JAX)",
                unit="batch",
            )
        except ImportError:
            print(
                "Tip: pip install tqdm for a live minibatch progress bar.",
                flush=True,
            )

    rng = np.random.default_rng(config.seed)
    first_minibatch = True

    for step in range(config.steps):
        perm = rng.permutation(n)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            if idx.size == 0:
                break
            xb = x[idx]
            yb = y_eye[idx]
            if batch_pbar is not None and first_minibatch:
                batch_pbar.write(
                    "JAX: compiling train_step (first minibatch can take a while)…",
                )
                first_minibatch = False
            if idx.size == bs:
                params, opt_state, _ = train_step(params, opt_state, xb, yb)
            else:
                _, g = loss_grad(params, xb, yb)
                updates, opt_state = optimizer.update(g, opt_state)
                params = optax.apply_updates(params, updates)
            if batch_pbar is not None:
                batch_pbar.set_postfix(epoch=f"{step + 1}/{config.steps}", refresh=False)
                batch_pbar.update(1)

        loss_f = full_data_loss(params)
        losses_out.append(loss_f)
        _params_to_model(model, params)

        if after_step is not None:
            after_step(step, loss_f, model)
        if log_n and ((step + 1) % log_n == 0 or step == config.steps - 1):
            line = f"  epoch {step + 1}/{config.steps}  train_loss={loss_f:.6f}"
            if batch_pbar is not None:
                batch_pbar.write(line)
            else:
                print(line, flush=True)

    if batch_pbar is not None:
        batch_pbar.close()

    return model, np.asarray(losses_out, dtype=float)
