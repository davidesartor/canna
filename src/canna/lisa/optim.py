"""Schedule-free NorMuon: Muon (+ NorMuon rows) on the matrices, AdamW on the rest, averaged weights x."""

from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .config import Config
from .network import LisaFlow

CLIP_NORM = 1.0
# Muon's update RMS per unit lr, the AdamW-matching scale
MUON_RMS = 0.2
ADAM_B2 = 0.95
# x follows z for this many warmups before averaging
WARM_START_WARMUPS = 2


class ScheduleFreeState(NamedTuple):
    step: jax.Array
    weight_sum: jax.Array
    base_state: optax.OptState
    z: optax.Params
    x: optax.Params


def interpolation(
    step: jax.Array,
    beta_start: float,
    beta_end: float,
    warm_start: int,
    total_steps: int,
) -> jax.Array:
    """beta at a step: beta_start until the warm start, then log-linear to beta_end."""
    t = jnp.maximum(step.astype(jnp.float32), 1.0)
    t0 = float(max(warm_start, 1))
    frac = jnp.log(jnp.maximum(t, t0) / t0) / jnp.log(
        max(total_steps, warm_start + 2) / t0
    )
    return beta_start + (beta_end - beta_start) * jnp.clip(frac, 0.0, 1.0)


def schedule_free(
    base: optax.GradientTransformation,
    learning_rate: Callable[[jax.Array], jax.Array],
    *,
    weight_decay: float = 0.0,
    decay_lr: Callable[[jax.Array, optax.Params], optax.Params] | None = None,
    beta_start: float = 0.9,
    beta_end: float = 0.9,
    total_steps: int = 1,
    warm_start: int = 0,
    r: float = 1.0,
) -> optax.GradientTransformation:
    """Schedule-free (Defazio et al. 2024) around `base`, decay on z (SF-NorMuon), x the t^r lr^2 mean after warm_start."""

    def init(params):
        return ScheduleFreeState(
            step=jnp.zeros((), jnp.int32),
            weight_sum=jnp.zeros((), jnp.float32),
            base_state=base.init(params),
            z=params,
            x=params,
        )

    def update(grads, state, params):
        step = state.step + 1
        lr = jnp.asarray(learning_rate(state.step), jnp.float32)
        base_updates, base_state = base.update(grads, state.base_state, params)

        # the base step, and the decay on z itself (matrices only), each at its own lr
        lrs = (
            decay_lr(state.step, params)
            if decay_lr is not None
            else jax.tree.map(lambda _: lr, params)
        )
        z = jax.tree.map(
            lambda zi, ui, li: (
                zi + ui - (li * weight_decay * zi if zi.ndim >= 2 else 0.0)
            ).astype(zi.dtype),
            state.z,
            base_updates,
            lrs,
        )

        # x follows z until the warm start, then a t^r lr^2 weighted mean of it
        warm = step <= warm_start
        weight = step.astype(jnp.float32) ** r * lr**2
        weight_sum = jnp.where(warm, 0.0, state.weight_sum + weight)
        c = jnp.where(warm, 1.0, weight / jnp.maximum(weight_sum, 1e-30))
        x = jax.tree.map(
            lambda xi, zi: ((1.0 - c) * xi + c * zi).astype(xi.dtype), state.x, z
        )

        beta = interpolation(step, beta_start, beta_end, warm_start, total_steps)
        y = jax.tree.map(lambda xi, zi: beta * xi + (1.0 - beta) * zi, x, z)
        updates = jax.tree.map(lambda yi, pi: (yi - pi).astype(pi.dtype), y, params)
        return updates, ScheduleFreeState(step, weight_sum, base_state, z, x)

    return optax.GradientTransformation(init, update)


# Muon takes the matrices of these fields (and the head's first layer), AdamW the rest
MUON_FIELDS = (
    "backbone",
    "c_mlp",
    "f_mlp",
    "f0_mlp",
    "y_mlp",
    "x_modulation",
    "frame_modulation",
)


def is_muon(path, leaf) -> bool:
    """Whether Muon takes this parameter: a Linear weight of a MUON_FIELDS module or of the head's hidden layer."""
    names = [k.name for k in path if isinstance(k, jax.tree_util.GetAttrKey)]
    hidden = names[0] in MUON_FIELDS or names[:2] == ["x_unembed", "linear1"]
    return names[-1] == "weight" and hidden


def normuon_rows(
    beta2: float = 0.95, eps: float = 1e-8
) -> optax.GradientTransformation:
    """NorMuon's per-row second-moment normalisation of Muon's update, keeping each matrix's norm."""

    def init(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, a: jnp.zeros(
                a.shape[:-1] if is_muon(path, a) else (), jnp.float32
            ),
            params,
        )

    def normalise(u, v):
        u32 = u.astype(jnp.float32)
        frob = lambda a: jnp.sqrt(jnp.sum(jnp.square(a), axis=(-2, -1), keepdims=True))
        unit = u32 / jnp.maximum(
            frob(u32) / jnp.sqrt(u32.shape[-2] * u32.shape[-1]), 1e-30
        )
        v = beta2 * v + (1.0 - beta2) * jnp.mean(jnp.square(unit), axis=-1)
        rows = unit / (jnp.sqrt(v)[..., None] + eps)
        out = rows * frob(u32) / jnp.maximum(frob(rows), 1e-30)
        return out.astype(u.dtype), v

    def update(updates, state, params=None):
        pairs = jax.tree.map(
            lambda muon, u, v: normalise(u, v) if muon else (u, v),
            jax.tree_util.tree_map_with_path(is_muon, updates),
            updates,
            state,
        )
        is_pair = lambda t: isinstance(t, tuple)
        return (
            jax.tree.map(lambda t: t[0], pairs, is_leaf=is_pair),
            jax.tree.map(lambda t: t[1], pairs, is_leaf=is_pair),
        )

    return optax.GradientTransformation(init, update)


def optimizer(config: Config) -> optax.GradientTransformation:
    """Schedule-free NorMuon, AdamW for what Muon does not take; only Muon's matrices decay (F48)."""
    warmup = lambda peak: optax.linear_schedule(0.0, peak, config.warmup_steps)
    # linear to 0 over the last cooldown_frac of the run, if any; x's averaging weights keep the uncooled lr
    cooldown_steps = round(config.cooldown_frac * config.total_steps)
    cooled = lambda peak: (
        (
            lambda step: warmup(peak)(step)
            * jnp.clip((config.total_steps - step) / cooldown_steps, 0.0, 1.0)
        )
        if cooldown_steps
        else warmup(peak)
    )
    muon_lr, adam_lr = cooled(config.learning_rate), cooled(config.adam_learning_rate)
    base = optax.chain(
        # Muon's gradients in its momentum dtype: with bf16, Newton-Schulz runs in bf16 as in the reference Muon
        optax.stateless(
            lambda updates, params: jax.tree_util.tree_map_with_path(
                lambda path, u: (
                    u.astype(jnp.dtype(config.momentum_dtype))
                    if is_muon(path, u)
                    else u
                ),
                updates,
            )
        ),
        optax.partition(
            dict(
                muon=optax.contrib.muon(
                    muon_lr,
                    beta=config.muon_momentum,
                    weight_decay=0.0,
                    mu_dtype=jnp.dtype(config.momentum_dtype),
                    nesterov=config.nesterov,
                    consistent_rms=MUON_RMS,
                    # Muon's matrix axes, (out, in) last; leading axes are batch axes
                    muon_weight_dimension_numbers=lambda params: jax.tree.map(
                        lambda leaf: optax.contrib.MuonDimensionNumbers(
                            reduction_axis=leaf.ndim - 1, output_axis=leaf.ndim - 2
                        ),
                        params,
                    ),
                ),
                # a float32 first moment whatever Muon's: bf16 drops a smooth gradient's small increments
                adam=optax.adamw(
                    adam_lr,
                    b1=config.adam_b1,
                    b2=ADAM_B2,
                    eps=1e-8,
                    weight_decay=0.0,
                    mu_dtype=jnp.float32,
                    nesterov=config.nesterov,
                ),
            ),
            lambda params: jax.tree_util.tree_map_with_path(
                lambda path, leaf: "muon" if is_muon(path, leaf) else "adam", params
            ),
        ),
        normuon_rows(),
    )

    return optax.chain(
        optax.clip_by_global_norm(CLIP_NORM),
        schedule_free(
            base,
            warmup(config.learning_rate),
            weight_decay=config.weight_decay,
            decay_lr=lambda step, params: jax.tree_util.tree_map_with_path(
                lambda path, a: muon_lr(step) * is_muon(path, a), params
            ),
            beta_start=config.sf_beta,
            beta_end=config.sf_beta_end,
            total_steps=config.total_steps,
            warm_start=WARM_START_WARMUPS * config.warmup_steps,
        ),
    )


def schedule_free_state(opt_state: optax.OptState) -> ScheduleFreeState:
    return next(s for s in opt_state if isinstance(s, ScheduleFreeState))


def eval_flow(flow: LisaFlow, opt_state: optax.OptState) -> LisaFlow:
    """The averaged weights x that schedule-free evaluates."""
    x = schedule_free_state(opt_state).x
    return eqx.combine(x, eqx.filter(flow, eqx.is_inexact_array, inverse=True))


def routing_report(flow: LisaFlow) -> str:
    """How many parameters Muon and AdamW each take; Muon must have the backbone but its qk gains."""
    params = eqx.filter(flow, eqx.is_inexact_array)
    split = lambda muon: jax.tree_util.tree_map_with_path(
        lambda path, a: a.size * (is_muon(path, a) == muon), params
    )
    muon, adam = (
        sum(jax.tree.leaves(split(True))),
        sum(jax.tree.leaves(split(False))),
    )
    stray = jax.tree_util.tree_map_with_path(
        lambda path, a: a.size * (not is_muon(path, a) and path[-1].name != "qk_gain"),
        params,
    )
    assert (
        sum(jax.tree.leaves(stray.backbone)) == 0
    ), "a backbone parameter is not on Muon"
    return (
        f"[optimizer] Muon {muon / 1e6:.2f}M of {(muon + adam) / 1e6:.2f}M parameters"
        f" ({muon / (muon + adam):.1%}), AdamW the other {adam / 1e6:.2f}M"
    )
