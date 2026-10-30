"""Muon on the matrices, AdamW on the rest; linear warmup, a constant lr, then a 1 - sqrt cooldown to 0."""

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


def schedule(config: Config, peak: float) -> optax.Schedule:
    """Linear warmup to peak, constant, then 1 - sqrt to 0 over the last cooldown_frac of the run (Hägele+24).

    Without a cooldown the lr stays at peak past total_steps, so the run can be extended.
    """
    cooldown_steps = round(config.cooldown_frac * config.total_steps)
    cooldown_start = config.total_steps - cooldown_steps

    def lr(step):
        warm = jnp.minimum(step / max(config.warmup_steps, 1), 1.0)
        if not cooldown_steps:
            return peak * warm
        cooled = jnp.clip((step - cooldown_start) / cooldown_steps, 0.0, 1.0)
        return peak * warm * (1.0 - jnp.sqrt(cooled))

    return lr


def optimizer(config: Config) -> optax.GradientTransformation:
    """Muon with decoupled decay on its matrices (Moonlight), AdamW without decay on the rest (F48), on clipped gradients."""
    return optax.chain(
        optax.clip_by_global_norm(CLIP_NORM),
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
                    schedule(config, config.learning_rate),
                    beta=config.muon_momentum,
                    weight_decay=config.weight_decay,
                    mu_dtype=jnp.dtype(config.momentum_dtype),
                    nesterov=True,
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
                    schedule(config, config.adam_learning_rate),
                    b1=config.adam_b1,
                    b2=ADAM_B2,
                    eps=1e-8,
                    weight_decay=0.0,
                    mu_dtype=jnp.float32,
                ),
            ),
            lambda params: jax.tree_util.tree_map_with_path(
                lambda path, leaf: "muon" if is_muon(path, leaf) else "adam", params
            ),
        ),
    )


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
