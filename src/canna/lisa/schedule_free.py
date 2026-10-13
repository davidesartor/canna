"""Schedule-free averaging with the fixes of SF-NorMuon and ScheduleFree+ (research log F37).

The method of Defazio et al. 2024 ("The Road Less Scheduled", arXiv:2405.15682): a fast
sequence z driven by a base optimizer, its weighted average x (the model to evaluate), and
the gradient taken at y = (1 - beta) z + beta x. optax.contrib.schedule_free implements it;
this module replaces it, because XS-fd showed its failure mode with Muon (F36): z ran away
(loss 2.7e21) and contaminated x. The changes:

- Weight decay on z, as part of the base dynamics: z <- z - eta lambda z + step. SF-NorMuon
  (Apte et al. 2026, arXiv:2605.23061) finds that "the validation loss at X diverges without
  weight decaying specifically at Z", and that decay at y -- what optax does, since it hands
  y to the base optimizer -- is unstable as well. Decay here uses each parameter's own lr
  (Muon's or AdamW's), and only touches matrices and embedding tables.
- Averaging weights proportional to t^r lr_t^p with r = 1. The default r = 0 is uniform
  after the warmup and "underweights later iterates" in long runs (ScheduleFree+, Defazio
  2026, arXiv:2605.19095); on XS-fd it left x carrying the iterates from before the late
  loss drop (F33).
- Warm-started averaging: x follows z for the first `warm_start` steps (~2x the warmup),
  while the weights change fastest, and the averaging starts after (ScheduleFree+).
- The interpolation beta annealed log-linearly in t, from beta_start to beta_end over
  `total_steps` (ScheduleFree+: 0.9 to 0.965), trading early speed for late stability (as
  AMUSE, arXiv:2605.22432, also does).
- x is stored, not rebuilt from y and z as optax does, which a varying beta would break
  (and which costs some float32 accuracy late in a long run).

The base optimizer must carry its own momentum (ScheduleFree+'s inner momentum; SF-NorMuon's
ablation loses 0.14 nats without it) and no weight decay of its own.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax


class ScheduleFreeState(NamedTuple):
    step: jax.Array  # steps taken, int32
    weight_sum: jax.Array  # sum of the averaging weights since the warm start
    base_state: optax.OptState
    z: optax.Params
    x: optax.Params


def interpolation(
    step: jax.Array, beta_start: float, beta_end: float, warm_start: int, total_steps: int
) -> jax.Array:
    """beta at a step: beta_start until the warm start, then log-linear to beta_end."""
    t = jnp.maximum(step.astype(jnp.float32), 1.0)
    t0 = float(max(warm_start, 1))
    frac = jnp.log(jnp.maximum(t, t0) / t0) / jnp.log(max(total_steps, warm_start + 2) / t0)
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
    weight_lr_power: float = 2.0,
) -> optax.GradientTransformation:
    """Wrap `base` (which must not decay the weights itself) into schedule-free training.

    `learning_rate` is the schedule that weights the average (lr^weight_lr_power).
    `decay_lr(step, params)` gives, per parameter, the lr its decay is scaled by -- each group's own;
    it defaults to `learning_rate` for every parameter. Decay applies to arrays of 2 or more
    dimensions only.
    """

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

        # the base step, and the decay on z itself (SF-NorMuon), each at its own lr
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

        # the average: x follows z until the warm start, then a t^r lr^p weighted mean
        warm = step <= warm_start
        weight = step.astype(jnp.float32) ** r * lr**weight_lr_power
        weight_sum = jnp.where(warm, 0.0, state.weight_sum + weight)
        c = jnp.where(warm, 1.0, weight / jnp.maximum(weight_sum, 1e-30))
        x = jax.tree.map(lambda xi, zi: ((1.0 - c) * xi + c * zi).astype(xi.dtype), state.x, z)

        beta = interpolation(step, beta_start, beta_end, warm_start, total_steps)
        y = jax.tree.map(lambda xi, zi: beta * xi + (1.0 - beta) * zi, x, z)
        updates = jax.tree.map(lambda yi, pi: (yi - pi).astype(pi.dtype), y, params)
        return updates, ScheduleFreeState(step, weight_sum, base_state, z, x)

    return optax.GradientTransformation(init, update)
