"""The lr cooldown: its schedule, and that lr_scale scales the optimizer step exactly."""

import jax.numpy as jnp
import numpy as np

from canna.lisa.train import TrainState, lr_scale_schedule

from .test_checkpoint import params, small_args


def test_the_schedule_holds_then_ramps_linearly_to_zero():
    scales = [lr_scale_schedule(e, 10, 4) for e in range(10)]
    np.testing.assert_allclose(scales, [1] * 6 + [3.5 / 4, 2.5 / 4, 1.5 / 4, 0.5 / 4])
    assert np.mean(scales[6:]) == 0.5
    assert all(lr_scale_schedule(e, 10, 0) == 1.0 for e in range(10))


def test_lr_scale_multiplies_the_step():
    # the same state and key draw the same batch, so the gradients agree and only the
    # scale differs between the three steps
    state = TrainState.from_config(small_args())
    step = lambda scale: params(
        state.train_epoch(jnp.asarray(1.0), 2, 1, 1.0, jnp.asarray(scale))[0]
    )
    before, full, half, none = params(state), step(1.0), step(0.5), step(0.0)

    assert all(jnp.array_equal(a, b) for a, b in zip(none, before))
    moved = [np.asarray(f - b) for f, b in zip(full, before)]
    assert max(np.abs(m).max() for m in moved) > 0
    for h, b, m in zip(half, before, moved):
        np.testing.assert_allclose(np.asarray(h - b), 0.5 * m, rtol=1e-2, atol=1e-6)
