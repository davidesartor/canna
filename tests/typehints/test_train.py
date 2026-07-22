"""Shape-signature contracts for the trainer's shape-bearing helpers."""

import jax
from flax import nnx

from canna.problems.point import NoisyPoint
from canna.train import sample_batch, aux_weight_schedule


def test_sample_batch_prepends_batch_axis():
    B, dim = 8, 3
    problem = NoisyPoint(dim=dim)
    batch = sample_batch(problem, nnx.Rngs(0), B)
    assert batch.xt.shape == (B, dim)
    assert batch.dx.shape == (B, dim)
    assert batch.t.shape == (B,)
    assert batch.y.shape == (B, dim)
    assert batch.x_target.shape == (B, dim)
    assert batch.y_target.shape == (B, dim)


def test_aux_weight_schedule_is_scalar_in_unit_interval():
    for step in [0, 5, 50, 100]:
        w = aux_weight_schedule(step, total_steps=100, warmup_frac=0.5)
        assert isinstance(w, float)
        assert 0.0 <= w <= 1.0
