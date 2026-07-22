"""update_metrics folds a batch into running Welford stats, returning the 3 target variances."""

import jax.numpy as jnp
from flax import nnx

from canna.train import TrainSample, sample_batch


def test_variance_shape_is_three(tiny_state, fake_problem):
    batch = sample_batch(fake_problem, nnx.Rngs(1), 16)
    variances = tiny_state.update_metrics(batch)
    assert variances.shape == (3,)


def test_constant_batch_has_zero_variance(tiny_state):
    """every target constant across the batch => all three variances exactly 0,
    regardless of which of (dx, x_target, y_target) maps to which of (flow, x, y) metrics
    """
    n = 8
    const_batch = TrainSample(
        xt=jnp.zeros((n, 3)),
        dx=jnp.full((n, 3), 2.0),
        t=jnp.full((n,), 0.5),
        y=jnp.zeros((n, 2)),
        x_target=jnp.full((n, 3), 3.0),
        y_target=jnp.full((n, 2), -1.0),
    )
    variances = tiny_state.update_metrics(const_batch)
    assert jnp.allclose(variances, 0.0, atol=1e-6)


def test_variances_match_manual_batch_variance(tiny_state):
    """flow_metrics/x_metrics/y_metrics track dx/x_target/y_target respectively"""
    n = 32
    batch = TrainSample(
        xt=jnp.zeros((n, 3)),
        dx=jnp.arange(n * 3, dtype=jnp.float32).reshape(n, 3),
        t=jnp.full((n,), 0.5),
        y=jnp.zeros((n, 2)),
        x_target=jnp.arange(n * 3, dtype=jnp.float32).reshape(n, 3) * 2.0,
        y_target=jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2) * -1.0,
    )
    variances = tiny_state.update_metrics(batch)
    assert variances[0] == pytest_approx(jnp.var(batch.dx))
    assert variances[1] == pytest_approx(jnp.var(batch.x_target))
    assert variances[2] == pytest_approx(jnp.var(batch.y_target))


def pytest_approx(x, rel=1e-3):
    import pytest

    return pytest.approx(float(x), rel=rel)


def test_metrics_accumulate_across_calls(tiny_state):
    """running stats must accumulate, not reset each call -- two constant-but-different
    batches combined must show nonzero variance even though each batch alone has zero"""
    n = 4

    def make(dx_val):
        return TrainSample(
            xt=jnp.zeros((n, 3)),
            dx=jnp.full((n, 3), dx_val),
            t=jnp.full((n,), 0.5),
            y=jnp.zeros((n, 2)),
            x_target=jnp.zeros((n, 3)),
            y_target=jnp.zeros((n, 2)),
        )

    tiny_state.update_metrics(make(-1.0))
    variances = tiny_state.update_metrics(make(1.0))
    assert variances[0] > 0.0


def test_mutation_persists_across_jit_calls(real_state):
    """the same Welford objects, nested inside a NamedTuple, must accumulate in place
    across two separate @nnx.jit-traced calls -- not reset or silently discarded"""
    batch1 = sample_batch(real_state.problem, real_state.rngs, 4)
    real_state.update_metrics(batch1)
    count_after_first = int(real_state.flow_metrics.count[...])

    batch2 = sample_batch(real_state.problem, real_state.rngs, 4)
    real_state.update_metrics(batch2)
    count_after_second = int(real_state.flow_metrics.count[...])

    assert count_after_first > 0
    assert count_after_second == count_after_first + batch2.dx.size
