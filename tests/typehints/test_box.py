"""Shape-signature contracts for the box concretes (Uniform, LogUniform, Bounded, Squash)."""

import jax
import jax.numpy as jnp
import pytest

from canna.priors import Uniform, LogUniform
from canna.charts import Squash
from canna.geometries import Bounded


@pytest.mark.parametrize("D", [1, 3])
def test_uniform_call_returns_D(D):
    prior = Uniform(low=jnp.zeros(D), high=jnp.ones(D))
    assert prior(jax.random.key(0)).shape == (D,)
    assert isinstance(prior.geometry, Bounded)


@pytest.mark.parametrize("D", [1, 3])
def test_loguniform_call_returns_D(D):
    prior = LogUniform(low=jnp.ones(D), high=jnp.full((D,), 10.0))
    assert prior(jax.random.key(0)).shape == (D,)
    assert isinstance(prior.geometry, Bounded)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_squash_forward_backward_preserve_batch_D(batch_shape):
    D = 3
    chart = Squash(low=jnp.zeros(D), high=jnp.ones(D))
    p = jax.random.uniform(
        jax.random.key(0), batch_shape + (D,), minval=0.1, maxval=0.9
    )
    x = chart.forward(p)
    assert x.shape == batch_shape + (D,)
    assert chart.backward(x).shape == batch_shape + (D,)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_bounded_maps_preserve_batch_D(batch_shape):
    D = 5
    geo = Bounded()
    k0, k1 = jax.random.split(jax.random.key(0))
    x0 = jax.random.uniform(k0, batch_shape + (D,), minval=-1.0, maxval=1.0)
    x1 = jax.random.uniform(k1, batch_shape + (D,), minval=-1.0, maxval=1.0)
    assert geo.log_map(x0, x1).shape == batch_shape + (D,)
    assert geo.exp_map(x0, geo.log_map(x0, x1)).shape == batch_shape + (D,)
    assert geo.geodesic(jnp.array(0.3), x0, x1).shape == batch_shape + (D,)
