"""Shape-signature contracts for the angular concretes (Cosine, Sine, Reflected)."""

import jax
import jax.numpy as jnp
import pytest

from canna.priors import Cosine, Sine
from canna.geometries import Reflected, Bounded


@pytest.mark.parametrize("D", [1, 3])
def test_cosine_call_returns_D(D):
    prior = Cosine(dim=D)
    assert prior(jax.random.key(0)).shape == (D,)
    assert isinstance(prior.geometry, Bounded)


@pytest.mark.parametrize("D", [1, 3])
def test_sine_call_returns_D(D):
    prior = Sine(dim=D)
    assert prior(jax.random.key(0)).shape == (D,)
    assert isinstance(prior.geometry, Reflected)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_reflected_maps_preserve_batch_D(batch_shape):
    D = 5
    geo = Reflected()
    k0, k1 = jax.random.split(jax.random.key(0))
    x0 = jax.random.uniform(k0, batch_shape + (D,), minval=-1.0, maxval=1.0)
    x1 = jax.random.uniform(k1, batch_shape + (D,), minval=-1.0, maxval=1.0)
    assert geo.log_map(x0, x1).shape == batch_shape + (D,)
    assert geo.exp_map(x0, geo.log_map(x0, x1)).shape == batch_shape + (D,)
    assert geo.geodesic(jnp.array(0.3), x0, x1).shape == batch_shape + (D,)
