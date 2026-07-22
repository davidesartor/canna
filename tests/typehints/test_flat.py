"""Shape-signature contracts for the flat concretes (Normal, LogNormal, Affine, LogAffine, Euclidean)."""

import jax
import jax.numpy as jnp
import pytest

from canna.priors import Normal, LogNormal
from canna.charts import Affine, LogAffine
from canna.geometries import Euclidean


@pytest.mark.parametrize("Prior", [Normal, LogNormal])
@pytest.mark.parametrize("D", [1, 3])
def test_prior_call_returns_D(Prior, D):
    prior = Prior(mean=jnp.zeros(D), cov=jnp.ones(D))
    out = prior(jax.random.key(0))
    assert out.shape == (D,)


@pytest.mark.parametrize("Prior", [Normal, LogNormal])
def test_prior_call_full_cov_returns_D(Prior):
    D = 3
    prior = Prior(mean=jnp.zeros(D), cov=jnp.eye(D))
    out = prior(jax.random.key(0))
    assert out.shape == (D,)


@pytest.mark.parametrize(
    "Prior,Geo,Ch", [(Normal, Euclidean, Affine), (LogNormal, Euclidean, LogAffine)]
)
def test_prior_geometry_and_chart_types(Prior, Geo, Ch):
    prior = Prior(mean=jnp.zeros(2), cov=jnp.ones(2))
    assert isinstance(prior.geometry, Geo)
    assert isinstance(prior.chart, Ch)


@pytest.mark.parametrize("Ch", [Affine, LogAffine])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_chart_forward_backward_preserve_batch_D(Ch, batch_shape):
    D = 3
    chart = Ch(shift=jnp.zeros(D), scale=jnp.ones(D))
    p = jnp.abs(jax.random.normal(jax.random.key(0), batch_shape + (D,))) + 0.1
    x = chart.forward(p)
    assert x.shape == batch_shape + (D,)
    assert chart.backward(x).shape == batch_shape + (D,)


@pytest.mark.parametrize("Ch", [Affine, LogAffine])
@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_chart_full_scale_preserve_batch_D(Ch, batch_shape):
    D = 3
    chart = Ch(shift=jnp.zeros(D), scale=jnp.eye(D))
    p = jnp.abs(jax.random.normal(jax.random.key(1), batch_shape + (D,))) + 0.1
    assert chart.forward(p).shape == batch_shape + (D,)
    assert chart.backward(chart.forward(p)).shape == batch_shape + (D,)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_euclidean_maps_preserve_batch_D(batch_shape):
    D = 5
    geo = Euclidean()
    k0, k1 = jax.random.split(jax.random.key(0))
    x0 = jax.random.normal(k0, batch_shape + (D,))
    x1 = jax.random.normal(k1, batch_shape + (D,))
    assert geo.log_map(x0, x1).shape == batch_shape + (D,)
    assert geo.exp_map(x0, geo.log_map(x0, x1)).shape == batch_shape + (D,)
    assert geo.geodesic(jnp.array(0.3), x0, x1).shape == batch_shape + (D,)
