"""LogNormal: exp(N(mean,cov)) draws, .chart whitens in log-space."""

import jax
import jax.numpy as jnp

from canna.geometries import Euclidean
from canna.charts import LogAffine
from canna.priors import LogNormal


def test_default_call_shape():
    prior = LogNormal()
    out = prior(jax.random.key(0))
    assert out.shape == (1,)


def test_call_shape_matches_mean_dim():
    prior = LogNormal(mean=jnp.array([0.0, 1.0, -1.0]), cov=jnp.eye(3))
    out = prior(jax.random.key(0))
    assert out.shape == (3,)


def test_call_strictly_positive():
    prior = LogNormal(mean=jnp.array([0.0, -5.0, 5.0]), cov=jnp.array([1.0, 4.0, 9.0]))
    keys = jax.random.split(jax.random.key(9), 200)
    draws = jax.vmap(prior)(keys)
    assert jnp.all(draws > 0.0)


def test_call_deterministic_same_key():
    prior = LogNormal(mean=jnp.array([1.0, 2.0]), cov=jnp.array([1.0, 2.0]))
    key = jax.random.key(42)
    a = prior(key)
    b = prior(key)
    assert jnp.array_equal(a, b)


def test_call_differs_across_keys():
    prior = LogNormal(mean=jnp.array([1.0, 2.0]), cov=jnp.array([1.0, 2.0]))
    a = prior(jax.random.key(1))
    b = prior(jax.random.key(2))
    assert not jnp.array_equal(a, b)


def test_geometry_is_euclidean_instance():
    prior = LogNormal(mean=jnp.array([0.0, 0.0]), cov=jnp.eye(2))
    assert isinstance(prior.geometry, Euclidean)


def test_chart_is_logaffine_instance():
    prior = LogNormal(mean=jnp.array([0.0, 0.0]), cov=jnp.eye(2))
    assert isinstance(prior.chart, LogAffine)


def test_chart_scale_squared_equals_inverse_diagonal_cov():
    cov = jnp.array([4.0, 9.0, 0.25])
    prior = LogNormal(mean=jnp.zeros(3), cov=cov)
    scale = prior.chart.scale
    assert jnp.allclose(scale**2, 1.0 / cov, atol=1e-5)


def test_chart_scale_gram_equals_inverse_full_cov():
    A = jnp.array([[2.0, 0.0], [0.5, 1.5]])
    cov = A @ A.T
    prior = LogNormal(mean=jnp.zeros(2), cov=cov)
    scale = prior.chart.scale
    assert jnp.allclose(scale.T @ scale, jnp.linalg.inv(cov), atol=1e-4)


def test_chart_scale_shift_diag_branch_matches_vector_branch():
    mean = jnp.array([0.0, 1.0, -1.0])
    diag = jnp.array([4.0, 1.0, 9.0])
    prior_1d = LogNormal(mean=mean, cov=diag)
    prior_full = LogNormal(mean=mean, cov=jnp.diag(diag))
    assert jnp.allclose(
        jnp.diag(prior_full.chart.scale), prior_1d.chart.scale, atol=1e-5
    )
    assert jnp.allclose(prior_full.chart.shift, prior_1d.chart.shift, atol=1e-5)


def test_full_cov_chart_forward_on_stacked_draws_shape():
    D = 5
    mean = jnp.zeros(D)
    cov = jnp.eye(D) * 2.0 + 0.1
    prior = LogNormal(mean=mean, cov=cov)
    keys = jax.random.split(jax.random.key(0), 7)
    draws = jax.vmap(prior)(keys)
    whitened = prior.chart.forward(draws)
    assert whitened.shape == draws.shape


def test_diag_cov_1d_vs_full_matrix_agree_on_draws():
    mean = jnp.array([0.0, 1.0, -1.0])
    diag = jnp.array([4.0, 1.0, 9.0])
    prior_1d = LogNormal(mean=mean, cov=diag)
    prior_full = LogNormal(mean=mean, cov=jnp.diag(diag))
    key = jax.random.key(7)
    assert jnp.allclose(prior_1d(key), prior_full(key), atol=1e-4)


def test_chart_forward_backward_roundtrip_on_draw():
    prior = LogNormal(mean=jnp.array([1.0, -1.0]), cov=jnp.array([2.0, 5.0]))
    draw = prior(jax.random.key(3))
    whitened = prior.chart.forward(draw)
    back = prior.chart.backward(whitened)
    assert jnp.allclose(back, draw, atol=1e-4)


def test_whitened_draws_are_standard_normal_diag_cov():
    D = 3
    mean = jnp.array([0.0, -1.0, 0.5])
    cov = jnp.array([1.0, 4.0, 0.25])
    prior = LogNormal(mean=mean, cov=cov)
    keys = jax.random.split(jax.random.key(123), 4000)
    draws = jax.vmap(prior)(keys)
    whitened = jax.vmap(prior.chart.forward)(draws)
    sample_mean = jnp.mean(whitened, axis=0)
    sample_cov = jnp.cov(whitened.T)
    assert jnp.allclose(sample_mean, jnp.zeros(D), atol=0.1)
    assert jnp.allclose(sample_cov, jnp.eye(D), atol=0.15)


def test_log_of_draws_distributed_as_underlying_normal():
    D = 2
    mean = jnp.array([0.5, -1.0])
    cov = jnp.array([1.0, 2.0])
    log_prior = LogNormal(mean=mean, cov=cov)
    keys = jax.random.split(jax.random.key(999), 4000)
    log_draws = jnp.log(jax.vmap(log_prior)(keys))
    sample_mean = jnp.mean(log_draws, axis=0)
    sample_var = jnp.var(log_draws, axis=0)
    assert jnp.allclose(sample_mean, mean, atol=0.15)
    assert jnp.allclose(sample_var, cov, atol=0.3)
