"""The template residual and the Fisher loss metric (train.with_residual, train.fisher_metric)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa import LisaGB
from canna.lisa.train import fisher_metric, train_sample, with_residual

XS = dict(n_sources=4, t_obs=63115200.0, response_points=16, f0_range=(1.0e-4, 12.0e-3))


def test_residual_at_the_truth_is_the_whitened_noise():
    problem = LisaGB(**XS)
    f = problem.sample_f(jr.key(0))
    p = problem.sample_physical(jr.key(1), f)
    o = problem.sample_observation(jr.key(2), p, f)
    y = problem.preprocess(o, f)
    aug = with_residual(problem, problem.physical_to_flow(p, f), y, f)
    noise = problem.preprocess(o - problem.clean_signal(p, f), f)
    assert aug.shape == (problem.window_bins, 12)
    assert jnp.allclose(aug[:, :6], y)
    assert jnp.allclose(aug[:, 6:], noise, atol=1e-6 * float(jnp.abs(y).max()))


def test_fisher_block_is_the_likelihood_hessian():
    problem = LisaGB(**XS)
    f = problem.sample_f(jr.key(3))
    p = problem.sample_physical(jr.key(4), f)
    x = problem.physical_to_flow(p, f)
    clean = problem.preprocess(problem.clean_signal(p, f), f)
    # at s = 0 and no clip, W = I + G
    g = fisher_metric(problem, x, jnp.asarray(0.0), f, jnp.inf) - jnp.eye(11)

    k = 1  # -log L = |y - h(x)|^2 / 2 in whitened units, noise-free data: its Hessian is J^T J
    def nll(x_k):
        h = problem.preprocess(problem.clean_signal(problem.flow_to_physical(x.at[k].set(x_k), f), f), f)
        return 0.5 * jnp.sum(jnp.square(clean - h))

    hess = jax.hessian(nll)(x[k])
    scale = jnp.sqrt(jnp.outer(jnp.diag(hess), jnp.diag(hess))) + 1e-30
    assert jnp.allclose(g[k] / scale, hess / scale, atol=1e-6)


def test_metric_limits_and_shape():
    problem = LisaGB(**XS)
    key = jr.key(5)
    sample = train_sample(problem, key, 3.0, fisher_cap=100.0)
    w = sample.metric
    assert w.shape == (4, 11, 11)
    assert jnp.allclose(w, jnp.swapaxes(w, -1, -2))
    lam = jnp.linalg.eigvalsh(w)
    assert bool(jnp.all(lam >= 0.5 - 1e-9)) and bool(jnp.all(lam <= 101.0 + 1e-6))
    # at the end of the path the metric is the identity
    x = problem.sample_flow(key, problem.sample_f(key))
    assert jnp.allclose(fisher_metric(problem, x, jnp.asarray(1.0), problem.sample_f(key), 100.0), jnp.eye(11))


def test_template_sample_is_the_plain_one_plus_the_residual():
    problem = LisaGB(**XS)
    plain = train_sample(problem, jr.key(6), 3.0)
    aug = train_sample(problem, jr.key(6), 3.0, templates=True)
    assert plain.metric is None and aug.y.shape[-1] == 12
    assert jnp.allclose(aug.y[:, :6], plain.y, rtol=1e-12, atol=1e-12) and jnp.allclose(aug.xt, plain.xt, rtol=1e-12, atol=1e-12)
