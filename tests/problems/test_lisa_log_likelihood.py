"""log_likelihood is the unnormalised Gaussian: the quadratic residual term only, so
it vanishes at the true parameters and its gap to the null model is exactly snr**2."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.problems import LisaGB

# a narrow band-limited config keeps the rfft/WDM grids tiny
SMALL = dict(
    n_sources=2,
    t_obs=1.0e6,
    sampling_step=0.25,
    wdm_freq_bands=32,
    patch_downsample=4,
    response_points=8,
    f0_range=(3.0e-3, 3.2e-3),
)


def test_log_likelihood_is_scalar():
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(0))
    o = problem.sample_observation(jr.key(0), p)
    assert problem.log_likelihood(p, o).shape == ()


def test_log_likelihood_is_finite():
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(1))
    o = problem.sample_observation(jr.key(1), p)
    assert jnp.isfinite(problem.log_likelihood(p, o))


def test_log_likelihood_is_zero_at_the_true_parameters():
    # unnormalised gaussian: at o = clean_signal(p) the residual vanishes and there is
    # no -D/2 log(2 pi sigma^2) constant, so the density is exactly 0.
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(2))
    ll = problem.log_likelihood(p, problem.clean_signal(p))
    assert jnp.allclose(ll, 0.0, atol=1e-6)


def test_log_likelihood_is_maximised_at_the_true_parameters():
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(3))
    o = problem.clean_signal(p)
    p_wrong = p.at[..., 2].multiply(0.5)
    assert problem.log_likelihood(p, o) > problem.log_likelihood(p_wrong, o)


def test_log_likelihood_preserves_leading_batch_axis():
    problem = LisaGB(**SMALL)
    keys = jr.split(jr.key(4), 3)
    ps = jax.vmap(problem.sample_physical)(keys)
    os_ = jax.vmap(lambda k, p: problem.sample_observation(k, p))(keys, ps)
    batched = problem.log_likelihood(ps, os_)
    looped = jnp.stack([problem.log_likelihood(ps[i], os_[i]) for i in range(3)])
    assert jnp.allclose(batched, looped, rtol=1e-5)


def test_log_likelihood_gap_to_the_null_model_is_snr_squared():
    # at o = clean_signal(p): ll(p, o) = 0, and ll(p_silent, o) = -sum|o|^2/power =
    # -snr(p)**2 because a zero-amplitude source radiates nothing. The gap pins the
    # matched-filter relationship independent of the shared power normalisation.
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(5))
    o = problem.clean_signal(p)
    p_silent = p.at[..., 2].set(0.0)
    gap = problem.log_likelihood(p, o) - problem.log_likelihood(p_silent, o)
    assert jnp.allclose(gap, problem.snr(p) ** 2, rtol=1e-4)
