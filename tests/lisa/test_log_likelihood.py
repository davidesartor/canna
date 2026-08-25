"""log_likelihood is the unnormalised Gaussian: the quadratic residual term only, so
it vanishes at the true parameters and its gap to the null model is exactly snr**2."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa import LisaGB
from ._helpers import window


# a narrow band-limited config keeps the rfft/WDM grids tiny
SMALL = dict(
    n_sources=2,
    t_obs=1.0e6,
    sampling_step=0.25,
    wdm_freq_bands=64,
    patch_downsample=4,
    f0_range=(3.0e-3, 3.2e-3),
)


def test_log_likelihood_is_scalar():
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(0), window(problem))
    o = problem.sample_observation(jr.key(0), p, window(problem))
    assert problem.log_likelihood(p, o, window(problem)).shape == ()


def test_log_likelihood_is_finite():
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(1), window(problem))
    o = problem.sample_observation(jr.key(1), p, window(problem))
    assert jnp.isfinite(problem.log_likelihood(p, o, window(problem)))


def test_log_likelihood_is_zero_at_the_true_parameters():
    # unnormalised gaussian: at o = clean_signal(p) the residual vanishes and there is
    # no -D/2 log(2 pi sigma^2) constant, so the density is exactly 0.
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(2), window(problem))
    ll = problem.log_likelihood(p, problem.clean_signal(p, window(problem)), window(problem))
    assert jnp.allclose(ll, 0.0, atol=1e-6)


def test_log_likelihood_is_maximised_at_the_true_parameters():
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(3), window(problem))
    o = problem.clean_signal(p, window(problem))
    p_wrong = p.at[..., 2].multiply(0.5)
    assert problem.log_likelihood(p, o, window(problem)) > problem.log_likelihood(p_wrong, o, window(problem))


def test_log_likelihood_preserves_leading_batch_axis():
    problem = LisaGB(**SMALL)
    keys = jr.split(jr.key(4), 3)
    ps = jax.vmap(problem.sample_physical, in_axes=(0, None))(keys, window(problem))
    os_ = jax.vmap(lambda k, p: problem.sample_observation(k, p, window(problem)))(keys, ps)
    batched = problem.log_likelihood(ps, os_, window(problem))
    looped = jnp.stack([problem.log_likelihood(ps[i], os_[i], window(problem)) for i in range(3)])
    assert jnp.allclose(batched, looped, rtol=1e-5)


def test_log_likelihood_gap_to_the_null_model_is_snr_squared():
    # at o = clean_signal(p): ll(p, o) = 0, and ll(p_silent, o) = -sum|o|^2/power because
    # a zero-amplitude source radiates nothing. So the gap is <h|h>/2 and the canonical
    # rho^2 = 2 [ll(h) - ll(0)] pins snr against the shared power normalisation.
    problem = LisaGB(**SMALL)
    p = problem.sample_physical(jr.key(5), window(problem))
    o = problem.clean_signal(p, window(problem))
    p_silent = p.at[..., 2].set(0.0)
    gap = problem.log_likelihood(p, o, window(problem)) - problem.log_likelihood(p_silent, o, window(problem))
    assert jnp.allclose(2.0 * gap, problem.snr(p, window(problem)) ** 2, rtol=1e-4)
