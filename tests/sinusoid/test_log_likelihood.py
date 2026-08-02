"""NoisySinusoid.log_likelihood: the unnormalized Gaussian log-density (quadratic
term only) and its consistency with clean_signal/sample_observation and snr."""

import math

import jax.numpy as jnp
import jax.random as jr

from canna.sinusoid import NoisySinusoid

# --- shape / broadcast ---


def test_log_likelihood_unbatched_scalar():
    problem = NoisySinusoid(n_sources=1, t_obs=64.0)
    p = jnp.array([[0.4, 0.03, 0.6]])
    assert problem.log_likelihood(p, problem.clean_signal(p)).shape == ()


def test_log_likelihood_batched_p_unbatched_o():
    problem = NoisySinusoid(n_sources=1, t_obs=64.0)
    p = jnp.ones((5, 1, 3)) * 0.3
    o = problem.clean_signal(p[0])
    assert problem.log_likelihood(p, o).shape == (5,)


def test_log_likelihood_batched_o_unbatched_p():
    problem = NoisySinusoid(n_sources=1, t_obs=64.0)
    p = jnp.array([[0.4, 0.03, 0.6]])
    o = jnp.ones((5,) + problem.clean_signal(p).shape) * 0.1
    assert problem.log_likelihood(p, o).shape == (5,)


def test_log_likelihood_double_batched():
    problem = NoisySinusoid(n_sources=1, t_obs=64.0)
    p = jnp.ones((3, 5, 1, 3)) * 0.3
    assert problem.log_likelihood(p, problem.clean_signal(p)).shape == (3, 5)


def test_log_likelihood_permutation_invariance():
    problem = NoisySinusoid(n_sources=3, t_obs=64.0)
    p = jr.uniform(jr.key(2), (3, 3), minval=0.05, maxval=0.5)
    permuted = p[jnp.array([2, 0, 1])]
    o = problem.clean_signal(p) + 0.05
    assert jnp.isclose(
        problem.log_likelihood(p, o), problem.log_likelihood(permuted, o), rtol=1e-6
    )


# --- maximized at truth, decreasing with residual ---


def test_log_likelihood_zero_at_exact_match():
    problem = NoisySinusoid(n_sources=2, t_obs=64.0)
    p = jnp.array([[0.4, 0.03, 0.6], [0.2, 0.07, 1.9]])
    assert jnp.isclose(
        problem.log_likelihood(p, problem.clean_signal(p)), 0.0, atol=1e-4
    )


def test_log_likelihood_monotonically_decreasing_in_residual_magnitude():
    problem = NoisySinusoid(n_sources=1, t_obs=64.0)
    p = jnp.array([[0.4, 0.03, 0.6]])
    signal = problem.clean_signal(p)
    assert problem.log_likelihood(p, signal + 0.05) > problem.log_likelihood(
        p, signal + 0.2
    )


# --- exact Gaussian value ---


def test_log_likelihood_exact_value_constant_residual():
    """T=64, variance=noise_level/(2*sampling_step)=0.2; constant residual 0.1 on every
    (t, channel): sum(residual**2)=64*2*0.01=1.28, log_likelihood=-0.5*1.28/0.2=-3.2."""
    problem = NoisySinusoid(n_sources=1, t_obs=64.0, sampling_step=1.0, noise_level=0.4)
    p = jnp.array([[0.3, 0.02, 0.1]])
    o = problem.clean_signal(p) + 0.1
    assert jnp.isclose(problem.log_likelihood(p, o), -3.2, rtol=1e-5)


def test_log_likelihood_matches_hand_recomputed_variance_on_real_noise_draw():
    problem = NoisySinusoid(n_sources=1, t_obs=64.0, sampling_step=1.0, noise_level=0.4)
    p = jnp.array([[0.3, 0.02, 0.1]])
    o = problem.sample_observation(jr.key(11), p)
    residual = o - problem.clean_signal(p)
    expected = -0.5 * jnp.sum(residual**2) / (0.4 / 2.0)
    assert jnp.isclose(problem.log_likelihood(p, o), expected, rtol=1e-5)


# --- relation to snr ---


def test_log_likelihood_energy_matches_snr_squared_when_t_obs_divisible():
    problem = NoisySinusoid(n_sources=1, t_obs=64.0, sampling_step=1.0, noise_level=0.4)
    p = jnp.array([[0.4, 0.03, 0.6]])
    variance = problem.noise_level / (2.0 * problem.sampling_step)
    energy_over_variance = jnp.sum(problem.clean_signal(p) ** 2) / variance
    assert jnp.isclose(energy_over_variance, problem.snr(p) ** 2, rtol=1e-5)


def test_signal_energy_diverges_from_snr_squared_when_t_obs_not_divisible():
    """snr uses continuous t_obs while clean_signal grids to T=ceil(t_obs/sampling_step);
    the ratio is T*sampling_step/t_obs, only 1 when t_obs is a multiple of sampling_step.
    """
    problem = NoisySinusoid(n_sources=1, t_obs=1.5, sampling_step=1.0, noise_level=0.4)
    p = jnp.array([[0.4, 0.03, 0.6]])
    assert (
        math.ceil(problem.t_obs / problem.sampling_step) * problem.sampling_step
        != problem.t_obs
    )
    variance = problem.noise_level / (2.0 * problem.sampling_step)
    ratio = jnp.sum(problem.clean_signal(p) ** 2) / variance / problem.snr(p) ** 2
    assert not jnp.isclose(ratio, 1.0, rtol=0.1)
