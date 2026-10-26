"""The (f0, fdot, A) prior: the catalogue's Gaussian in log10, truncated to the window in f0."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy import stats

from canna.lisa.physics import (
    FDOT_STD,
    SOURCE_COVARIANCE,
    SOURCE_MEAN,
    log10_fdot_mean,
    sample_sources,
    source_log_density,
)

jax.config.update("jax_enable_x64", True)

DRAWS = 200_000


def test_an_untruncating_window_draws_the_parent_gaussian():
    # a window 8 sigma either side of the mean truncates nothing
    sigma = np.sqrt(SOURCE_COVARIANCE[0, 0])
    lo, hi = 10.0 ** (SOURCE_MEAN[0] + np.array([-8, 8]) * sigma)
    x = np.log10(np.asarray(sample_sources(jr.key(0), lo, hi, DRAWS)))
    np.testing.assert_allclose(x.mean(0), SOURCE_MEAN, atol=0.01)
    np.testing.assert_allclose(np.cov(x.T), SOURCE_COVARIANCE, rtol=0.02, atol=2e-3)


def test_draws_stay_in_the_window():
    lo, hi = 1.0e-4, 2.0e-4
    f0, fdot, amp = np.asarray(sample_sources(jr.key(1), lo, hi, DRAWS)).T
    assert np.all((f0 >= lo) & (f0 <= hi))
    assert np.all(fdot > 0)
    assert np.all(amp > 0)


def test_the_conditionals_follow_the_regression_on_log_f0():
    # inside a window, (log fdot, log A) given f0 is the parent's conditional, whatever the window
    lo, hi = 2.0e-3, 4.0e-3
    x = np.log10(np.asarray(sample_sources(jr.key(2), lo, hi, DRAWS)))
    design = np.column_stack([np.ones(DRAWS), x[:, 0]])
    beta, *_ = np.linalg.lstsq(design, x[:, 1:], rcond=None)
    slope = SOURCE_COVARIANCE[0, 1:] / SOURCE_COVARIANCE[0, 0]
    np.testing.assert_allclose(beta[1], slope, atol=0.05)
    np.testing.assert_allclose((x[:, 1] - log10_fdot_mean(10 ** x[:, 0])).std(), FDOT_STD, rtol=0.01)


def test_the_window_density_is_the_truncated_marginal():
    # f0 at the top of the full band, 4.7 sigma out, still fills its window as the Gaussian says
    lo, hi = 11.9e-3, 12.0e-3
    log_f0 = np.log10(np.asarray(sample_sources(jr.key(3), lo, hi, DRAWS))[:, 0])
    mean, sigma = SOURCE_MEAN[0], np.sqrt(SOURCE_COVARIANCE[0, 0])
    a, b = (np.log10([lo, hi]) - mean) / sigma
    result = stats.kstest(log_f0, stats.truncnorm(a, b, loc=mean, scale=sigma).cdf)
    assert result.pvalue > 1e-3


def test_the_log_density_is_the_parent_gaussian():
    rng = np.random.default_rng(4)
    x = rng.multivariate_normal(SOURCE_MEAN, SOURCE_COVARIANCE, 64)
    log_p = np.asarray(source_log_density(jnp.asarray(x)))
    exact = stats.multivariate_normal(SOURCE_MEAN, SOURCE_COVARIANCE).logpdf(x)
    np.testing.assert_allclose(log_p - exact, (log_p - exact)[0], atol=1e-9)
